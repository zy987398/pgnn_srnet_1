"""
工具函数和数据预处理
包含数据准备、增强和辅助函数
"""

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from scipy.stats.mstats import winsorize
from typing import Optional

# 全局常量
GRAD_CLIP = 5.0


def prepare_material_data(
    df,
    scaler: Optional[RobustScaler] = None,
    le: Optional[LabelEncoder] = None,
    winsorize_limits=(0.02, 0.02),  # 2%-98%分位数裁剪
    verbose: bool = True,
):
    """
    准备材料数据，包括预处理和特征工程
    
    参数:
        df: 原始数据DataFrame
        scaler: 特征标准化器（如果为None则创建新的）
        le: 标签编码器（如果为None则创建新的）
        winsorize_limits: Winsorize裁剪限制
        verbose: 是否打印处理信息
    
    返回:
        包含处理后数据的字典
    """
    numeric_cols = [
        "莫氏硬度", "弹性模量", "断裂韧性", 
        "晶粒粒径", "压溃压力", "磨粒速度"
    ]
    
    df_clean = df.copy()
    
    # 1. 处理缺失值
    for col in numeric_cols + ["亚表面裂纹深度"]:
        if df_clean[col].isna().any():
            if verbose:
                print(f"Warning: Found NaN values in {col}, filling with median")
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    # 2. Winsorize处理极端值
    for col in numeric_cols:
        original_values = df_clean[col].values
        winsorized_values = winsorize(original_values, limits=winsorize_limits)
        outlier_count = np.sum(original_values != winsorized_values)
        if outlier_count > 0:
            if verbose:
                print(f"Winsorized {outlier_count} outliers in {col}")
            df_clean[col] = winsorized_values
    
    # 3. 对数变换处理长尾分布
    log_transform_cols = ["弹性模量", "断裂韧性", "晶粒粒径"]
    df_log = df_clean.copy()
    
    for col in log_transform_cols:
        if f"{col}_logged" not in df.columns:
            min_val = df_clean[col].min()
            if min_val <= 0:
                df_log[col] = np.log1p(df_clean[col] - min_val + 1e-6)
            else:
                df_log[col] = np.log1p(df_clean[col])
            df_log[f"{col}_logged"] = True
            if verbose:
                print(f"Applied log1p transform to {col}")
    
    # 4. 编码材料种类
    if le is None:
        le = LabelEncoder()
        material_ids = le.fit_transform(df_clean["材料种类"])
    else:
        material_ids = le.transform(df_clean["材料种类"])
    
    # 5. 提取特征
    # 原始特征（用于物理约束）
    numeric_raw = df_clean[numeric_cols].values.astype(np.float32)
    
    # 用于主网络的特征（使用对数变换后的版本）
    numeric_for_scaling = df_log[numeric_cols].values.astype(np.float32)
    
    # 6. RobustScaler标准化
    if scaler is None:
        scaler = RobustScaler(quantile_range=(5.0, 95.0))
        scaler.fit(numeric_for_scaling)
    
    numeric_scaled = scaler.transform(numeric_for_scaling)
    
    # 7. 最终安全检查
    numeric_scaled = np.clip(numeric_scaled, -10, 10)
    
    target = df_clean["亚表面裂纹深度"].values.astype(np.float32)
    
    return {
        "material_ids": material_ids,
        "numeric_features": numeric_scaled,  # 标准化后的特征（主网络）
        "numeric_raw": numeric_raw,          # 原始特征（物理约束）
        "target": target,
        "label_encoder": le,
        "scaler": scaler,
        "num_materials": len(le.classes_),
    }


def smart_augment(df: pd.DataFrame, model_predictions=None):
    """
    智能数据增强，使用mixup策略
    优先增强预测误差大的样本
    """
    df = df.copy()
    features = df[[
        "莫氏硬度",
        "弹性模量",
        "断裂韧性",
        "晶粒粒径",
        "压溃压力",
        "磨粒速度",
    ]].values
    targets = df["亚表面裂纹深度"].values

    # 根据预测误差计算权重
    if model_predictions is not None:
        errors = np.abs(targets - model_predictions)
        weights = errors / (errors.sum() + 1e-6)
    else:
        weights = np.ones(len(df)) / len(df)

    # Mixup增强
    shuffled_idx = np.random.permutation(len(df))
    alpha = 0.2
    lam = np.random.beta(alpha, alpha)
    
    mixed_features = lam * features + (1 - lam) * features[shuffled_idx]
    mixed_targets = lam * targets + (1 - lam) * targets[shuffled_idx]

    # 创建增强数据DataFrame
    aug_df = pd.DataFrame(mixed_features, columns=[
        "莫氏硬度",
        "弹性模量",
        "断裂韧性",
        "晶粒粒径",
        "压溃压力",
        "磨粒速度",
    ])
    aug_df["材料种类"] = df["材料种类"].values
    aug_df["亚表面裂纹深度"] = mixed_targets

    return pd.concat([df, aug_df], ignore_index=True)


def augment_dataset(orig_df, noise_scale=0.05):
    """
    简单的噪声数据增强
    在数值特征上添加高斯噪声
    """
    n_samples = len(orig_df)
    n_aug = n_samples // 4  # 增加25%的样本
    aug_indices = np.random.choice(n_samples, n_aug, replace=False)
    aug_df = orig_df.iloc[aug_indices].copy()

    numeric_cols = [
        "莫氏硬度",
        "弹性模量",
        "断裂韧性",
        "晶粒粒径",
        "压溃压力",
        "磨粒速度",
        "亚表面裂纹深度",
    ]
    
    # 添加高斯噪声
    aug_df[numeric_cols] *= 1 + np.random.normal(
        0, noise_scale, aug_df[numeric_cols].shape
    )

    return pd.concat([orig_df, aug_df], ignore_index=True)


def reconstruct_df(data_dict):
    """
    将预处理后的数据还原成DataFrame
    用于数据增强等操作
    """
    le = data_dict["label_encoder"]
    names = le.inverse_transform(data_dict["material_ids"])
    raw = data_dict["numeric_raw"]
    
    df = pd.DataFrame({
        "材料种类": names,
        "莫氏硬度": raw[:, 0],
        "弹性模量": raw[:, 1],
        "断裂韧性": raw[:, 2],
        "晶粒粒径": raw[:, 3],
        "压溃压力": raw[:, 4],
        "磨粒速度": raw[:, 5],
        "亚表面裂纹深度": data_dict["target"],
    })
    
    return df