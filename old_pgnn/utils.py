"""
工具函数和数据预处理
包含数据准备、增强和辅助函数
"""

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from scipy.stats.mstats import winsorize
from typing import Optional, Union

# 全局常量
GRAD_CLIP = 5.0
# 安全常数阈值
MIN_SAFE_EPS = 1e-10  # 增大最小值，避免过小数导致的问题
MAX_SAFE_VAL = 1e6    # 显著降低，≈ 2^20
SQRT_MAX = 1e6        # 更保守的最大开方值 
LOG_MAX = 1e6         # 更保守的对数输入上限
EXP_MAX = 20.0        # 降低指数上限，e^20 ≈ 5*10^8


def check_and_fix_tensor(tensor, name="tensor", verbose=True):
    """
    检查并修复tensor中的NaN、Inf和过大/过小值
    
    参数:
        tensor: 输入张量
        name: 张量名称（用于日志）
        verbose: 是否打印警告信息
        
    返回:
        修复后的张量
    """
    try:
        # 首先检查是否是None
        if tensor is None:
            if verbose:
                print(f"Error: {name} is None!")
            # 返回一个小的全零tensor
            return torch.zeros(1, 1)
            
        # 检查并替换NaN和Inf值
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            if verbose:
                nan_count = torch.isnan(tensor).sum().item()
                inf_count = torch.isinf(tensor).sum().item()
                print(f"Warning: {nan_count} NaN and {inf_count} Inf detected in {name}, replacing with safe values")
            tensor = torch.nan_to_num(tensor, nan=0.0, posinf=MAX_SAFE_VAL/100, neginf=-MAX_SAFE_VAL/100)
        
        # 检测极端值
        abs_tensor = torch.abs(tensor)
        max_val = abs_tensor.max().item()
        too_large = (abs_tensor > MAX_SAFE_VAL/10).any()
        
        if too_large and verbose:
            print(f"Warning: Extreme value {max_val} detected in {name}, normalizing")
            
        # 对特别大的值使用缩放
        if max_val > MAX_SAFE_VAL/10:
            scale_factor = MAX_SAFE_VAL/100 / max_val
            if verbose:
                print(f"Scaling {name} by factor {scale_factor:.8f}")
            tensor = tensor * scale_factor
        
        # 应用更保守的裁剪
        return torch.clamp(tensor, min=-MAX_SAFE_VAL/10, max=MAX_SAFE_VAL/10)
    except Exception as e:
        if verbose:
            print(f"Error processing tensor {name}: {e}")
        # 返回一个安全的零张量，保持原始形状
        return torch.zeros_like(tensor) if isinstance(tensor, torch.Tensor) else torch.zeros(1, 1)


def safe_division(numerator: torch.Tensor, denominator: torch.Tensor, eps: float = 1e-6, max_val: float = 1e6) -> torch.Tensor:
    """
    安全的除法操作，避免除零和过大数值
    
    参数:
        numerator: 分子
        denominator: 分母
        eps: 防止除零的小值
        max_val: 结果的最大允许值
        
    返回:
        安全处理后的除法结果
    """
    # 确保分母非零
    safe_denom = torch.clamp(denominator, min=eps)
    
    # 计算结果
    result = numerator / safe_denom
    
    # 处理无穷大和NaN
    result = torch.nan_to_num(result, nan=0.0, posinf=max_val, neginf=-max_val)
    
    # 限制最大值
    return torch.clamp(result, min=-max_val, max=max_val)


def safe_log(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    安全的对数计算，避免对负数或零取对数
    """
    x_safe = torch.clamp(x, min=eps)
    result = torch.log(x_safe)
    return torch.nan_to_num(result, nan=0.0, posinf=10.0, neginf=-10.0)


def safe_sqrt(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    安全的平方根计算，避免对负数取平方根
    """
    x_safe = torch.clamp(x, min=eps)
    result = torch.sqrt(x_safe)
    return torch.nan_to_num(result, nan=0.0, posinf=10.0, neginf=0.0)


def safe_power(x: torch.Tensor, power: float, max_val: float = 1e6) -> torch.Tensor:
    """
    安全的幂运算，避免过大的结果
    """
    # 如果是偶数幂，确保不使用负数
    if power % 2 == 0:
        x_safe = torch.abs(x)
    else:
        x_safe = x
    
    # 限制基数范围以避免过大结果
    if power > 1:
        max_base = torch.pow(torch.tensor(max_val, device=x.device), 1/power)
        x_safe = torch.clamp(x_safe, -max_base, max_base)
    
    result = torch.pow(x_safe, power)
    return torch.nan_to_num(result, nan=0.0, posinf=max_val, neginf=-max_val)


def safe_exp(x: torch.Tensor, max_val: float = 30.0) -> torch.Tensor:
    """
    安全的指数函数，避免溢出
    """
    x_safe = torch.clamp(x, max=max_val)
    result = torch.exp(x_safe)
    return torch.nan_to_num(result, nan=0.0, posinf=1e10, neginf=0.0)


def adaptive_scale(x: torch.Tensor, target_max: float = 10.0, name: str = None, 
                  verbose: bool = False) -> torch.Tensor:
    """
    自适应缩放张量到合理范围
    
    参数:
        x: 输入张量
        target_max: 目标最大值
        name: 张量名称（用于日志）
        verbose: 是否打印缩放信息
        
    返回:
        缩放后的张量
    """
    # 检查输入
    if torch.isnan(x).any() or torch.isinf(x).any():
        if verbose and name:
            nan_count = torch.isnan(x).sum().item()
            inf_count = torch.isinf(x).sum().item()
            print(f"Warning: {nan_count} NaN和{inf_count} Inf在{name}中，替换为安全值")
        x = torch.nan_to_num(x, nan=0.0, posinf=target_max, neginf=-target_max)
    
    # 获取当前最大绝对值
    max_abs = torch.max(torch.abs(x)).item()
    
    # 如果最大值超过目标，进行缩放
    if max_abs > target_max:
        scale_factor = target_max / max_abs
        if verbose and name:
            print(f"自动缩放 {name}，因子: {scale_factor:.8f}")
        return x * scale_factor
    
    return x


def prepare_material_data(
    df,
    scaler: Optional[RobustScaler] = None,
    le: Optional[LabelEncoder] = None,
    winsorize_limits=(0.02, 0.02),  # 2%-98%分位数裁剪
    verbose: bool = True,
):
    """
    准备材料数据，包括预处理和特征工程
    """
    numeric_cols = [
        "莫氏硬度", "弹性模量", "断裂韧性", 
        "晶粒粒径", "压溃压力", "磨粒速度"
    ]
    
    # 创建深拷贝以避免修改原始数据
    df_clean = df.copy()
    
    # 确保数值列是浮点型，避免类型转换警告
    for col in numeric_cols + ["亚表面裂纹深度"]:
        df_clean[col] = df_clean[col].astype(float)
    
    # 1. 单位统一 (在最开始进行，确保所有后续计算使用统一单位)
    df_clean.loc[:, "莫氏硬度"] = 0.00981 * df_clean["莫氏硬度"]**3  # Mohs → Pa
    df_clean.loc[:, "晶粒粒径"] = df_clean["晶粒粒径"] * 1e-6        # μm  → m
    df_clean.loc[:, "压溃压力"] = df_clean["压溃压力"] * 1e3         # kPa → Pa
    df_clean.loc[:, "断裂韧性"] = df_clean["断裂韧性"] * 1e6         # MPa√m → Pa√m
    df_clean.loc[:, "弹性模量"] = df_clean["弹性模量"] * 1e9      # GPa → Pa
    # 2. 处理缺失值
    for col in numeric_cols + ["亚表面裂纹深度"]:
        if df_clean[col].isna().any():
            if verbose:
                print(f"Warning: Found NaN values in {col}, filling with median")
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    # 3. Winsorize处理极端值
    for col in numeric_cols:
        original_values = df_clean[col].values
        winsorized_values = winsorize(original_values, limits=winsorize_limits)
        outlier_count = np.sum(original_values != winsorized_values)
        if outlier_count > 0:
            if verbose:
                print(f"Winsorized {outlier_count} outliers in {col}")
            df_clean[col] = winsorized_values
    
    # 4. 对数变换处理长尾分布
    log_transform_cols = ["弹性模量", "断裂韧性", "晶粒粒径"]
    df_log = df_clean.copy()
    
    for col in log_transform_cols:
        if f"{col}_logged" not in df.columns:
            min_val = df_clean[col].min()
            if min_val <= 0:
                df_log.loc[:, col] = np.log1p(df_clean[col] - min_val + 1e-6)
            else:
                df_log.loc[:, col] = np.log1p(df_clean[col])
            df_log[f"{col}_logged"] = True
            if verbose:
                print(f"Applied log1p transform to {col}")
    
    # 5. 编码材料种类
    if le is None:
        le = LabelEncoder()
        material_ids = le.fit_transform(df_clean["材料种类"])
    else:
        material_ids = le.transform(df_clean["材料种类"])
    
    # 6. 提取特征
    # 原始特征（用于物理约束）- 已经做过单位统一
    numeric_raw = df_clean[numeric_cols].values.astype(np.float32)
    
    # 用于主网络的特征（使用对数变换后的版本）
    numeric_for_scaling = df_log[numeric_cols].values.astype(np.float32)
    
    # 7. RobustScaler标准化
    if scaler is None:
        scaler = RobustScaler(quantile_range=(5.0, 95.0))
        scaler.fit(numeric_for_scaling)
    
    numeric_scaled = scaler.transform(numeric_for_scaling)
    
    # 8. 最终安全检查
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