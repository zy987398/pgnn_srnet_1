import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Dict, List, Tuple, Optional


def prepare_material_data(
    df,
    scaler: Optional[StandardScaler] = None,
    le: Optional[LabelEncoder] = None,
    augment: bool = False,
):
    """准备材料数据，可选数据增强"""
    # 编码材料种类
    if le is None:
        le = LabelEncoder()
        material_ids = le.fit_transform(df["材料种类"])
    else:
        material_ids = le.transform(df["材料种类"])

    # 提取数值特征
    numeric_cols = [
        "莫氏硬度",
        "弹性模量",
        "断裂韧性",
        "晶粒粒径",
        "压溃压力",
        "磨粒速度",
    ]
    numeric_raw = df[numeric_cols].values.astype(np.float32)

    # 目标变量
    target = df["亚表面裂纹深度"].values

    # 使用StandardScaler
    if scaler is None:
        scaler = StandardScaler().fit(numeric_raw)
    numeric_scaled = scaler.transform(numeric_raw)

    # 数据增强 - 只在训练时进行
    if augment:
        # 1. 样本数量
        n_samples = len(material_ids)
        n_augment = n_samples // 4  # 增加25%的样本

        # 2. 随机选择要增强的样本索引
        aug_indices = np.random.choice(n_samples, n_augment, replace=False)

        # 3. 为每个特征添加微小噪声
        aug_material_ids = material_ids[aug_indices].copy()
        aug_numeric_scaled = numeric_scaled[aug_indices].copy()
        aug_numeric_raw = numeric_raw[aug_indices].copy()
        aug_target = target[aug_indices].copy()

        # 4. 添加高斯噪声
        noise_scale = 0.05  # 5%的噪声
        aug_numeric_scaled += np.random.normal(0, noise_scale, aug_numeric_scaled.shape)

        # 为原始特征也添加相应的噪声（确保一致性）
        aug_numeric_raw *= 1 + np.random.normal(0, noise_scale, aug_numeric_raw.shape)

        # 目标变量也略微调整，以保持与特征的相关性
        aug_target *= 1 + np.random.normal(0, noise_scale * 0.5, aug_target.shape)

        # 5. 合并原始数据和增强数据
        material_ids = np.concatenate([material_ids, aug_material_ids])
        numeric_scaled = np.concatenate([numeric_scaled, aug_numeric_scaled])
        numeric_raw = np.concatenate([numeric_raw, aug_numeric_raw])
        target = np.concatenate([target, aug_target])

    return {
        "material_ids": material_ids,
        "numeric_features": numeric_scaled,
        "numeric_raw": numeric_raw,
        "target": target,
        "label_encoder": le,
        "scaler": scaler,
        "num_materials": len(le.classes_),
    }


def reconstruct_df(data_dict):
    """将预处理后的数据还原成DataFrame"""
    le = data_dict["label_encoder"]
    names = le.inverse_transform(data_dict["material_ids"])
    raw = data_dict["numeric_raw"]
    df = pd.DataFrame(
        {
            "材料种类": names,
            "莫氏硬度": raw[:, 0],
            "弹性模量": raw[:, 1],
            "断裂韧性": raw[:, 2],
            "晶粒粒径": raw[:, 3],
            "压溃压力": raw[:, 4],
            "磨粒速度": raw[:, 5],
            "亚表面裂纹深度": data_dict["target"],
        }
    )
    return df


def augment_dataset(orig_df, noise_scale=0.05):
    """在数值列上添加噪声的数据增强"""
    if orig_df is None or len(orig_df) == 0:
        print("警告：传入的数据框为空，无法进行数据增强")
        return orig_df
        
    try:
        n_samples = len(orig_df)
        n_aug = max(1, n_samples // 4)  # 确保至少增强一个样本
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
        
        # 检查所有列是否都存在
        missing_cols = [col for col in numeric_cols if col not in aug_df.columns]
        if missing_cols:
            print(f"警告：以下列在数据中不存在: {missing_cols}")
            numeric_cols = [col for col in numeric_cols if col in aug_df.columns]
            
        if not numeric_cols:
            print("警告：没有可用于增强的数值列")
            return orig_df
            
        aug_df[numeric_cols] *= 1 + np.random.normal(
            0, noise_scale, aug_df[numeric_cols].shape
        )

        return pd.concat([orig_df, aug_df], ignore_index=True)
    except Exception as e:
        print(f"数据增强过程出错: {str(e)}")
        return orig_df  # 出错时返回原始数据


def safe_division(a, b, eps=1e-6, max_val=1e6):
    """安全除法操作，防止除零"""
    # 确保分母非零
    safe_denom = torch.clamp(b, min=eps)
    
    # 计算结果
    result = a / safe_denom
    
    # 处理无穷大和NaN
    result = torch.nan_to_num(result, nan=0.0, posinf=max_val, neginf=-max_val)
    
    # 限制最大值
    return torch.clamp(result, min=-max_val, max=max_val)


def safe_log(x, eps=1e-6):
    """安全对数操作，防止对负数或零取对数"""
    x_safe = torch.clamp(x, min=eps)
    result = torch.log(x_safe)
    return torch.nan_to_num(result, nan=0.0, posinf=10.0, neginf=-10.0)


def safe_sqrt(x, eps=1e-6):
    """安全平方根操作，防止对负数取平方根"""
    x_safe = torch.clamp(torch.abs(x), min=eps)
    result = torch.sqrt(x_safe)
    return torch.nan_to_num(result, nan=0.0, posinf=10.0, neginf=0.0)


def safe_power(x, power, eps=1e-6, max_val=1e6):
    """安全幂运算，处理边界情况"""
    # 如果是偶数幂，确保不使用负数
    if power % 2 == 0:
        x_safe = torch.abs(x)
    else:
        x_safe = x
    
    # 限制基数范围以避免过大结果
    if power > 1:
        max_base = torch.pow(torch.tensor(max_val, device=x.device), 1/power)
        x_safe = torch.clamp(x_safe, -max_base, max_base)
    
    result = torch.pow(x_safe + eps, power)
    return torch.nan_to_num(result, nan=0.0, posinf=max_val, neginf=-max_val)


def adaptive_scale(x, target_max=10.0, name=None):
    """自适应缩放，防止数值过大或过小"""
    if torch.isnan(x).any() or torch.isinf(x).any():
        if name:
            print(f"Warning: NaN/Inf in {name}, fixing")
        x = torch.nan_to_num(x, nan=0.0, posinf=1e3, neginf=-1e3)
    
    # 计算当前范围
    abs_max = torch.max(torch.abs(x))
    
    # 如果范围太大或太小，进行缩放
    if abs_max > target_max or abs_max < (target_max / 100):
        if abs_max > 0:
            scale = target_max / abs_max
            x = x * scale
    
    return x