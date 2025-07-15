"""
模型架构定义
包含所有神经网络模型组件和主模型类
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import traceback
import matplotlib.pyplot as plt
from utils import (safe_division, safe_log, safe_sqrt, safe_power, 
                  safe_exp, adaptive_scale, check_and_fix_tensor,
                  MIN_SAFE_EPS, MAX_SAFE_VAL, SQRT_MAX, LOG_MAX, EXP_MAX)

# 全局梯度裁剪阈值 - 降低以增加稳定性
GRAD_CLIP = 3.0

def safe_sqrt(x, eps=MIN_SAFE_EPS):
    """安全开方运算"""
    # 确保输入为非负且有合理上限
    x_safe = torch.clamp(torch.abs(x) + eps, min=eps, max=SQRT_MAX)
    return torch.sqrt(x_safe)

def safe_log(x, eps=MIN_SAFE_EPS):
    """安全对数运算"""
    # 确保输入严格为正且有合理上限
    x_safe = torch.clamp(torch.abs(x) + eps, min=eps, max=LOG_MAX)
    result = torch.log(x_safe)
    # 额外检查结果
    if torch.isnan(result).any() or torch.isinf(result).any():
        print("Warning: NaN/Inf detected in safe_log output, replacing")
        result = torch.nan_to_num(result, nan=0.0, posinf=10.0, neginf=-10.0)
    return torch.clamp(result, min=-20.0, max=20.0)

def safe_exp(x, max_val=EXP_MAX):
    """安全指数运算，输出范围更加有限"""
    # 使用更严格的输入范围限制
    x_safe = torch.clamp(x, min=-max_val/2, max=max_val/2)
    # 尝试计算指数
    try:
        result = torch.exp(x_safe)
        # 额外检查并处理可能的数值异常
        if torch.isnan(result).any() or torch.isinf(result).any():
            print("Warning: NaN/Inf detected in safe_exp output, replacing")
            result = torch.nan_to_num(result, nan=1.0, posinf=1e6, neginf=1e-6)
        # 应用更严格的上限
        return torch.clamp(result, min=0.0, max=1e6)
    except Exception as e:
        print(f"Error in safe_exp: {e}")
        # 失败时返回安全值
        return torch.ones_like(x)

def safe_power(x, power, max_val=MAX_SAFE_VAL):
    """安全幂运算，预处理输入防止数值爆炸"""
    if power > 1:
        # 对于大于1的幂，需要更严格的限制
        # 使用n次根计算安全范围：x^n ≤ max_val => x ≤ max_val^(1/n)
        safe_range = max_val ** (1.0/power)
        x_safe = torch.clamp(x, min=-safe_range, max=safe_range)
    else:
        # 对于小于等于1的幂，使用标准限制
        x_safe = torch.clamp(x, min=-max_val/2, max=max_val/2)
    
    # 对于非整数幂，确保输入为正
    if power != int(power):
        x_safe = torch.abs(x_safe) + MIN_SAFE_EPS
    
    # 计算幂并限制结果
    result = x_safe ** power
    return torch.clamp(result, min=-max_val, max=max_val)

# check_and_fix_tensor函数已移至utils.py


# ===== 基础组件 =====
class MaterialEmbedding(nn.Module):
    """材料种类嵌入层"""
    def __init__(self, num_materials, embedding_dim=16, init_gain=0.5):
        super().__init__()
        self.embedding = nn.Embedding(num_materials, embedding_dim)
        # 保持与原始代码相同的初始化方式
        nn.init.xavier_uniform_(self.embedding.weight, gain=init_gain)
    
    def forward(self, material_ids):
        return self.embedding(material_ids)


class ResidualBlock(nn.Module):
    """残差连接块，提高训练稳定性"""
    def __init__(self, dim, dropout_rate=0.1):
        super().__init__()
        # 修改为与主类中相同的结构
        self.linear = nn.Linear(dim, dim)
        self.layer_norm = nn.LayerNorm(dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # 注意：不在这里应用额外的GELU，因为主类会在残差连接后应用
        out = self.linear(x)
        out = self.layer_norm(out)
        out = self.act(out)
        out = self.dropout(out)
        return out + x  # 残差连接


# ===== 物理特征提取器 =====
class PhysicsInformedFeatureExtractor(nn.Module):
    """
    物理特征提取器
    基于材料科学原理提取物理意义明确的特征
    """
    def __init__(self, input_dim=6):
        super().__init__()
        self.input_dim = input_dim
        # 可学习的缩放因子，用于平衡不同物理特征的尺度
        self.feature_scales = nn.Parameter(torch.ones(12))
    
    def forward(self, x):
        """
        输入x: [batch_size, 6] - 原始尺度的特征
        返回: [batch_size, 18] - 原始特征(6) + 物理特征(12) = 总共18个特征
        """
        # 首先检查输入
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("Warning: NaN/Inf in physics extractor input, fixing")
            x = torch.nan_to_num(x, nan=0.0, posinf=1e3, neginf=0.0)
        
        # 首先进行自适应缩放，确保数据在合理范围内
        x = adaptive_scale(x, target_max=1e6, name="physics_extractor_input")
        
        # 保存原始特征用于输出
        features = [x]
        
        # -------- 基本变量与安全裁剪 - 使用更保守的限制 --------
        try:
            # 使用更严格的范围约束
            hardness         = x[:, 0:1].clamp(min=1.0, max=1e7)        # Pa
            elastic_modulus  = x[:, 1:2].clamp(min=1.0, max=1e9)        # Pa
            fracture_tough   = x[:, 2:3].clamp(min=0.1, max=1e7)        # Pa·√m
            grain_size       = x[:, 3:4].clamp(min=1e-9, max=1e-4)      # m
            stress           = x[:, 4:5].clamp(min=0.0, max=1e7)        # Pa
            velocity         = x[:, 5:6].clamp(min=0.0, max=1e2)        # m·s⁻¹
            
            # 额外的自适应缩放确保数值稳定
            hardness = adaptive_scale(hardness, target_max=1e6, name="hardness")
            elastic_modulus = adaptive_scale(elastic_modulus, target_max=1e8, name="elastic_modulus")
            fracture_tough = adaptive_scale(fracture_tough, target_max=1e6, name="fracture_tough")
            stress = adaptive_scale(stress, target_max=1e6, name="stress")

            # 增加数值稳定性的常数
            eps = 1e-6
            π = 3.14159  # 使用常数而不是torch.pi，避免自动微分带来的额外复杂性
            feats = []
        except Exception as e:
            print(f"Error in physics feature extraction (variable setup): {e}")
            # 创建安全的默认值
            hardness = torch.ones_like(x[:, 0:1]) * 1e6
            elastic_modulus = torch.ones_like(x[:, 1:2]) * 1e10
            fracture_tough = torch.ones_like(x[:, 2:3]) * 1e6
            grain_size = torch.ones_like(x[:, 3:4]) * 1e-6
            stress = torch.ones_like(x[:, 4:5]) * 1e6
            velocity = torch.ones_like(x[:, 5:6]) * 10.0
            eps = 1e-6
            π = 3.14159
            feats = []

        # 计算物理特征，使用更安全的方式
        try:
            # 1. Griffith 能量释放率  G = σ² π d / E       [ J·m⁻² ]
            stress_squared = torch.clamp(stress**2, max=1e12)  # 限制平方值
            G = safe_division(stress_squared * π * grain_size, elastic_modulus, eps=eps, max_val=1e6)
            # 使用更小的缩放以避免大值
            G_scaled = torch.tanh(torch.clamp(G / 1e3, -10, 10))
            feats.append(G_scaled * self.feature_scales[0])

            # 2. 应力强度比  K/Kc = σ √(π d) / K_IC       [ — ]
            sqrt_term = torch.sqrt(torch.clamp(π * grain_size, min=eps, max=1.0))
            k_ratio = safe_division(stress * sqrt_term, fracture_tough, eps=eps, max_val=10.0)
            feats.append(torch.tanh(torch.clamp(k_ratio, -10, 10)) * self.feature_scales[1])

            # 3. Hall-Petch 尺度项  d^{-1/2}               [ m^{-½} ]
            hall_petch = torch.rsqrt(torch.clamp(grain_size + eps, min=eps, max=1.0))  # 使用rsqrt更稳定
            hall_petch = torch.clamp(hall_petch, max=1e6)  # 额外限制
            feats.append(torch.tanh(torch.clamp(hall_petch / 1e3, -10, 10)) * self.feature_scales[2])

            # 4. 裂纹屏蔽因子  χ = (K_IC / E) √d          [ m ]
            k_ic_over_e = safe_division(fracture_tough, elastic_modulus, eps=eps, max_val=1.0)
            sqrt_grain = torch.sqrt(torch.clamp(grain_size, min=eps, max=1.0))
            shielding = k_ic_over_e * sqrt_grain
            shielding = torch.clamp(shielding, max=1e-2)  # 限制最大值
            feats.append(torch.tanh(torch.clamp(shielding / 1e-4, -10, 10)) * self.feature_scales[3])

            # 5. 载荷-强度比  ρ₁ = σ / H                  [ — ]
            rho1 = safe_division(stress, hardness, eps=eps, max_val=1.0)
            feats.append(torch.tanh(torch.clamp(rho1, -10, 10)) * self.feature_scales[4])
        
        except Exception as e:
            print(f"Error in physics feature extraction (features 1-5): {e}")
            # 添加安全默认值
            for i in range(5):
                feats.append(torch.zeros_like(hardness) * self.feature_scales[i])

        try:
            # 6. 功率-强度比  ρ₂ = σ V / H               [ m·s⁻¹ ]
            rho2 = safe_division(stress * velocity, hardness, eps=eps, max_val=100.0)
            feats.append(torch.tanh(torch.clamp(rho2 / 10, -10, 10)) * self.feature_scales[5])

            # 7. 延-脆转临界深度  d_c = η (E/H) (K_IC/H)²   [ m ]  (η≈0.15)
            eta = 0.15
            e_over_h = safe_division(elastic_modulus, hardness, eps=eps, max_val=100.0)
            k_over_h = safe_division(fracture_tough, hardness, eps=eps, max_val=10.0)
            k_over_h_squared = torch.clamp(k_over_h**2, max=100.0)
            d_c = eta * e_over_h * k_over_h_squared
            d_c = torch.clamp(d_c, max=1e-3)  # 限制最大值
            feats.append(torch.tanh(torch.clamp(d_c / 1e-5, -10, 10)) * self.feature_scales[6])

            # 8. 临界裂纹长度比  a_c / d,  a_c = (K_IC/σ)² / π   [ — ]
            k_over_s = safe_division(fracture_tough, stress, eps=eps, max_val=10.0)
            k_over_s_squared = torch.clamp(k_over_s**2, max=100.0)
            a_c = k_over_s_squared / π
            ac_over_d = safe_division(a_c, grain_size, eps=eps, max_val=100.0)
            feats.append(torch.tanh(torch.clamp(ac_over_d, -10, 10)) * self.feature_scales[7])

            # 9. 塑性区尺寸比  r_p / d,  r_p = (K_IC / H)² / (2π) [ — ]
            k_over_h = safe_division(fracture_tough, hardness, eps=eps, max_val=10.0)
            k_over_h_squared = torch.clamp(k_over_h**2, max=100.0)
            r_p = k_over_h_squared / (2.0 * π)
            rp_over_d = safe_division(r_p, grain_size, eps=eps, max_val=100.0)
            feats.append(torch.tanh(torch.clamp(rp_over_d, -10, 10)) * self.feature_scales[8])

            # 10. 脆性指数  B = H / K_IC                    [ √m⁻¹ ]
            B = safe_division(hardness, fracture_tough, eps=eps, max_val=100.0)
        except Exception as e:
            print(f"Error in physics feature extraction (features 6-10): {e}")
            # 添加安全默认值
            for i in range(5, 10):
                feats.append(torch.zeros_like(hardness) * self.feature_scales[i])
        feats.append(torch.tanh(B / 10) * self.feature_scales[9])

        try:
            # 11. 初-剖比  K_IC / E                       [ √m ]
            ki_over_e = safe_division(fracture_tough, elastic_modulus, eps=eps, max_val=1.0)
            feats.append(torch.tanh(torch.clamp(ki_over_e * 100, -10, 10)) * self.feature_scales[10])

            # 12. 弹性应变能密度  U = 0.5 σ² / E          [ J·m⁻³ ]
            stress_squared = torch.clamp(stress**2, max=1e12)  # 限制平方值
            U = 0.5 * safe_division(stress_squared, elastic_modulus, eps=eps, max_val=1e6)
            feats.append(torch.tanh(torch.clamp(U / 1e3, -10, 10)) * self.feature_scales[11])
        except Exception as e:
            print(f"Error in physics feature extraction (features 11-12): {e}")
            # 添加安全默认值
            for i in range(10, 12):
                feats.append(torch.zeros_like(hardness) * self.feature_scales[i])
        
        # 将原始特征与提取的物理特征合并
        all_features = torch.cat(features + feats, dim=1)  # 6个原始特征 + 12个物理特征 = 18个总特征
        
        # 最终输出前进行自适应缩放，避免传递极端值到下一层
        all_features = adaptive_scale(all_features, target_max=100.0, name="physics_features_final")
        
        return all_features


# ===== 符号层（复杂版本） =====
class AdaptiveSymbolicLayer(nn.Module):
    """
    自适应符号层 - 完整版
    整合物理特征提取器，使用完整的Transformer架构（双残差连接）
    实现交叉注意力机制从物理特征中提取与潜在表示相关的信息
    """
    def __init__(self, latent_dim: int = 32):
        super().__init__()
        
        # 内置物理特征提取器
        self.physics_extractor = PhysicsInformedFeatureExtractor()
        
        # 确保注意力头数能整除latent_dim
        self.num_heads = max(1, (latent_dim // 8) * 2)
        if latent_dim % self.num_heads != 0:
            self.num_heads = 1
        
        # --- Transformer架构组件 ---
        # 1. 输入投影
        self.query_proj = nn.Linear(latent_dim, latent_dim)
        self.key_proj = nn.Linear(1, latent_dim)
        self.value_proj = nn.Linear(1, latent_dim)
        
        # 2. 多头交叉注意力层
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=self.num_heads,
            batch_first=True,
            dropout=0.1
        )
        
        # 3. 第一个残差连接后的层归一化
        self.attn_norm = nn.LayerNorm(latent_dim)
        
        # 4. 前馈神经网络（Transformer标准组件）
        self.feed_forward = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 4),  # 扩展维度
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(latent_dim * 4, latent_dim)   # 恢复维度
        )
        
        # 5. 第二个残差连接后的层归一化
        self.ffn_norm = nn.LayerNorm(latent_dim)
        
        # --- 符号表达式生成组件 ---
        self.operators = ["*", "/", "^2", "^3", "sqrt", "log", "exp", "sin"]
        self.num_operators = len(self.operators)

        # 特征选择器（哪些物理特征重要）
        self.variable_selector = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.GELU(),
            nn.LayerNorm(64),
            nn.Dropout(0.1),
            nn.Linear(64, 18),  # 更新为18个特征：6个原始特征 + 12个物理特征
        )

        # 运算符选择器（使用什么数学运算）
        self.operator_selector = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.GELU(),
            nn.LayerNorm(32),
            nn.Dropout(0.2),
            nn.Linear(32, self.num_operators),
            nn.Tanh(),  # 添加Tanh限制输出范围
        )

        # 系数生成网络（多项式系数）
        self.coefficient_net = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.GELU(),
            nn.LayerNorm(32),
            nn.Dropout(0.1),
            nn.Linear(32, 4),
        )

        # 门控机制（平衡多项式与运算符表达式）
        self.gate = nn.Sequential(
            nn.Linear(latent_dim, 1), 
            nn.Sigmoid()
        )
        
        # 用于可解释性分析的属性
        self.last_attn_weights = None
        self.last_var_weights = None
        self.last_op_weights = None
        self.last_coeffs = None
        self.complexity = 0.0

    def forward(self, z: torch.Tensor, numeric_raw: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        z: [batch_size, latent_dim] - 潜在表示
        numeric_raw: [batch_size, 6] - 原始尺度的物理特征
        """
        # 输入预处理和缩放
        z = adaptive_scale(z, target_max=10.0, name="symbolic_z_input")
        numeric_raw = adaptive_scale(numeric_raw, target_max=1e6, name="symbolic_numeric_raw")
        
        # 1. 提取物理特征
        physics_features = self.physics_extractor(numeric_raw)
        
        # 使用自适应缩放而不是简单的check_and_fix_tensor
        physics_features = adaptive_scale(physics_features, target_max=10.0, name="physics_features")
        
        # 2. 准备交叉注意力的输入
        # z作为query
        query = self.query_proj(z).unsqueeze(1)  # [B, 1, latent_dim]
        
        # 物理特征作为key和value - 每个特征单独编码
        batch_size = physics_features.size(0)
        phys_features_seq = physics_features.unsqueeze(2)  # [B, num_features, 1]
        key = self.key_proj(phys_features_seq)    # [B, num_features, latent_dim]
        value = self.value_proj(phys_features_seq)  # [B, num_features, latent_dim]            # 3. 应用交叉注意力（第一个Transformer块），增强稳健性
        try:
            # 先检查输入
            query = adaptive_scale(query, name="attention_query")
            key = adaptive_scale(key, name="attention_key")
            value = adaptive_scale(value, name="attention_value")
            
            # 应用注意力机制
            attn_output, attn_weights = self.cross_attention(
                query=query,
                key=key,
                value=value,
                need_weights=True
            )
            
            # 检查注意力权重
            if torch.isnan(attn_weights).any() or torch.isinf(attn_weights).any():
                print(f"Warning: NaN/Inf in attention weights, replacing with uniform attention")
                attn_weights = torch.ones_like(attn_weights) / attn_weights.size(-1)
                
            # 保存注意力权重用于可解释性分析
            self.last_attn_weights = attn_weights
            
            # 检查注意力输出
            if torch.isnan(attn_output).any() or torch.isinf(attn_output).any():
                print(f"Warning: NaN/Inf in attention output, replacing with safe values")
                attn_output = torch.zeros_like(attn_output)
            
            # 提取注意力输出
            attn_output = attn_output.squeeze(1)
            attn_output = torch.clamp(attn_output, -5.0, 5.0)  # 使用保守的裁剪
            
            # 第一个残差连接 + 层归一化，采用更安全的方式
            z_safe = adaptive_scale(z, name="z_pre_residual")
            z_attn = z_safe + 0.1 * attn_output  # 减少注意力输出的影响
            
            # 检查残差连接结果
            z_attn = adaptive_scale(z_attn, name="pre_norm_attention")
            
            # 应用层归一化，包含错误处理
            try:
                z_attn = self.attn_norm(z_attn)
            except Exception as e:
                print(f"Warning: Layer normalization failed: {e}")
                # 如果归一化失败，使用简单的缩放和中心化
                mean = z_attn.mean(dim=-1, keepdim=True)
                std = z_attn.std(dim=-1, keepdim=True) + 1e-5
                z_attn = (z_attn - mean) / std
                
            # 最终检查
            z_attn = adaptive_scale(z_attn, name="attention_output")
            
            # 4. 前馈网络（第二个Transformer块）
            ffn_output = self.feed_forward(z_attn)
            
            # 第二个残差连接 + 层归一化
            z = z_attn + ffn_output  # 残差连接 2
            z = self.ffn_norm(z)
            z = adaptive_scale(z, name="ffn_output")
            
        except Exception as e:
            print(f"Warning: Transformer processing failed ({e}), using original z")
            traceback.print_exc()
            z = adaptive_scale(z, name="transformer_fallback")

        # 5. 特征选择权重 - 使用更稳定的版本
        try:
            # 通过自适应缩放处理选择器的输入
            var_selector_input = adaptive_scale(z, target_max=5.0, name="var_selector_input")
            op_selector_input = adaptive_scale(z, target_max=5.0, name="op_selector_input")
            coeff_input = adaptive_scale(z, target_max=5.0, name="coeff_input")
            
            # 计算权重和系数
            var_logits = self.variable_selector(var_selector_input)
            var_logits = torch.clamp(var_logits, -5, 5)  # 限制logits范围
            var_weights = F.softmax(var_logits, dim=-1)
            
            op_logits = self.operator_selector(op_selector_input)
            op_logits = torch.clamp(op_logits, -5, 5)  # 限制logits范围
            op_weights = F.softmax(op_logits, dim=-1)
            
            # 限制系数范围，避免过大值
            coeffs = torch.tanh(self.coefficient_net(coeff_input)) * 3.0  # 使用tanh限制在[-3,3]范围
            
            # 保存权重用于可解释性
            self.last_var_weights = var_weights
            self.last_op_weights = op_weights
            self.last_coeffs = coeffs
    
            # 仅存储非梯度版本的复杂度 - 用于可视化和记录，不参与梯度计算
            complexity_value = -torch.sum(var_weights * safe_log(var_weights), dim=-1).mean()
            self.complexity = float(complexity_value.detach().item()) if isinstance(complexity_value, torch.Tensor) else float(complexity_value)
        except Exception as e:
            print(f"Warning: Failed to compute weights: {e}")
            # 提供默认值
            batch_size = z.size(0)
            var_weights = torch.ones(batch_size, 18, device=z.device) / 18
            op_weights = torch.ones(batch_size, self.num_operators, device=z.device) / self.num_operators
            coeffs = torch.zeros(batch_size, 4, device=z.device)
            self.last_var_weights = var_weights
            self.last_op_weights = op_weights
            self.last_coeffs = coeffs
            self.complexity = 0.0

        # 6. 获取一个池化版本的物理特征用于符号表达式
        try:
            pool_size = min(18, physics_features.size(1))  # 调整为最大18个特征
            if physics_features.size(1) > pool_size:
                pooled_features = F.adaptive_avg_pool1d(
                    physics_features.unsqueeze(1), pool_size
                ).squeeze(1)
            else:
                pooled_features = physics_features[:, :pool_size]
                if pooled_features.size(1) < pool_size:
                    padding = torch.zeros(batch_size, pool_size - pooled_features.size(1), 
                                        device=pooled_features.device)
                    pooled_features = torch.cat([pooled_features, padding], dim=1)
            
            # 使用自适应缩放而不是check_and_fix_tensor
            pooled_features = adaptive_scale(pooled_features, target_max=5.0, name="pooled_features")
            
            # 7. 加权特征组合
            selected_features = (var_weights * pooled_features).sum(dim=1, keepdim=True)
            selected_features = adaptive_scale(selected_features, target_max=5.0, name="selected_features")
        except Exception as e:
            print(f"Warning: Feature pooling failed: {e}")
            # 提供安全的后备值
            selected_features = torch.zeros(batch_size, 1, device=z.device)

        # 8.符号表达式计算 - 使用更严格的缩放
        try:
            expressions = []
            
            # 线性项 - 直接乘法
            expr1 = coeffs[:, 0:1] * selected_features
            expr1 = adaptive_scale(expr1, target_max=5.0, name="expression")
            expressions.append(expr1)
            
            # 二次项 - 使用safe_power并缩放
            expr2_base = safe_power(selected_features, 2, max_val=25.0)
            expr2 = coeffs[:, 1:2] * expr2_base
            expr2 = adaptive_scale(expr2, target_max=5.0, name="expression")
            expressions.append(expr2)
            
            # 三次项 - 使用safe_power并缩放
            expr3_base = safe_power(selected_features, 3, max_val=25.0)
            expr3 = coeffs[:, 2:3] * expr3_base
            expr3 = adaptive_scale(expr3, target_max=5.0, name="expression")
            expressions.append(expr3)
            
            # 平方根项 - 使用safe_sqrt并缩放
            expr4_base = safe_sqrt(selected_features)
            expr4 = coeffs[:, 3:4] * expr4_base
            expr4 = adaptive_scale(expr4, target_max=5.0, name="expression")
            expressions.append(expr4)
        except Exception as e:
            print(f"Warning: Expression calculation failed: {e}")
            # 提供安全的后备值
            for _ in range(4):
                expressions.append(torch.zeros_like(selected_features))

        # 9算子操作 - 严格控制每个算子的范围
        try:
            ops = [
                selected_features,  # 原始特征
                safe_division(torch.ones_like(selected_features), selected_features, max_val=5.0),  # 倒数
                adaptive_scale(safe_power(selected_features, 2, max_val=10.0), target_max=5.0, name="op_2"),  # 平方
                adaptive_scale(safe_power(selected_features, 3, max_val=10.0), target_max=5.0, name="op_3"),  # 立方
                safe_sqrt(selected_features),  # 平方根
                safe_log(torch.abs(selected_features) + 1e-6),  # 对数
                torch.tanh(selected_features),  # 用tanh替代exp，避免值爆炸
                torch.sin(torch.clamp(selected_features, -3.14, 3.14))  # 正弦，限制在[-π,π]
            ]
            
            # 为每个操作应用自适应缩放
            ops = [adaptive_scale(op, target_max=5.0, name=f"op_{i}") for i, op in enumerate(ops)]
        except Exception as e:
            print(f"Warning: Operator calculation failed: {e}")
            # 提供安全的后备值
            ops = [torch.zeros_like(selected_features) for _ in range(8)]
        
        # 10. 算子组合，增强稳健性
        try:
            # 检查每个操作并修复
            safe_ops = []
            for i, op in enumerate(ops):
                if torch.isnan(op).any() or torch.isinf(op).any():
                    print(f"Warning: Fixing NaN/Inf in operator {i}")
                    safe_op = torch.nan_to_num(op, nan=0.0, posinf=1.0, neginf=-1.0)
                else:
                    safe_op = op
                # 额外应用裁剪，每个操作都单独限制范围
                safe_op = torch.clamp(safe_op, min=-5.0, max=5.0)
                safe_ops.append(safe_op)
            
            # 将操作堆叠成矩阵，确保形状正确
            try:
                op_stack = torch.stack(safe_ops, dim=1)
                if len(op_stack.shape) == 3 and op_stack.shape[2] == 1:
                    op_stack = op_stack.squeeze(-1)
                op_stack = adaptive_scale(op_stack, name="op_stack")
            except Exception as e:
                print(f"Error in stacking operations: {e}")
                # 使用备选方法创建op_stack
                batch_size = safe_ops[0].shape[0]
                op_stack = torch.zeros(batch_size, len(safe_ops), device=safe_ops[0].device)
                for i, op in enumerate(safe_ops):
                    op_stack[:, i] = op.view(batch_size)
            
            # 检查操作堆栈的形状
            if len(op_stack.shape) != 2:
                print(f"Warning: Unexpected op_stack shape {op_stack.shape}, reshaping")
                batch_size = op_stack.shape[0]
                op_stack = op_stack.reshape(batch_size, -1)
            
            # 确保权重不含NaN/Inf并且形状匹配
            weights_clean = adaptive_scale(op_weights, name="op_weights")
            
            # 如果维度不匹配，尝试调整
            if op_stack.shape[1] != weights_clean.shape[1]:
                print(f"Warning: Shape mismatch - op_stack: {op_stack.shape}, weights: {weights_clean.shape}")
                # 调整权重形状以匹配操作堆栈
                min_dim = min(op_stack.shape[1], weights_clean.shape[1])
                op_stack = op_stack[:, :min_dim]
                weights_clean = weights_clean[:, :min_dim]
                
            # 执行矩阵乘法，使用更安全的方法
            try:
                weights_clean_unsqueezed = weights_clean.unsqueeze(-1)
                op_comb = torch.bmm(op_stack.unsqueeze(1), weights_clean_unsqueezed)
                op_comb = op_comb.squeeze(1)
            except Exception as e:
                print(f"Error in matrix multiplication: {e}, trying simpler method")
                # 使用更简单的方法: 加权和
                op_comb = torch.zeros_like(selected_features)
                for i in range(min(op_stack.shape[1], weights_clean.shape[1])):
                    op_comb = op_comb + op_stack[:, i:i+1] * weights_clean[:, i:i+1]
            
            # 检查并修复结果
            op_comb = adaptive_scale(op_comb, name="op_comb")
            
            # 应用更严格的裁剪
            op_comb = torch.clamp(op_comb, min=-5.0, max=5.0)
        except Exception as e:
            print(f"Warning: Operator combination failed ({e}), using fallback")
            traceback.print_exc()
            op_comb = selected_features

        # 11. 最终输出，增强稳健性
        try:
            # 计算门控值
            gate_input = adaptive_scale(z, target_max=5.0, name="gate_input")
            gate_value = self.gate(gate_input)
            gate_value = torch.clamp(gate_value, 0.2, 0.8)  # 使用更保守的范围避免极端值
            
            # 单独计算表达式总和，使用自适应缩放
            expressions_sum = torch.zeros_like(selected_features)
            for i, expr in enumerate(expressions):
                # 在添加之前进行自适应缩放
                safe_expr = adaptive_scale(expr, target_max=1.0, name=f"expr_{i}")
                expressions_sum = expressions_sum + safe_expr
            
            # 对总和再次应用自适应缩放
            expressions_sum = adaptive_scale(expressions_sum, target_max=5.0, name="expressions_sum")
            
            # 安全地计算门控组合
            gated_expr = expressions_sum * gate_value
            
            # 确保操作符组合也进行了自适应缩放
            op_comb_safe = adaptive_scale(op_comb, target_max=5.0, name="op_comb_final")
            
            # 组合最终输出并进行最终缩放
            symbolic_output = gated_expr + (1.0 - gate_value) * op_comb_safe
            symbolic_output = adaptive_scale(symbolic_output, target_max=5.0, name="symbolic_output")
            
            # 应用tanh激活函数，确保输出在[-1,1]范围内
            symbolic_output = torch.tanh(symbolic_output)
        except Exception as e:
            print(f"Warning: Symbolic output calculation failed ({e}), using fallback")
            # 提供一个安全的回退值 - 使用tanh确保在合理范围内
            symbolic_output = torch.tanh(torch.zeros_like(selected_features))

        # 确保输出形状正确 [batch_size, 1]
        batch_size = symbolic_output.size(0)
        symbolic_output = symbolic_output.reshape(batch_size, -1)
        if symbolic_output.size(1) != 1:
            symbolic_output = symbolic_output[:, :1]

        # 12. 复杂度计算
        try:
            self.complexity = -torch.sum(
                var_weights * safe_log(var_weights), dim=-1
            ).mean()
            if isinstance(self.complexity, torch.Tensor):
                self.complexity = float(self.complexity.detach().item())
            else:
                self.complexity = float(self.complexity)
        except:
            self.complexity = 0.0

        return symbolic_output

    def compute_complexity(self):
        """计算并返回带有梯度的复杂度张量，用于损失函数优化"""
        if not hasattr(self, 'last_var_weights') or self.last_var_weights is None:
            # 如果尚未调用forward方法，返回一个默认张量
            return torch.tensor(0.5, requires_grad=True)
                
        try:
            # 重新计算复杂度，保留梯度连接
            # 注意：使用已有的 safe_log 函数而不是直接使用 torch.log
            complexity = -torch.sum(self.last_var_weights * safe_log(self.last_var_weights), dim=-1).mean()
            
            # 平滑下限，避免硬截断可能导致的梯度问题
            min_complexity = 0.1
            # 使用平滑过渡函数确保梯度始终能流动
            return complexity + min_complexity * torch.exp(-complexity/min_complexity)
        except Exception as e:
            print(f"Error computing complexity tensor: {e}")
            # 返回默认值，确保训练能继续
            return torch.tensor(0.5, device=self.last_var_weights.device, requires_grad=True)
    def get_interpretation(self):
        """返回符号层的可解释性信息"""
        interpretation = {
            "attention_weights": self.last_attn_weights.detach().cpu() if self.last_attn_weights is not None else None,
            "variable_weights": self.last_var_weights.detach().cpu() if hasattr(self, 'last_var_weights') else None,
            "operator_weights": self.last_op_weights.detach().cpu() if hasattr(self, 'last_op_weights') else None,
            "coefficients": self.last_coeffs.detach().cpu() if hasattr(self, 'last_coeffs') else None,
            "complexity": self.complexity if hasattr(self, 'complexity') else 0.0,
            "operators": self.operators
        }
        return interpretation
    
    def visualize_attention(self, feature_names=None):
        """可视化注意力权重"""
        if self.last_attn_weights is None:
            print("No attention weights available yet. Run forward pass first.")
            return None

        
        # 默认物理特征名称
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(self.last_attn_weights.shape[-1])]

        # 绘制热力图
        plt.figure(figsize=(10, 4))
        plt.imshow(self.last_attn_weights.detach().cpu().numpy().squeeze(), cmap='viridis')
        plt.colorbar(label='Attention Weight')
        plt.xlabel('Physical Features')
        plt.ylabel('Query')
        plt.title('Cross-Attention Weights')
        plt.xticks(np.arange(len(feature_names)), feature_names, rotation=90)
        plt.tight_layout()
        
        return plt


# ===== 主模型 =====
class MaterialCrackPGNN(nn.Module):
    """
    材料裂纹预测的物理信息神经网络
    使用双输入设计：标准化特征用于主网络，原始特征用于物理约束
    """
    def __init__(
        self,
        num_materials,
        material_embed_dim: int = 16,
        latent_dim: int = 32,
        dropout_rate: float = 0.1,
        num_blocks: int = 2,
        use_advanced_symbolic: bool = True,  # 增加参数控制是否使用高级符号层
    ):
        super().__init__()
        
        # 材料嵌入
        self.material_embedding = MaterialEmbedding(
            num_materials, 
            embedding_dim=material_embed_dim,
            init_gain=0.5 
        )
        
        # 物理特征提取（使用原始尺度数据）
        self.physics_extractor = PhysicsInformedFeatureExtractor()
        physics_dim = 6 + 12  # 原始特征 + 物理特征
        
        # 特征融合层
        input_dim = material_embed_dim + 6  # 只使用标准化的6个特征
        self.feature_projection = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        # 物理特征投影（单独处理物理特征）
        self.physics_projection = nn.Sequential(
            nn.Linear(physics_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        # 残差块
        self.blocks = nn.ModuleList([
            ResidualBlock(latent_dim, dropout_rate) 
            for _ in range(num_blocks)
        ])
        
        # 符号层 - 选择使用高级符号层或简化符号层
        self.use_advanced_symbolic = use_advanced_symbolic
        if use_advanced_symbolic:
            self.symbolic_layer = AdaptiveSymbolicLayer(latent_dim=latent_dim)
        else:
            self.symbolic_layer = nn.Sequential(
                nn.Linear(latent_dim, 32),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(32, 1),
                nn.Tanh()  # 限制输出范围
            )
        
        # 残差预测头
        self.residual_head = nn.Sequential(
            nn.Linear(latent_dim + 1, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 1)
        )
        
        # 不确定性预测
        self.uncertainty_head = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.LayerNorm(16),
            nn.GELU(),
            nn.Linear(16, 1)
        )

        # 控制MC Dropout与梯度监控
        self.mc_dropout_enabled = False
        self.max_grad = 0.0

    def forward(self, material_ids, numeric_scaled, numeric_raw):
        """
        前向传播，增强数值稳定性
        material_ids: [batch_size] - 材料类别ID
        numeric_scaled: [batch_size, 6] - 标准化后的特征（主网络）
        numeric_raw: [batch_size, 6] - 原始尺度特征（物理约束）
        """
        # 检查输入数据的有效性
        if torch.isnan(numeric_scaled).any() or torch.isinf(numeric_scaled).any():
            print("Warning: NaN/Inf detected in numeric_scaled, fixing...")
            numeric_scaled = torch.nan_to_num(numeric_scaled, nan=0.0, posinf=5.0, neginf=-5.0)
            numeric_scaled = torch.clamp(numeric_scaled, -10.0, 10.0)
            
        if torch.isnan(numeric_raw).any() or torch.isinf(numeric_raw).any():
            print("Warning: NaN/Inf detected in numeric_raw, fixing...")
            numeric_raw = torch.nan_to_num(numeric_raw, nan=0.0, posinf=1e6, neginf=1.0)
            
        # 1. 材料嵌入
        material_embed = self.material_embedding(material_ids)
        material_embed = adaptive_scale(material_embed, name="material_embed")
        
        # 2. 主网络特征处理（使用标准化数据）
        main_features = torch.cat([material_embed, numeric_scaled], dim=1)
        main_features = adaptive_scale(main_features, name="main_features")
        z_main = self.feature_projection(main_features)
        z_main = adaptive_scale(z_main, name="z_main")
        
        # 3. 物理特征提取（使用原始数据）
        physics_features = self.physics_extractor(numeric_raw)
        physics_features = adaptive_scale(physics_features, name="physics_features")
        z_physics = self.physics_projection(physics_features)
        z_physics = adaptive_scale(z_physics, name="z_physics")
        
        # 4. 特征融合
        fusion_input = torch.cat([z_main, z_physics], dim=1)
        fusion_input = adaptive_scale(fusion_input, name="fusion_input")
        z = self.fusion(fusion_input)
        z = adaptive_scale(z, name="z_fusion")
        
        # 5. 通过残差块，添加额外的数值检查
        for i, block in enumerate(self.blocks):
            z_input = adaptive_scale(z, name=f"block_{i}_input")
            try:
                z = block(z_input)  # 使用ResidualBlock类
            except Exception as e:
                print(f"Error in residual block {i}: {e}")
                z = z_input  # 出错时使用原始输入
            z = adaptive_scale(z, name=f"block_{i}_output")
            z = F.gelu(z)  # 额外的GELU激活
            z = adaptive_scale(z, name=f"block_{i}_gelu")
        
        # 6. 符号输出 - 根据使用的符号层类型调用不同的前向传播
        # 确保z在传给符号层之前没有数值问题
        z = adaptive_scale(z, name="symbolic_z")
        
        try:
            if self.use_advanced_symbolic:
                # 高级符号层需要潜在向量z和原始物理特征
                symbolic = self.symbolic_layer(z, numeric_raw)
                
                # 检查符号输出有效性
                if torch.isnan(symbolic).any() or torch.isinf(symbolic).any():
                    print("Warning: NaN/Inf detected in symbolic output, using fallback")
                    symbolic = torch.zeros(z.size(0), 1, device=z.device)
                
                # 检查并修复维度问题
                if symbolic.dim() > 2:
                    # 保留批次维度，展平其他维度
                    batch_size = symbolic.size(0)
                    symbolic = symbolic.reshape(batch_size, -1)
                    
                # 确保是 [batch_size, 1] 形状
                if symbolic.size(1) != 1:
                    symbolic = symbolic[:, :1]
            else:
                # 简化符号层只需要潜在向量z
                symbolic = self.symbolic_layer(z)
                
            # 应用自适应缩放并裁剪
            symbolic = adaptive_scale(symbolic, target_max=5.0, name="symbolic_final")
            symbolic = torch.clamp(symbolic, -5.0, 5.0)  # 使用更保守的限制
                
        except Exception as e:
            print(f"Error in symbolic layer: {e}")
            # 符号层失败时提供安全的回退值
            symbolic = torch.zeros(z.size(0), 1, device=z.device)
        
        # 7. 残差预测
        try:
            # 保证z和symbolic数值稳定，使用自适应缩放
            z_safe = adaptive_scale(z, name="residual_z")
            symbolic_safe = adaptive_scale(symbolic, name="residual_symbolic")
            
            residual_input = torch.cat([z_safe, symbolic_safe], dim=-1)
            residual_input = adaptive_scale(residual_input, name="residual_input")
            residual = self.residual_head(residual_input)
            residual = adaptive_scale(residual, name="residual_output")
            
        except Exception as e:
            print(f"Error in residual head: {e}")
            residual = torch.zeros_like(symbolic)
        
        # 8. 最终预测（确保正值），采用更稳定的方式
        try:
            # 先应用裁剪，防止softplus溢出
            combined = torch.clamp(symbolic + residual, -20.0, 20.0)
            prediction = F.softplus(combined, beta=1) + 0.1  # 降低beta值增加稳定性
            
            # 检查预测结果
            if torch.isnan(prediction).any() or torch.isinf(prediction).any():
                print("Warning: NaN/Inf in final prediction, using safe values")
                prediction = torch.ones_like(symbolic) * 0.5  # 安全的默认预测值
            
            # 确保预测是正值且范围合理
            prediction = torch.clamp(prediction, min=0.1, max=100.0)
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            prediction = torch.ones_like(symbolic) * 0.5
        
        # 9. 不确定性估计，增强稳定性
        try:
            # 使用adaptive_scale代替check_and_fix_tensor
            z_safe = adaptive_scale(z, name="uncertainty_z")
            log_var = self.uncertainty_head(z_safe)
            log_var = torch.clamp(log_var, -10.0, 2.0)  # 放宽下限，增加稳定性
            uncertainty = torch.exp(log_var) + 0.01
            
            # 使用adaptive_scale确保不确定性在合理范围内
            uncertainty = adaptive_scale(uncertainty, target_max=5.0, name="uncertainty")
            uncertainty = torch.clamp(uncertainty, min=0.01, max=10.0)
            
        except Exception as e:
            print(f"Error in uncertainty estimation: {e}")
            uncertainty = torch.ones_like(symbolic) * 0.1
        
        return prediction, uncertainty, symbolic, residual
    
    # 获取符号层的解释性信息（如果使用高级符号层）
    def get_symbolic_interpretation(self):
        """获取符号层的解释性信息"""
        if self.use_advanced_symbolic:
            return self.symbolic_layer.get_interpretation()
        else:
            return {"message": "使用的是简化符号层，没有可解释性信息"}
    
    # 可视化注意力权重（如果使用高级符号层）
    def visualize_symbolic_attention(self, feature_names=None):
        """可视化符号层的注意力权重"""
        if self.use_advanced_symbolic:
            return self.symbolic_layer.visualize_attention(feature_names)
        else:
            print("使用的是简化符号层，没有注意力权重可视化")
            return None

    def predict_with_uncertainty(self, material_ids, num_scaled, num_raw, n_samples=10):
        """使用MC Dropout进行带不确定性的预测，增强数值稳定性"""
        self.enable_mc_dropout(True)
        self.train()  # 保持train模式以激活Dropout

        predictions = []
        uncertainties = []
        
        # 首先检查输入数据
        if torch.isnan(num_scaled).any() or torch.isinf(num_scaled).any():
            print("Warning: NaN/Inf in input data for uncertainty prediction, fixing...")
            num_scaled = torch.nan_to_num(num_scaled, nan=0.0, posinf=5.0, neginf=-5.0)
            num_scaled = torch.clamp(num_scaled, -10.0, 10.0)

        with torch.no_grad():
            for i in range(n_samples):
                try:
                    # 获取预测并立即检查有效性
                    pred, uncertainty, _, _ = self(material_ids, num_scaled, num_raw)
                    
                    # 检查并修复异常值
                    if torch.isnan(pred).any() or torch.isinf(pred).any():
                        print(f"Warning: NaN/Inf in MC sample {i}, using safe values")
                        pred = torch.ones_like(pred) * 0.5
                        
                    if torch.isnan(uncertainty).any() or torch.isinf(uncertainty).any():
                        print(f"Warning: NaN/Inf in uncertainty for MC sample {i}, using safe values")
                        uncertainty = torch.ones_like(uncertainty) * 0.1
                    
                    # 确保值在合理范围内
                    pred = torch.clamp(pred, 0.1, 100.0)
                    uncertainty = torch.clamp(uncertainty, 0.01, 10.0)
                    
                    predictions.append(pred)
                    uncertainties.append(uncertainty)
                except Exception as e:
                    print(f"Error in MC Dropout sample {i}: {e}")
                    # 跳过出错的样本
                    continue

        # 恢复正常模式
        self.enable_mc_dropout(False)
        self.eval()
        
        # 处理特殊情况：如果所有样本都失败
        if len(predictions) == 0:
            print("Warning: All MC samples failed, using default values")
            default_pred = torch.ones_like(num_scaled[:, :1]) * 0.5
            default_unc = torch.ones_like(num_scaled[:, :1]) * 0.1
            return default_pred, default_unc, default_unc, default_unc

        # 计算均值和不确定性，添加安全检查
        try:
            # 堆叠预测结果
            stacked_predictions = torch.stack(predictions, dim=0)
            stacked_uncertainties = torch.stack(uncertainties, dim=0)
            
            # 计算均值和不确定性，使用更稳健的方法
            mean_pred = torch.median(stacked_predictions, dim=0)[0]  # 使用中位数而不是均值
            aleatoric_uncertainty = torch.median(stacked_uncertainties, dim=0)[0]
            
            # 使用范围而非方差来计算认知不确定性
            pred_max = torch.max(stacked_predictions, dim=0)[0]
            pred_min = torch.min(stacked_predictions, dim=0)[0]
            epistemic_uncertainty = (pred_max - pred_min) / (2.0 * mean_pred + 1e-6)
            
            # 限制在合理范围内
            epistemic_uncertainty = torch.clamp(epistemic_uncertainty, 0.01, 1.0)
            total_uncertainty = aleatoric_uncertainty + epistemic_uncertainty
        except Exception as e:
            print(f"Error computing uncertainty statistics: {e}")
            # 提供默认值
            mean_pred = predictions[0] if predictions else torch.ones_like(num_scaled[:, :1]) * 0.5
            aleatoric_uncertainty = uncertainties[0] if uncertainties else torch.ones_like(num_scaled[:, :1]) * 0.1
            epistemic_uncertainty = torch.ones_like(aleatoric_uncertainty) * 0.05
            total_uncertainty = aleatoric_uncertainty + epistemic_uncertainty

        return (
            mean_pred,
            total_uncertainty,
            aleatoric_uncertainty,
            epistemic_uncertainty,
        )
    
    def enable_mc_dropout(self, enable: bool = True):
        """在评估模式下显式启用或关闭Dropout"""
        self.mc_dropout_enabled = enable
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train(enable)

    def check_grad_norm(self) -> float:
        """计算并记录当前梯度范数"""
        total_norm = 0.0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        self.max_grad = max(self.max_grad, total_norm)
        return total_norm