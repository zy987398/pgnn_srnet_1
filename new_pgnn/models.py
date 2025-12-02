import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import traceback
from matplotlib import pyplot as plt
from utils import (safe_division, safe_log, safe_sqrt, safe_power, 
                  adaptive_scale,)
from light_symbolic_layer import LightweightAttentionSymbolicLayer
from ImprovedSymbolicLayer import ImprovedSymbolicLayer


class MaterialEmbedding(nn.Module):
    """材料种类嵌入层"""

    def __init__(self, num_materials, embedding_dim=16, init_gain=0.5):
        super().__init__()
        self.embedding = nn.Embedding(num_materials, embedding_dim)
        # 保持与原始代码相同的初始化方式
        nn.init.xavier_uniform_(self.embedding.weight, gain=init_gain)

    def forward(self, material_ids):
        return self.embedding(material_ids)


class PhysicsInformedFeatureExtractor(nn.Module):
    """
    物理特征提取器
    基于材料科学原理提取物理意义明确的特征
    使用改进的物理特征计算和归一化方法
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
        使用更优化的计算方法和归一化策略
        """
        # 首先检查输入
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("Warning: NaN/Inf in physics extractor input, fixing")
            x = torch.nan_to_num(x, nan=0.0, posinf=1e3, neginf=0.0)
        
        # 保存原始特征用于输出
        features = [x]
        
        # -------- 基本变量与安全裁剪 --------
        try:
            # 提取各个原始特征
            hardness = x[:, 0:1]          # 莫氏硬度
            elastic_modulus = x[:, 1:2]   # 弹性模量
            fracture_tough = x[:, 2:3]    # 断裂韧性
            grain_size = x[:, 3:4]        # 晶粒粒径
            stress = x[:, 4:5]            # 压溃压力
            velocity = x[:, 5:6]          # 磨粒速度
            
            # 增加数值稳定性的常数
            eps = 1e-6
            π = 3.14159
            
            # 计算12个物理特征
            feats = []
            
            # 1. Griffith 能量释放率 G = σ² π d / E [J·m⁻²]
            stress_squared = torch.clamp(stress**2, max=1e12)
            G = stress_squared * π * grain_size / (elastic_modulus + eps)
            G_scaled = torch.tanh(torch.clamp(G / 1e3, -10, 10))
            feats.append(G_scaled * self.feature_scales[0])
            
            # 2. 应力强度比 K/Kc = σ √(π d) / K_IC [—]
            sqrt_term = torch.sqrt(torch.clamp(π * grain_size, min=eps, max=1.0))
            k_ratio = stress * sqrt_term / (fracture_tough + eps)
            feats.append(torch.tanh(torch.clamp(k_ratio, -10, 10)) * self.feature_scales[1])
            
            # 3. Hall-Petch 尺度项 d^{-1/2} [m^{-½}]
            hall_petch = torch.rsqrt(torch.clamp(grain_size + eps, min=eps, max=1.0))
            hall_petch = torch.clamp(hall_petch, max=1e6)
            feats.append(torch.tanh(torch.clamp(hall_petch / 1e3, -10, 10)) * self.feature_scales[2])
            
            # 4. 裂纹屏蔽因子 χ = (K_IC / E) √d [m]
            k_ic_over_e = fracture_tough / (elastic_modulus + eps)
            sqrt_grain = torch.sqrt(torch.clamp(grain_size, min=eps, max=1.0))
            shielding = k_ic_over_e * sqrt_grain
            shielding = torch.clamp(shielding, max=1e-2)
            feats.append(torch.tanh(torch.clamp(shielding / 1e-4, -10, 10)) * self.feature_scales[3])
            
            # 5. 载荷-强度比 ρ₁ = σ / H [—]
            rho1 = stress / (hardness + eps)
            # 使用Min-Max缩放代替tanh，保留更多信息
            rho1_min = torch.min(rho1)
            rho1_max = torch.max(rho1)
            rho1_range = rho1_max - rho1_min + eps
            rho1_norm = (rho1 - rho1_min) / rho1_range
            feats.append(rho1_norm * self.feature_scales[4])
            
            # 6. 功率-强度比 ρ₂ = σ V / H [m·s⁻¹]
            rho2 = stress * velocity / (hardness + eps)
            # 使用Min-Max缩放
            rho2_min = torch.min(rho2)
            rho2_max = torch.max(rho2)
            rho2_range = rho2_max - rho2_min + eps
            rho2_norm = (rho2 - rho2_min) / rho2_range
            feats.append(rho2_norm * self.feature_scales[5])
            
            # 7. 延-脆转临界深度 d_c = η (E/H) (K_IC/H)² [m] (η≈0.15)
            eta = 0.15
            e_over_h = elastic_modulus / (hardness + eps)
            k_over_h = fracture_tough / (hardness + eps)
            k_over_h_squared = k_over_h**2  # 移除过度截断
            d_c = eta * e_over_h * k_over_h_squared
            # 使用标准化缩放而非tanh
            d_c_mean = torch.mean(d_c)
            d_c_std = torch.std(d_c, unbiased=False) + eps  # 添加unbiased=False
            d_c_norm = (d_c - d_c_mean) / d_c_std * 0.3 + 0.5

            feats.append(torch.clamp(d_c_norm, 0, 1) * self.feature_scales[6])
            
            # 8. 临界裂纹长度比 a_c / d, a_c = (K_IC/σ)² / π [—]
            k_over_s = fracture_tough / (stress + eps)
            k_over_s_squared = k_over_s**2  # 移除过度截断
            a_c = k_over_s_squared / π
            ac_over_d = a_c / (grain_size + eps)
            # 对数缩放后再标准化
            ac_log = torch.log1p(torch.clamp(ac_over_d, min=eps))
            ac_log_mean = torch.mean(ac_log)
            ac_log_std = torch.std(ac_log, unbiased=False) + eps  # 添加unbiased=False
            ac_norm = (ac_log - ac_log_mean) / ac_log_std * 0.3 + 0.5

            feats.append(torch.clamp(ac_norm, 0, 1) * self.feature_scales[7])
            
            # 9. 塑性区尺寸比 r_p / d, r_p = (K_IC / H)² / (2π) [—]
            r_p = k_over_h_squared / (2.0 * π)
            rp_over_d = r_p / (grain_size + eps)
            # 对数缩放后再使用Min-Max归一化
            rp_log = torch.log1p(torch.clamp(rp_over_d, min=eps))
            rp_log_min = torch.min(rp_log)
            rp_log_max = torch.max(rp_log)
            rp_log_range = rp_log_max - rp_log_min + eps
            rp_norm = (rp_log - rp_log_min) / rp_log_range
            feats.append(rp_norm * self.feature_scales[8])
            
            # 10. 脆性指数 B = H / K_IC [√m⁻¹]
            B = hardness / (fracture_tough + eps)
            # 使用Min-Max缩放代替tanh，避免信息损失
            B_min = torch.min(B)
            B_max = torch.max(B)
            B_range = B_max - B_min + eps
            B_norm = (B - B_min) / B_range
            feats.append(B_norm * self.feature_scales[9])
            
            # 11. 初-剖比 K_IC / E [√m]
            ki_over_e = fracture_tough / (elastic_modulus + eps)
            # 使用标准化归一化
            ki_over_e_mean = torch.mean(ki_over_e)
            ki_over_e_std = torch.std(ki_over_e, unbiased=False) + eps  # 添加unbiased=False
            ki_over_e_norm = (ki_over_e - ki_over_e_mean) / ki_over_e_std * 0.3 + 0.5  # 缩放到大约[0.2-0.8]范围
            feats.append(torch.clamp(ki_over_e_norm, 0, 1) * self.feature_scales[10])
            
            # 12. 弹性应变能密度 U = 0.5 σ² / E [J·m⁻³]
            U = 0.5 * stress_squared / (elastic_modulus + eps)
            # 对数缩放后使用Min-Max归一化
            U_log = torch.log1p(torch.clamp(U, min=eps))
            U_log_min = torch.min(U_log)
            U_log_max = torch.max(U_log)
            U_log_range = U_log_max - U_log_min + eps
            U_norm = (U_log - U_log_min) / U_log_range
            feats.append(U_norm * self.feature_scales[11])
            
        except Exception as e:
            print(f"Error in physics feature extraction: {e}")
            traceback.print_exc()
            # 创建安全的默认值
            feats = [torch.zeros_like(x) for _ in range(12)]
            
        # 将原始特征与提取的物理特征合并
        all_features = torch.cat(features + feats, dim=1)  # 6个原始特征 + 12个物理特征 = 18个总特征
        
        return all_features


class ResidualBlock(nn.Module):
    """残差连接块，提高训练稳定性"""

    def __init__(self, dim, dropout_rate=0.1):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * 2)
        self.act = nn.GELU()  # 替换ReLU为GELU获得更平滑的梯度
        self.layer_norm2 = nn.LayerNorm(dim * 2)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(dim * 2, dim)

    def forward(self, x):
        residual = x
        out = self.layer_norm1(x)
        out = self.fc1(out)
        out = self.act(out)
        out = self.layer_norm2(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out + residual  # 残差连接




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
        # 仅对输入z进行一次自适应缩放
        # z = adaptive_scale(z, target_max=10.0, name="symbolic_z_input")
        
        # 1. 提取物理特征
        physics_features = self.physics_extractor(numeric_raw)
        
        # 2. 准备交叉注意力的输入
        # z作为query
        query = self.query_proj(z).unsqueeze(1)  # [B, 1, latent_dim]
        
        # 物理特征作为key和value - 每个特征单独编码
        batch_size = physics_features.size(0)
        phys_features_seq = physics_features.unsqueeze(2)  # [B, num_features, 1]
        key = self.key_proj(phys_features_seq)    # [B, num_features, latent_dim]
        value = self.value_proj(phys_features_seq)  # [B, num_features, latent_dim]            # 3. 应用交叉注意力（第一个Transformer块），增强稳健性
        try:
            # 直接应用注意力机制，无需中间缩放
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
            
            # 第一个残差连接 + 层归一化，简化方式
            z_attn = z + 0.1 * attn_output  # 减少注意力输出的影响
            
            # 应用层归一化，包含错误处理
            try:
                z_attn = self.attn_norm(z_attn)
            except Exception as e:
                print(f"Warning: Layer normalization failed: {e}")
                # 如果归一化失败，使用简单的缩放和中心化
                mean = z_attn.mean(dim=-1, keepdim=True)
                std = z_attn.std(dim=-1, keepdim=True) + 1e-5
                z_attn = (z_attn - mean) / std
            
            # 4. 前馈网络（第二个Transformer块）
            ffn_output = self.feed_forward(z_attn)
            
            # 第二个残差连接 + 层归一化
            z = z_attn + ffn_output  # 残差连接 2
            z = self.ffn_norm(z)
            
        except Exception as e:
            print(f"Warning: Transformer processing failed ({e}), using original z")
            traceback.print_exc()
            z = adaptive_scale(z, name="transformer_fallback")

        # 5. 特征选择权重 - 使用更稳定的版本
        try:
            # 直接使用z而不进行额外缩放
            
            # 计算权重和系数
            var_logits = self.variable_selector(z)
            var_logits = torch.clamp(var_logits, -5, 5)  # 限制logits范围
            var_weights = F.softmax(var_logits, dim=-1)
            
            op_logits = self.operator_selector(z)
            op_logits = torch.clamp(op_logits, -5, 5)  # 限制logits范围
            op_weights = F.softmax(op_logits, dim=-1)
            
            # 限制系数范围，避免过大值
            coeffs = torch.tanh(self.coefficient_net(z)) * 3.0  # 使用tanh限制在[-3,3]范围
            
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
            
            # 7. 加权特征组合
            selected_features = (var_weights * pooled_features).sum(dim=1, keepdim=True)
        except Exception as e:
            print(f"Warning: Feature pooling failed: {e}")
            # 提供安全的后备值
            selected_features = torch.zeros(batch_size, 1, device=z.device)

        # 8.符号表达式计算 - 使用更严格的缩放
        try:
            expressions = []
            
            # 线性项 - 直接乘法
            expr1 = coeffs[:, 0:1] * selected_features
            expressions.append(expr1)
            
            # 二次项 - 使用safe_power
            expr2_base = safe_power(selected_features, 2, max_val=25.0)
            expr2 = coeffs[:, 1:2] * expr2_base
            expressions.append(expr2)
            
            # 三次项 - 使用safe_power
            expr3_base = safe_power(selected_features, 3, max_val=25.0)
            expr3 = coeffs[:, 2:3] * expr3_base
            expressions.append(expr3)
            
            # 平方根项 - 使用safe_sqrt
            expr4_base = safe_sqrt(selected_features)
            expr4 = coeffs[:, 3:4] * expr4_base
            expressions.append(expr4)
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
            
            # 不再为每个操作应用自适应缩放
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
            
            # 确保权重形状匹配
            weights_clean = op_weights
            
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
            
            # 只应用裁剪，不做多余缩放
            op_comb = torch.clamp(op_comb, min=-5.0, max=5.0)
        except Exception as e:
            print(f"Warning: Operator combination failed ({e}), using fallback")
            traceback.print_exc()
            op_comb = selected_features

        # 11. 最终输出，增强稳健性
        try:
            # 计算门控值
            gate_value = self.gate(z)
            gate_value = torch.clamp(gate_value, 0.2, 0.8)  # 使用更保守的范围避免极端值
            
            # 计算表达式总和，无需中间缩放
            expressions_sum = torch.zeros_like(selected_features)
            for expr in expressions:
                expressions_sum = expressions_sum + expr
            
            # 计算门控组合
            gated_expr = expressions_sum * gate_value
            
            # 组合最终输出并只在最终输出进行一次自适应缩放
            symbolic_output = gated_expr + (1.0 - gate_value) * op_comb
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
            print("Warning: Failed to compute complexity value.")
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


class MaterialCrackPGNN(nn.Module):
    """专门用于材料裂纹深度预测的PGNN"""

    def __init__(
        self, num_materials, material_embed_dim=16, latent_dim=32, dropout_rate=0.1,
        symbolic_type="lightweight"  # 符号层类型：'advanced', 'improved', 'lightweight'
    ):
        super().__init__()

        # 材料嵌入
        self.material_embedding = MaterialEmbedding(num_materials, material_embed_dim)

        # 物理特征提取
        self.physics_extractor = PhysicsInformedFeatureExtractor()

        # 改进：计算特征总维度 (使用新的物理特征提取器)
        physics_dim = 18  # 6个原始特征 + 12个物理特征 = 18个总特征
        input_dim = material_embed_dim + physics_dim

        # 编码器 (使用残差连接)
        self.input_layer = nn.Linear(input_dim, latent_dim)
        self.encoder_norm = nn.LayerNorm(latent_dim)

        # 添加残差块
        self.residual_blocks = nn.ModuleList(
            [ResidualBlock(latent_dim, dropout_rate) for _ in range(2)]
        )

        # 符号层 - 选择使用不同的符号层实现
        self.symbolic_type = symbolic_type
        
        if symbolic_type == "advanced":
            # 完整版高级符号层（最复杂，带Transformer架构）
            self.symbolic_layer = AdaptiveSymbolicLayer(latent_dim=latent_dim)
            self.use_advanced_symbolic = True
        elif symbolic_type == "lightweight":
            # 轻量级注意力符号层（平衡性能与复杂度）
            self.symbolic_layer = LightweightAttentionSymbolicLayer(latent_dim=latent_dim)
            self.use_advanced_symbolic = True  # 仍视为高级符号层（有注意力机制和可解释性）
        else:
            # 基础版符号层（最简单）
            self.symbolic_layer = ImprovedSymbolicLayer(latent_dim)
            self.use_advanced_symbolic = False

        # 残差网络（捕获非线性效应）
        self.residual_net = nn.Sequential(
            nn.Linear(latent_dim + 1, 64),
            nn.GELU(),
            nn.LayerNorm(64),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

        # 新增：符号输出对齐网络（学习最佳的scale和bias）
        self.sym_scale_net = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.GELU(),
            nn.Linear(16, 2)  # 输出 [scale_raw, bias_raw]
        )
        
        # 新增：融合门控网络（决定符号层vs残差的权重）
        self.fusion_gate = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Sigmoid()  # gate ∈ (0,1)
        )
        
        # 记录最近一次的门控值（用于可解释性分析）
        self.last_gate = None

        # 不确定性预测 - 分离认知不确定性和随机不确定性
        self.aleatoric_net = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),  # 输出log_var (随机不确定性)
        )
        
        # 加权自适应组合网络 - 为符号层和残差网络输出动态分配权重
        self.alpha_net = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.GELU(),
            nn.LayerNorm(16),
            nn.Linear(16, 1),
            nn.Sigmoid()  # 输出范围[0,1]，表示符号层的权重
        )

        # 输出层（确保正值）
        self.output_activation = nn.Softplus()

        # MC Dropout的开关
        self.mc_dropout_enabled = False

    def enable_mc_dropout(self, enable=True):
        """启用/禁用MC Dropout"""
        self.mc_dropout_enabled = enable

    def forward(self, material_ids, numeric_features):
        """
        material_ids: [batch_size] - 材料类别ID
        numeric_features: [batch_size, 6] - 数值特征
        """
        # 材料嵌入
        material_embed = self.material_embedding(material_ids)

        # 提取物理特征
        physics_features = self.physics_extractor(numeric_features)

        # 组合所有特征
        combined_features = torch.cat([material_embed, physics_features], dim=1)

        # 编码到潜在空间 (使用残差结构)
        z = self.input_layer(combined_features)
        z = self.encoder_norm(z)

        # 通过残差块
        for block in self.residual_blocks:
            z = block(z)
            # 在推理时如果启用MC Dropout，保持dropout激活
            if self.mc_dropout_enabled and not self.training:
                z = F.dropout(z, p=0.1, training=True)

        # 符号表达式输出 - 根据使用的符号层类型调用不同的前向传播
        try:
            if self.symbolic_type == "improved":
                # 基础符号层，传入潜在向量z和物理特征
                symbolic_output = self.symbolic_layer(z, physics_features)
            else:
                # 高级符号层和轻量级符号层都需要潜在向量z和物理特征
                if self.symbolic_type == "lightweight":
                    # 轻量级注意力符号层 - 需要使用physics_features (18维)，而不是numeric_features
                    symbolic_output = self.symbolic_layer(z, physics_features)
                else:
                    # 原始高级符号层
                    symbolic_output = self.symbolic_layer(z, numeric_features)
                
                # 检查符号输出有效性
                if torch.isnan(symbolic_output).any() or torch.isinf(symbolic_output).any():
                    print(f"Warning: NaN/Inf detected in {self.symbolic_type} symbolic output, using fallback")
                    symbolic_output = torch.zeros(z.size(0), 1, device=z.device)
                
                # 检查是否返回了元组（包含注意力权重和操作符权重）
                if isinstance(symbolic_output, tuple):
                    symbolic_output = symbolic_output[0]  # 取第一个元素作为输出值
                
                # 检查并修复维度问题
                if symbolic_output.dim() > 2:
                    # 保留批次维度，展平其他维度
                    batch_size = symbolic_output.size(0)
                    symbolic_output = symbolic_output.reshape(batch_size, -1)
                    
                # 确保是 [batch_size, 1] 形状
                if symbolic_output.size(1) != 1:
                    symbolic_output = symbolic_output[:, :1]
        except Exception as e:
            print(f"Error in symbolic layer: {e}")
            # 符号层失败时提供安全的回退值
            symbolic_output = torch.zeros(z.size(0), 1, device=z.device)

        # 新增：符号输出对齐（学习最佳的幅值缩放和偏置）
        scale_bias = self.sym_scale_net(z)
        scale_raw = scale_bias[:, 0:1]
        bias_raw = scale_bias[:, 1:2]
        
        # 使用exp确保scale为正，用tanh限制bias范围
        scale = torch.exp(scale_raw * 0.5)  # scale ∈ (0, ∞)，但通常在(0.6, 1.6)范围
        bias = torch.tanh(bias_raw) * 0.5   # bias ∈ (-0.5, 0.5)
        
        # 对齐符号输出
        symbolic_aligned = scale * symbolic_output + bias

        # 新增：计算融合门控
        gate = self.fusion_gate(z)  # gate ∈ (0,1)
        self.last_gate = gate.detach().clone()  # 保存用于分析
        
        # 修改：残差网络输入 - 移除detach()，让梯度双向流动
        residual_input = torch.cat([z, symbolic_output], dim=-1)  # 不再使用sym_detached
        residual_output = self.residual_net(residual_input)

        # 新增：门控加权融合（这是核心改进）
        prediction = gate * symbolic_aligned + (1 - gate) * residual_output
        
        # 应用激活函数确保非负
        prediction = F.softplus(prediction, beta=5) + 1e-3

        # 不确定性估计 - 随机不确定性
        log_var = self.aleatoric_net(z).clamp(-10, 6)
        uncertainty = torch.exp(log_var) + 1e-6

        return prediction, uncertainty, symbolic_output, residual_output

    def predict_with_uncertainty(self, material_ids, numeric_features, n_samples=10):
        """使用MC Dropout进行带不确定性的预测"""
        self.enable_mc_dropout(True)  # 启用MC Dropout
        self.eval()  # 设置为评估模式

        predictions = []
        uncertainties = []
        alphas = []  # 存储每次采样的权重值

        with torch.no_grad():
            for _ in range(n_samples):
                pred, uncertainty, _, _ = self(material_ids, numeric_features)
                predictions.append(pred)
                uncertainties.append(uncertainty)
                
                # 存储当前的alpha权重（如果可用）
                if hasattr(self, 'last_alpha'):
                    alphas.append(self.last_alpha.clone())

        # 恢复正常模式
        self.enable_mc_dropout(False)
        
        # 如果收集了alpha值，计算其平均值和方差
        if alphas:
            self.mc_alpha_mean = torch.mean(torch.stack(alphas), dim=0)
            self.mc_alpha_var = torch.var(torch.stack(alphas), dim=0)

        # 计算均值和不确定性
        mean_pred = torch.mean(torch.stack(predictions), dim=0)
        # 总不确定性 = 认知不确定性 + 随机不确定性
        aleatoric_uncertainty = torch.mean(
            torch.stack(uncertainties), dim=0
        )  # 随机不确定性
        epistemic_uncertainty = torch.var(
            torch.stack(predictions), dim=0
        )  # 认知不确定性
        total_uncertainty = aleatoric_uncertainty + epistemic_uncertainty

        return (
            mean_pred,
            total_uncertainty,
            aleatoric_uncertainty,
            epistemic_uncertainty,
        )
        
    # 获取符号层的解释性信息（如果使用高级符号层）
    def get_symbolic_interpretation(self):
        """获取符号层的解释性信息"""
        interpretation = {}
        
        # 添加自适应权重信息
        if hasattr(self, 'last_alpha'):
            interpretation["alpha_weight"] = float(self.last_alpha.detach().mean().item())
            
        # 添加符号层解释
        if self.use_advanced_symbolic:
            symbolic_interpretation = self.symbolic_layer.get_interpretation()
            interpretation.update(symbolic_interpretation)
        else:
            interpretation["symbolic_message"] = "使用的是简化符号层，没有可解释性信息"
            
        return interpretation
    
    # 可视化注意力权重（如果使用高级符号层）
    def visualize_symbolic_attention(self, feature_names=None):
        """可视化符号层的注意力权重"""
        if self.use_advanced_symbolic:
            return self.symbolic_layer.visualize_attention(feature_names)
        else:
            print("使用的是简化符号层，没有注意力权重可视化")
            return None
    def get_fusion_interpretation(self):
        """分析融合门的行为"""
        if self.last_gate is None:
            return "模型尚未进行前向传播"
        
        stats = self.get_gate_statistics()
        mean_gate = stats['mean']
        
        if mean_gate > 0.7:
            return f"模型强烈依赖符号层（物理先验主导），平均信任度: {mean_gate:.2f}"
        elif mean_gate > 0.5:
            return f"模型偏向符号层，但残差仍有贡献，平均信任度: {mean_gate:.2f}"
        elif mean_gate > 0.3:
            return f"模型在符号层和残差之间较好平衡，平均信任度: {mean_gate:.2f}"
        else:
            return f"模型更依赖残差网络（数据驱动为主），平均信任度: {mean_gate:.2f}"
    
    def get_gate_statistics(self):
        """获取门控统计信息"""
        if self.last_gate is None:
            return {
                'mean': 0.5,
                'std': 0.0,
                'min': 0.5,
                'max': 0.5,
                'symbolic_weight': 0.5,
                'residual_weight': 0.5
            }
        
        gate = self.last_gate
        mean_gate = float(gate.mean().item())
        std_gate = float(gate.std().item())
        min_gate = float(gate.min().item())
        max_gate = float(gate.max().item())
        
        return {
            'mean': mean_gate,
            'std': std_gate,
            'min': min_gate,
            'max': max_gate,
            'symbolic_weight': mean_gate,
            'residual_weight': 1.0 - mean_gate
        }
