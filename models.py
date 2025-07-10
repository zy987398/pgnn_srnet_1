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
import numpy as np
# 全局梯度裁剪阈值
GRAD_CLIP = 5.0
# 全局安全常量
MAX_SAFE_VAL = 1e4      # 降低最大安全值
MIN_SAFE_EPS = 1e-6     # 确保最小安全值
SQRT_MAX = 1e3          # 开方最大值
LOG_MAX = 1e3           # 对数最大值
EXP_MAX = 4.0           # 指数最大值，e^4 ≈ 54.6，比e^10小得多

# ===== 安全数值操作函数 =====
def safe_division(numerator, denominator, eps=MIN_SAFE_EPS, max_val=MAX_SAFE_VAL):
    """安全除法，防止除零和数值爆炸"""
    # 预处理分子，避免过大值
    numerator = torch.clamp(numerator, min=-max_val, max=max_val)
    # 确保分母不为零且不过大
    denominator = torch.clamp(torch.abs(denominator) + eps, min=eps, max=max_val)
    # 计算结果并再次限制范围
    result = numerator / denominator
    return torch.clamp(result, min=-max_val/10, max=max_val/10)  # 额外限制结果范围

def safe_sqrt(x, eps=MIN_SAFE_EPS):
    """安全开方运算"""
    # 确保输入为非负且有合理上限
    x_safe = torch.clamp(torch.abs(x) + eps, min=eps, max=SQRT_MAX)
    return torch.sqrt(x_safe)

def safe_log(x, eps=MIN_SAFE_EPS):
    """安全对数运算"""
    # 确保输入严格为正且有合理上限
    x_safe = torch.clamp(torch.abs(x) + eps, min=eps, max=LOG_MAX)
    return torch.log(x_safe)

def safe_exp(x, max_val=EXP_MAX):
    """安全指数运算，输出范围更加有限"""
    # 严格限制输入范围，避免e^x爆炸
    x_safe = torch.clamp(x, min=-max_val, max=max_val)
    return torch.exp(x_safe)

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

def check_and_fix_tensor(tensor, name="tensor"):
    """检查并修复tensor中的NaN和Inf"""
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        print(f"Warning: NaN/Inf detected in {name}, replacing with safe values")
        tensor = torch.nan_to_num(tensor, nan=0.0, posinf=MAX_SAFE_VAL/100, neginf=-MAX_SAFE_VAL/100)
    return torch.clamp(tensor, min=-MAX_SAFE_VAL, max=MAX_SAFE_VAL)


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
        self.feature_scales = nn.Parameter(torch.ones(13))
    
    def forward(self, x):
        """
        输入x: [batch_size, 6] - 原始尺度的特征
        返回: [batch_size, 19] - 原始特征 + 13个物理特征
        """
        features = [x]
        
        # 提取各个特征（原始尺度）
        hardness = x[:, 0:1].clamp(min=0.0)  # 硬度应为正值
        elastic_modulus = x[:, 1:2].clamp(min=1.0)
        fracture_toughness = x[:, 2:3].clamp(min=0.1)
        grain_size = x[:, 3:4].clamp(min=0.001)
        pressure = x[:, 4:5]
        velocity = x[:, 5:6].clamp(min=0.0, max=1e3) 
        
        eps = 1e-6
        # 各种物理特征计算
        # 1. Griffith能量释放率 G = σ²πa/E
        energy_release = (pressure ** 2 * np.pi * grain_size) / (elastic_modulus + eps)
        energy_release = torch.tanh(energy_release / 100) * self.feature_scales[0]
        features.append(energy_release)
        
        # 2. 应力强度因子比 K/Kc
        k_ratio = (pressure * torch.sqrt(np.pi * grain_size)) / (fracture_toughness + eps)
        k_ratio = torch.tanh(k_ratio) * self.feature_scales[1]
        features.append(k_ratio)
        
        # 3. Hertz接触应力
        contact_stress = torch.pow(pressure * elastic_modulus, 1/3)
        contact_stress = torch.tanh(contact_stress / 10) * self.feature_scales[2]
        features.append(contact_stress)
        
        # 4. Hall-Petch关系 σy ∝ 1/√d
        hall_petch = 1.0 / torch.sqrt(grain_size + eps)
        hall_petch = torch.tanh(hall_petch) * self.feature_scales[3]
        features.append(hall_petch)
        
        # 5. 脆性指数 B = H/Kc
        brittleness = hardness / (fracture_toughness + eps)
        brittleness = torch.tanh(brittleness / 10) * self.feature_scales[4]
        features.append(brittleness)
        
        # 6. 能量输入率
        energy_input = pressure * velocity
        energy_input = torch.tanh(energy_input / 1000) * self.feature_scales[5]
        features.append(energy_input)
        
        # 7. 韧性模量比 Kc/E
        toughness_modulus = fracture_toughness / (elastic_modulus + eps)
        toughness_modulus = torch.tanh(toughness_modulus * 100) * self.feature_scales[6]
        features.append(toughness_modulus)
        
        # 8. H/E比值（耐磨性指标）
        he_ratio = hardness / (elastic_modulus + eps)
        he_ratio = torch.tanh(he_ratio * 10) * self.feature_scales[7]
        features.append(he_ratio)
        
        # 9. 断裂倾向 G/Gc
        fracture_tendency = energy_release / (fracture_toughness ** 2 / elastic_modulus + eps)
        fracture_tendency = torch.tanh(fracture_tendency) * self.feature_scales[8]
        features.append(fracture_tendency)
        
        # 10. 应变能密度
        strain_energy = 0.5 * pressure ** 2 / (elastic_modulus + eps)
        strain_energy = torch.tanh(strain_energy) * self.feature_scales[9]
        features.append(strain_energy)
        
        # 11. 临界裂纹长度
        critical_crack = (fracture_toughness / (pressure + eps)) ** 2 / np.pi
        critical_crack = torch.tanh(critical_crack / grain_size) * self.feature_scales[10]
        features.append(critical_crack)
        
        # 12. 塑性区尺寸
        plastic_zone = (fracture_toughness / (hardness + eps)) ** 2
        plastic_zone = torch.tanh(plastic_zone / grain_size) * self.feature_scales[11]
        features.append(plastic_zone)
        
        # 13. 疲劳参数
        fatigue_param = velocity * pressure / (fracture_toughness + eps)
        fatigue_param = torch.tanh(fatigue_param / 100) * self.feature_scales[12]
        features.append(fatigue_param)
        
        return torch.cat(features, dim=1)


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
            nn.Linear(64, 8),
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
        # 输入检查
        z = check_and_fix_tensor(z, "symbolic_z")
        numeric_raw = check_and_fix_tensor(numeric_raw, "symbolic_numeric_raw")
        
        # 1. 提取物理特征
        physics_features = self.physics_extractor(numeric_raw)
        physics_features = check_and_fix_tensor(physics_features, "extracted_physics_features")
        
        # 2. 准备交叉注意力的输入
        # z作为query
        query = self.query_proj(z).unsqueeze(1)  # [B, 1, latent_dim]
        
        # 物理特征作为key和value - 每个特征单独编码
        batch_size = physics_features.size(0)
        phys_features_seq = physics_features.unsqueeze(2)  # [B, num_features, 1]
        key = self.key_proj(phys_features_seq)    # [B, num_features, latent_dim]
        value = self.value_proj(phys_features_seq)  # [B, num_features, latent_dim]
        
        # 3. 应用交叉注意力（第一个Transformer块）
        try:
            attn_output, attn_weights = self.cross_attention(
                query=query,
                key=key,
                value=value,
                need_weights=True
            )
            
            # 保存注意力权重用于可解释性分析
            self.last_attn_weights = attn_weights
            
            # 提取注意力输出
            attn_output = attn_output.squeeze(1)
            
            # 第一个残差连接 + 层归一化
            z_attn = z + attn_output  # 残差连接 1
            z_attn = self.attn_norm(z_attn)
            z_attn = check_and_fix_tensor(z_attn, "attention_output")
            
            # 4. 前馈网络（第二个Transformer块）
            ffn_output = self.feed_forward(z_attn)
            
            # 第二个残差连接 + 层归一化
            z = z_attn + ffn_output  # 残差连接 2
            z = self.ffn_norm(z)
            z = check_and_fix_tensor(z, "ffn_output")
            
        except Exception as e:
            print(f"Warning: Transformer processing failed ({e}), using original z")
            traceback.print_exc()
            z = check_and_fix_tensor(z, "transformer_fallback")

        # 5. 特征选择权重
        var_weights = F.softmax(torch.clamp(self.variable_selector(z), -5, 5), dim=-1)
        op_weights = F.softmax(torch.clamp(self.operator_selector(z), -5, 5), dim=-1)
        coeffs = torch.clamp(self.coefficient_net(z), -5, 5)
        
        # 保存权重用于可解释性
        self.last_var_weights = var_weights
        self.last_op_weights = op_weights
        self.last_coeffs = coeffs
        
        # 6. 获取一个池化版本的物理特征用于符号表达式
        pool_size = min(8, physics_features.size(1))
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
        
        pooled_features = check_and_fix_tensor(pooled_features, "pooled_features")
        
        # 7. 加权特征组合
        selected_features = (var_weights * pooled_features).sum(dim=1, keepdim=True)
        selected_features = check_and_fix_tensor(selected_features, "selected_features")

        # 8.符号表达式计算
        expressions = []
        expressions.append(coeffs[:, 0:1] * selected_features)
        expressions.append(coeffs[:, 1:2] * safe_power(selected_features, 2))
        expressions.append(coeffs[:, 2:3] * safe_power(selected_features, 3))
        expressions.append(coeffs[:, 3:4] * safe_sqrt(selected_features))

        # 9算子操作
        ops = [
            selected_features,
            safe_division(torch.ones_like(selected_features), selected_features),
            safe_power(selected_features, 2),
            safe_power(selected_features, 3),
            safe_sqrt(selected_features),
            safe_log(selected_features),
            safe_exp(torch.clamp(selected_features, -5, 5)),
            torch.sin(torch.clamp(selected_features, -10, 10))
        ]
        
        ops = [check_and_fix_tensor(op, f"op_{i}") for i, op in enumerate(ops)]
        
        # 10. 算子组合
        try:
            op_stack = torch.stack(ops, dim=1).squeeze(-1)
            op_stack = check_and_fix_tensor(op_stack, "op_stack")
            op_comb = op_stack @ op_weights.unsqueeze(-1)
            op_comb = check_and_fix_tensor(op_comb, "op_comb")
        except Exception as e:
            print(f"Warning: Operator combination failed ({e}), using fallback")
            op_comb = selected_features

        # 11. 最终输出
        gate_value = self.gate(z)
        gate_value = check_and_fix_tensor(gate_value, "gate_value")

        symbolic_output = sum(expressions) * gate_value + op_comb
        symbolic_output = check_and_fix_tensor(symbolic_output, "symbolic_output")

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
        physics_dim = 6 + 13  # 原始特征 + 物理特征
        
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
        前向传播
        material_ids: [batch_size] - 材料类别ID
        numeric_scaled: [batch_size, 6] - 标准化后的特征（主网络）
        numeric_raw: [batch_size, 6] - 原始尺度特征（物理约束）
        """
        # 1. 材料嵌入
        material_embed = self.material_embedding(material_ids)
        
        # 2. 主网络特征处理（使用标准化数据）
        main_features = torch.cat([material_embed, numeric_scaled], dim=1)
        z_main = self.feature_projection(main_features)
        
        # 3. 物理特征提取（使用原始数据）
        physics_features = self.physics_extractor(numeric_raw)
        z_physics = self.physics_projection(physics_features)
        
        # 4. 特征融合
        z = self.fusion(torch.cat([z_main, z_physics], dim=1))
        
        # 5. 通过残差块
        for block in self.blocks:
            z = block(z)  # 使用ResidualBlock类
            z = F.gelu(z)  # 额外的GELU激活
        
        # 6. 符号输出 - 根据使用的符号层类型调用不同的前向传播
        if self.use_advanced_symbolic:
            # 高级符号层需要潜在向量z和原始物理特征
            symbolic = self.symbolic_layer(z, numeric_raw)
            
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
        
        # 7. 残差预测
        residual_input = torch.cat([z, symbolic], dim=-1)
        residual = self.residual_head(residual_input)
        
        # 8. 最终预测（确保正值）
        prediction = F.softplus(symbolic + residual, beta=2) + 0.1
        
        # 9. 不确定性估计
        log_var = self.uncertainty_head(z).clamp(-4, 2)
        uncertainty = torch.exp(log_var) + 0.01
        
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
        """使用MC Dropout进行带不确定性的预测"""
        self.enable_mc_dropout(True)
        self.train()  # 保持train模式以激活Dropout

        predictions = []
        uncertainties = []

        with torch.no_grad():
            for _ in range(n_samples):
                pred, uncertainty, _, _ = self(material_ids, num_scaled, num_raw)
                predictions.append(pred)
                uncertainties.append(uncertainty)

        # 恢复正常模式
        self.enable_mc_dropout(False)
        self.eval()

        # 计算均值和不确定性
        mean_pred = torch.mean(torch.stack(predictions), dim=0)
        aleatoric_uncertainty = torch.mean(torch.stack(uncertainties), dim=0)
        epistemic_uncertainty = torch.var(torch.stack(predictions), dim=0)
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