"""
模型架构定义
包含所有神经网络模型组件和主模型类
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 全局梯度裁剪阈值
GRAD_CLIP = 5.0

# ===== 安全数值操作函数 =====
def safe_division(numerator, denominator, eps=1e-6, max_val=1e6):
    """安全除法，防止除零和数值爆炸"""
    denominator = torch.clamp(torch.abs(denominator) + eps, min=eps, max=max_val)
    result = numerator / denominator
    return torch.clamp(result, min=-max_val, max=max_val)

def safe_sqrt(x, eps=1e-8):
    """安全开方运算"""
    return torch.sqrt(torch.clamp(torch.abs(x) + eps, min=eps, max=1e6))

def safe_log(x, eps=1e-8):
    """安全对数运算"""
    return torch.log(torch.clamp(torch.abs(x) + eps, min=eps, max=1e6))

def safe_exp(x, max_val=10):
    """安全指数运算"""
    return torch.exp(torch.clamp(x, min=-max_val, max=max_val))

def check_and_fix_tensor(tensor, name="tensor"):
    """检查并修复tensor中的NaN和Inf"""
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        print(f"Warning: NaN/Inf detected in {name}, replacing with safe values")
        tensor = torch.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=-1.0)
    return torch.clamp(tensor, min=-1e6, max=1e6)


# ===== 基础组件 =====
class MaterialEmbedding(nn.Module):
    """材料种类嵌入层"""
    def __init__(self, num_materials, embedding_dim=16):
        super().__init__()
        self.embedding = nn.Embedding(num_materials, embedding_dim)

    def forward(self, material_ids):
        return self.embedding(material_ids)


class ResidualBlock(nn.Module):
    """残差连接块，提高训练稳定性"""
    def __init__(self, dim, dropout_rate=0.1):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * 2)
        self.act = nn.GELU()
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
        return out + residual


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
        hardness = x[:, 0:1]
        elastic_modulus = x[:, 1:2].clamp(min=1.0)
        fracture_toughness = x[:, 2:3].clamp(min=0.1)
        grain_size = x[:, 3:4].clamp(min=0.001)
        pressure = x[:, 4:5]
        velocity = x[:, 5:6]
        
        eps = 1e-6
        
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
    自适应符号层
    使用注意力机制和多种数学操作符生成符号表达式
    """
    def __init__(self, latent_dim: int = 32):
        super().__init__()
        
        self.num_features = 19
        # 确保注意力头数能整除latent_dim
        self.num_heads = max(1, (latent_dim // 8) * 2)
        if latent_dim % self.num_heads != 0:
            self.num_heads = 1
            
        self.attention = nn.MultiheadAttention(
            latent_dim, num_heads=self.num_heads, batch_first=True
        )
        self.attn_norm = nn.LayerNorm(latent_dim)
        
        self.operators = ["*", "/", "^2", "^3", "sqrt", "log", "exp", "sin"]
        self.num_operators = len(self.operators)

        self.variable_selector = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.GELU(),
            nn.LayerNorm(64),
            nn.Dropout(0.1),
            nn.Linear(64, 8),
        )

        self.operator_selector = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.GELU(),
            nn.LayerNorm(32),
            nn.Dropout(0.1),
            nn.Linear(32, self.num_operators),
        )

        self.coefficient_net = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.GELU(),
            nn.LayerNorm(32),
            nn.Dropout(0.1),
            nn.Linear(32, 4),
        )

        self.gate = nn.Sequential(
            nn.Linear(latent_dim, 1), 
            nn.Sigmoid()
        )

    def forward(self, z: torch.Tensor, physics_features: torch.Tensor) -> torch.Tensor:
        # 输入检查
        z = check_and_fix_tensor(z, "symbolic_z")
        physics_features = check_and_fix_tensor(physics_features, "symbolic_physics_features")
        
        # 注意力机制
        try:
            attn_out, _ = self.attention(z.unsqueeze(1), z.unsqueeze(1), z.unsqueeze(1))
            z = attn_out.squeeze(1)
            z = self.attn_norm(z)
            z = check_and_fix_tensor(z, "attention_output")
        except Exception as e:
            print(f"Warning: Attention failed ({e}), using original z")
            z = check_and_fix_tensor(z, "attention_fallback")

        # 特征选择权重
        var_weights = F.softmax(torch.clamp(self.variable_selector(z), -5, 5), dim=-1)
        op_weights = F.softmax(torch.clamp(self.operator_selector(z), -5, 5), dim=-1)
        coeffs = torch.clamp(self.coefficient_net(z), -5, 5)
        
        # 特征降维处理
        batch_size = physics_features.size(0)
        feature_dim = physics_features.size(1)
        
        pool_size = min(8, feature_dim)
        if feature_dim > pool_size:
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
        
        # 加权特征组合
        selected_features = (var_weights * pooled_features).sum(dim=1, keepdim=True)
        selected_features = check_and_fix_tensor(selected_features, "selected_features")

        # 符号表达式计算
        expressions = []
        expressions.append(coeffs[:, 0:1] * selected_features)
        expressions.append(coeffs[:, 1:2] * (selected_features ** 2))
        expressions.append(coeffs[:, 2:3] * (selected_features ** 3))
        expressions.append(coeffs[:, 3:4] * safe_sqrt(selected_features))

        # 算子操作
        ops = [
            selected_features,
            safe_division(torch.ones_like(selected_features), selected_features),
            selected_features ** 2,
            selected_features ** 3,
            safe_sqrt(selected_features),
            safe_log(selected_features),
            safe_exp(torch.clamp(selected_features, -5, 5)),
            torch.sin(torch.clamp(selected_features, -10, 10))
        ]
        
        ops = [check_and_fix_tensor(op, f"op_{i}") for i, op in enumerate(ops)]
        
        # 算子组合
        try:
            op_stack = torch.stack(ops, dim=1).squeeze(-1)
            op_stack = check_and_fix_tensor(op_stack, "op_stack")
            op_comb = op_stack @ op_weights.unsqueeze(-1)
            op_comb = check_and_fix_tensor(op_comb, "op_comb")
        except Exception as e:
            print(f"Warning: Operator combination failed ({e}), using fallback")
            op_comb = selected_features

        # 最终输出
        gate_value = self.gate(z)
        gate_value = check_and_fix_tensor(gate_value, "gate_value")
        
        symbolic_output = sum(expressions) * gate_value + op_comb
        symbolic_output = check_and_fix_tensor(symbolic_output, "symbolic_output")

        # 复杂度计算
        try:
            self.complexity = -torch.sum(
                var_weights * safe_log(var_weights), dim=-1
            ).mean()
            self.complexity = check_and_fix_tensor(
                torch.tensor(self.complexity), "complexity"
            ).item()
        except:
            self.complexity = 0.0

        return symbolic_output


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
    ):
        super().__init__()
        
        # 材料嵌入
        self.material_embedding = nn.Embedding(num_materials, material_embed_dim)
        nn.init.xavier_uniform_(self.material_embedding.weight, gain=0.5)
        
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
            nn.Sequential(
                nn.Linear(latent_dim, latent_dim),
                nn.LayerNorm(latent_dim),
                nn.GELU(),
                nn.Dropout(dropout_rate)
            ) for _ in range(num_blocks)
        ])
        
        # 简化的符号层
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
            z = z + block(z)  # 残差连接
            z = F.gelu(z)
        
        # 6. 符号输出
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