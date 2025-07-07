import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, RobustScaler
from typing import Dict, List, Tuple, Optional
import pandas as pd
from scipy.stats.mstats import winsorize

GRAD_CLIP = 5.0


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


class MaterialEmbedding(nn.Module):
    """材料种类嵌入层"""

    def __init__(self, num_materials, embedding_dim=16):
        super().__init__()
        self.embedding = nn.Embedding(num_materials, embedding_dim)

    def forward(self, material_ids):
        return self.embedding(material_ids)


# 增加安全数值操作的函数
def safe_divide(a, b, eps=1e-4):
    return a / (torch.abs(b) + eps)


class FeatureInteractionModule(nn.Module):
    """二阶物理交互特征提取"""

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        # 修改点 ①：Linear 输入维度 = 2 * input_dim * input_dim
        self.interaction_net = nn.Sequential(
            nn.Linear(input_dim * input_dim * 2, hidden_dim),  # ←★★
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, input_dim)
        输出与 x 同形状，包含二阶交互校正
        """
        b, d = x.shape
        x1 = x.unsqueeze(2).expand(-1, -1, d)  # (b, d, d)
        x2 = x.unsqueeze(1).expand(-1, d, -1)  # (b, d, d)
        # 拼接得到 (b, d, d, 2) → view 为 (b, 2*d*d)
        interactions = torch.cat([x1, x2], dim=-1).contiguous()
        interactions = self.interaction_net(interactions.view(b, -1))
        return x + interactions



class PhysicsInformedFeatureExtractor(nn.Module):
    """物理特征提取器，使用原始尺度的数据确保物理意义"""
    
    def __init__(self, input_dim=6):
        super().__init__()
        self.input_dim = input_dim
        
        # 可学习的缩放因子，用于平衡不同物理特征的尺度
        self.feature_scales = nn.Parameter(torch.ones(13))
    
    def forward(self, x):
        """
        x: [batch_size, 6] - 原始尺度的特征
        """
        features = [x]
        
        # 提取各个特征（原始尺度）
        hardness = x[:, 0:1]
        elastic_modulus = x[:, 1:2].clamp(min=1.0)  # GPa，避免除零
        fracture_toughness = x[:, 2:3].clamp(min=0.1)  # MPa·m^0.5
        grain_size = x[:, 3:4].clamp(min=0.001)  # mm
        pressure = x[:, 4:5]  # GPa
        velocity = x[:, 5:6]  # m/s
        
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
        
        # 组合所有特征
        result = torch.cat(features, dim=1)
        
        return result


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


class FiLMLayer(nn.Module):
    """FiLM调制层，用物理特征调制隐藏表示"""

    def __init__(self, cond_dim: int, feature_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, feature_dim * 2),
            nn.GELU(),
            nn.Linear(feature_dim * 2, feature_dim * 2),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma_beta = self.net(cond)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        return gamma * x + beta


class AdaptiveSymbolicLayer(nn.Module):
    """数值稳定的符号层"""

    def __init__(self, latent_dim: int = 32):
        super().__init__()
        
        self.num_features = 19
        # 确保注意力头数能整除latent_dim
        self.num_heads = max(1, (latent_dim // 8) * 2)
        if latent_dim % self.num_heads != 0:
            self.num_heads = 1  # 如果不能整除，使用单头注意力
            
        self.attention = nn.MultiheadAttention(
            latent_dim, num_heads=self.num_heads, batch_first=True
        )
        self.attn_norm = nn.LayerNorm(latent_dim)  # 新增LayerNorm
        
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
        
        # 注意力机制 - 添加异常处理
        try:
            attn_out, _ = self.attention(z.unsqueeze(1), z.unsqueeze(1), z.unsqueeze(1))
            z = attn_out.squeeze(1)
            z = self.attn_norm(z)  # 在这里加LayerNorm
            z = check_and_fix_tensor(z, "attention_output")
        except Exception as e:
            print(f"Warning: Attention failed ({e}), using original z")
            z = check_and_fix_tensor(z, "attention_fallback")

        # 特征选择权重
        var_weights = F.softmax(torch.clamp(self.variable_selector(z), -5, 5), dim=-1)
        op_weights = F.softmax(torch.clamp(self.operator_selector(z), -5, 5), dim=-1)
        coeffs = torch.clamp(self.coefficient_net(z), -5, 5)
        
        var_weights = check_and_fix_tensor(var_weights, "var_weights")
        op_weights = check_and_fix_tensor(op_weights, "op_weights")
        coeffs = check_and_fix_tensor(coeffs, "coeffs")

        # 特征降维处理
        batch_size = physics_features.size(0)
        feature_dim = physics_features.size(1)
        
        # 使用自适应平均池化确保输出维度一致
        pool_size = min(8, feature_dim)
        if feature_dim > pool_size:
            pooled_features = F.adaptive_avg_pool1d(
                physics_features.unsqueeze(1), pool_size
            ).squeeze(1)
        else:
            pooled_features = physics_features[:, :pool_size]
            if pooled_features.size(1) < pool_size:
                # 用零填充
                padding = torch.zeros(batch_size, pool_size - pooled_features.size(1), 
                                    device=pooled_features.device)
                pooled_features = torch.cat([pooled_features, padding], dim=1)
        
        pooled_features = check_and_fix_tensor(pooled_features, "pooled_features")
        
        # 加权特征组合
        selected_features = (var_weights * pooled_features).sum(dim=1, keepdim=True)
        selected_features = check_and_fix_tensor(selected_features, "selected_features")

        # 符号表达式计算 - 使用安全运算
        expressions = []
        expressions.append(coeffs[:, 0:1] * selected_features)
        expressions.append(coeffs[:, 1:2] * (selected_features ** 2))
        expressions.append(coeffs[:, 2:3] * (selected_features ** 3))
        expressions.append(coeffs[:, 3:4] * safe_sqrt(selected_features))

        # 算子操作 - 使用安全运算
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
        
        # 检查每个操作的结果
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


class MaterialCrackPGNN(nn.Module):
    """改进的PGNN，使用双输入设计"""
    
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
        # 注意：这里使用标准化后的特征维度 + 材料嵌入维度
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
        self.enable_mc_dropout(True)  # 启用MC Dropout
        self.train()  # 关键：保持train模式以激活Dropout

        predictions = []
        uncertainties = []

        with torch.no_grad():
            for _ in range(n_samples):
                pred, uncertainty, _, _ = self(material_ids, num_scaled, num_raw)
                predictions.append(pred)
                uncertainties.append(uncertainty)

        # 恢复正常模式
        self.enable_mc_dropout(False)
        self.eval()  # 恢复评估模式

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


class PhysicsInformedLoss(nn.Module):
    """改进的物理信息损失函数"""
    
    def __init__(self, lambda_complexity=1e-4, lambda_physics=1e-2):
        super().__init__()
        self.lambda_complexity = lambda_complexity
        self.lambda_physics = lambda_physics
        self.current_epoch = 0
        self.total_epochs = 100
    
    def update_epoch(self, current, total):
        self.current_epoch = current
        self.total_epochs = total
    
    def forward(self, pred, uncertainty, target, model, numeric_raw):
        """
        注意：numeric_raw是原始尺度的数据，用于物理约束计算
        """
        # 1. NLL损失（使用Huber损失增强鲁棒性）
        uncertainty = uncertainty.clamp(min=0.01, max=10.0)
        residual = (target - pred).abs()
        huber_delta = 1.0
        huber_loss = torch.where(
            residual < huber_delta,
            0.5 * residual ** 2,
            huber_delta * (residual - 0.5 * huber_delta)
        )
        nll_loss = (0.5 * torch.log(2 * np.pi * uncertainty) + huber_loss / uncertainty).mean()
        
        # 2. 复杂度损失（简化）
        complexity_loss = 0.0
        
        # 3. 物理约束（使用原始尺度数据）
        physics_loss = self.compute_physics_constraints(pred, numeric_raw)
        
        # 4. 动态权重调整
        warm_progress = max(0.0, self.current_epoch - 30) / 20
        physics_weight = self.lambda_physics * min(1.0, warm_progress)
        
        # 5. 总损失
        total_loss = nll_loss + self.lambda_complexity * complexity_loss + physics_weight * physics_loss
        
        return total_loss, {
            "nll": nll_loss.item(),
            "complexity": complexity_loss,
            "physics": physics_loss.item(),
            "physics_weight": physics_weight,
            "total": total_loss.item()
        }
    
    def compute_physics_constraints(self, pred, features):
        """基于原始尺度数据的物理约束"""
        # features是原始尺度：[硬度, 弹性模量, 断裂韧性, 晶粒粒径, 压溃压力, 磨粒速度]
        
        grain_size = features[:, 3]  # mm
        pressure = features[:, 4]     # GPa
        hardness = features[:, 0]     # GPa
        fracture_toughness = features[:, 2]  # MPa·m^0.5
        
        # 约束1：裂纹深度不应超过晶粒尺寸的合理倍数
        max_crack_ratio = 5.0  # 经验值
        constraint1 = F.relu(pred.squeeze() - max_crack_ratio * grain_size)
        
        # 约束2：低压力下裂纹应该较小
        pressure_threshold = 0.5  # GPa
        low_pressure_mask = (pressure < pressure_threshold).float()
        constraint2 = low_pressure_mask * pred.squeeze()
        
        # 约束3：高脆性材料更容易产生裂纹
        brittleness = hardness / (fracture_toughness + 1e-6)
        brittleness_normalized = (brittleness - brittleness.mean()) / (brittleness.std() + 1e-6)
        high_brittleness_mask = (brittleness_normalized > 1.0).float()
        expected_crack_depth = pred.mean()
        constraint3 = high_brittleness_mask * F.relu(expected_crack_depth - pred.squeeze())
        
        # 组合约束
        total_constraint = constraint1.mean() + 0.5 * constraint2.mean() + 0.3 * constraint3.mean()
        
        return total_constraint

class AdaptiveLoss(PhysicsInformedLoss):
    """动态权重的物理损失函数"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_weights = nn.Parameter(torch.ones(3))
    
    def safety_check(self, loss_tensor):
        """检查损失值是否有数值问题"""
        if torch.isnan(loss_tensor) or torch.isinf(loss_tensor):
            return torch.tensor(1.0, device=loss_tensor.device)
        return loss_tensor

    def forward(self, pred, uncertainty, target, model, numeric_features):
        nll_loss = 0.5 * (
            torch.log(2 * np.pi * uncertainty) + (target - pred) ** 2 / uncertainty
        )
        nll_loss = nll_loss.mean()
        complexity_loss = (
            model.symbolic_layer.complexity if hasattr(model.symbolic_layer, "complexity") else 0
        )
        
        # 数值安全检查
        nll_loss = self.safety_check(nll_loss)
        if isinstance(complexity_loss, torch.Tensor):
            complexity_loss = self.safety_check(complexity_loss)
            
        physics_loss = self.compute_physics_constraints(pred, numeric_features)
        physics_loss = self.safety_check(physics_loss)
        weights = F.softmax(self.loss_weights, dim=0)
        total_loss = (
            weights[0] * nll_loss + weights[1] * complexity_loss + weights[2] * physics_loss
        )
        
        # 限制总损失范围，防止梯度爆炸
        if total_loss > 1000:
            print(f"Warning: Large loss detected ({total_loss.item():.2f}), clipping to 1000")
            total_loss = torch.clamp(total_loss, max=1000)
            
        # 确保返回合法损失值
        total_loss = self.safety_check(total_loss)
        return total_loss, {
            "nll": nll_loss.item(),
            "complexity": complexity_loss.item() if torch.is_tensor(complexity_loss) else complexity_loss,
            "physics": physics_loss.item(),
            "physics_weight": float(weights[2].item()),
            "total": total_loss.item(),
        }


def smart_augment(df: pd.DataFrame, model_predictions=None):
    """根据预测误差进行mixup的数据增强"""
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

    if model_predictions is not None:
        errors = np.abs(targets - model_predictions)
        weights = errors / (errors.sum() + 1e-6)
    else:
        weights = np.ones(len(df)) / len(df)

    shuffled_idx = np.random.permutation(len(df))
    alpha = 0.2
    lam = np.random.beta(alpha, alpha)
    mixed_features = lam * features + (1 - lam) * features[shuffled_idx]
    mixed_targets = lam * targets + (1 - lam) * targets[shuffled_idx]

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


# 数据预处理与增强
def prepare_material_data(
    df,
    scaler: Optional[RobustScaler] = None,
    le: Optional[LabelEncoder] = None,
    winsorize_limits=(0.02, 0.02),  # 2%-98%分位数裁剪
    verbose: bool = True,
):
    """改进的数据预处理，使用Winsorize和对数变换处理长尾分布"""
    
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
    
    # 2. Winsorize处理极端值（特别是弹性模量）
    for col in numeric_cols:
        original_values = df_clean[col].values
        winsorized_values = winsorize(original_values, limits=winsorize_limits)
        outlier_count = np.sum(original_values != winsorized_values)
        if outlier_count > 0:
            if verbose:
                print(f"Winsorized {outlier_count} outliers in {col}")
            df_clean[col] = winsorized_values
    
    # 3. 对数变换处理长尾分布（重点是弹性模量）
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


# 训练函数
def train_material_model(
    model, train_data, val_data, epochs=100, lr=5e-4, device="cuda"
):
    """改进的训练函数，保持双输入一致性"""
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = PhysicsInformedLoss()
    
    # 准备数据 - 注意现在有两个输入
    train_materials = torch.LongTensor(train_data["material_ids"]).to(device)
    train_scaled = torch.FloatTensor(train_data["numeric_features"]).to(device)
    train_raw = torch.FloatTensor(train_data["numeric_raw"]).to(device)
    train_target = torch.FloatTensor(train_data["target"]).unsqueeze(1).to(device)
    
    val_materials = torch.LongTensor(val_data["material_ids"]).to(device)
    val_scaled = torch.FloatTensor(val_data["numeric_features"]).to(device)
    val_raw = torch.FloatTensor(val_data["numeric_raw"]).to(device)
    val_target = torch.FloatTensor(val_data["target"]).unsqueeze(1).to(device)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        loss_fn.update_epoch(epoch, epochs)
        
        # 训练
        model.train()
        optimizer.zero_grad()
        
        # 前向传播 - 使用双输入
        pred, uncertainty, _, _ = model(train_materials, train_scaled, train_raw)
        loss, metrics = loss_fn(pred, uncertainty, train_target, model, train_raw)
        
        # 梯度裁剪
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        scheduler.step()
        
        train_losses.append(loss.item())
        
        # 验证
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_pred, val_uncertainty, _, _ = model(val_materials, val_scaled, val_raw)
                val_loss, _ = loss_fn(val_pred, val_uncertainty, val_target, model, val_raw)
                val_losses.append(val_loss.item())
                
                print(f"Epoch {epoch}: Train Loss = {loss.item():.4f}, Val Loss = {val_loss.item():.4f}")
    
    return model, train_losses, val_losses