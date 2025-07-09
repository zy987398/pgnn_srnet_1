import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Optional
import pandas as pd
from scipy.stats.mstats import winsorize
from torch.utils.data import TensorDataset, DataLoader
from torch.cuda.amp import GradScaler

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


def robust_pow(x, p):
    """
    A robust power function that clamps the input to prevent overflow.
    """
    return torch.clamp(x, -1e4, 1e4).pow(p)

def _check_tensor(tensor, name):
    """检查张量是否存在NaN或Inf，并打印调试信息"""
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        print(f"Warning: NaN/Inf detected in {name, tensor.shape}, replacing with safe values")
        tensor = torch.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=-1.0)
    return tensor


class AdaptiveSymbolicLayer(nn.Module):
    """数值稳定的符号层"""

    def __init__(self, input_dim, num_symbolic_units=2, num_physics_features=1):
        super(AdaptiveSymbolicLayer, self).__init__()
        self.num_symbolic_units = num_symbolic_units
        self.input_dim = input_dim
        self.num_physics_features = num_physics_features
        
        self.input_norm = nn.LayerNorm(input_dim)
        self.physics_feature_norm = nn.LayerNorm(num_physics_features)

        # Define base functions for symbolic regression
        self.base_funcs = [
            lambda x: x,
            lambda x: robust_pow(x, 2),
            lambda x: robust_pow(x, 3),
            lambda x: safe_sqrt(torch.abs(x)),
            lambda x: safe_log(torch.abs(x) + 1e-8),
            lambda x: torch.sin(x),
            lambda x: torch.cos(x),
            lambda x: torch.exp(-torch.abs(x))
        ]
        self.num_operators = len(self.base_funcs)

        # Attention mechanism for feature selection
        self.symbolic_transform = nn.Linear(input_dim, input_dim)
        self.physics_feature_transform = nn.Linear(num_physics_features, num_physics_features)
        
        combined_dim = input_dim + num_physics_features
        self.attention = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.GELU(),
            nn.Linear(64, combined_dim),
            nn.Softmax(dim=1)
        )

        # Generators for symbolic layer parameters
        self.var_weight_generator = nn.Linear(combined_dim, num_symbolic_units * input_dim)
        self.op_weight_generator = nn.Linear(combined_dim, num_symbolic_units * self.num_operators)
        self.coeff_generator = nn.Linear(combined_dim, num_symbolic_units)
        self.gate = nn.Linear(combined_dim, num_symbolic_units)

        # Final prediction layer
        self.output_layer = nn.Linear(num_symbolic_units, 1)
        
        # Uncertainty prediction layers
        self.uncertainty_layer = nn.Sequential(
            nn.Linear(input_dim + num_symbolic_units, 64),
            nn.GELU(),
            nn.Linear(64, 3)  # v, alpha, beta
        )
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, physics_features):
        x = _check_tensor(x, "input x to symbolic layer")
        physics_features = _check_tensor(physics_features, "input physics_features to symbolic layer")

        x = self.input_norm(x)
        physics_features = self.physics_feature_norm(physics_features)
        
        x = _check_tensor(x, "normalized x in symbolic layer")
        physics_features = _check_tensor(physics_features, "normalized physics_features in symbolic layer")

        symbolic_z = torch.tanh(self.symbolic_transform(x))
        _check_tensor(symbolic_z, "symbolic_z")
        
        symbolic_physics_features = torch.tanh(self.physics_feature_transform(physics_features))
        _check_tensor(symbolic_physics_features, "symbolic_physics_features")

        combined_features = torch.cat([symbolic_z, symbolic_physics_features], dim=1)
        _check_tensor(combined_features, "combined_features")

        attention_weights = self.attention(combined_features)
        _check_tensor(attention_weights, "attention_weights")
        
        attention_output = combined_features * attention_weights
        _check_tensor(attention_output, "attention_output")

        var_weights = self.var_weight_generator(attention_output).view(-1, self.num_symbolic_units, self.input_dim)
        op_weights = self.op_weight_generator(attention_output).view(-1, self.num_symbolic_units, self.num_operators)
        coeffs = self.coeff_generator(attention_output)
        gate_value = torch.sigmoid(self.gate(attention_output))

        _check_tensor(var_weights, "var_weights")
        _check_tensor(op_weights, "op_weights")
        _check_tensor(coeffs, "coeffs")
        _check_tensor(gate_value, "gate_value")

        symbolic_outputs = []
        for i in range(self.num_symbolic_units):
            var_weights_i = var_weights[:, i, :]
            op_weights_i = torch.softmax(op_weights[:, i, :], dim=1)

            # Select variables using weighted sum
            # 修正：使用点积来创建符号变量，确保形状正确
            selected_vars = torch.sum(var_weights_i * x, dim=1, keepdim=True)
            _check_tensor(selected_vars, f"selected_vars_{i}")

            base_outputs = []
            for op_idx, op_func in enumerate(self.base_funcs):
                op_output = op_func(selected_vars)
                _check_tensor(op_output, f"op_output_{op_idx}")
                base_outputs.append(op_output * op_weights_i[:, op_idx].unsqueeze(1))

            symbolic_output_i = torch.sum(torch.stack(base_outputs, dim=1), dim=1)
            _check_tensor(symbolic_output_i, f"symbolic_output_i_{i}")
            symbolic_outputs.append(symbolic_output_i)

        symbolic_output = torch.cat(symbolic_outputs, dim=1)
        _check_tensor(symbolic_output, "symbolic_output_concat")
        
        symbolic_output = symbolic_output * coeffs * gate_value
        _check_tensor(symbolic_output, "symbolic_output_gated")

        pred = self.output_layer(symbolic_output)
        _check_tensor(pred, "final_pred")

        # For uncertainty, combine GNN output with symbolic layer output
        uncertainty_input = torch.cat([x, symbolic_output], dim=1)
        uncertainty = self.uncertainty_layer(uncertainty_input)
        _check_tensor(uncertainty, "final_uncertainty")

        # For interpretability, we can try to return a simplified equation string
        # equation = "not_implemented" # 移除未实现的 equation

        return pred, uncertainty, None # 返回 None 代替 equation

class ResNetBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout_rate=0.2):
        super(ResNetBlock, self).__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.linear2 = nn.Linear(out_features, out_features)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.GELU() # Using GELU for smoother activation
        if in_features != out_features:
            self.shortcut = nn.Linear(in_features, out_features)
        else:
            self.shortcut = None

    def forward(self, x):
        identity = x
        out = self.linear1(x)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.linear2(out)
        
        if self.shortcut is not None:
            identity = self.shortcut(identity)
        
        out += identity
        return out

class PGNN_SRNet(nn.Module):
    def __init__(self, num_materials, numeric_feature_dim, physics_feature_dim,
                 material_embed_dim=32, latent_dim=128, num_blocks=2, dropout_rate=0.2,
                 num_symbolic_units=2):
        super(PGNN_SRNet, self).__init__()
        self.dropout_rate = dropout_rate
        
        # 材料嵌入
        self.material_embedding = nn.Embedding(num_materials, material_embed_dim)
        nn.init.xavier_uniform_(self.material_embedding.weight, gain=0.5)
        self.emb_dropout = nn.Dropout(0.1)
        
        # 物理特征提取器
        self.physics_extractor = PhysicsInformedFeatureExtractor()
        physics_dim = 6 + 13 # 原始6个 + 提取的13个
        
        # 特征融合层
        # 注意：这里使用标准化后的特征维度 + 材料嵌入维度
        input_dim = material_embed_dim + 6  # 只使用标准化的6个特征
        self.feature_projection = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        # FiLM调制层
        self.film_layer = FiLMLayer(cond_dim=physics_dim, feature_dim=latent_dim)

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
                ResidualBlock(latent_dim, dropout_rate) # 使用残差块
            ) for _ in range(num_blocks)
        ])
        
        # 符号层，提供复杂度度量
        # 物理特征维度：6原始 + 13提取的 = 19
        self.symbolic_layer = AdaptiveSymbolicLayer(latent_dim, num_physics_features=19)
        
        # 残差预测头
        self.residual_head = nn.Sequential(
            nn.Linear(latent_dim, 32), # 输入应该是 latent_dim
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 1)
        )
        
        # 不确定性预测现在由符号层处理，这里不再需要
        # self.uncertainty_head = self.create_head(latent_dim, activation=None)

    def forward(self, material_ids, x_numeric, x_physics_raw):
        """
        双输入前向传播:
        - material_ids: 材料ID (LongTensor)
        - x_numeric: 标准化后的数值特征 (FloatTensor)
        - x_physics_raw: 原始尺度的物理特征 (FloatTensor)
        """
        try:
            # 1. 特征嵌入和融合
            material_emb = self.material_embedding(material_ids)
            material_emb = self.emb_dropout(material_emb)

            if x_numeric.dim() > 2:
                x_numeric = x_numeric.squeeze()

            combined_features = torch.cat([material_emb, x_numeric], dim=1)
            z = self.feature_projection(combined_features)
            
            # 2. 物理特征提取
            physics_features = self.physics_extractor(x_physics_raw)
            
            # 3. 主干网络 (Residual Blocks)
            for block in self.blocks:
                z = block(z)
            
            # 4. FiLM调制
            z = self.film_layer(z, physics_features)
            
            # 5. 分支预测
            # symbolic_layer现在返回 pred, uncertainty, equation
            symbolic_pred, uncertainty, _ = self.symbolic_layer(z, physics_features)
            residual_pred = self.residual_head(z)
            
            # 组合预测
            prediction = symbolic_pred + residual_pred
            
            # 不确定性预测 (由symbolic_layer提供)
            # uncertainty = self.uncertainty_head(z)
            
            # 确保所有输出都是 [batch, 1]
            prediction = prediction.view(prediction.size(0), -1)
            uncertainty = uncertainty.view(uncertainty.size(0), -1)
            if symbolic_pred is not None:
                symbolic_pred = symbolic_pred.view(symbolic_pred.size(0), -1)
            if residual_pred is not None:
                residual_pred = residual_pred.view(residual_pred.size(0), -1)


            return prediction, uncertainty, symbolic_pred, residual_pred
        except Exception as e:
            print("\n--- ERROR in MaterialCrackPGNN.forward ---")
            print(f"Exception: {e}")
            # Print shapes of all relevant tensors at the time of error
            print(f"Shapes at error:")
            if 'material_emb' in locals(): print(f"  material_emb: {material_emb.shape}")
            if 'x_numeric' in locals(): print(f"  x_numeric: {x_numeric.shape}")
            if 'combined_features' in locals(): print(f"  combined_features: {combined_features.shape}")
            if 'z' in locals(): print(f"  z: {z.shape}")
            if 'physics_features' in locals(): print(f"  physics_features: {physics_features.shape}")
            if 'symbolic_pred' in locals(): print(f"  symbolic_pred: {symbolic_pred.shape}")
            if 'residual_pred' in locals(): print(f"  residual_pred: {residual_pred.shape}")
            if 'prediction' in locals(): print(f"  prediction: {prediction.shape}")
            if 'uncertainty' in locals(): print(f"  uncertainty: {uncertainty.shape}")
            raise e

    def create_head(self, latent_dim, activation=None):
        layers = [
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(latent_dim, 1)
        ]
        return nn.Sequential(*layers)

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
    
    def __init__(self, lambda_complexity=1e-3, lambda_physics=0.2):
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
        
        # 2. 复杂度损失
        complexity_loss = getattr(model.symbolic_layer, "complexity", 0.0)
        
        # 3. 物理约束（使用原始尺度数据）
        physics_loss = self.compute_physics_constraints(pred, numeric_raw)
        
        # 4. 动态权重调整
        progress = (self.current_epoch / max(1, self.total_epochs)) ** 0.5
        physics_weight = min(self.lambda_physics, progress * self.lambda_physics)
        
        # 5. 总损失
        total_loss = (
            nll_loss
            + self.lambda_complexity * complexity_loss
            + physics_weight * physics_loss
        )
        
        loss_components = {
            "nll": nll_loss.item(),
            "complexity": complexity_loss,
            "physics": physics_loss.item(),
            "physics_weight": physics_weight,
            "total": total_loss.item()
        }

        return total_loss, loss_components

    def compute_physics_constraints(self, pred, features):
        """基于原始尺度数据的物理约束"""
        # 确保pred是二维的
        if pred.dim() == 1:
            pred = pred.unsqueeze(1)
        
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
        
        loss_components = {
            "nll": nll_loss.item(),
            "complexity": complexity_loss.item() if torch.is_tensor(complexity_loss) else complexity_loss,
            "physics": physics_loss.item(),
            "physics_weight": float(weights[2].item()),
            "total": total_loss.item(),
        }
        return total_loss, loss_components


def smart_augment(df: pd.DataFrame, noise_scale: float = 0.04):
    """简单的高斯噪声增广"""
    df = df.copy()
    numeric_cols = [
        "莫氏硬度",
        "弹性模量",
        "断裂韧性",
        "晶粒粒径",
        "压溃压力",
        "磨粒速度",
    ]

    aug_df = df.sample(frac=0.5, replace=True).reset_index(drop=True)
    noise = np.random.normal(0, noise_scale, (len(aug_df), len(numeric_cols)))
    aug_df[numeric_cols] += noise
    
    return pd.concat([df, aug_df], ignore_index=True)


def prepare_material_data(
    df: pd.DataFrame,
    scaler: Optional[StandardScaler] = None,
    le: Optional[LabelEncoder] = None,
) -> Dict:
    """
    预处理材料数据，处理数值和类别特征。
    现在移除 verbose 参数。
    """
    df = df.copy()
    
    if "亚表面裂纹深度" not in df.columns:
        raise ValueError("亚表面裂纹深度 is required for training.")

    target = df["亚表面裂纹深度"].values

    # 1. 标签编码
    if le is None:
        le = LabelEncoder()
        material_ids = le.fit_transform(df["材料种类"])
    else:
        material_ids = le.transform(df["材料种类"])

    # 2. 数值特征缩放
    numeric_cols = ["莫氏硬度", "弹性模量", "断裂韧性", "晶粒粒径", "压溃压力", "磨粒速度"]
    numeric_features = df[numeric_cols].values.astype(np.float32)

    # 保存原始特征用于物理模型
    numeric_raw = numeric_features.copy()

    if scaler is None:
        scaler = RobustScaler() # 使用更稳健的缩放器
        numeric_scaled = scaler.fit_transform(numeric_features)
    else:
        numeric_scaled = scaler.transform(numeric_features)

    # 提取用于符号模型的物理特征（原始尺度）
    physics_features = numeric_raw

    return {
        "material_ids": material_ids,
        "numeric_features": numeric_scaled,
        "physics_features": physics_features,
        "numeric_raw": numeric_raw,
        "target": target,
        "scaler": scaler,
        "label_encoder": le,
    }


# 为了与现有的超参数优化脚本兼容，创建MaterialCrackPGNN别名
MaterialCrackPGNN = PGNN_SRNet
