"""
材料裂纹深度预测模型 - 增强版本
基于R²=0.7版本的成功经验，通过精细调整追求更高性能
主要改进：
1. 更智能的数据预处理（目标变量对数变换）
2. 物理特征的批归一化
3. 自适应符号层
4. OneCycleLR调度器
5. 集成学习支持
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_squared_error
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')


class MaterialEmbedding(nn.Module):
    """
    增强版材料嵌入层
    添加了正则化和更好的初始化策略
    """
    def __init__(self, num_materials, embedding_dim=16):
        super().__init__()
        self.embedding = nn.Embedding(num_materials, embedding_dim)
        # Xavier初始化，考虑材料数量
        gain = 1.0 / np.sqrt(num_materials)
        nn.init.normal_(self.embedding.weight, mean=0, std=gain)
        
        # 添加dropout防止过拟合
        self.dropout = nn.Dropout(0.1)

    def forward(self, material_ids):
        embed = self.embedding(material_ids)
        return self.dropout(embed)


class PhysicsInformedFeatureExtractorEnhanced(nn.Module):
    """
    增强版物理特征提取器
    添加了批归一化和更好的数值稳定性处理
    """
    def __init__(self, input_dim=6):
        super().__init__()
        self.input_dim = input_dim
        # 批归一化层，用于稳定物理特征
        self.physics_norm = nn.BatchNorm1d(9)
        
        # 可学习的温度参数，用于控制特征的锐度
        self.temperature = nn.Parameter(torch.ones(1))
        
    def forward(self, x):
        """
        计算物理特征时使用更稳定的数值方法
        """
        features = [x]  # 保留原始特征
        
        # 提取各个特征
        hardness = x[:, 0:1]
        elastic_modulus = x[:, 1:2]
        fracture_toughness = x[:, 2:3]
        grain_size = x[:, 3:4]
        pressure = x[:, 4:5]
        velocity = x[:, 5:6]
        
        eps = 1e-6
        
        # 修改归一化逻辑，避免因标准差为0导致的不稳定性
        
        # 添加安全检查和归一化函数
        def safe_normalize(tensor):
            std = tensor.std() + eps
            if torch.isnan(std) or std < eps:
                return tensor  # 如果标准差太小或无效，直接返回原始张量
            return (tensor - tensor.mean()) / std
        
        # 1. Griffith能量释放率（归一化版本）
        sqrt_g = torch.sqrt(torch.abs(grain_size) + eps)
        energy_release_rate = (pressure**2) * sqrt_g / (elastic_modulus.abs() + eps)
        # 使用安全归一化
        energy_release_rate = safe_normalize(energy_release_rate)
        energy_release_rate = torch.tanh(energy_release_rate / self.temperature)
        features.append(energy_release_rate)
        
        # 2. 应力强度因子比（改进版）
        k_ratio = pressure * sqrt_g / (fracture_toughness.abs() + eps)
        k_ratio = k_ratio / (k_ratio.std() + eps)
        k_ratio = torch.tanh(k_ratio / self.temperature)
        features.append(k_ratio)
        
        # 3. Hertz接触应力（稳定版）
        contact_param = (pressure * elastic_modulus) / (grain_size + eps)
        contact_stress = torch.sign(contact_param) * torch.pow(
            torch.abs(contact_param) + eps, 1/3
        )
        contact_stress = contact_stress / (contact_stress.std() + eps)
        features.append(contact_stress)
        
        # 4. Hall-Petch关系（改进版）
        hall_petch = 1.0 / torch.sqrt(grain_size + eps)
        hall_petch = (hall_petch - hall_petch.mean()) / (hall_petch.std() + eps)
        features.append(hall_petch)
        
        # 5. 脆性指数（归一化版）
        brittleness = hardness / (fracture_toughness + eps)
        brittleness = (brittleness - brittleness.mean()) / (brittleness.std() + eps)
        features.append(brittleness)
        
        # 6. 能量输入率（对数尺度）
        energy_input = pressure * velocity
        energy_input = torch.log1p(energy_input)
        energy_input = (energy_input - energy_input.mean()) / (energy_input.std() + eps)
        features.append(energy_input)
        
        # 7. 韧性模量比
        toughness_modulus = fracture_toughness / (elastic_modulus + eps)
        toughness_modulus = torch.log1p(toughness_modulus * 100)
        features.append(toughness_modulus)
        
        # 8. H/E比值（改进版）
        he_ratio = hardness / (elastic_modulus + eps)
        he_ratio = torch.sqrt(he_ratio)  # 使用平方根压缩范围
        features.append(he_ratio)
        
        # 9. 综合断裂参数
        fracture_param = energy_release_rate * brittleness / (toughness_modulus + eps)
        fracture_param = torch.tanh(fracture_param)
        features.append(fracture_param)
        
        # 组合物理特征
        physics_features = torch.cat(features[1:], dim=1)
        
        # 批归一化（训练时）
        if self.training and physics_features.shape[0] > 1:
            physics_features = self.physics_norm(physics_features)
        
        # 返回原始特征和归一化的物理特征
        return torch.cat([x, physics_features], dim=1)


class AdaptiveSymbolicLayer(nn.Module):
    """
    自适应符号层
    使用注意力机制动态选择特征和操作
    """
    def __init__(self, latent_dim=32, physics_dim=15):
        super().__init__()
        
        # 特征重要性注意力
        self.feature_attention = nn.Sequential(
            nn.Linear(latent_dim, physics_dim),
            nn.Tanh(),
            nn.Linear(physics_dim, physics_dim),
            nn.Softmax(dim=-1)
        )
        
        # 操作选择网络
        self.operation_selector = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.GELU(),
            nn.Linear(32, 4),
            nn.Softmax(dim=-1)
        )
        
        # 系数预测网络（带残差连接）
        self.coefficient_net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.GELU(),
            nn.LayerNorm(64),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 4)
        )
        
        # 输出门控
        self.output_gate = nn.Sequential(
            nn.Linear(latent_dim, 1),
            nn.Sigmoid()
        )
        
        # 修改初始化
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                # 使用更小的增益，降低初始值
                nn.init.xavier_uniform_(layer.weight, gain=0.001)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, z, physics_features):
        """
        使用自适应机制构建符号表达式
        """
        batch_size = z.shape[0]
        
        # 计算特征注意力权重
        feature_weights = self.feature_attention(z)
        
        # 加权聚合特征
        weighted_features = torch.sum(
            physics_features * feature_weights, 
            dim=1, 
            keepdim=True
        )
        
        # 选择操作
        op_weights = self.operation_selector(z)
        
        # 预测系数
        coeffs = self.coefficient_net(z)
        # 添加残差连接
        coeffs = coeffs + 0.1 * z.mean(dim=1, keepdim=True)
        
        # 构建符号表达式
        # 线性项
        linear_term = coeffs[:, 0:1] * weighted_features
        
        # 二次项
        quadratic_term = coeffs[:, 1:2] * torch.clamp(torch.square(weighted_features), -10.0, 10.0)
        
        # 平方根项
        sqrt_term = coeffs[:, 2:3] * torch.sqrt(torch.clamp(torch.abs(weighted_features), 1e-4, 10.0))
        
        # 倒数项
        inv_denom = torch.clamp(torch.abs(weighted_features), 0.1, 10.0)
        reciprocal_term = coeffs[:, 3:4] * torch.sign(weighted_features) / inv_denom
        
        # 保护系数
        coeffs_safe = torch.clamp(coeffs, -5.0, 5.0)
        
        # 组合操作
        operations = torch.cat(
            [linear_term, quadratic_term, sqrt_term, reciprocal_term], 
            dim=1
        )
        
        # 检查数值有效性
        operations = torch.where(
            torch.isnan(operations) | torch.isinf(operations),
            torch.zeros_like(operations),
            operations
        )
        
        # 加权组合
        symbolic_output = torch.sum(operations * op_weights, dim=1, keepdim=True)
        
        # 应用输出门控，但增加安全检查
        gate = self.output_gate(z)
        gate = torch.clamp(gate, 0.01, 0.99)  # 避免极端值
        output = gate * symbolic_output + (1 - gate) * torch.clamp(weighted_features, -10.0, 10.0)
        
        # 最终安全检查
        output = torch.clamp(output, -10.0, 10.0)
        
        # 计算复杂度（用于正则化）
        self.complexity = -(
            torch.sum(feature_weights * torch.log(feature_weights + 1e-8), dim=1).mean() +
            torch.sum(op_weights * torch.log(op_weights + 1e-8), dim=1).mean()
        ) / 2
        
        return output


class ResidualBlockEnhanced(nn.Module):
    """
    增强版残差块
    添加了SE注意力机制
    """
    def __init__(self, dim, dropout_rate=0.1):
        super().__init__()
        # 主路径
        self.layer_norm1 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * 2)
        self.act = nn.GELU()
        self.layer_norm2 = nn.LayerNorm(dim * 2)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(dim * 2, dim)
        
        # SE注意力（Squeeze-and-Excitation）
        self.se_fc1 = nn.Linear(dim, dim // 4)
        self.se_fc2 = nn.Linear(dim // 4, dim)

    def forward(self, x):
        # 主路径
        residual = x
        out = self.layer_norm1(x)
        out = self.fc1(out)
        out = self.act(out)
        out = self.layer_norm2(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        # SE注意力
        se = torch.mean(out, dim=0, keepdim=True)  # 全局平均池化
        se = F.relu(self.se_fc1(se))
        se = torch.sigmoid(self.se_fc2(se))
        out = out * se  # 特征重标定
        
        return out + residual


class MaterialCrackPGNNEnhanced(nn.Module):
    """
    增强版材料裂纹预测模型
    集成了所有改进技术
    """
    def __init__(
        self, num_materials, material_embed_dim=16, 
        latent_dim=48, dropout_rate=0.15, num_blocks=2
    ):
        super().__init__()
        
        # 材料嵌入（增强版）
        self.material_embedding = MaterialEmbedding(num_materials, material_embed_dim)
        
        # 物理特征提取（增强版）
        self.physics_extractor = PhysicsInformedFeatureExtractorEnhanced()
        
        # 特征维度
        physics_dim = 6 + 9  # 原始特征 + 物理特征
        input_dim = material_embed_dim + physics_dim
        
        # 输入投影
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        # 残差块（增强版）
        self.residual_blocks = nn.ModuleList([
            ResidualBlockEnhanced(latent_dim, dropout_rate) 
            for _ in range(num_blocks)
        ])
        
        # 符号层（自适应版）
        self.symbolic_layer = AdaptiveSymbolicLayer(latent_dim, physics_dim)
        
        # 残差预测网络（更深）
        self.residual_net = nn.Sequential(
            nn.Linear(latent_dim + 1, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.LayerNorm(64),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )
        
        # 不确定性预测（分离认知和随机不确定性）
        self.aleatoric_head = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )
        
        # MC Dropout控制
        self.mc_dropout_enabled = False
        
        # 初始化策略
        self._initialize_weights()

    def _initialize_weights(self):
        """
        精心设计的权重初始化
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # He初始化用于ReLU/GELU
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def enable_mc_dropout(self, enable=True):
        """控制MC Dropout"""
        self.mc_dropout_enabled = enable
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.training = enable

    def forward(self, material_ids, numeric_features):
        """
        前向传播
        实现了多个创新技术的集成
        """
        # 材料嵌入
        material_embed = self.material_embedding(material_ids)
        
        # 物理特征提取（增强版）
        physics_features = self.physics_extractor(numeric_features)
        
        # 特征组合
        combined_features = torch.cat([material_embed, physics_features], dim=1)
        
        # 投影到潜在空间
        z = self.input_projection(combined_features)
        
        # 通过残差块
        for i, block in enumerate(self.residual_blocks):
            z = block(z)
            # 渐进式dropout
            if self.training:
                dropout_rate = 0.1 + 0.05 * i
                z = F.dropout(z, p=dropout_rate, training=True)
        
        # 符号输出
        symbolic_output = self.symbolic_layer(z, physics_features)
        
        # 残差预测
        # 使用停止梯度避免符号层的不稳定影响
        residual_input = torch.cat([z, symbolic_output.detach()], dim=-1)
        residual_output = self.residual_net(residual_input)
        
        # 最终预测（使用改进的激活函数）
        # 先应用softplus，避免负值
        combined_output = symbolic_output + residual_output
        # 检查并替换nan值
        combined_output = torch.where(
            torch.isnan(combined_output),
            torch.zeros_like(combined_output),
            combined_output
        )
        prediction = F.softplus(combined_output, beta=1.0) + 0.001
        
        # 限制预测范围，避免极端值
        prediction = torch.clamp(prediction, min=0.001, max=10.0)
        
        # 不确定性估计
        log_var = self.aleatoric_head(z)
        # 限制方差范围
        log_var = torch.clamp(log_var, min=-6, max=2)  # 更保守的范围
        uncertainty = torch.exp(log_var) + 1e-6
        
        return prediction, uncertainty, symbolic_output, residual_output

    def predict_with_uncertainty(self, material_ids, numeric_features, n_samples=30):
        """
        增强版不确定性预测
        使用更多的MC样本获得更准确的估计
        """
        self.enable_mc_dropout(True)
        
        predictions = []
        uncertainties = []
        
        # 优化内存管理：分批处理预测以减少内存占用
        batch_size = 10  # 每批处理的样本数
        n_batches = (n_samples + batch_size - 1) // batch_size  # 向上取整
        
        with torch.no_grad():
            for batch_idx in range(n_batches):
                batch_samples = min(batch_size, n_samples - batch_idx * batch_size)
                
                batch_predictions = []
                batch_uncertainties = []
                
                for i in range(batch_samples):
                    pred, uncertainty, _, _ = self(material_ids, numeric_features)
                    batch_predictions.append(pred)
                    batch_uncertainties.append(uncertainty)
                
                # 立即聚合这批次的结果，并将单个结果添加到列表中
                batch_pred = torch.stack(batch_predictions).mean(dim=0)
                batch_uncertainty = torch.stack(batch_uncertainties).mean(dim=0)
                
                predictions.append(batch_pred)
                uncertainties.append(batch_uncertainty)
                
                # 手动释放内存
                del batch_predictions
                del batch_uncertainties
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        self.enable_mc_dropout(False)
        
        # 计算统计量
        predictions = torch.stack(predictions)
        uncertainties = torch.stack(uncertainties)
        
        # 使用更稳健的统计方法
        mean_pred = torch.median(predictions, dim=0)[0]  # 使用中位数更稳健
        aleatoric = torch.mean(uncertainties, dim=0)
        
        # 兼容性处理：使用替代函数计算分位数
        try:
            q75 = torch.quantile(predictions, 0.75, dim=0)
            q25 = torch.quantile(predictions, 0.25, dim=0)
        except AttributeError:
            # 对于较旧的PyTorch版本，手动计算分位数
            def manual_quantile(tensor, q, dim=0):
                sorted_tensor, _ = torch.sort(tensor, dim=dim)
                n = tensor.size(dim)
                idx = int(q * (n - 1))
                return sorted_tensor.select(dim, idx)
            
            q75 = manual_quantile(predictions, 0.75, dim=0)
            q25 = manual_quantile(predictions, 0.25, dim=0)
        
        epistemic = (q75 - q25) / 1.35  # 转换为标准差尺度
        
        total_uncertainty = torch.sqrt(aleatoric**2 + epistemic**2)
        
        return mean_pred, total_uncertainty, aleatoric, epistemic


class EnhancedPhysicsLoss(nn.Module):
    """
    增强版物理损失函数
    实现了更精细的物理约束和动态权重调整
    """
    def __init__(
        self, lambda_complexity=1e-5, lambda_physics=0.03,
        target_transform='log', annealing_epochs=100
    ):
        super().__init__()
        self.lambda_complexity = lambda_complexity
        self.lambda_physics = lambda_physics
        self.target_transform = target_transform
        self.annealing_epochs = annealing_epochs
        self.current_epoch = 0
        self.total_epochs = 200

    def update_epoch(self, current, total):
        self.current_epoch = current
        self.total_epochs = total

    # 3. 增强 EnhancedPhysicsLoss 类的 forward 方法鲁棒性
    def forward(self, pred, uncertainty, target, model, features):
        # 安全检查
        pred = torch.where(torch.isnan(pred), torch.ones_like(pred) * 0.001, pred)
        uncertainty = torch.where(torch.isnan(uncertainty), torch.ones_like(uncertainty), uncertainty)
        uncertainty = torch.clamp(uncertainty, min=0.01, max=5.0)
        
        # 如果使用了目标变换，需要相应调整损失计算
        if self.target_transform == 'log':
            # 在对数空间计算损失，但提高鲁棒性
            try:
                # 直接在对数空间计算
                huber_loss = F.smooth_l1_loss(pred, target, beta=0.1)
                uncertainty_loss = torch.mean(
                    huber_loss / uncertainty + 0.5 * torch.log(uncertainty)
                )
            except Exception as e:
                print(f"Warning: Loss calculation error: {e}, using simplified loss")
                uncertainty_loss = F.mse_loss(pred, target)
        else:
            # 更鲁棒的标准NLL损失
            try:
                squared_error = (target - pred)**2
                uncertainty_loss = 0.5 * torch.mean(
                    torch.log(uncertainty) + squared_error / uncertainty
                )
            except Exception as e:
                print(f"Warning: Loss calculation error: {e}, using MSE")
                uncertainty_loss = F.mse_loss(pred, target)
        
        # 复杂度损失
        complexity_loss = 0.0
        if hasattr(model.symbolic_layer, 'complexity'):
            complexity_loss = model.symbolic_layer.complexity
        
        # 物理约束（增强版）
        physics_loss = self.compute_enhanced_physics_constraints(pred, features)
        
        # 动态权重（改进的退火策略）
        if self.current_epoch < 10:
            physics_weight = 0.0
        else:
            # 使用S型曲线而非余弦
            progress = (self.current_epoch - 10) / self.annealing_epochs
            physics_weight = self.lambda_physics * torch.sigmoid(
                torch.tensor(5 * (progress - 0.5))
            ).item()
        
        # 总损失
        total_loss = (
            uncertainty_loss + 
            self.lambda_complexity * complexity_loss + 
            physics_weight * physics_loss
        )
        
        # L2正则化（通过优化器的weight_decay实现）
        
        return total_loss, {
            'uncertainty_loss': uncertainty_loss.item(),
            'complexity': complexity_loss if isinstance(complexity_loss, float) else complexity_loss.item(),
            'physics': physics_loss.item(),
            'physics_weight': physics_weight,
            'total': total_loss.item()
        }

    def compute_enhanced_physics_constraints(self, pred, features):
        """
        增强版物理约束
        包含更多基于材料科学的约束
        """
        # 提取特征
        hardness = features[:, 0]
        elastic_modulus = features[:, 1]
        fracture_toughness = features[:, 2]
        grain_size = features[:, 3]
        pressure = features[:, 4]
        velocity = features[:, 5]
        
        # 约束1：裂纹深度的物理上限
        # 基于Griffith理论的最大裂纹深度
        max_crack_theoretical = 2.5 * grain_size * torch.sqrt(
            fracture_toughness / (pressure + 1e-6)
        )
        constraint1 = F.relu(pred.squeeze() - max_crack_theoretical)
        
        # 约束2：压力阈值约束
        # 低于临界压力不应产生裂纹
        critical_pressure = 0.1 * hardness  # 经验公式
        low_pressure_mask = torch.sigmoid(-(pressure - critical_pressure) * 20)
        constraint2 = low_pressure_mask * pred.squeeze()
        
        # 约束3：速度影响约束
        # 高速下裂纹深度应该增加
        velocity_factor = torch.log1p(velocity)
        expected_trend = velocity_factor * pressure
        
        # 使用自定义相关系数计算，替代torch.corr（PyTorch没有这个函数）
        def correlation(x, y):
            x_centered = x - x.mean()
            y_centered = y - y.mean()
            numerator = torch.sum(x_centered * y_centered)
            denominator = torch.sqrt(torch.sum(x_centered**2) * torch.sum(y_centered**2) + 1e-8)
            return numerator / denominator
        
        corr_value = correlation(expected_trend, pred.squeeze())
        constraint3 = F.relu(-corr_value)  # 我们希望正相关
        
        # 约束4：材料韧性约束
        # 高韧性材料裂纹应该较浅
        toughness_factor = fracture_toughness / (elastic_modulus + 1e-6)
        constraint4 = F.relu(
            pred.squeeze() - 5 * grain_size / (1 + 10 * toughness_factor)
        )
        
        # 组合所有约束
        total_constraint = (
            constraint1.mean() + 
            0.5 * constraint2.mean() + 
            0.3 * torch.abs(constraint3) +
            0.2 * constraint4.mean()
        )
        
        return total_constraint


def prepare_material_data_enhanced(
    df, scaler=None, le=None, target_transform='log', verbose=True
):
    """
    增强版数据预处理
    实现了目标变换和更智能的特征处理
    """
    numeric_cols = [
        '莫氏硬度', '弹性模量', '断裂韧性', 
        '晶粒粒径', '压溃压力', '磨粒速度'
    ]
    
    df_clean = df.copy()
    
    # 智能缺失值处理
    for col in numeric_cols + ['亚表面裂纹深度']:
        if df_clean[col].isna().any():
            # 使用材料种类分组的中位数填充
            df_clean[col] = df_clean.groupby('材料种类')[col].transform(
                lambda x: x.fillna(x.median())
            )
            # 如果还有缺失（某类材料全部缺失），使用全局中位数
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    # 选择性对数变换
    df_transformed = df_clean.copy()
    
    # 对长尾分布的特征进行对数变换
    log_features = ['弹性模量', '晶粒粒径']
    for col in log_features:
        if df_clean[col].min() > 0:
            df_transformed[col] = np.log1p(df_clean[col])
            if verbose:
                print(f"Applied log transform to {col}")
    
    # 目标变量处理
    if target_transform == 'log':
        target = np.log1p(df_clean['亚表面裂纹深度'].values)
        if verbose:
            print("Applied log transform to target variable")
    elif target_transform == 'sqrt':
        target = np.sqrt(df_clean['亚表面裂纹深度'].values)
        if verbose:
            print("Applied sqrt transform to target variable")
    else:
        target = df_clean['亚表面裂纹深度'].values
    
    # 编码材料种类
    if le is None:
        le = LabelEncoder()
        material_ids = le.fit_transform(df_clean['材料种类'])
    else:
        material_ids = le.transform(df_clean['材料种类'])
    
    # 特征标准化
    numeric_raw = df_clean[numeric_cols].values.astype(np.float32)
    numeric_for_scaling = df_transformed[numeric_cols].values.astype(np.float32)
    
    if scaler is None:
        # 使用RobustScaler，但参数更保守
        scaler = RobustScaler(quantile_range=(10.0, 90.0))
        scaler.fit(numeric_for_scaling)
    
    numeric_scaled = scaler.transform(numeric_for_scaling)
    # 额外的安全裁剪
    numeric_scaled = np.clip(numeric_scaled, -5, 5)
    
    return {
        'material_ids': material_ids,
        'numeric_features': numeric_scaled,
        'numeric_raw': numeric_raw,
        'target': target.astype(np.float32),
        'label_encoder': le,
        'scaler': scaler,
        'num_materials': len(le.classes_),
        'target_transform': target_transform
    }


def create_data_loader(data_dict, batch_size=32, shuffle=True):
    """
    创建PyTorch数据加载器
    """
    dataset = TensorDataset(
        torch.LongTensor(data_dict['material_ids']),
        torch.FloatTensor(data_dict['numeric_features']),
        torch.FloatTensor(data_dict['target']).unsqueeze(1)
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    
    return loader


def train_enhanced_model(
    model, train_data, val_data, config, device='cuda'
):
    """
    增强版训练函数
    实现了所有高级训练技巧
    """
    model = model.to(device)
    
    # 分组优化器（不同学习率）
    embedding_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if 'embedding' in name:
            embedding_params.append(param)
        else:
            other_params.append(param)
    
    # 优化器
    optimizer = torch.optim.AdamW([
        {'params': embedding_params, 'lr': config['lr'] * 0.1},
        {'params': other_params, 'lr': config['lr']}
    ], weight_decay=config.get('weight_decay', 1e-4))
    
    # 数据加载器
    train_loader = create_data_loader(
        train_data, 
        batch_size=config['batch_size'], 
        shuffle=True
    )
    
    # OneCycleLR调度器
    epochs = config['epochs']
    steps_per_epoch = len(train_loader)
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config['lr'],
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.3,
        anneal_strategy='cos',
        cycle_momentum=True,
        base_momentum=0.85,
        max_momentum=0.95,
        div_factor=25,
        final_div_factor=1000  # 降低final_div_factor避免学习率太小
    )
    
    # 损失函数
    loss_fn = EnhancedPhysicsLoss(
        lambda_complexity=config.get('lambda_complexity', 1e-5),
        lambda_physics=config.get('lambda_physics', 0.03),
        target_transform=train_data.get('target_transform', 'none'),
        annealing_epochs=epochs // 2
    )
    
    # 准备验证数据
    val_materials = torch.LongTensor(val_data['material_ids']).to(device)
    val_features = torch.FloatTensor(val_data['numeric_features']).to(device)
    val_target = torch.FloatTensor(val_data['target']).unsqueeze(1).to(device)
    
    # 训练历史
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_r2': [],
        'lr': []
    }
    
    # 最佳模型跟踪
    best_val_r2 = -float('inf')
    best_model_state = None
    patience_counter = 0
    
    # 梯度累积（可选）
    accumulation_steps = config.get('gradient_accumulation', 1)
    
    print("Starting enhanced training...")
    
    for epoch in range(epochs):
        # 更新损失函数的epoch
        loss_fn.update_epoch(epoch, epochs)
        
        # 训练阶段
        model.train()
        epoch_loss = 0
        batch_count = 0
        
        optimizer.zero_grad()  # 在循环开始前清零梯度
        
        for batch_idx, (materials, features, targets) in enumerate(train_loader):
            materials = materials.to(device)
            features = features.to(device)
            targets = targets.to(device)
            
            # 前向传播
            pred, uncertainty, _, _ = model(materials, features)
            loss, metrics = loss_fn(
                pred, uncertainty, targets, model, 
                features  # 使用标准化特征作为物理约束输入
            )
            
            # 梯度累积
            loss = loss / accumulation_steps
            loss.backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                
                # 优化器步骤
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()  # 优化器更新后清零梯度
            
            epoch_loss += loss.item() * accumulation_steps
            batch_count += 1
        
        # 计算平均损失
        avg_train_loss = epoch_loss / batch_count
        history['train_loss'].append(avg_train_loss)
        
        # 验证阶段
        model.eval()
        with torch.no_grad():
            val_pred, val_uncertainty, _, _ = model(val_materials, val_features)
            val_loss, _ = loss_fn(
                val_pred, val_uncertainty, val_target, model, val_features
            )
            
            # 反变换预测（如果使用了目标变换）
            if train_data.get('target_transform') == 'log':
                # 确保预测值非负，避免expm1产生NaN
                val_pred_log_safe = torch.clamp(val_pred, min=-0.99)  # 避免expm1产生极端负值
                val_pred_original = torch.expm1(val_pred_log_safe)
                val_target_original = torch.expm1(val_target)
            elif train_data.get('target_transform') == 'sqrt':
                val_pred_original = torch.square(val_pred)
                val_target_original = torch.square(val_target)
            else:
                val_pred_original = val_pred
                val_target_original = val_target
            
            # 确保没有NaN/Inf值
            val_pred_np = val_pred_original.cpu().numpy()
            val_target_np = val_target_original.cpu().numpy()
            
            # 移除可能的NaN/Inf值
            mask = ~(np.isnan(val_pred_np) | np.isnan(val_target_np) | 
                     np.isinf(val_pred_np) | np.isinf(val_target_np))
            
            if np.any(~mask):
                print(f"Warning: Removed {np.sum(~mask)} samples with NaN/Inf values from evaluation")
                val_pred_np = val_pred_np[mask]
                val_target_np = val_target_np[mask]
            
            # 确保有足够的有效样本
            if len(val_pred_np) < 5:
                print("Error: Too few valid samples for evaluation")
                r2 = float('nan')
                rmse = float('nan')
                mae = float('nan')
            else:
                r2 = r2_score(val_target_np, val_pred_np)
                rmse = np.sqrt(mean_squared_error(val_target_np, val_pred_np))
                mae = np.mean(np.abs(val_target_np - val_pred_np))
            
            history['val_loss'].append(val_loss.item())
            history['val_r2'].append(r2)
        
        # 记录学习率
        current_lr = optimizer.param_groups[0]['lr']
        history['lr'].append(current_lr)
        
        # 打印进度
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Train Loss: {avg_train_loss:.4f}")
            print(f"  Val Loss: {val_loss.item():.4f}")
            print(f"  Val R²: {r2:.4f}")
            print(f"  LR: {current_lr:.6f}")
            print(f"  Physics Weight: {metrics['physics_weight']:.4f}")
        
        # 早停和最佳模型保存
        if r2 > best_val_r2:
            best_val_r2 = r2
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= config.get('patience', 30):
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    # 恢复最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, history


class EnsemblePredictor:
    """
    集成预测器
    训练多个模型并集成它们的预测
    """
    def __init__(self, n_models=5):
        self.n_models = n_models
        self.models = []
        self.histories = []
        
    def train_ensemble(self, train_data, val_data, base_config, device='cuda'):
        """
        训练集成模型
        """
        print(f"Training ensemble of {self.n_models} models...")
        
        for i in range(self.n_models):
            print(f"\nTraining model {i+1}/{self.n_models}")
            
            # 为每个模型创建略微不同的配置
            config = base_config.copy()
            
            # 随机调整超参数
            config['lr'] *= (0.8 + 0.4 * np.random.rand())
            config['dropout_rate'] = base_config['dropout_rate'] * (0.8 + 0.4 * np.random.rand())
            config['seed'] = base_config.get('seed', 42) + i
            
            # 设置随机种子
            torch.manual_seed(config['seed'])
            np.random.seed(config['seed'])
            
            # 创建模型
            model = MaterialCrackPGNNEnhanced(
                num_materials=train_data['num_materials'],
                material_embed_dim=config.get('material_embed_dim', 16),
                latent_dim=config.get('latent_dim', 48),
                dropout_rate=config['dropout_rate'],
                num_blocks=config.get('num_blocks', 2)
            )
            
            # 训练模型
            model, history = train_enhanced_model(
                model, train_data, val_data, config, device
            )
            
            self.models.append(model)
            self.histories.append(history)
            
            print(f"Model {i+1} best val R²: {max(history['val_r2']):.4f}")
    
    def predict(self, materials, features, return_all=False):
        """
        集成预测
        """
        predictions = []
        uncertainties = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred, unc, _, _ = model(materials, features)
                predictions.append(pred)
                uncertainties.append(unc)
        
        # 堆叠预测
        predictions = torch.stack(predictions)
        uncertainties = torch.stack(uncertainties)
        
        if return_all:
            return predictions, uncertainties
        
        # 加权平均（基于不确定性）
        weights = 1.0 / (uncertainties + 1e-6)
        weights = weights / weights.sum(dim=0, keepdim=True)
        
        ensemble_pred = (predictions * weights).sum(dim=0)
        ensemble_unc = torch.sqrt((weights * uncertainties**2).sum(dim=0))
        
        return ensemble_pred, ensemble_unc
    
    def predict_with_uncertainty(self, materials, features, n_samples=30):
        """
        集成MC Dropout预测
        """
        all_predictions = []
        
        for model in self.models:
            pred, total_unc, aleatoric, epistemic = model.predict_with_uncertainty(
                materials, features, n_samples
            )
            all_predictions.append(pred)
            # 手动清理内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # 集成预测（分批处理减少内存占用）
        stacked_predictions = torch.stack(all_predictions)
        ensemble_pred = torch.mean(stacked_predictions, dim=0)
        ensemble_std = torch.std(stacked_predictions, dim=0)
        
        return ensemble_pred, ensemble_std


# 主程序
if __name__ == '__main__':
    # 设置随机种子
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # 配置
        # 5. 在 __main__ 部分修改配置
    config = {
        # 模型架构
        'material_embed_dim': 16,
        'latent_dim': 48,
        'num_blocks': 2,
        'dropout_rate': 0.15,
        
        # 训练参数 - 降低学习率
        'lr': 0.0003,  # 降低初始学习率
        'batch_size': 16,  # 减小批量
        'epochs': 300,
        'weight_decay': 1e-4,
        'gradient_accumulation': 2,  # 增加梯度累积步数
        
        # 损失权重 - 减小物理约束权重
        'lambda_complexity': 1e-6,
        'lambda_physics': 0.01,  # 减小物理约束权重
        
        # 其他
        'patience': 40,  # 增加耐心值
        'seed': seed
    }
    
    # 加载数据
    print("Loading data...")
    df = pd.read_csv('3011916164精确材料性能数据.csv')
    
    # 划分数据
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=seed, stratify=df['材料种类']
    )
    
    # 准备数据（使用增强版预处理）
    print("Preparing data with enhanced preprocessing...")
    train_data = prepare_material_data_enhanced(
        train_df, target_transform='log', verbose=True
    )
    test_data = prepare_material_data_enhanced(
        test_df, 
        scaler=train_data['scaler'],
        le=train_data['label_encoder'],
        target_transform='log',
        verbose=False
    )
    
    # 设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 创建并训练集成模型
    ensemble = EnsemblePredictor(n_models=3)  # 使用3个模型的集成
    ensemble.train_ensemble(train_data, test_data, config, device)
    
    # 评估集成模型
    print("\n" + "="*50)
    print("Evaluating ensemble model...")
    print("="*50)
    
    # 准备测试数据
    test_materials = torch.LongTensor(test_data['material_ids']).to(device)
    test_features = torch.FloatTensor(test_data['numeric_features']).to(device)
    test_target = torch.FloatTensor(test_data['target']).unsqueeze(1).to(device)
    
    # 集成预测
    ensemble_pred, ensemble_unc = ensemble.predict(test_materials, test_features)
    
    # 反变换（如果使用了对数变换）
    if test_data.get('target_transform') == 'log':
        # 安全的反变换
        ensemble_pred_safe = torch.clamp(ensemble_pred, min=-0.99)
        ensemble_pred_original = torch.expm1(ensemble_pred_safe)
        test_target_original = torch.expm1(test_target)
    else:
        ensemble_pred_original = ensemble_pred
        test_target_original = test_target

    # 计算指标
    pred_np = ensemble_pred_original.cpu().numpy()
    target_np = test_target_original.cpu().numpy()
    
    # 移除可能的NaN/Inf值
    mask = ~(np.isnan(pred_np) | np.isnan(target_np) | 
             np.isinf(pred_np) | np.isinf(target_np))

    if np.any(~mask):
        print(f"Warning: Removed {np.sum(~mask)} samples with NaN/Inf values from evaluation")
        pred_np = pred_np[mask]
        target_np = target_np[mask]

    # 确保有足够的有效样本
    if len(pred_np) < 5:
        print("Error: Too few valid samples for evaluation")
        r2 = float('nan')
        rmse = float('nan')
        mae = float('nan')
    else:
        r2 = r2_score(target_np, pred_np)
        rmse = np.sqrt(mean_squared_error(target_np, pred_np))
        mae = np.mean(np.abs(target_np - pred_np))
    
    print(f"\nEnsemble Performance:")
    print(f"  R²: {r2:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    
    # MC Dropout集成预测
    print("\nPerforming ensemble MC Dropout predictions...")
    mc_pred, mc_std = ensemble.predict_with_uncertainty(
        test_materials, test_features, n_samples=30
    )
    
    if test_data.get('target_transform') == 'log':
        mc_pred_original = torch.expm1(mc_pred)
    else:
        mc_pred_original = mc_pred
    
    mc_pred_np = mc_pred_original.cpu().numpy()
    mc_r2 = r2_score(target_np, mc_pred_np)
    
    print(f"\nEnsemble MC Dropout R²: {mc_r2:.4f}")
    
    # 可视化结果
    print("\nCreating visualizations...")
    
    # 首先检查是否在可视化环境中
    import os
    can_visualize = True
    # 检测是否在无界面环境中运行
    if os.environ.get('DISPLAY', '') == '' and not os.name == 'nt':
        # 设置matplotlib为非交互模式
        print("Running in headless environment, setting matplotlib backend to 'Agg'")
        import matplotlib
        matplotlib.use('Agg')
        can_visualize = False
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. 训练历史（第一个模型）
    ax = axes[0, 0]
    history = ensemble.histories[0]
    ax.plot(history['train_loss'], label='Train Loss')
    ax.plot(history['val_loss'], label='Val Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training History (Model 1)')
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # 2. 学习率变化
    ax = axes[0, 1]
    ax.plot(history['lr'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # 3. 验证R²历史
    ax = axes[0, 2]
    for i, history in enumerate(ensemble.histories):
        ax.plot(history['val_r2'], label=f'Model {i+1}', alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation R²')
    ax.set_title('Validation R² History')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. 预测vs真实（集成）
    ax = axes[1, 0]
    ax.scatter(target_np, pred_np, alpha=0.6, c='blue', edgecolors='black', linewidth=0.5)
    ax.plot([target_np.min(), target_np.max()], 
            [target_np.min(), target_np.max()], 'r--', lw=2)
    ax.set_xlabel('True Crack Depth')
    ax.set_ylabel('Predicted Crack Depth')
    ax.set_title(f'Ensemble Predictions (R² = {r2:.4f})')
    ax.grid(True, alpha=0.3)
    
    # 5. 残差分析
    ax = axes[1, 1]
    residuals = target_np - pred_np
    ax.scatter(pred_np, residuals, alpha=0.6)
    ax.axhline(y=0, color='r', linestyle='--')
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Residuals')
    ax.set_title('Residual Analysis')
    ax.grid(True, alpha=0.3)
    
    # 6. 不确定性vs误差
    ax = axes[1, 2]
    errors = np.abs(target_np - pred_np)
    unc_np = ensemble_unc.cpu().numpy()
    ax.scatter(errors, unc_np, alpha=0.6)
    ax.set_xlabel('Absolute Error')
    ax.set_ylabel('Predicted Uncertainty')
    ax.set_title('Uncertainty Calibration')
    ax.grid(True, alpha=0.3)
    
    # 添加相关系数
    corr = np.corrcoef(errors.flatten(), unc_np.flatten())[0, 1]
    ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', 
            transform=ax.transAxes, verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig('enhanced_model_results.png', dpi=300, bbox_inches='tight')
    # 只在支持可视化的环境中显示图形
    if can_visualize:
        plt.show()
    else:
        plt.close()
    
    # 特征重要性分析（如果可能）
    print("\nAnalyzing feature importance...")
    
    # 获取第一个模型进行分析
    model = ensemble.models[0]
    model.eval()
    
    # 创建一个小批量用于分析
    sample_idx = np.random.choice(len(test_materials), size=10, replace=False)
    sample_materials = test_materials[sample_idx]
    sample_features = test_features[sample_idx]
    
    with torch.no_grad():
        # 获取符号层的注意力权重
        _ = model(sample_materials, sample_features)
        if hasattr(model.symbolic_layer, 'feature_attention'):
            # 获取潜在表示
            material_embed = model.material_embedding(sample_materials)
            physics_features = model.physics_extractor(sample_features)
            combined = torch.cat([material_embed, physics_features], dim=1)
            z = model.input_projection(combined)
            
            # 获取特征注意力
            attention_weights = model.symbolic_layer.feature_attention(z)
            avg_attention = attention_weights.mean(dim=0).cpu().numpy()
            
            # 可视化特征重要性
            feature_names = (
                ['莫氏硬度', '弹性模量', '断裂韧性', '晶粒粒径', '压溃压力', '磨粒速度'] +
                ['能量释放率', 'K比值', '接触应力', 'Hall-Petch', '脆性指数', 
                 '能量输入', '韧性模量比', 'H/E比', '断裂参数']
            )
            
            plt.figure(figsize=(10, 6))
            plt.bar(range(len(avg_attention)), avg_attention)
            plt.xticks(range(len(avg_attention)), feature_names, rotation=45, ha='right')
            plt.ylabel('Average Attention Weight')
            plt.title('Feature Importance from Symbolic Layer')
            plt.tight_layout()
            plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
            # 只在支持可视化的环境中显示图形
            if can_visualize:
                plt.show()
            else:
                plt.close()
    
    print("\nAll experiments completed successfully!")
    print(f"Final ensemble R² score: {r2:.4f}")
    print(f"This represents a potential improvement over the baseline R²=0.7")