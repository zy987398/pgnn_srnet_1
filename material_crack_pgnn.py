import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from typing import Dict, List, Tuple, Optional
import pandas as pd

class MaterialEmbedding(nn.Module):
    """材料种类嵌入层"""
    def __init__(self, num_materials, embedding_dim=16):
        super().__init__()
        self.embedding = nn.Embedding(num_materials, embedding_dim)
        
    def forward(self, material_ids):
        return self.embedding(material_ids)

class PhysicsInformedFeatureExtractor(nn.Module):
    """物理信息特征提取器"""
    def __init__(self, input_dim=6):
        super().__init__()
        self.input_dim = input_dim
        
    def forward(self, x):
        """
        x: [batch_size, 6] - [硬度, 弹性模量, 断裂韧性, 晶粒粒径, 压溃压力, 磨粒速度]
        """
        features = [x]
        
        # 基于断裂力学的特征工程
        hardness = x[:, 0:1]
        elastic_modulus = x[:, 1:2]
        fracture_toughness = x[:, 2:3]
        grain_size = x[:, 3:4]
        pressure = x[:, 4:5]
        velocity = x[:, 5:6]
        
        # 1. Griffith能量释放率相关特征
        # Griffith G = (P² √a)/E
        sqrt_g = torch.sqrt(torch.abs(grain_size) + 1e-6)
        energy_release_rate = (pressure ** 2) * sqrt_g / (elastic_modulus.abs() + 1e-6)
        features.append(energy_release_rate)

        # K_ratio   = P √a / K_IC
        k_ratio = pressure * sqrt_g / (fracture_toughness.abs() + 1e-6)
        features.append(k_ratio)

        # Hertz contact stress ∝ (P·E²/a²)^{1/3}
        expr = pressure * elastic_modulus ** 2 / (grain_size ** 2 + 1e-6)
        contact_stress = torch.sign(expr) * torch.pow(expr.abs() + 1e-6, 1/3)
        features.append(contact_stress)

        # Hall-Petch
        hall_petch = 1 / torch.sqrt(torch.abs(grain_size) + 1e-6)
        features.append(hall_petch)

        
        # 3. 材料脆性指数
        # B = H/K_IC (Lawn's brittleness index)
        brittleness = hardness / (fracture_toughness + 1e-6)
        features.append(brittleness)
        
        
        # 5. 能量输入
        energy_input = pressure * velocity
        features.append(energy_input)
        
        # 6. 韧性模量比
        toughness_modulus_ratio = fracture_toughness / (elastic_modulus + 1e-6)
        features.append(toughness_modulus_ratio)
        
        
        return torch.cat(features, dim=1)

class ImprovedSymbolicLayer(nn.Module):
    """改进的符号表达式层，专门针对裂纹深度预测"""
    def __init__(self, latent_dim=32):
        super().__init__()
        
        # 预定义的物理操作符
        self.operators = ['*', '/', '^', 'sqrt', 'log']
        self.num_operators = len(self.operators)
        
        # 物理变量选择网络
        self.variable_selector = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 13)  # 6原始 + 7衍生特征
        )
        
        # 操作符选择网络
        self.operator_selector = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, self.num_operators)
        )
        
        # 系数预测网络
        self.coefficient_net = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 4)  # 预测4个系数
        )
        
    def forward(self, z, physics_features):
        """
        z: 潜在表示
        physics_features: 物理特征
        """
        batch_size = z.shape[0]
        
        # 选择变量
        var_weights = F.softmax(self.variable_selector(z), dim=-1)
        
        # 选择操作符
        op_weights = F.softmax(self.operator_selector(z), dim=-1)
        
        # 预测系数
        coeffs = self.coefficient_net(z)
        
        # 构建符号表达式（加权组合）
        # 使用软选择机制
        selected_features = torch.matmul(var_weights.unsqueeze(1), physics_features.unsqueeze(2)).squeeze(-1)
        
        # 应用不同的操作符（软操作）
        expressions = []
        
        # 线性项
        linear = coeffs[:, 0:1] * selected_features
        expressions.append(linear * op_weights[:, 0:1])
        
        # 平方项
        square = coeffs[:, 1:2] * (selected_features ** 2)
        expressions.append(square * op_weights[:, 1:2])
        
        # 平方根项
        sqrt_term = coeffs[:, 2:3] * torch.sqrt(torch.abs(selected_features) + 1e-6)
        expressions.append(sqrt_term * op_weights[:, 2:3])
        
        # 对数项
        log_term = coeffs[:, 3:4] * torch.log(torch.abs(selected_features) + 1)
        expressions.append(log_term * op_weights[:, 3:4])
        
        # 组合所有项
        symbolic_output = sum(expressions)
        
        # 计算复杂度（用于正则化）- 修复：移除负号
        self.complexity = torch.sum(var_weights * torch.log(var_weights + 1e-8), dim=-1).mean()
        
        return symbolic_output

class MaterialCrackPGNN(nn.Module):
    """专门用于材料裂纹深度预测的PGNN"""
    def __init__(self, num_materials, material_embed_dim=16, latent_dim=32):
        super().__init__()
        
        # 材料嵌入
        self.material_embedding = MaterialEmbedding(num_materials, material_embed_dim)
        
        # 物理特征提取
        self.physics_extractor = PhysicsInformedFeatureExtractor()
        
        # 编码器（处理材料嵌入 + 数值特征 + 物理特征）
        # 输入维度：16(材料) + 6(原始) + 7(物理) = 29
        self.encoder = nn.Sequential(
            nn.Linear(material_embed_dim + 13, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, latent_dim)
        )
        
        # 符号表达式层
        self.symbolic_layer = ImprovedSymbolicLayer(latent_dim)
        
        # 残差网络（捕获非线性效应）
        self.residual_net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # 不确定性预测 - 修改：输出log_var而不是直接使用Softplus
        self.uncertainty_net = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # 输出log_var
        )
        
        # 输出层（确保正值）
        self.output_activation = nn.Softplus()
        
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
        
        # 编码到潜在空间
        z = self.encoder(combined_features)
        
        # 符号表达式输出
        symbolic_output = self.symbolic_layer(z, physics_features)
        
        # 残差输出
        residual_output = self.residual_net(z)
        
        # 组合预测（确保正值）
        prediction = self.output_activation(symbolic_output + residual_output)
        
        # 不确定性估计 - 修改：转换log_var为方差
        log_var = self.uncertainty_net(z)
        uncertainty = torch.exp(log_var)
        
        return prediction, uncertainty, symbolic_output, residual_output

class PhysicsInformedLoss(nn.Module):
    """物理信息损失函数"""
    def __init__(self, lambda_complexity=1e-3, lambda_physics=1e-2):
        super().__init__()
        self.lambda_complexity = lambda_complexity
        self.lambda_physics = lambda_physics
        
    def forward(self, pred, uncertainty, target, model, numeric_features):
        batch_size = pred.shape[0]
        
        # 1. NLL损失（异方差高斯）
        nll_loss = 0.5 * (torch.log(2 * np.pi * uncertainty) + 
                         (target - pred) ** 2 / uncertainty)
        nll_loss = nll_loss.mean()
        
        # 2. 复杂度正则化
        complexity_loss = model.symbolic_layer.complexity if hasattr(model.symbolic_layer, 'complexity') else 0
        
        # 3. 物理约束损失
        physics_loss = self.compute_physics_constraints(pred, numeric_features)
        
        # 总损失
        total_loss = nll_loss + self.lambda_complexity * complexity_loss + self.lambda_physics * physics_loss
        
        return total_loss, {
            'nll': nll_loss.item(),
            'complexity': complexity_loss.item() if torch.is_tensor(complexity_loss) else complexity_loss,
            'physics': physics_loss.item(),
            'total': total_loss.item()
        }
    
    def compute_physics_constraints(self, pred, features):
        """计算物理约束"""
        # 约束1：裂纹深度应该与晶粒尺寸有关（不能超过某个比例）
        grain_size = features[:, 3]
        constraint1 = F.relu(pred.squeeze() - 5 * grain_size)  # 惩罚过大的裂纹深度
        
        # 约束2：零压力应该产生很小的裂纹
        pressure = features[:, 4]
        zero_pressure_mask = (pressure < 0.1).float()
        constraint2 = zero_pressure_mask * pred.squeeze()
        
        return constraint1.mean() + constraint2.mean()

# 数据预处理函数
def prepare_material_data(df,
                          scaler: Optional[StandardScaler] = None,
                          le: Optional[LabelEncoder] = None):
    """准备材料数据"""
    # 编码材料种类
    if le is None:
        le = LabelEncoder()
        material_ids = le.fit_transform(df['材料种类'])
    else:                       # 复用已有编码器
        material_ids = le.transform(df['材料种类'])
    
    # 提取数值特征
    numeric_cols = ['莫氏硬度', '弹性模量', '断裂韧性', '晶粒粒径', '压溃压力', '磨粒速度']
    numeric_raw = df[numeric_cols].values.astype(np.float32)
    
    # 目标变量
    target = df['亚表面裂纹深度'].values
    
    # 使用MinMaxScaler避免负值
    if scaler is None:
        scaler = StandardScaler().fit(numeric_raw)
    numeric_scaled = scaler.transform(numeric_raw)
    
    return {
        'material_ids': material_ids,
        'numeric_features': numeric_scaled,   # 喂给网络
        'numeric_raw':      numeric_raw,      # 喂给 PhysicsLos
        'target': target,
        'label_encoder': le,
        'scaler': scaler,
        'num_materials': len(le.classes_),
        'scaler': scaler,
        'label_encoder': le,
    }

# 训练函数
def train_material_model(model, train_data, val_data, epochs=200, lr=1e-3, device='cuda'):
    """训练材料裂纹预测模型"""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    loss_fn = PhysicsInformedLoss()
    
    # 准备数据
    train_materials = torch.LongTensor(train_data['material_ids']).to(device)
    train_features = torch.FloatTensor(train_data['numeric_features']).to(device)
    train_raw      = torch.FloatTensor(train_data['numeric_raw']).to(device)
    train_target = torch.FloatTensor(train_data['target']).unsqueeze(1).to(device)
    
    val_materials = torch.LongTensor(val_data['material_ids']).to(device)
    val_features   = torch.FloatTensor(val_data['numeric_features']).to(device)
    val_raw        = torch.FloatTensor(val_data['numeric_raw']).to(device)
    val_target = torch.FloatTensor(val_data['target']).unsqueeze(1).to(device)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # 训练
        model.train()
        optimizer.zero_grad()
        
        pred, uncertainty, symbolic, residual = model(train_materials, train_features)
        loss, metrics = loss_fn(pred, uncertainty, train_target, model, train_raw)  # 使用train_raw
        
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        
        # 验证
        model.eval()
        with torch.no_grad():
            val_pred, val_uncertainty, _, _ = model(val_materials, val_features)
            val_loss, _ = loss_fn(val_pred, val_uncertainty, val_target, model, val_raw)  # 使用val_raw
            val_losses.append(val_loss.item())
        
        scheduler.step(val_loss)
        
        if (epoch + 1) % 20 == 0:
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'  Train Loss: {loss.item():.4f} (NLL: {metrics["nll"]:.4f}, Physics: {metrics["physics"]:.4f})')
            print(f'  Val Loss: {val_loss.item():.4f}')
    
    return train_losses, val_losses