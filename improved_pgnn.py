import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd

class EnhancedPhysicsFeatureExtractor(nn.Module):
    """增强的物理特征提取器"""
    def __init__(self, input_dim=6):
        super().__init__()
        self.input_dim = input_dim
        
        # 可学习的特征变换
        self.feature_transform = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )
        
    def forward(self, x):
        """增强的物理特征工程"""
        features = []
        
        hardness = x[:, 0:1]
        elastic_modulus = x[:, 1:2]
        fracture_toughness = x[:, 2:3]
        grain_size = x[:, 3:4]
        pressure = x[:, 4:5]
        velocity = x[:, 5:6]
        
        # 原始特征
        features.append(x)
        
        # 核心物理特征（基于数据分析的强相关特征）
        # 1. 晶粒尺寸的多种变换（最强相关r=0.778）
        features.append(grain_size)
        features.append(torch.sqrt(torch.abs(grain_size) + 1e-6))
        features.append(torch.log(torch.abs(grain_size) + 1))
        features.append(grain_size ** 2)
        
        # 2. 脆性相关特征
        brittleness1 = hardness / (fracture_toughness + 1e-6)
        brittleness2 = hardness / (elastic_modulus + 1e-6)
        features.append(brittleness1)
        features.append(brittleness2)
        
        # 3. 能量和应力特征
        energy_release = (pressure ** 2) * torch.sqrt(torch.abs(grain_size) + 1e-6) / (torch.abs(elastic_modulus) + 1e-6)
        stress_intensity = pressure * torch.sqrt(torch.abs(grain_size) + 1e-6) / (torch.abs(fracture_toughness) + 1e-6)
        features.append(energy_release)
        features.append(stress_intensity)
        
        # 4. 交互特征
        features.append(pressure * velocity)
        features.append(elastic_modulus * fracture_toughness / (hardness + 1e-6))
        
        # 5. 比例特征
        features.append(fracture_toughness / (torch.abs(elastic_modulus) + 1e-6))
        features.append(1 / torch.sqrt(torch.abs(grain_size) + 1e-6))  # Hall-Petch
        
        # 可学习的特征变换
        learned_features = self.feature_transform(x)
        features.append(learned_features)
        
        return torch.cat(features, dim=1)

class SimplifiedPGNN(nn.Module):
    """简化但更有效的PGNN架构"""
    def __init__(self, num_materials, material_embed_dim=16, hidden_dim=64):
        super().__init__()
        
        # 材料嵌入
        self.material_embedding = nn.Embedding(num_materials, material_embed_dim)
        
        # 物理特征提取
        self.physics_extractor = EnhancedPhysicsFeatureExtractor()
        
        # 特征融合网络（简化架构）
        # 输入: 16(材料) + 34(物理特征) = 50
        self.feature_fusion = nn.Sequential(
            nn.Linear(material_embed_dim + 34, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU()
        )
        
        # 物理引导分支（直接预测）
        self.physics_branch = nn.Sequential(
            nn.Linear(hidden_dim // 2, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
        # 残差分支
        self.residual_branch = nn.Sequential(
            nn.Linear(hidden_dim // 2, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
        # 门控机制
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # 输出激活
        self.output_activation = nn.Softplus()
        
    def forward(self, material_ids, numeric_features):
        # 材料嵌入
        material_embed = self.material_embedding(material_ids)
        
        # 物理特征
        physics_features = self.physics_extractor(numeric_features)
        
        # 特征融合
        combined = torch.cat([material_embed, physics_features], dim=1)
        hidden = self.feature_fusion(combined)
        
        # 双分支预测
        physics_pred = self.physics_branch(hidden)
        residual_pred = self.residual_branch(hidden)
        
        # 门控组合
        gate_weight = self.gate(hidden)
        prediction = gate_weight * physics_pred + (1 - gate_weight) * residual_pred
        
        # 确保正值输出
        prediction = self.output_activation(prediction)
        
        return prediction, gate_weight, physics_pred, residual_pred

class AdaptiveLoss(nn.Module):
    """自适应损失函数"""
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.huber = nn.HuberLoss(delta=1.0)
        
    def forward(self, pred, target, gate_weight=None):
        # 主损失：Huber损失（对异常值更鲁棒）
        main_loss = self.huber(pred, target)
        
        # 相对误差损失（处理不同尺度）- 添加稳定性处理
        relative_error = torch.abs((pred - target) / (torch.abs(target) + 1.0))
        relative_loss = torch.mean(torch.clamp(relative_error, max=10.0))
        
        # 门控正则化（鼓励使用物理分支）
        gate_reg = 0 if gate_weight is None else torch.mean((gate_weight - 0.7) ** 2)
        
        total_loss = main_loss + 0.05 * relative_loss + 0.01 * gate_reg
        
        # 检查NaN
        if torch.isnan(total_loss):
            print(f"NaN detected! main: {main_loss}, relative: {relative_loss}, gate: {gate_reg}")
            return main_loss, {
                'main': main_loss.item(),
                'relative': 0,
                'gate_reg': 0
            }
        
        return total_loss, {
            'main': main_loss.item(),
            'relative': relative_loss.item(),
            'gate_reg': gate_reg.item() if torch.is_tensor(gate_reg) else gate_reg
        }

# 数据增强函数
def augment_data(X, y, noise_level=0.05, n_augment=2):
    """数据增强"""
    X_aug = []
    y_aug = []
    
    for _ in range(n_augment):
        # 添加高斯噪声
        noise = np.random.normal(0, noise_level, X.shape)
        X_noisy = X + noise * np.std(X, axis=0)
        
        # 对目标也添加小噪声
        y_noise = np.random.normal(0, noise_level * np.std(y), y.shape)
        y_noisy = y + y_noise
        
        X_aug.append(X_noisy)
        y_aug.append(y_noisy)
    
    return np.vstack([X] + X_aug), np.hstack([y] + y_aug)

# 集成预测
class PGNNEnsemble:
    """PGNN集成模型"""
    def __init__(self, n_models=5, **model_kwargs):
        self.n_models = n_models
        self.models = []
        self.model_kwargs = model_kwargs
        
    def fit(self, train_data, val_data, epochs=300, lr=1e-3, device='cuda'):
        """训练多个模型"""
        self.models = []
        
        for i in range(self.n_models):
            print(f"\nTraining model {i+1}/{self.n_models}")
            
            # 创建模型
            model = SimplifiedPGNN(**self.model_kwargs).to(device)
            
            # 不同的初始化种子
            torch.manual_seed(42 + i)
            
            # 训练
            model = self._train_single_model(
                model, train_data, val_data, 
                epochs=epochs, lr=lr * (0.8 + 0.4 * np.random.rand()), 
                device=device
            )
            
            self.models.append(model)
        
        return self
    
    def _train_single_model(self, model, train_data, val_data, epochs, lr, device):
        """训练单个模型"""
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        loss_fn = AdaptiveLoss()
        
        # 准备数据
        train_materials = torch.LongTensor(train_data['material_ids']).to(device)
        train_features = torch.FloatTensor(train_data['numeric_features']).to(device)
        train_target = torch.FloatTensor(train_data['target']).unsqueeze(1).to(device)
        
        val_materials = torch.LongTensor(val_data['material_ids']).to(device)
        val_features = torch.FloatTensor(val_data['numeric_features']).to(device)
        val_target = torch.FloatTensor(val_data['target']).unsqueeze(1).to(device)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # 训练
            model.train()
            optimizer.zero_grad()
            
            pred, gate, _, _ = model(train_materials, train_features)
            loss, _ = loss_fn(pred, train_target, gate)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            # 验证
            if epoch % 50 == 0:
                model.eval()
                with torch.no_grad():
                    val_pred, _, _, _ = model(val_materials, val_features)
                    val_loss, _ = loss_fn(val_pred, val_target)
                    
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter > 10:
                    break
        
        return model
    
    def predict(self, material_ids, features, device='cuda'):
        """集成预测"""
        material_ids = torch.LongTensor(material_ids).to(device)
        features = torch.FloatTensor(features).to(device)
        
        predictions = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred, _, _, _ = model(material_ids, features)
                predictions.append(pred.cpu().numpy())
        
        # 平均预测
        return np.mean(predictions, axis=0).squeeze()

# 混合模型：PGNN特征 + XGBoost
def create_hybrid_model(pgnn_model, train_data, device='cuda'):
    """使用PGNN提取特征，XGBoost进行最终预测"""
    from xgboost import XGBRegressor
    
    # 提取PGNN特征
    pgnn_model.eval()
    with torch.no_grad():
        materials = torch.LongTensor(train_data['material_ids']).to(device)
        features = torch.FloatTensor(train_data['numeric_features']).to(device)
        
        # 获取中间层特征
        material_embed = pgnn_model.material_embedding(materials)
        physics_features = pgnn_model.physics_extractor(features)
        combined = torch.cat([material_embed, physics_features], dim=1)
        hidden_features = pgnn_model.feature_fusion(combined)
        
        # 转换为numpy
        pgnn_features = hidden_features.cpu().numpy()
    
    # 组合原始特征和PGNN特征
    combined_features = np.hstack([train_data['numeric_features'], pgnn_features])
    
    # 训练XGBoost
    xgb_model = XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )
    xgb_model.fit(combined_features, train_data['target'])
    
    return xgb_model, pgnn_model
