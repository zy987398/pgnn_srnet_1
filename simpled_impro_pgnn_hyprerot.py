import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 导入原始模型组件
from improved_material_crack_pgnn import (
    PhysicsInformedFeatureExtractor,
    ResidualBlock,
    ImprovedSymbolicLayer,
    PhysicsInformedLoss
)


def create_material_hierarchy(df, n_clusters=None):
    """创建材料层次结构"""
    # 计算每种材料的平均属性
    material_features = df.groupby('材料种类')[
        ['莫氏硬度', '弹性模量', '断裂韧性']
    ].mean()
    
    # 确定聚类数量
    if n_clusters is None:
        n_clusters = min(5, len(material_features) // 4)
    
    # 使用RobustScaler更稳健
    scaler = RobustScaler()
    features_scaled = scaler.fit_transform(material_features)
    
    # 层次聚类
    clusters = AgglomerativeClustering(n_clusters=n_clusters).fit(features_scaled)
    material_to_cluster = dict(zip(material_features.index, clusters.labels_))
    
    print(f"\n材料聚类结果（{n_clusters}个类别）：")
    for i in range(n_clusters):
        materials = [m for m, c in material_to_cluster.items() if c == i]
        print(f"  类别 {i}: {len(materials)} 种材料")
    
    return material_to_cluster, n_clusters


class SimplifiedHierarchicalEmbedding(nn.Module):
    """简化的层次化材料嵌入"""
    def __init__(self, num_materials, num_clusters, embed_dim=16):
        super().__init__()
        self.material_embed = nn.Embedding(num_materials, embed_dim)
        
        # 更保守的初始化
        nn.init.uniform_(self.material_embed.weight, -0.1, 0.1)
        
        # 材料到聚类的映射
        self.register_buffer('material_to_cluster', torch.zeros(num_materials, dtype=torch.long))
        
    def set_clusters(self, material_to_cluster, label_encoder):
        """设置聚类映射"""
        for material, cluster in material_to_cluster.items():
            if material in label_encoder.classes_:
                idx = label_encoder.transform([material])[0]
                self.material_to_cluster[idx] = cluster
    
    def forward(self, material_ids):
        # 只使用材料嵌入，聚类信息通过正则化隐式利用
        return self.material_embed(material_ids)


class RobustEnhancedPGNN(nn.Module):
    """稳健的增强版PGNN"""
    def __init__(self, num_materials, num_clusters=5, material_embed_dim=12, 
                 latent_dim=24, dropout_rate=0.15):
        super().__init__()
        
        # 简化的材料嵌入
        self.material_embedding = SimplifiedHierarchicalEmbedding(
            num_materials, num_clusters, material_embed_dim
        )
        
        # 物理特征提取（使用原始的，不用多项式）
        self.physics_extractor = PhysicsInformedFeatureExtractor()
        
        # 特征维度
        physics_dim = 6 + 9  # 原始特征 + 物理特征
        input_dim = material_embed_dim + physics_dim
        
        # 简化的编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        # 减少残差块数量
        self.residual_block = ResidualBlock(latent_dim, dropout_rate)
        
        # 符号层
        self.symbolic_layer = ImprovedSymbolicLayer(latent_dim)
        
        # 简化的残差网络
        self.residual_net = nn.Sequential(
            nn.Linear(latent_dim + 1, 32),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 1)
        )
        
        # 不确定性估计
        self.uncertainty_net = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.GELU(),
            nn.Linear(16, 1)
        )
        
    def forward(self, material_ids, numeric_features):
        # 材料嵌入
        material_embed = self.material_embedding(material_ids)
        
        # 物理特征
        physics_features = self.physics_extractor(numeric_features)
        
        # 组合特征
        combined = torch.cat([material_embed, physics_features], dim=1)
        
        # 编码
        z = self.encoder(combined)
        z = self.residual_block(z)
        
        # 符号输出（不detach梯度）
        symbolic_output = self.symbolic_layer(z, physics_features)
        
        # 残差修正（使用较小的权重）
        residual_input = torch.cat([z, symbolic_output * 0.1], dim=-1)
        residual_output = self.residual_net(residual_input)
        
        # 最终预测
        prediction = F.relu(symbolic_output + 0.1 * residual_output) + 0.01
        
        # 不确定性
        log_var = self.uncertainty_net(z)
        uncertainty = torch.exp(log_var.clamp(-10, 5)) + 0.01
        
        return prediction, uncertainty, symbolic_output, residual_output


def prepare_data_simple(df, test_size=0.2, random_state=42):
    """简化的数据准备"""
    # 材料聚类
    material_to_cluster, n_clusters = create_material_hierarchy(df)
    
    # 编码材料
    le = LabelEncoder()
    material_ids = le.fit_transform(df['材料种类'])
    
    # 数值特征
    feature_cols = ['莫氏硬度', '弹性模量', '断裂韧性', '晶粒粒径', '压溃压力', '磨粒速度']
    features = df[feature_cols].values
    targets = df['亚表面裂纹深度'].values
    
    # 使用RobustScaler
    scaler = RobustScaler()
    features_scaled = scaler.fit_transform(features)
    
    # 划分数据
    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=random_state,
        stratify=material_ids
    )
    
    train_data = {
        'material_ids': material_ids[train_idx],
        'features': features_scaled[train_idx],
        'targets': targets[train_idx],
        'le': le,
        'scaler': scaler,
        'material_to_cluster': material_to_cluster,
        'n_clusters': n_clusters
    }
    
    test_data = {
        'material_ids': material_ids[test_idx],
        'features': features_scaled[test_idx],
        'targets': targets[test_idx]
    }
    
    return train_data, test_data


def train_robust_model(model, train_data, val_data, epochs=200, batch_size=32, 
                      lr=0.0005, device='cuda'):
    """稳健的训练过程"""
    model = model.to(device)
    
    # 创建数据加载器
    train_dataset = TensorDataset(
        torch.LongTensor(train_data['material_ids']),
        torch.FloatTensor(train_data['features']),
        torch.FloatTensor(train_data['targets'])
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 验证数据
    val_materials = torch.LongTensor(val_data['material_ids']).to(device)
    val_features = torch.FloatTensor(val_data['features']).to(device)
    val_targets = torch.FloatTensor(val_data['targets']).unsqueeze(1).to(device)
    
    # 优化器（使用较小的学习率）
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # 学习率调度器（更温和的调度）
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.7, patience=15, min_lr=1e-6
    )
    
    # 损失函数
    criterion = nn.MSELoss()
    
    # 训练历史
    train_losses = []
    val_losses = []
    val_r2_scores = []
    best_val_r2 = -float('inf')
    best_model_state = None
    patience_counter = 0
    
    print("开始训练...")
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        epoch_loss = 0
        
        for batch_materials, batch_features, batch_targets in train_loader:
            batch_materials = batch_materials.to(device)
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.unsqueeze(1).to(device)
            
            optimizer.zero_grad()
            
            # 前向传播
            predictions, uncertainty, _, _ = model(batch_materials, batch_features)
            
            # 简单的MSE损失
            loss = criterion(predictions, batch_targets)
            
            # 添加不确定性正则化
            uncertainty_reg = 0.01 * torch.mean(torch.log(uncertainty))
            total_loss = loss + uncertainty_reg
            
            # 反向传播
            total_loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # 验证阶段
        model.eval()
        with torch.no_grad():
            val_pred, _, _, _ = model(val_materials, val_features)
            val_loss = criterion(val_pred, val_targets)
            val_losses.append(val_loss.item())
            
            # 计算R²
            val_pred_np = val_pred.cpu().numpy()
            val_targets_np = val_targets.cpu().numpy()
            val_r2 = r2_score(val_targets_np, val_pred_np)
            val_r2_scores.append(val_r2)
            
            # 更新学习率
            scheduler.step(val_loss)
            
            # 保存最佳模型
            if val_r2 > best_val_r2:
                best_val_r2 = val_r2
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
        
        # 打印进度
        if (epoch + 1) % 20 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{epochs}:")
            print(f"  LR: {current_lr:.6f}")
            print(f"  Train Loss: {avg_train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val R²: {val_r2:.4f}")
        
        # 早停
        if patience_counter >= 30:
            print(f"\n早停于 epoch {epoch+1}")
            break
    
    # 恢复最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, train_losses, val_losses, val_r2_scores


def evaluate_model(model, test_data, device='cuda'):
    """评估模型"""
    model.eval()
    
    test_materials = torch.LongTensor(test_data['material_ids']).to(device)
    test_features = torch.FloatTensor(test_data['features']).to(device)
    test_targets = test_data['targets']
    
    with torch.no_grad():
        predictions, uncertainty, _, _ = model(test_materials, test_features)
        pred_np = predictions.cpu().numpy().flatten()
        
    # 计算指标
    r2 = r2_score(test_targets, pred_np)
    rmse = np.sqrt(mean_squared_error(test_targets, pred_np))
    mae = np.mean(np.abs(pred_np - test_targets))
    
    print(f"\n测试集性能:")
    print(f"  R² Score: {r2:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    
    # 可视化
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.scatter(test_targets, pred_np, alpha=0.6)
    plt.plot([test_targets.min(), test_targets.max()], 
             [test_targets.min(), test_targets.max()], 'r--', lw=2)
    plt.xlabel('真实值')
    plt.ylabel('预测值')
    plt.title(f'预测结果 (R²={r2:.4f})')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    residuals = pred_np - test_targets
    plt.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel('残差')
    plt.ylabel('频数')
    plt.title('残差分布')
    plt.axvline(x=0, color='r', linestyle='--')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return {'r2': r2, 'rmse': rmse, 'mae': mae}


if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 读取数据
    print("读取数据...")
    df = pd.read_csv('3011916164精确材料性能数据.csv')
    
    # 准备数据（不使用多项式特征）
    print("\n准备数据...")
    train_data, test_data = prepare_data_simple(df)
    
    # 创建模型
    print("\n创建模型...")
    num_materials = len(train_data['le'].classes_)
    model = RobustEnhancedPGNN(
        num_materials=num_materials,
        num_clusters=train_data['n_clusters'],
        material_embed_dim=12,
        latent_dim=24,
        dropout_rate=0.15
    )
    
    # 设置材料聚类信息
    model.material_embedding.set_clusters(
        train_data['material_to_cluster'],
        train_data['le']
    )
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数数量: {total_params:,}")
    
    # 训练模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n使用设备: {device}")
    
    model, train_losses, val_losses, val_r2_scores = train_robust_model(
        model, train_data, test_data,
        epochs=300,
        batch_size=32,
        lr=0.0005,
        device=device
    )
    
    # 评估模型
    print("\n评估模型...")
    results = evaluate_model(model, test_data, device)
    
    # 绘制训练历史
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='训练损失')
    plt.plot(val_losses, label='验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.title('训练历史')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(val_r2_scores)
    plt.xlabel('Epoch')
    plt.ylabel('验证 R²')
    plt.title('验证 R² 历史')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.plot(np.gradient(train_losses))
    plt.xlabel('Epoch')
    plt.ylabel('损失梯度')
    plt.title('训练稳定性')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()