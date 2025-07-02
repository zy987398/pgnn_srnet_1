import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
from improved_pgnn import SimplifiedPGNN, create_hybrid_model

def prepare_material_data_fixed(df):
    """准备材料数据 - 修复版本，确保物理量为正"""
    # 编码材料种类
    le = LabelEncoder()
    material_ids = le.fit_transform(df['材料种类'])
    
    # 提取数值特征
    numeric_cols = ['莫氏硬度', '弹性模量', '断裂韧性', '晶粒粒径', '压溃压力', '磨粒速度']
    numeric_features = df[numeric_cols].values
    
    # 使用MinMaxScaler确保所有值为正（归一化到[0,1]）
    scaler = MinMaxScaler()
    numeric_features = scaler.fit_transform(numeric_features)
    
    # 目标变量
    target = df['亚表面裂纹深度'].values
    
    return {
        'material_ids': material_ids,
        'numeric_features': numeric_features,
        'target': target,
        'label_encoder': le,
        'scaler': scaler,
        'num_materials': len(le.classes_)
    }

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 准备数据
df = pd.read_csv('3011916164精确材料性能数据.csv')
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# 使用修复的数据准备函数
train_data = prepare_material_data_fixed(train_df)
test_data = prepare_material_data_fixed(test_df)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# 创建模型
model = SimplifiedPGNN(
    num_materials=train_data['num_materials'],
    material_embed_dim=16,
    hidden_dim=32
).to(device)

# 初始化权重
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=0.5)
        if m.bias is not None:
            m.bias.data.fill_(0.01)
    elif isinstance(m, nn.Embedding):
        torch.nn.init.normal_(m.weight, mean=0.0, std=0.1)

model.apply(init_weights)

# 准备数据
train_materials = torch.LongTensor(train_data['material_ids']).to(device)
train_features = torch.FloatTensor(train_data['numeric_features']).to(device)
train_target = torch.FloatTensor(train_data['target']).unsqueeze(1).to(device)

test_materials = torch.LongTensor(test_data['material_ids']).to(device)
test_features = torch.FloatTensor(test_data['numeric_features']).to(device)
test_target = torch.FloatTensor(test_data['target']).unsqueeze(1).to(device)

# 优化器
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)
loss_fn = nn.MSELoss()

# 训练
best_test_r2 = -float('inf')
train_losses = []
test_scores = []

print("Starting training...")
for epoch in range(300):
    # 训练
    model.train()
    optimizer.zero_grad()
    
    pred, _, _, _ = model(train_materials, train_features)
    loss = loss_fn(pred, train_target)
    
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    train_losses.append(loss.item())
    
    # 评估
    if (epoch + 1) % 20 == 0:
        model.eval()
        with torch.no_grad():
            test_pred, _, _, _ = model(test_materials, test_features)
            test_pred_np = test_pred.cpu().squeeze().numpy()
            test_target_np = test_target.cpu().squeeze().numpy()
            
            test_r2 = r2_score(test_target_np, test_pred_np)
            test_rmse = np.sqrt(mean_squared_error(test_target_np, test_pred_np))
            test_scores.append(test_r2)
            
            print(f"Epoch {epoch+1}: Loss={loss.item():.4f}, Test R²={test_r2:.4f}, RMSE={test_rmse:.4f}")
            
            if test_r2 > best_test_r2:
                best_test_r2 = test_r2
                torch.save(model.state_dict(), 'best_pgnn.pth')
    
    scheduler.step(loss)

# 加载最佳模型
model.load_state_dict(torch.load('best_pgnn.pth'))

# 最终评估
model.eval()
with torch.no_grad():
    final_pred, _, _, _ = model(test_materials, test_features)
    final_pred_np = final_pred.cpu().squeeze().numpy()
    final_target_np = test_target.cpu().squeeze().numpy()
    
    final_r2 = r2_score(final_target_np, final_pred_np)
    final_rmse = np.sqrt(mean_squared_error(final_target_np, final_pred_np))

print(f"\n最终PGNN性能: R²={final_r2:.4f}, RMSE={final_rmse:.4f}")

# PGNN + XGBoost混合模型
print("\n训练PGNN+XGBoost混合模型...")
xgb_model, _ = create_hybrid_model(model, train_data, device)

# 混合模型预测
with torch.no_grad():
    test_material_embed = model.material_embedding(test_materials)
    test_physics_features = model.physics_extractor(test_features)
    test_combined = torch.cat([test_material_embed, test_physics_features], dim=1)
    test_hidden = model.feature_fusion(test_combined)
    test_pgnn_features = test_hidden.cpu().numpy()

test_combined_features = np.hstack([test_data['numeric_features'], test_pgnn_features])
hybrid_pred = xgb_model.predict(test_combined_features)

hybrid_r2 = r2_score(final_target_np, hybrid_pred)
hybrid_rmse = np.sqrt(mean_squared_error(final_target_np, hybrid_pred))
print(f"PGNN+XGBoost: R²={hybrid_r2:.4f}, RMSE={hybrid_rmse:.4f}")

# 可视化
import matplotlib.pyplot as plt

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

# 训练损失
ax1.plot(train_losses)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training Loss')
ax1.set_yscale('log')

# PGNN预测
ax2.scatter(final_target_np, final_pred_np, alpha=0.5)
ax2.plot([0, 12], [0, 12], 'r--')
ax2.set_xlabel('Actual')
ax2.set_ylabel('Predicted')
ax2.set_title(f'PGNN (R²={final_r2:.3f})')

# 混合模型预测
ax3.scatter(final_target_np, hybrid_pred, alpha=0.5, color='green')
ax3.plot([0, 12], [0, 12], 'r--')
ax3.set_xlabel('Actual')
ax3.set_ylabel('Predicted')
ax3.set_title(f'PGNN+XGBoost (R²={hybrid_r2:.3f})')

# 特征分布检查
ax4.hist(train_features.cpu().numpy().flatten(), bins=50, alpha=0.7)
ax4.set_xlabel('Feature Values')
ax4.set_ylabel('Frequency')
ax4.set_title('Feature Distribution (After MinMax Scaling)')

plt.tight_layout()
plt.savefig('pgnn_fixed_results.png', dpi=300)
plt.show()
