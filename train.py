import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 导入之前定义的模型类（假设已经导入）
from model import PGNNSymbolic, PGNNLoss, train_pgnn, create_physics_features

class PhysicsDataset(Dataset):
    """物理数据集类"""
    def __init__(self, X, y, transform=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).reshape(-1, 1)
        self.transform = transform
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        
        if self.transform:
            x = self.transform(x)
            
        return x, y

def generate_synthetic_physics_data(n_samples=1000, noise_level=0.1):
    """生成合成物理数据用于测试"""
    np.random.seed(42)
    
    # 生成基础特征
    x1 = np.random.uniform(-5, 5, n_samples)
    x2 = np.random.uniform(-5, 5, n_samples)
    x3 = np.random.uniform(0.1, 10, n_samples)
    
    # 创建物理关系（例如：简单的物理定律）
    # y = a * x1^2 + b * log(x3) + c * x1 * x2 + noise
    y = 2.5 * x1**2 + 3.0 * np.log(x3) + 0.5 * x1 * x2
    
    # 添加噪声
    y += np.random.normal(0, noise_level, n_samples)
    
    # 创建额外的噪声特征
    noise_features = np.random.randn(n_samples, 57)
    
    # 组合所有特征
    X = np.column_stack([x1, x2, x3, noise_features])
    
    return X, y

def prepare_data_loaders(X, y, batch_size=32, val_split=0.2, test_split=0.1):
    """准备数据加载器"""
    # 分割数据
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_split, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_split/(1-test_split), random_state=42
    )
    
    # 标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # 创建数据集
    train_dataset = PhysicsDataset(X_train, y_train)
    val_dataset = PhysicsDataset(X_val, y_val)
    test_dataset = PhysicsDataset(X_test, y_test)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, scaler

def evaluate_model(model, test_loader, device='cuda'):
    """评估模型性能"""
    model.eval()
    predictions = []
    uncertainties = []
    actuals = []
    symbolic_preds = []
    residuals = []
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            
            y_pred, sigma_squared, c, r = model(x)
            
            predictions.extend(y_pred.cpu().numpy())
            uncertainties.extend(sigma_squared.cpu().numpy())
            actuals.extend(y.cpu().numpy())
            symbolic_preds.extend(c.cpu().numpy())
            residuals.extend(r.cpu().numpy())
    
    predictions = np.array(predictions)
    uncertainties = np.array(uncertainties)
    actuals = np.array(actuals)
    symbolic_preds = np.array(symbolic_preds)
    residuals = np.array(residuals)
    
    # 计算指标
    mse = np.mean((predictions - actuals)**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - actuals))
    
    # R^2分数
    ss_res = np.sum((actuals - predictions)**2)
    ss_tot = np.sum((actuals - np.mean(actuals))**2)
    r2 = 1 - (ss_res / ss_tot)
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'predictions': predictions,
        'uncertainties': uncertainties,
        'actuals': actuals,
        'symbolic_preds': symbolic_preds,
        'residuals': residuals
    }

def plot_results(results, save_path=None):
    """可视化结果"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. 预测 vs 实际
    ax = axes[0, 0]
    ax.scatter(results['actuals'], results['predictions'], alpha=0.5)
    ax.plot([results['actuals'].min(), results['actuals'].max()], 
            [results['actuals'].min(), results['actuals'].max()], 'r--')
    ax.set_xlabel('Actual Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title(f'Predictions vs Actuals (R² = {results["r2"]:.3f})')
    
    # 2. 残差图
    ax = axes[0, 1]
    residuals = results['actuals'] - results['predictions']
    ax.scatter(results['predictions'], residuals, alpha=0.5)
    ax.axhline(y=0, color='r', linestyle='--')
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Residuals')
    ax.set_title('Residual Plot')
    
    # 3. 不确定性分布
    ax = axes[1, 0]
    ax.hist(np.sqrt(results['uncertainties']), bins=30, alpha=0.7)
    ax.set_xlabel('Predicted Standard Deviation')
    ax.set_ylabel('Frequency')
    ax.set_title('Uncertainty Distribution')
    
    # 4. 符号表达式 vs 残差贡献
    ax = axes[1, 1]
    ax.scatter(results['symbolic_preds'], results['residuals'], alpha=0.5)
    ax.set_xlabel('Symbolic Expression Output')
    ax.set_ylabel('Residual Branch Output')
    ax.set_title('Symbolic vs Residual Contributions')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def main():
    """主训练流程"""
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 生成数据
    print("Generating synthetic physics data...")
    X, y = generate_synthetic_physics_data(n_samples=5000)
    print(f"Data shape: X={X.shape}, y={y.shape}")
    
    # 准备数据加载器
    train_loader, val_loader, test_loader, scaler = prepare_data_loaders(
        X, y, batch_size=64
    )
    
    # 创建模型
    model = PGNNSymbolic(input_dim=60, latent_dim=16)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 训练模型
    print("\nTraining model...")
    train_losses, val_losses = train_pgnn(
        model, train_loader, val_loader, 
        epochs=100, lr=1e-3, device=device
    )
    
    # 评估模型
    print("\nEvaluating model...")
    results = evaluate_model(model, test_loader, device)
    
    print(f"\nTest Results:")
    print(f"  MSE: {results['mse']:.4f}")
    print(f"  RMSE: {results['rmse']:.4f}")
    print(f"  MAE: {results['mae']:.4f}")
    print(f"  R²: {results['r2']:.4f}")
    
    # 绘制结果
    plot_results(results)
    
    # 绘制训练曲线
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_losses[-50:], label='Train Loss (last 50)')
    plt.plot(val_losses[-50:], label='Val Loss (last 50)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History (Zoomed)')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # 保存模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler': scaler,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'test_results': results
    }, 'pgnn_model.pth')
    print("\nModel saved to pgnn_model.pth")

# 高级功能：符号表达式提取和简化
def extract_and_simplify_expression(model, sample_data, device='cuda'):
    """提取并简化符号表达式"""
    model.eval()
    model = model.to(device)
    sample_data = sample_data.to(device)
    
    with torch.no_grad():
        z = model.encoder(sample_data)
        
        # 获取符号层的输出
        op_logits = model.symbolic_layer.operator_net(z)
        var_logits = model.symbolic_layer.variable_net(z)
        constants = model.symbolic_layer.constant_net(z)
        
        # 获取最可能的操作和变量
        op_probs = torch.softmax(op_logits, dim=-1)
        var_probs = torch.softmax(var_logits, dim=-1)
        
        # 提取表达式结构
        top_ops = torch.topk(op_probs, k=3, dim=-1)
        top_vars = torch.topk(var_probs, k=5, dim=-1)
        
        expression_info = {
            'top_operators': [(model.symbolic_layer.operators[idx], prob.item()) 
                             for idx, prob in zip(top_ops.indices[0], top_ops.values[0])],
            'top_variables': [(idx.item(), prob.item()) 
                             for idx, prob in zip(top_vars.indices[0], top_vars.values[0])],
            'constants': constants[0].cpu().numpy()
        }
        
    return expression_info

# 使用示例
if __name__ == "__main__":
    # 运行主训练流程
    main()
    
    # 或者加载已保存的模型
    # checkpoint = torch.load('pgnn_model.pth')
    # model = PGNNSymbolic(input_dim=60, latent_dim=16)
    # model.load_state_dict(checkpoint['model_state_dict'])