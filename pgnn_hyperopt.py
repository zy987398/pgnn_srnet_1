import optuna
from optuna.trial import TrialState
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import warnings
import os
warnings.filterwarnings("ignore")

from material_crack_pgnn import (
    MaterialCrackPGNN,
    PhysicsInformedLoss,
    prepare_material_data,
)

def create_model(trial, num_materials):
    """根据trial创建模型"""
    # 模型架构超参数
    material_embed_dim = trial.suggest_int("material_embed_dim", 8, 64, step=8)
    latent_dim = trial.suggest_int("latent_dim", 16, 128, step=16)
    
    # 创建模型并修改部分架构
    model = MaterialCrackPGNN(
        num_materials=num_materials,
        material_embed_dim=material_embed_dim,
        latent_dim=latent_dim
    )
    
    # 动态调整dropout率
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5, step=0.1)
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = dropout_rate
    
    return model

def train_with_trial(trial, train_data, val_data, num_materials, device='cuda'):
    """使用给定的超参数训练模型"""
    
    # 训练超参数
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    epochs = trial.suggest_int("epochs", 200, 1000, step=100)
    
    # 损失函数超参数
    lambda_complexity = trial.suggest_float("lambda_complexity", 1e-5, 1e-2, log=True)
    lambda_phys_base = trial.suggest_float("lambda_physics", 0.05, 0.5, step=0.025)  # {{ 调整搜索范围符合文献推荐值 }}
    
    # 优化器选择
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW", "RMSprop"])
    weight_decay = trial.suggest_float("weight_decay", 0, 0.01) if optimizer_name == "AdamW" else 0
    
    # 创建模型
    model = create_model(trial, num_materials).to(device)
    
    # 创建优化器
    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    
    # 学习率调度器
    scheduler_type = trial.suggest_categorical("scheduler", ["plateau", "cosine", "none"])
    if scheduler_type == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=20, factor=0.5, min_lr=1e-6
        )
    elif scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    else:
        scheduler = None
    
    # 损失函数
    loss_fn = PhysicsInformedLoss(
        lambda_complexity=lambda_complexity,
        lambda_physics=lambda_phys_base  # 使用基础值初始化
    )
    
    # 准备数据
    train_materials = torch.LongTensor(train_data['material_ids']).to(device)
    train_features = torch.FloatTensor(train_data['numeric_features']).to(device)
    train_target = torch.FloatTensor(train_data['target']).unsqueeze(1).to(device)
    
    val_materials = torch.LongTensor(val_data['material_ids']).to(device)
    val_features = torch.FloatTensor(val_data['numeric_features']).to(device)
    val_target = torch.FloatTensor(val_data['target']).unsqueeze(1).to(device)
    
    # 训练循环
    best_val_r2 = -float('inf')
    patience = 40  # {{ 修改早停机制的patience为40 }}
    patience_counter = 0
    
    train_losses = []
    val_losses = []  # 添加验证损失列表
    val_r2_history = []
    
    for epoch in range(epochs):
        # 动态调整lambda_physics
        lambda_phys = lambda_phys_base * min(1, epoch / 100)  # {{ 动态调整lambda_physics }}
        
        # 训练
        model.train()
        optimizer.zero_grad()
        
        pred, uncertainty, symbolic, residual = model(train_materials, train_features)
        loss, metrics = loss_fn(pred, uncertainty, train_target, model, train_features)
        
        # 动态更新lambda_physics
        loss_fn.lambda_physics = lambda_phys  # {{ 动态更新lambda_physics }}
        
        # 检查NaN
        if torch.isnan(loss):
            return float('inf')
        
        loss.backward()
        
        # 梯度裁剪（降低阈值至0.5）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        
        optimizer.step()
        
        # 记录训练损失
        train_losses.append(loss.item())
        
        # 验证
        model.eval()
        with torch.no_grad():
            val_pred, val_uncertainty, _, _ = model(val_materials, val_features)
            val_loss, _ = loss_fn(val_pred, val_uncertainty, val_target, model, val_features)
            val_losses.append(val_loss.item())
            
            val_pred_np = val_pred.cpu().squeeze().numpy()
            val_target_np = val_target.cpu().squeeze().numpy()
            
            val_r2 = r2_score(val_target_np, val_pred_np)
            val_r2_history.append(val_r2)  # 记录每个epoch的val_r2
            
            # 定义val_rmse用于学习率调度
            val_rmse = np.sqrt(mean_squared_error(val_target_np, val_pred_np))

        # 更新学习率
        if scheduler:
            if scheduler_type == "plateau":
                scheduler.step(val_rmse)
            else:
                scheduler.step()
        
        # 早停检查
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            patience_counter = 0
            # 保存最佳模型（在100~150个epoch之间）
            if 100 <= epoch <= 150:  # {{ 保存最佳模型（在100~150个epoch之间） }}
                torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            break
        
        # 向Optuna报告中间结果
        trial.report(val_r2, epoch)
        
        # 如果trial应该被剪枝则停止
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return best_val_r2, train_losses, val_r2_history  # 返回训练损失和验证R²历史

def objective(trial, train_data, val_data, num_materials, device):
    """Optuna目标函数"""
    try:
        # 仅获取val_r2作为返回值
        val_r2, _, _ = train_with_trial(trial, train_data, val_data, num_materials, device)
        return val_r2
    except Exception as e:
        print(f"Trial failed with error: {e}")
        return -float('inf')

def run_hyperopt(data_path="3011916164精确材料性能数据.csv", n_trials=100, seed=42):
    """运行超参数优化"""
    
    # 设置随机种子
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 读取数据
    df = pd.read_csv(data_path)
    
    # 分割数据
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=seed)
    
    # 准备数据
    train_data = prepare_material_data(train_df)
    val_data = prepare_material_data(val_df)
    num_materials = train_data['num_materials']
    
    # 设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 创建study
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=10, n_warmup_steps=50
        ),
        sampler=optuna.samplers.TPESampler(seed=seed)
    )
    
    # 优化
    study.optimize(
        lambda trial: objective(trial, train_data, val_data, num_materials, device),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    # 打印结果
    print("\n" + "="*50)
    print("优化完成!")
    print("="*50)
    
    print(f"\n最佳验证R²: {study.best_value:.4f}")
    print("\n最佳超参数:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # 统计信息
    completed_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
    print(f"\n完成的trials: {len(completed_trials)}/{n_trials}")
    
    # 超参数重要性
    try:
        importance = optuna.importance.get_param_importances(study)
        print("\n超参数重要性:")
        for key, value in sorted(importance.items(), key=lambda x: x[1], reverse=True):
            print(f"  {key}: {value:.3f}")
    except:
        pass
    
    return study

# 使用最佳超参数重新训练
def retrain_with_best_params(study, data_path="3011916164精确材料性能数据.csv", seed=42):
    """使用最佳超参数重新训练并评估"""
    
    # 读取数据
    df = pd.read_csv(data_path)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=seed)
    
    train_data = prepare_material_data(train_df)
    test_data = prepare_material_data(test_df)
    
    # 创建模型
    best_params = study.best_params
    model = MaterialCrackPGNN(
        num_materials=train_data['num_materials'],
        material_embed_dim=best_params['material_embed_dim'],
        latent_dim=best_params['latent_dim']
    )
    
    # 设置dropout
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = best_params['dropout_rate']
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    # 优化器
    if best_params['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=best_params['lr'])
    elif best_params['optimizer'] == 'AdamW':
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=best_params['lr'],
            weight_decay=best_params.get('weight_decay', 0)
        )
    else:
        optimizer = torch.optim.RMSprop(model.parameters(), lr=best_params['lr'])
    
    # 损失函数
    loss_fn = PhysicsInformedLoss(
        lambda_complexity=best_params['lambda_complexity'],
        lambda_physics=best_params['lambda_physics']
    )
    
    # 准备数据
    train_materials = torch.LongTensor(train_data['material_ids']).to(device)
    train_features = torch.FloatTensor(train_data['numeric_features']).to(device)
    train_target = torch.FloatTensor(train_data['target']).unsqueeze(1).to(device)
    
    test_materials = torch.LongTensor(test_data['material_ids']).to(device)
    test_features = torch.FloatTensor(test_data['numeric_features']).to(device)
    test_target = torch.FloatTensor(test_data['target']).unsqueeze(1).to(device)
    
    # 添加早停参数
    best_test_r2 = -float('inf')
    patience = 50
    patience_counter = 0
    
    # 训练
    print("\n使用最佳超参数重新训练...")
    train_losses = []
    test_r2_history = []
    
    # 创建保存模型的目录
    os.makedirs('best_model', exist_ok=True)

    for epoch in range(best_params['epochs']):
        model.train()
        optimizer.zero_grad()
        
        pred, uncertainty, _, _ = model(train_materials, train_features)
        loss, _ = loss_fn(pred, uncertainty, train_target, model, train_features)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # 评估
        if (epoch + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                # 修正模型输出解包，接收全部四个返回值
                test_pred, _, _, _ = model(test_materials, test_features)
                
                # 添加NaN检测与替换
                test_pred = torch.nan_to_num(test_pred, nan=0.0, posinf=1e5, neginf=-1e5)
                
                test_pred_np = test_pred.cpu().squeeze().numpy()
                test_target_np = test_target.cpu().squeeze().numpy()
                
                test_r2 = r2_score(test_target_np, test_pred_np)
                test_rmse = np.sqrt(mean_squared_error(test_target_np, test_pred_np))
                test_r2_history.append(test_r2)
                
                print(f"Epoch {epoch+1}: Test R²={test_r2:.4f}, RMSE={test_rmse:.4f}")

            # 早停检查
            if test_r2 > best_test_r2:
                best_test_r2 = test_r2
                patience_counter = 0
                # 保存最佳模型
                torch.save(model.state_dict(), 'best_model/best_model.pth')
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # 加载最佳模型
    model.load_state_dict(torch.load('best_model/best_model.pth'))
    os.remove('best_model/best_model.pth')
    
    # 最终评估
    model.eval()
    with torch.no_grad():
        test_pred, test_uncertainty, _, _ = model(test_materials, test_features)
        test_pred_np = test_pred.cpu().squeeze().numpy()
        test_target_np = test_target.cpu().squeeze().numpy()
        
        final_r2 = r2_score(test_target_np, test_pred_np)
        final_rmse = np.sqrt(mean_squared_error(test_target_np, test_pred_np))
    
    print(f"\n最终测试集性能:")
    print(f"  R²: {final_r2:.4f}")
    print(f"  RMSE: {final_rmse:.4f}")
    
    return model, train_losses, test_r2_history

if __name__ == "__main__":
    # 运行超参数优化
    study = run_hyperopt(n_trials=50)  # 可以增加trials获得更好结果
    
    # 使用最佳参数重新训练
    # 重新加载数据
    df = pd.read_csv("3011916164精确材料性能数据.csv")
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_data = prepare_material_data(train_df)
    val_data = prepare_material_data(test_df)  # 定义val_data
    
    model, train_losses, test_r2_history = retrain_with_best_params(study)
    
    # 可视化优化历史
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 优化历史
    ax1.plot([t.value for t in study.trials if t.value is not None])
    ax1.set_xlabel('Trial')
    ax1.set_ylabel('Validation R²')
    ax1.set_title('Hyperparameter Optimization History')
    ax1.grid(True, alpha=0.3)
    
    # 训练损失（使用重新训练时的损失）
    ax2.plot(train_losses)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training Loss with Best Hyperparameters')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/hyperopt_results.png', dpi=300)
    plt.show()
