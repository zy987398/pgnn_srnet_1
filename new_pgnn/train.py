import os
import time
import warnings
import copy
import torch
from datetime import datetime
from torch.optim.lr_scheduler import OneCycleLR
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 设置中文字体为 SimSun，解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['axes.unicode_minus'] = False

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from torch.utils.data import TensorDataset, DataLoader

# 导入自定义模块
from models import MaterialCrackPGNN
from losses import PhysicsInformedLoss
from utils import prepare_material_data, augment_dataset

# 忽略警告
warnings.filterwarnings("ignore")

# 数据路径
DATA_PATH = "3011916164精确材料性能数据.csv"

# 创建带时间戳的结果目录
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_DIR = f"results/run_{TIMESTAMP}"

# 固定随机种子，确保结果可重复
SEED = 66
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# 最佳超参数 (从best_hyperopt_params.json提取)
PARAMS = {
    # 模型架构参数
    "material_embed_dim": 8,     # 材料嵌入维度
    "latent_dim": 48,            # 隐藏层维度
    "dropout_rate": 0.35,        # Dropout率
    "symbolic_type": "advanced",  # 符号层类型，可选: "lightweight", "improved", "advanced"等

    # 训练参数
    "epochs": 350,               # 训练轮数
    "batch_size": 16,            # 批次大小
    "lr": 0.0011173842752625436, # 学习率
    "patience": 55,              # 早停参数
    "augmentation": False,       # 是否使用数据增强
    "optimizer": "AdamW",        # 优化器
    "weight_decay": 0.0066923795400554925,      # 权重衰减
    "lambda_complexity": 3.4069890537972935e-05,  # 复杂度损失权重
    "lambda_physics": 0.2        # 物理损失权重
}


def train_with_fixed_params(params, data_path=DATA_PATH, seed=SEED):
    """使用固定超参数训练和评估模型"""
    print("\n" + "="*50)
    print("使用固定超参数训练模型")
    print("="*50)
    
    # 打印使用的参数
    for key, value in params.items():
        print(f"{key}: {value}")
    
    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n使用设备: {device}")
    
    # 读取和分割数据
    print("\n加载数据...")
    df = pd.read_csv(data_path)
    print(f"数据集大小: {len(df)} 样本")
    
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=seed, stratify=df["材料种类"])
    print(f"训练集: {len(train_df)} 样本, 测试集: {len(test_df)} 样本")
    
    # 准备数据
    train_data = prepare_material_data(train_df)
    test_data = prepare_material_data(
        test_df,
        scaler=train_data["scaler"],
        le=train_data["label_encoder"],
    )
    
    num_materials = train_data["num_materials"]
    print(f"共 {num_materials} 种材料类型")
    
    # 数据增强 (如果启用)
    train_df_for_aug = train_df.copy()
    if params["augmentation"]:
        print("\n应用数据增强...")
        train_df_for_aug = augment_dataset(train_df_for_aug)
        print(f"数据集大小: {len(train_df)} -> {len(train_df_for_aug)} (增加 {len(train_df_for_aug)-len(train_df)} 个样本)")
    
    augmented_train_data = prepare_material_data(
        train_df_for_aug,
        scaler=train_data["scaler"],
        le=train_data["label_encoder"],
        augment=False,
    )
    
    # 创建模型
    print("\n创建模型...")
    model = MaterialCrackPGNN(
        num_materials=num_materials,
        material_embed_dim=params.get("material_embed_dim", 16),
        latent_dim=params.get("latent_dim", 32),
        dropout_rate=params.get("dropout_rate", 0.2),
        symbolic_type=params.get("symbolic_type", "advanced")
    )
    model = model.to(device)
    
    # 打印模型参数总数
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数总数: {total_params:,}")
    
    # 创建DataLoader
    batch_size = params["batch_size"]
    train_dataset = TensorDataset(
        torch.LongTensor(augmented_train_data["material_ids"]),
        torch.FloatTensor(augmented_train_data["numeric_features"]),
        torch.FloatTensor(augmented_train_data["numeric_raw"]),
        torch.FloatTensor(augmented_train_data["target"]).unsqueeze(1),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=True,
        num_workers=0,  # 单进程加载
    )
    
    # 准备测试数据
    test_materials = torch.LongTensor(test_data["material_ids"]).to(device)
    test_features = torch.FloatTensor(test_data["numeric_features"]).to(device)
    test_raw = torch.FloatTensor(test_data["numeric_raw"]).to(device)
    test_target = torch.FloatTensor(test_data["target"]).unsqueeze(1).to(device)
    
    # 优化器
    lr = params["lr"]
    optimizer_name = params["optimizer"]
    weight_decay = params["weight_decay"]
    
    if optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "RMSprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "RAdam":
        optimizer = torch.optim.RAdam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, 
                                   weight_decay=weight_decay, nesterov=True)
    elif optimizer_name == "NAdam":
        optimizer = torch.optim.NAdam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        print(f"未知的优化器: {optimizer_name}, 使用默认的AdamW")
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # 学习率调度
    epochs = params["epochs"]
    total_steps = epochs * len(train_loader)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=lr,
        total_steps=total_steps,
        pct_start=0.3,      # 30% 的步数用于预热
        div_factor=25.0,    # 初始学习率 = max_lr/25
        final_div_factor=1e4,  # 最终学习率 = max_lr/(25*10000)
        anneal_strategy='cos'
    )
    
    # 损失函数
    loss_fn = PhysicsInformedLoss(
        lambda_complexity=params["lambda_complexity"],
        lambda_physics=params["lambda_physics"],
        weight_decay=0.0,
    )
    loss_fn.total_epochs = epochs
    
    # 训练循环
    print("\n开始训练...")
    start_time = time.time()
    best_val_r2 = -float("inf")
    patience_counter = 0
    best_model_wts = None
    train_losses = []
    val_losses = []
    val_r2_scores = []
    
    for epoch in range(epochs):
        # 更新当前epoch
        loss_fn.update_epoch(epoch, epochs)
        
        # 训练
        model.train()
        epoch_loss = 0
        
        for batch_materials, batch_features, batch_raw, batch_target in train_loader:
            batch_materials = batch_materials.to(device, non_blocking=True)
            batch_features = batch_features.to(device, non_blocking=True)
            batch_raw = batch_raw.to(device, non_blocking=True)
            batch_target = batch_target.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            pred, uncertainty, symbolic, residual = model(batch_materials, batch_raw)
            loss, metrics = loss_fn(pred, uncertainty, batch_target, model, batch_raw)
            
            # 检查NaN
            if torch.isnan(loss) or torch.isinf(loss):
                print("警告：检测到NaN或Inf损失，跳过此批次")
                continue
                
            loss.backward()
            # 更严格的梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            scheduler.step()  # OneCycleLR 需要在每个批次后更新
            
            epoch_loss += loss.item() * batch_target.size(0)
        
        epoch_loss /= len(train_dataset)
        train_losses.append(epoch_loss)
        
        # 验证
        model.eval()
        with torch.no_grad():
            val_pred, val_uncertainty, _, _ = model(test_materials, test_raw)
            val_loss, _ = loss_fn(val_pred, val_uncertainty, test_target, model, test_raw)
            val_losses.append(val_loss.item())
            
            val_pred_np = val_pred.cpu().squeeze().numpy()
            val_target_np = test_target.cpu().squeeze().numpy()
            
            val_r2 = r2_score(val_target_np, val_pred_np)
            val_rmse = np.sqrt(mean_squared_error(val_target_np, val_pred_np))
            val_r2_scores.append(val_r2)
            
            # 每10个epoch打印一次进度
            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch+1}/{epochs} - 训练损失: {epoch_loss:.4f}, 验证损失: {val_loss.item():.4f}, 验证R²: {val_r2:.4f}, 验证RMSE: {val_rmse:.4f}")
                
                # 门控监控
                if hasattr(model, 'get_gate_statistics'):
                    gate_stats = model.get_gate_statistics()
                    print(f"         门控统计 - 平均值: {gate_stats['mean']:.3f}, "
                          f"标准差: {gate_stats['std']:.3f}, "
                          f"符号层权重: {gate_stats['symbolic_weight']:.3f}")
        
        # 早停检查
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            patience_counter = 0
            # 保存最佳模型状态
            best_model_wts = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
        
        if patience_counter >= params["patience"]:
            print(f"\n早停：{params['patience']} 轮未改善，在第 {epoch+1} 轮停止")
            break
    
    # 恢复最佳模型状态
    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)
    
    # 训练时间
    train_time = time.time() - start_time
    print(f"\n训练完成，用时: {train_time:.2f} 秒 ({train_time/60:.2f} 分钟)")
    
    # 评估模型
    print("\n在测试集上评估模型...")
    model.eval()
    with torch.no_grad():
        test_pred, test_uncertainty, test_symbolic, test_residual = model(test_materials, test_raw)
        
        # 转换为NumPy数组
        test_pred_np = test_pred.cpu().squeeze().numpy()
        test_target_np = test_target.cpu().squeeze().numpy()
        test_uncertainty_np = test_uncertainty.cpu().squeeze().numpy()
        
        # 计算指标
        test_r2 = r2_score(test_target_np, test_pred_np)
        test_rmse = np.sqrt(mean_squared_error(test_target_np, test_pred_np))
        test_mae = mean_absolute_error(test_target_np, test_pred_np)
        
        print(f"测试集 R²: {test_r2:.4f}")
        print(f"测试集 RMSE: {test_rmse:.4f}")
        print(f"测试集 MAE: {test_mae:.4f}")
        
        # 门控融合分析
        if hasattr(model, 'get_fusion_interpretation'):
            print(f"\n门控融合分析:")
            print(model.get_fusion_interpretation())
            
            gate_stats = model.get_gate_statistics()
            print(f"门控详细统计: 平均={gate_stats['mean']:.3f}, "
                  f"最小={gate_stats['min']:.3f}, 最大={gate_stats['max']:.3f}")
        
        # 符号层可解释性分析（如果支持）
        if hasattr(model, 'get_symbolic_interpretation'):
            print(f"\n符号层可解释性分析:")
            interp = model.get_symbolic_interpretation()
            if 'complexity' in interp:
                print(f"符号表达式复杂度: {interp['complexity']:.4f}")
    
    # 可视化训练过程
    print("\n生成训练过程可视化...")
    plt.figure(figsize=(12, 10))
    
    # 绘制损失曲线
    plt.subplot(2, 1, 1)
    plt.plot(train_losses, label='训练损失')
    plt.plot(val_losses, label='验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.legend()
    plt.title('训练和验证损失')
    
    # 绘制R²曲线
    plt.subplot(2, 1, 2)
    plt.plot(val_r2_scores, label='验证R²')
    plt.axhline(y=test_r2, color='r', linestyle='-', label=f'测试R²: {test_r2:.4f}')
    plt.xlabel('Epoch')
    plt.ylabel('R²')
    plt.legend()
    plt.title('验证和测试R²')
    
    # 保存图表
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plt.tight_layout()
    training_plots_path = os.path.join(RESULTS_DIR, "training_plots.png")
    plt.savefig(training_plots_path)
    plt.close()
    print(f"训练图表已保存至 {training_plots_path}")
    
    # 门控权重可视化（如果模型支持）
    if hasattr(model, 'last_gate') and model.last_gate is not None:
        plt.figure(figsize=(10, 6))
        gate_values = model.last_gate.cpu().detach().numpy().flatten()
        
        plt.subplot(1, 2, 1)
        plt.hist(gate_values, bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel('门控权重值')
        plt.ylabel('频次')
        plt.title('门控权重分布')
        plt.axvline(x=0.5, color='r', linestyle='--', label='平衡点 (0.5)')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(gate_values[:50], 'o-', alpha=0.7)  # 显示前50个样本
        plt.xlabel('样本索引')
        plt.ylabel('门控权重')
        plt.title('门控权重示例 (前50个样本)')
        plt.axhline(y=0.5, color='r', linestyle='--', label='平衡点')
        plt.legend()
        
        plt.tight_layout()
        gate_path = os.path.join(RESULTS_DIR, "gate_analysis.png")
        plt.savefig(gate_path)
        plt.close()
        print(f"门控分析图已保存至 {gate_path}")
    
    # 预测可视化
    plt.figure(figsize=(10, 8))
    plt.scatter(test_target_np, test_pred_np, alpha=0.5)
    plt.plot([test_target_np.min(), test_target_np.max()], 
             [test_target_np.min(), test_target_np.max()], 'r--')
    plt.xlabel('实际裂纹深度')
    plt.ylabel('预测裂纹深度')
    plt.title(f'实际值 vs 预测值 (测试集 R² = {test_r2:.4f}')
    pred_vis_path = os.path.join(RESULTS_DIR, "predictions_visualization.png")
    plt.savefig(pred_vis_path)
    plt.close()
    print(f"预测可视化已保存至 {pred_vis_path}")
    
    # 保存模型
    model_path = os.path.join(RESULTS_DIR, "train_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"模型已保存至 {model_path}")
    
    # 不确定性可视化
    plt.figure(figsize=(10, 8))
    plt.errorbar(range(len(test_pred_np)), test_pred_np, 
                 yerr=np.sqrt(test_uncertainty_np), fmt='o', alpha=0.4)
    plt.plot(range(len(test_target_np)), test_target_np, 'ro', label='实际值')
    plt.xlabel('测试样本')
    plt.ylabel('裂纹深度')
    plt.title('预测值与不确定性')
    plt.legend()
    uncertainty_path = os.path.join(RESULTS_DIR, "uncertainty_visualization.png")
    plt.savefig(uncertainty_path)
    plt.close()
    print(f"不确定性可视化已保存至 {uncertainty_path}")
    
    # 返回模型和指标
    results = {
        "model": model,
        "test_r2": test_r2,
        "test_rmse": test_rmse,
        "test_mae": test_mae,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_r2_scores": val_r2_scores
    }
    
    return results


def predict_on_new_data(model, new_data_path, train_data_path=DATA_PATH):
    """使用训练好的模型对新数据进行预测"""
    print("\n" + "="*50)
    print("对新数据进行预测")
    print("="*50)
    
    # 加载原始训练数据以获取缩放器和标签编码器
    train_df = pd.read_csv(train_data_path)
    train_data = prepare_material_data(train_df)
    
    # 加载新数据
    new_df = pd.read_csv(new_data_path)
    print(f"新数据大小: {len(new_df)} 样本")
    
    # 使用训练数据的缩放器和标签编码器处理新数据
    new_data = prepare_material_data(
        new_df,
        scaler=train_data["scaler"],
        le=train_data["label_encoder"],
    )
    
    # 准备预测
    device = next(model.parameters()).device
    new_materials = torch.LongTensor(new_data["material_ids"]).to(device)
    new_raw = torch.FloatTensor(new_data["numeric_raw"]).to(device)
    
    # 进行预测
    model.eval()
    with torch.no_grad():
        preds, uncertainties, _, _ = model(new_materials, new_raw)
        
        # 转换为NumPy数组
        preds_np = preds.cpu().squeeze().numpy()
        uncertainties_np = uncertainties.cpu().squeeze().numpy()
    
    # 创建结果DataFrame
    results_df = new_df.copy()
    results_df["预测裂纹深度"] = preds_np
    results_df["预测不确定性"] = uncertainties_np
    
    # 如果有真实的裂纹深度，计算指标
    if "亚表面裂纹深度" in results_df.columns:
        r2 = r2_score(results_df["亚表面裂纹深度"], results_df["预测裂纹深度"])
        rmse = np.sqrt(mean_squared_error(results_df["亚表面裂纹深度"], results_df["预测裂纹深度"]))
        mae = mean_absolute_error(results_df["亚表面裂纹深度"], results_df["预测裂纹深度"])
        
        print(f"预测 R²: {r2:.4f}")
        print(f"预测 RMSE: {rmse:.4f}")
        print(f"预测 MAE: {mae:.4f}")
    
    # 可视化预测结果
    if "亚表面裂纹深度" in results_df.columns:
        plt.figure(figsize=(10, 8))
        plt.scatter(results_df["亚表面裂纹深度"], results_df["预测裂纹深度"], alpha=0.5)
        plt.plot([results_df["亚表面裂纹深度"].min(), results_df["亚表面裂纹深度"].max()], 
                [results_df["亚表面裂纹深度"].min(), results_df["亚表面裂纹深度"].max()], 'r--')
        plt.xlabel('实际裂纹深度')
        plt.ylabel('预测裂纹深度')
        plt.title(f'实际值 vs 预测值 (R² = {r2:.4f}')
        new_pred_path = os.path.join(RESULTS_DIR, "new_data_predictions.png")
        plt.savefig(new_pred_path)
        plt.close()
        print(f"新数据预测图已保存至 {new_pred_path}")
    
    # 保存预测结果
    pred_results_path = os.path.join(RESULTS_DIR, "prediction_results.csv")
    results_df.to_csv(pred_results_path, index=False)
    print(f"预测结果已保存至 {pred_results_path}")
    
    return results_df


def load_model_and_predict(model_path, data_path=DATA_PATH, new_data_path=None):
    """加载已保存的模型并进行预测"""
    print("\n" + "="*50)
    print("加载模型并进行预测")
    print("="*50)
    
    # 加载数据以获取缩放器和标签编码器
    df = pd.read_csv(data_path)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df["材料种类"])
    train_data = prepare_material_data(train_df)
    
    # 创建模型结构
    model = MaterialCrackPGNN(
        num_materials=train_data["num_materials"],
        material_embed_dim=PARAMS["material_embed_dim"],
        latent_dim=PARAMS["latent_dim"],
        dropout_rate=PARAMS["dropout_rate"],
        symbolic_type=PARAMS.get("symbolic_type", "advanced")
    )
    
    # 加载模型参数
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    print(f"已从 {model_path} 加载模型")
    
    # 在测试集上评估
    test_data = prepare_material_data(
        test_df,
        scaler=train_data["scaler"],
        le=train_data["label_encoder"],
    )
    
    test_materials = torch.LongTensor(test_data["material_ids"]).to(device)
    test_raw = torch.FloatTensor(test_data["numeric_raw"]).to(device)
    test_target = torch.FloatTensor(test_data["target"]).unsqueeze(1).to(device)
    
    model.eval()
    with torch.no_grad():
        test_pred, test_uncertainty, _, _ = model(test_materials, test_raw)
        
        # 转换为NumPy数组
        test_pred_np = test_pred.cpu().squeeze().numpy()
        test_target_np = test_target.cpu().squeeze().numpy()
        
        # 计算指标
        test_r2 = r2_score(test_target_np, test_pred_np)
        test_rmse = np.sqrt(mean_squared_error(test_target_np, test_pred_np))
        test_mae = mean_absolute_error(test_target_np, test_pred_np)
        
        print(f"测试集 R²: {test_r2:.4f}")
        print(f"测试集 RMSE: {test_rmse:.4f}")
        print(f"测试集 MAE: {test_mae:.4f}")
        
        # 门控融合分析
        if hasattr(model, 'get_fusion_interpretation'):
            print(f"\n门控融合分析:")
            print(model.get_fusion_interpretation())
    
    # 如果提供了新数据路径，对新数据进行预测
    if new_data_path:
        predict_on_new_data(model, new_data_path, data_path)
    
    return model


if __name__ == "__main__":
    print("\n" + "="*50)
    print("材料裂纹深度预测 - 测试与评估")
    print("使用固定超参数训练PGNN模型")
    print("="*50)
    
    # 确保带时间戳的结果目录存在
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"结果将保存至: {RESULTS_DIR}\n")
    
    # 使用固定超参数训练和评估模型
    results = train_with_fixed_params(PARAMS)
    
    print("\n" + "="*50)
    print(f"模型训练与评估完成，测试集 R² = {results['test_r2']:.4f}")
    print("="*50)
    
