import sys
import os

# 将old_pgnn目录添加到模块搜索路径
current_dir = os.path.dirname(os.path.abspath(__file__))
old_pgnn_path = os.path.join(current_dir, 'old_pgnn')
sys.path.append(old_pgnn_path)

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import copy
import os
import json
import pickle
import traceback
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from torch.utils.data import TensorDataset, DataLoader
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import OneCycleLR

from old_pgnn.models import MaterialCrackPGNN, GRAD_CLIP
from old_pgnn.losses import AdaptiveLoss
from old_pgnn.utils import prepare_material_data, augment_dataset

# =============================================
# 手动指定超参数（可以根据需要修改）
# =============================================
HYPERPARAMS = {
    # 模型架构参数
    "material_embed_dim": 16,    # 材料嵌入维度
    "latent_dim": 32,            # 隐藏层维度
    "num_blocks": 4,             # 模型块数量
    "dropout_rate": 0.05,         # Dropout率
    "use_advanced_symbolic": True,  # 是否使用高级符号层
    
    # 训练参数
    "lr": 0.00131792,                 # 学习率
    "epochs": 300,               # 训练轮次
    "batch_size": 32,            # 批次大小
    "weight_decay": 0.00308058,      # 权重衰减
    "lambda_complexity": 0.000186227,  # 复杂度损失权重
    "lambda_physics": 0.48,      # 物理损失权重
    "patience": 60,              # 早停耐心值
    
    # 其他选项
    "augmentation": True,        # 是否使用数据增强
    "optimizer": "SGD",        # 优化器类型: AdamW, Adam, RMSprop, RAdam, SGD, NAdam
}

# 数据和训练配置
DATA_PATH = "merged_normal_data_v3_sorted.csv"  # 数据路径
SEED = 66                       # 随机种子
SAVE_DIR = "results"            # 结果保存目录


def train_with_fixed_params(params, data_path=DATA_PATH, seed=SEED):
    """
    使用指定的超参数训练模型
    
    参数:
        params: 指定的超参数字典
        data_path: 数据文件路径
        seed: 随机种子
    
    返回:
        model: 训练好的模型
        train_losses: 训练损失历史
        test_r2_history: 测试R²历史
    """
    # 设置随机种子
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 读取数据
    df = pd.read_csv(data_path)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=seed)

    train_data = prepare_material_data(train_df, verbose=True)
    test_data = prepare_material_data(
        test_df,
        scaler=train_data["scaler"],
        le=train_data["label_encoder"],
        verbose=False,
    )

    # 创建模型
    model = MaterialCrackPGNN(
        num_materials=train_data["num_materials"],
        material_embed_dim=params.get("material_embed_dim", 16),
        latent_dim=params.get("latent_dim", 64),
        dropout_rate=params.get("dropout_rate", 0.1),
        num_blocks=params.get("num_blocks", 3),
        use_advanced_symbolic=params.get("use_advanced_symbolic", True),
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"使用设备: {device}")
    
    # 准备数据
    use_augmentation = params.get("augmentation", True)
    train_df_aug = train_df.copy()
    
    if use_augmentation:
        print("应用数据增强...")
        train_df_aug = augment_dataset(train_df_aug)
        
    augmented_train_data = prepare_material_data(
        train_df_aug,
        scaler=train_data["scaler"],
        le=train_data["label_encoder"],
        verbose=False,
    )

    # DataLoader
    batch_size = params.get("batch_size", 32)
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
    )

    test_materials = torch.LongTensor(test_data["material_ids"]).to(device)
    test_features = torch.FloatTensor(test_data["numeric_features"]).to(device)
    test_raw = torch.FloatTensor(test_data["numeric_raw"]).to(device)
    test_target = torch.FloatTensor(test_data["target"]).unsqueeze(1).to(device)

    # 优化器
    lr = params.get("lr", 1e-3)
    optimizer_name = params.get("optimizer", "AdamW")
    weight_decay = params.get("weight_decay", 1e-4)
    
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
        print(f"未知的优化器 {optimizer_name}，使用默认的AdamW")
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 学习率调度 - 使用 OneCycleLR
    epochs = params.get("epochs", 100)
    total_steps = epochs * len(train_loader)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=lr,
        total_steps=total_steps,
        pct_start=0.3,           # 30% 的步数用于预热
        div_factor=25.0,         # 初始学习率 = max_lr/25
        final_div_factor=1e4,    # 最终学习率 = max_lr/(25*10000)
        anneal_strategy='cos'
    )

    # 损失函数
    loss_fn = AdaptiveLoss(
        lambda_complexity=params.get("lambda_complexity", 1e-5),
        lambda_physics=params.get("lambda_physics", 0.05),
    )
    loss_fn.total_epochs = epochs
    scaler = GradScaler()

    # 训练
    print("\n开始训练...")
    train_losses = []
    test_r2_history = []
    best_model_wts, best_test_r2, patience_ctr = None, -1e9, 0
    best_epoch = 0

    for epoch in range(epochs):
        loss_fn.update_epoch(epoch, epochs)

        # 训练
        model.train()
        epoch_loss = 0

        for batch_materials, batch_features, batch_raw, batch_target in train_loader:
            batch_materials = batch_materials.to(device, non_blocking=True)
            batch_features = batch_features.to(device, non_blocking=True)
            batch_raw = batch_raw.to(device, non_blocking=True)
            batch_target = batch_target.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            try:
                with autocast(device_type=device):
                    pred, uncertainty, _, _ = model(
                        batch_materials, batch_features, batch_raw
                    )
                    loss, metrics = loss_fn(pred, uncertainty, batch_target, model, batch_raw)

                # 检查NaN
                if torch.isnan(loss) or torch.isinf(loss) or loss > 1000:
                    print(f"NaN/Inf loss detected in epoch {epoch}")
                    continue

                # 反向传播
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
                
                # 更新学习率
                scheduler.step()

                # 检查loss是否为NaN
                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    print(f"Warning: NaN/Inf loss detected, skipping batch")
                    continue
                
                epoch_loss += loss.item() * batch_target.size(0)
                
            except Exception as e:
                print(f"Error in batch training: {e}")
                import traceback
                traceback.print_exc()
                continue

        # 避免分母为0的情况
        num_processed = max(1, len(train_dataset))
        epoch_loss /= num_processed
        train_losses.append(epoch_loss)

        # 评估
        model.eval()
        with torch.no_grad(), autocast(device_type=device):
            test_pred, test_uncertainty, _, _ = model(
                test_materials, test_features, test_raw
            )
            # 检查并修复NaN/Inf值
            if torch.isnan(test_pred).any() or torch.isinf(test_pred).any():
                print(f"Warning: Detected NaN/Inf in predictions, fixing...")
                test_pred = torch.nan_to_num(test_pred, nan=0.0, posinf=1.0, neginf=-1.0)
            
            test_pred_np = test_pred.cpu().squeeze().numpy()
            test_target_np = test_target.cpu().squeeze().numpy()

            # 额外检查numpy数组中的NaN
            if np.isnan(test_pred_np).any() or np.isinf(test_pred_np).any():
                print(f"Warning: Detected NaN/Inf in numpy predictions, replacing with median")
                valid_preds = test_pred_np[~np.isnan(test_pred_np) & ~np.isinf(test_pred_np)]
                if len(valid_preds) > 0:
                    median_val = np.median(valid_preds)
                else:
                    median_val = 0.0
                test_pred_np = np.nan_to_num(test_pred_np, nan=median_val, posinf=1.0, neginf=-1.0)

            test_r2 = r2_score(test_target_np, test_pred_np)
            test_rmse = np.sqrt(mean_squared_error(test_target_np, test_pred_np))
            test_r2_history.append(test_r2)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch+1}: Loss={epoch_loss:.4f}, LR={current_lr:.6f}, "
                  f"Test R²={test_r2:.4f}, RMSE={test_rmse:.4f}")

        # 保存最佳模型
        if test_r2 > best_test_r2:
            best_test_r2 = test_r2
            best_model_wts = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            patience_ctr = 0
        else:
            patience_ctr += 1

        # 早停
        patience = params.get("patience", 20)
        if patience_ctr >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # 恢复最佳权重
    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)

    # 最终评估
    model.eval()
    with torch.no_grad(), autocast(device_type=device):
        test_pred, test_uncertainty, _, _ = model(
            test_materials, test_features, test_raw
        )
        # 检查并修复NaN/Inf值
        if torch.isnan(test_pred).any() or torch.isinf(test_pred).any():
            print(f"Warning: Detected NaN/Inf in final predictions, fixing...")
            test_pred = torch.nan_to_num(test_pred, nan=0.0, posinf=1.0, neginf=-1.0)
            
        test_pred_np = test_pred.cpu().squeeze().numpy()
        test_target_np = test_target.cpu().squeeze().numpy()
        
        # 额外检查numpy数组中的NaN
        if np.isnan(test_pred_np).any() or np.isinf(test_pred_np).any():
            print(f"Warning: Detected NaN/Inf in final numpy predictions, replacing with median")
            valid_preds = test_pred_np[~np.isnan(test_pred_np) & ~np.isinf(test_pred_np)]
            if len(valid_preds) > 0:
                median_val = np.median(valid_preds)
            else:
                median_val = 0.0
            test_pred_np = np.nan_to_num(test_pred_np, nan=median_val, posinf=1.0, neginf=-1.0)

        final_r2 = r2_score(test_target_np, test_pred_np)
        final_rmse = np.sqrt(mean_squared_error(test_target_np, test_pred_np))

    # MC Dropout预测
    mc_pred, mc_unc, aleatoric_unc, epistemic_unc = model.predict_with_uncertainty(
        test_materials, test_features, test_raw, n_samples=20
    )
    mc_pred_np = mc_pred.cpu().squeeze().numpy()
    mc_r2 = r2_score(test_target_np, mc_pred_np)
    mc_rmse = np.sqrt(mean_squared_error(test_target_np, mc_pred_np))

    print(f"\n最终测试集性能:")
    print(f"  标准预测 R²: {final_r2:.4f}, RMSE: {final_rmse:.4f}")
    print(f"  MC预测 R²: {mc_r2:.4f}, RMSE: {mc_rmse:.4f}")
    print(f"  最佳epoch: {best_epoch}")

    return model, train_losses, test_r2_history


def create_training_plots(train_losses, test_r2_history):
    """
    创建训练过程的可视化图表
    """
    try:
        matplotlib.use('Agg')  # 使用非交互式后端
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 1. 训练损失
        axes[0].plot(train_losses, 'g-', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Training Loss')
        axes[0].set_title('Training Loss History')
        axes[0].grid(True, alpha=0.3)
        
        # 2. 测试R²历史
        axes[1].plot(test_r2_history, 'r-', linewidth=2)
        axes[1].axhline(y=max(test_r2_history), color='b', linestyle='--',
                       label=f'Best: {max(test_r2_history):.4f}')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Test R²')
        axes[1].set_title('Test R² History')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{SAVE_DIR}/training_plots.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"训练图表已保存到: {SAVE_DIR}/training_plots.png")
        
    except ImportError:
        print("警告: matplotlib未安装，跳过图表生成")
    except Exception as e:
        print(f"生成图表时出错: {e}")


def main():
    """
    主函数：使用指定超参数执行训练和测试
    """
    print("=" * 60)
    print("材料裂纹预测PGNN模型 - 使用指定超参数训练")
    print("=" * 60)
    
    # 检查数据文件是否存在
    if not os.path.exists(DATA_PATH):
        print(f"错误: 数据文件 '{DATA_PATH}' 不存在!")
        return
    
    # 检查GPU可用性
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    if device == "cuda":
        print(f"GPU型号: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print()
    
    # 打印使用的超参数
    print("使用的超参数:")
    for key, value in HYPERPARAMS.items():
        print(f"  {key}: {value}")
    print()
    
    try:
        # 创建结果目录
        os.makedirs(SAVE_DIR, exist_ok=True)
        
        # 训练模型
        print("开始训练模型...")
        model, train_losses, test_r2_history = train_with_fixed_params(
            HYPERPARAMS, 
            data_path=DATA_PATH, 
            seed=SEED
        )
        
        if model is None:
            print("训练失败!")
            return
            
        # 保存结果
        print("\n保存模型和结果...")
        
        # 保存最佳模型
        torch.save(model.state_dict(), f"{SAVE_DIR}/best_model.pth")
        
        # 保存超参数
        with open(f"{SAVE_DIR}/hyperparams.json", "w", encoding="utf-8") as f:
            json.dump(HYPERPARAMS, f, indent=2, ensure_ascii=False)
        
        # 保存训练历史
        with open(f"{SAVE_DIR}/training_history.pkl", "wb") as f:
            pickle.dump({
                "train_losses": train_losses,
                "test_r2_history": test_r2_history,
                "hyperparams": HYPERPARAMS
            }, f)
        
        # 生成可视化结果
        create_training_plots(train_losses, test_r2_history)
        
        # 输出最终结果总结
        print("\n" + "=" * 60)
        print("训练完成! 结果总结:")
        print("=" * 60)
        print(f"最终测试集R²: {max(test_r2_history):.4f}")
        print(f"训练轮次: {len(train_losses)}")
        
        print(f"\n文件已保存到:")
        print(f"  模型权重: {SAVE_DIR}/best_model.pth")
        print(f"  超参数: {SAVE_DIR}/hyperparams.json")
        print(f"  训练历史: {SAVE_DIR}/training_history.pkl")
        print(f"  训练图表: {SAVE_DIR}/training_plots.png")
        
    except KeyboardInterrupt:
        print("\n用户中断训练")
    except Exception as e:
        print(f"\n训练过程中出现错误: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()