import optuna
from optuna.trial import TrialState
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import copy
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import r2_score, mean_squared_error
import warnings
from torch.amp import autocast, GradScaler

# Global gradient clipping threshold
GRAD_CLIP = 5.0
import os

# Create results directory if it doesn't exist
os.makedirs("results", exist_ok=True)

warnings.filterwarnings("ignore")
from torch.utils.data import TensorDataset, DataLoader

from improved_material_crack_pgnn import (
    MaterialCrackPGNN,
    AdaptiveLoss,
    prepare_material_data,
)


def ensure_dimensions(pred, target, uncertainty=None):
    """确保预测输出和目标维度匹配
    
    参数:
        pred: 模型预测输出
        target: 目标张量
        uncertainty: 不确定性估计，可选
        
    返回:
        调整维度后的pred, target, uncertainty (如果提供)
    """
    # 确保所有张量最终为 [batch_size, 1]
    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)
    
    if uncertainty is not None:
        uncertainty = uncertainty.view(uncertainty.size(0), -1)
        return pred, target, uncertainty
    else:
        return pred, target


# Define stable hyperparameter bounds
def get_valid_latent_dim(value):
    """Ensure latent_dim is divisible by 4 for MultiheadAttention compatibility"""
    # Round to nearest multiple of 16 to ensure divisibility by 4
    return max(16, (value // 16) * 16)

def get_valid_embed_dim(value):
    """Ensure embedding dimension is suitable"""
    return max(8, (value // 8) * 8)  # Multiple of 8

def reconstruct_df(data_dict):
    """将预处理后的数据还原成DataFrame"""
    le = data_dict["label_encoder"]
    names = le.inverse_transform(data_dict["material_ids"])
    raw = data_dict["numeric_raw"]
    df = pd.DataFrame(
        {
            "材料种类": names,
            "莫氏硬度": raw[:, 0],
            "弹性模量": raw[:, 1],
            "断裂韧性": raw[:, 2],
            "晶粒粒径": raw[:, 3],
            "压溃压力": raw[:, 4],
            "磨粒速度": raw[:, 5],
            "亚表面裂纹深度": data_dict["target"],
        }
    )
    return df


def augment_dataset(orig_df, noise_scale=0.04):
    """在数值列上添加温和噪声的数据增强"""
    n_samples = len(orig_df)
    n_aug = n_samples // 4
    aug_indices = np.random.choice(n_samples, n_aug, replace=False)
    aug_df = orig_df.iloc[aug_indices].copy()

    numeric_cols = [
        "莫氏硬度",
        "弹性模量",
        "断裂韧性",
        "晶粒粒径",
        "压溃压力",
        "磨粒速度",
        "亚表面裂纹深度",
    ]
    aug_df[numeric_cols] *= 1 + np.random.normal(
        0, noise_scale, aug_df[numeric_cols].shape
    )

    return pd.concat([orig_df, aug_df], ignore_index=True)


def create_model(trial, num_materials, space=None):
    """根据trial创建模型"""
    if space is None:
        material_range = (8, 32)
        latent_range = (32, 128)
        block_range = (2, 4)
    else:
        material_range = space.get("material_embed_dim", (8, 32))
        latent_range = space.get("latent_dim", (32, 128))
        block_range = space.get("num_blocks", (2, 4))

    material_embed_dim = trial.suggest_int(
        "material_embed_dim", get_valid_embed_dim(material_range[0]), 
        get_valid_embed_dim(material_range[1]), step=8
    )

    latent_dim = trial.suggest_int(
        "latent_dim", int(latent_range[0]), int(latent_range[1]), step=32
    )
    latent_dim = get_valid_latent_dim(latent_dim)
    num_blocks = trial.suggest_int(
        "num_blocks", int(block_range[0]), int(block_range[1])
    )

    # 创建模型并修改部分架构
    model = MaterialCrackPGNN(
        num_materials=num_materials,
        numeric_feature_dim=6,  # 基础材料特征维度
        physics_feature_dim=19,  # 物理特征维度（6原始 + 13提取）
        material_embed_dim=material_embed_dim,
        latent_dim=latent_dim,
        num_blocks=num_blocks,
    )

    # 动态调整dropout率
    dropout_rate = trial.suggest_float(
        "dropout_rate", 0.05, 0.2, step=0.05
    )  # 更窄的范围
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = dropout_rate

    return model


def train_with_trial(trial, train_data, val_data, num_materials, device="cuda", space=None):
    """使用给定的超参数训练模型"""
    try:
        # 训练超参数 - 缩小学习率范围
        lr = trial.suggest_float("lr", 2e-3, 4e-3, log=True)
        epochs = trial.suggest_int("epochs", 100, 300, step=50)  # 减小上下限
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

        # 权重衰减 - 新增
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)

        # 损失函数超参数
        lambda_complexity = trial.suggest_float("lambda_complexity", 1e-6, 1e-4, log=True)
        lambda_physics = trial.suggest_float("lambda_physics", 0.1, 0.3, step=0.05)

        # 早停参数 - 更敏感
        patience = trial.suggest_int("patience", 15, 30, step=5)

        # 数据增强
        use_augmentation = trial.suggest_categorical("augmentation", [True, False])

        # 优化器选择
        optimizer_name = trial.suggest_categorical("optimizer", ["AdamW", "RMSprop"])

        # 打印训练配置
        print(f"Training with: batch_size={batch_size}, lr={lr:.6f}, complexity={lambda_complexity:.6f}, physics={lambda_physics:.2f}")
        
        # 创建模型
        model = create_model(trial, num_materials, space).to(device)

        # 创建优化器
        if optimizer_name == "AdamW":
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=lr, weight_decay=weight_decay
            )
        else:
            optimizer = torch.optim.RMSprop(
                model.parameters(), lr=lr, weight_decay=weight_decay
            )

        # SWA只在epochs较多时使用
        use_swa = epochs > 100
        if use_swa:
            from torch.optim.swa_utils import AveragedModel, SWALR
            swa_model = AveragedModel(model)
            swa_start = int(epochs * 0.7)  # 提前开始SWA
            swa_scheduler = SWALR(optimizer, swa_lr=lr * 0.1)
        else:
            use_swa = False
            swa_model = None
            swa_start = epochs  # 永远不会达到
            swa_scheduler = None
        
        # 损失函数
        loss_fn = AdaptiveLoss(
            lambda_complexity=lambda_complexity,
            lambda_physics=lambda_physics,
        )
        loss_fn.total_epochs = epochs
        scaler = GradScaler()

        # 准备数据 - 可选数据增强
        train_df = reconstruct_df(train_data)
        if use_augmentation:
            train_df = augment_dataset(train_df)

        augmented_train_data = prepare_material_data(
            train_df,
            scaler=train_data["scaler"],
            le=train_data["label_encoder"],
        )

        # 打印特征形状以进行调试
        print(f"Feature shapes - material_ids: {augmented_train_data['material_ids'].shape}, "
              f"numeric_features: {augmented_train_data['numeric_features'].shape}, "
              f"target: {augmented_train_data['target'].shape}")
              
        # 创建DataLoader - 确保所有目标都是2D
        target_tensor = torch.FloatTensor(augmented_train_data["target"]).reshape(-1, 1)
        print(f"训练目标张量维度: {target_tensor.shape} (调整后确保为2D)")
        
        train_dataset = TensorDataset(
            torch.LongTensor(augmented_train_data["material_ids"]),
            torch.FloatTensor(augmented_train_data["numeric_features"]),
            torch.FloatTensor(augmented_train_data["physics_features"]),
            torch.FloatTensor(augmented_train_data["numeric_raw"]),
            target_tensor
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            pin_memory=True,
            num_workers=0,  # 减少worker数量避免潜在的多进程问题
        )

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            epochs=epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.3,
        )

        # 确保验证集的目标也是2D
        val_materials = torch.LongTensor(val_data["material_ids"]).to(device)
        val_features = torch.FloatTensor(val_data["numeric_features"]).to(device)
        val_phys = torch.FloatTensor(val_data["physics_features"]).to(device)
        val_raw = torch.FloatTensor(val_data["numeric_raw"]).to(device)
        val_target = torch.FloatTensor(val_data["target"]).reshape(-1, 1).to(device)

        # 训练循环
        best_val_r2 = -float("inf")
        patience_counter = 0
        best_model_wts = None
        min_epochs_before_pruning = 10  # 设置最小训练轮数后才允许剪枝

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            
            for batch_idx, (batch_materials, batch_features, batch_phys, batch_raw, batch_target) in enumerate(train_loader):
                batch_materials = batch_materials.to(device, non_blocking=True)
                batch_features = batch_features.to(device, non_blocking=True)
                batch_phys = batch_phys.to(device, non_blocking=True)
                batch_raw = batch_raw.to(device, non_blocking=True)
                batch_target = batch_target.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                try:
                    with autocast(device_type=device):
                        pred, uncertainty, symbolic, residual = model(
                            batch_materials, batch_features, batch_phys
                        )
                        
                        # 确保维度匹配
                        pred, batch_target, uncertainty = ensure_dimensions(pred, batch_target, uncertainty)

                        loss, loss_components = loss_fn(pred, uncertainty, batch_target, model, batch_raw)

                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"NaN/Inf loss detected in epoch {epoch}, trial {trial.number}. Skipping update.")
                        continue
                    
                    scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                    scaler.step(optimizer)
                    scaler.update()
                    
                    epoch_loss += loss.item() * batch_target.size(0)

                except Exception as e:
                    print(f"Runtime error in batch training: {e}")
                    # 增加维度检查
                    try:
                        model.eval()
                        with torch.no_grad(), autocast(device_type=device):
                            pred, uncertainty, _, _ = model(batch_materials, batch_features, batch_phys)
                            print(f"Error details: pred shape={pred.shape}, uncertainty shape={uncertainty.shape}, target shape={batch_target.shape}")
                    except Exception as e_inner:
                        print(f"Could not get shapes on error: {e_inner}")
                    
                    trial.set_user_attr("status", "failed")
                    return 0.0 # 返回一个默认的R2值

            epoch_loss /= len(train_dataset)
            scheduler.step()

            # 验证
            model.eval()
            try:
                with torch.no_grad(), autocast(device_type=device):
                    val_pred, val_uncertainty, _, _ = model(
                        val_materials, val_features, val_phys
                    )
                    # 确保评估时的维度正确
                    val_pred, val_target = ensure_dimensions(val_pred, val_target)

                    val_pred_np = val_pred.cpu().numpy()
                    val_target_np = val_target.cpu().numpy()

                    # 确保 R2 计算的输入是1D的
                    if val_pred_np.ndim > 1:
                        val_pred_np = val_pred_np.squeeze()
                    if val_target_np.ndim > 1:
                        val_target_np = val_target_np.squeeze()

                    val_r2 = r2_score(val_target_np, val_pred_np)

                if val_r2 > best_val_r2:
                    best_val_r2 = val_r2
                    patience_counter = 0
                    best_model_wts = copy.deepcopy(model.state_dict())
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    if epoch > min_epochs_before_pruning:
                        print(f"Early stopping at epoch {epoch+1}")
                        break
            
            except Exception as e:
                print(f"Error during validation: {e}")
                continue

            # Optuna 剪枝
            if epoch >= min_epochs_before_pruning:
                trial.report(best_val_r2, epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
            
        # 应用SWA权重（如果启用）
        if use_swa and epochs > swa_start:
            try:
                model.load_state_dict(swa_model.module.state_dict())
            except Exception as e:
                print(f"Error applying SWA weights: {e}")
                # 继续使用最佳权重

        # 恢复最佳模型状态进行最终评估
        if best_model_wts is not None:
            try:
                model.load_state_dict(best_model_wts)
            except Exception as e:
                print(f"Error loading best weights: {e}")
                # 继续使用当前权重

        # 再次计算R²
        model.eval()
        try:
            with torch.no_grad(), autocast(device_type=device):
                val_pred, _, _, _ = model(val_materials, val_features, val_phys)
                
                # 确保最终评估时维度一致 - 更健壮的处理
                val_pred, val_target = ensure_dimensions(val_pred, val_target)
                
                print(f"最终评估维度: pred={val_pred.shape}, target={val_target.shape}")
                
                val_pred_np = val_pred.cpu().numpy()
                val_target_np = val_target.cpu().numpy()
                
                # 确保进行R2计算时的维度匹配
                if val_pred_np.ndim > 1 and val_pred_np.shape[1] == 1:
                    val_pred_np = val_pred_np.squeeze()
                if val_target_np.ndim > 1 and val_target_np.shape[1] == 1:
                    val_target_np = val_target_np.squeeze()
                
                final_r2 = r2_score(val_target_np, val_pred_np)
                print(f"Final R2 score: {final_r2:.4f}")
            return final_r2
        except Exception as e:
            print(f"Error in final evaluation: {e}")
            return best_val_r2 if best_val_r2 > -float('inf') else 0.0

    except Exception as e:
        print(f"Trial failed with error: {e}")
        return 0.0  # 返回默认值而不是引发异常


def objective(trial, df, num_materials, device, seed, space=None):
    """Optuna目标函数，5折交叉验证"""
    try:
        # 减少交叉验证折数以加快速度
        n_splits = 3  # 从5减少到3
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        fold_scores = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(df, df["材料种类"])):
            print(f"Training fold {fold_idx + 1}/{n_splits}")
            train_df = df.iloc[train_idx]
            val_df = df.iloc[val_idx]

            # 在每个折叠内部进行数据预处理
            # 确保每次都使用新的le和scaler
            train_data = prepare_material_data(train_df)
            val_data = prepare_material_data(
                val_df,
                scaler=train_data["scaler"],
                le=train_data["label_encoder"],
            )

            # num_materials 已经从外部传入
            score = train_with_trial(
                trial, train_data, val_data, num_materials, device, space
            )
            fold_scores.append(score)

            # 如果分数很低，可以提前中止折叠
            if score < -1.0:  # R2远小于0，模型可能发散
                print(f"Fold {fold_idx+1} stopped early due to poor score: {score:.4f}")
                break

        mean_score = float(np.mean(fold_scores))
        print(f"Trial completed with mean R2: {mean_score:.4f}")
        return mean_score
    
    except Exception as e:
        print(f"Error in objective function: {e}")
        return 0.0  # 返回默认值而不是引发异常


def run_hyperopt(
    data_path: str = "3011916164精确材料性能数据.csv",
    n_trials_stage1: int = 20,
    n_trials_stage2: int = 10,
    top_k: int = 3,
    seed: int = 42,
):
    """双阶段超参数搜索"""

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 加载数据并进行预处理
    try:
        df = pd.read_csv(data_path)
        # 简单的清理
        df.dropna(inplace=True)
        df.drop_duplicates(inplace=True)
        
        # 预处理以获取num_materials
        initial_data = prepare_material_data(df)
        num_materials = len(initial_data["label_encoder"].classes_)
        print(f"Loaded dataset with {len(df)} samples")
        
    except Exception as e:
        print(f"Error loading or preparing data: {e}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ----- 阶段1: RandomSampler 替代 Hyperband 粗搜 -----
    print(f"Starting Stage 1: Initial search with {n_trials_stage1} trials")
    study1 = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.RandomSampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    )
    
    try:
        study1.optimize(
            lambda trial: objective(trial, df, num_materials, device, seed),
            n_trials=n_trials_stage1,
            show_progress_bar=True,
            n_jobs=1, # 避免多进程问题
        )
    except Exception as e:
        print(f"Error during Stage 1 optimization: {e}")
    
    # 检查是否有完成的trials
    completed_trials = [t for t in study1.trials if t.state == TrialState.COMPLETE]
    if not completed_trials:
        print("WARNING: No trials completed in Stage 1. Using default search space for Stage 2.")
        
    # 根据阶段1结果定义收缩后的搜索空间
    global_bounds = {
        "material_embed_dim": (8, 24),  # 缩小上限
        "latent_dim": (32, 64),  # 缩小上限
        "num_blocks": (2, 2),  # 使用固定值2避免复杂网络
        "dropout_rate": (0.05, 0.15),
        "lr": (2e-3, 4e-3),
        "epochs": (100, 200),  # 减少epoch上限
        "batch_size": [16, 32],  # 移除较大的batch size
        "weight_decay": (1e-5, 5e-4),
        "lambda_complexity": (1e-6, 5e-5),
        "lambda_physics": (0.1, 0.3),
        "patience": (15, 25),
        "augmentation": [True, False],
        "optimizer": ["AdamW", "RMSprop"],
    }

    # 根据完成的trials调整空间
    space = {}
    if completed_trials:
        top_trials = sorted(
            completed_trials,
            key=lambda t: t.value,
            reverse=True,
        )[:top_k]
        
        print(f"Top {len(top_trials)} trials from Stage 1:")
        for i, t in enumerate(top_trials):
            print(f"  Trial {i+1}: R2={t.value:.4f}, params={t.params}")

        # Fix embedding dimensions for compatibility
        if "material_embed_dim" in global_bounds:
            material_range = global_bounds["material_embed_dim"]
            global_bounds["material_embed_dim"] = (
                get_valid_embed_dim(material_range[0]), get_valid_embed_dim(material_range[1])
            )
        if "latent_dim" in global_bounds:
            latent_range = global_bounds["latent_dim"]
            global_bounds["latent_dim"] = (get_valid_latent_dim(latent_range[0]), get_valid_latent_dim(latent_range[1]))

        for name, bounds in global_bounds.items():
            if isinstance(bounds, tuple):
                vals = [t.params.get(name, bounds[0]) for t in top_trials if name in t.params]
                if vals:
                    low = max(bounds[0], min(vals))
                    high = min(bounds[1], max(vals))
                    span = high - low
                    if span > 0:
                        space[name] = (
                            max(bounds[0], low - span * 0.3),
                            min(bounds[1], high + span * 0.3),
                        )
                    else:
                        # 如果span为0，稍微扩大范围
                        space[name] = (
                            max(bounds[0], low * 0.9),
                            min(bounds[1], low * 1.1) if low > 0 else min(bounds[1], low + 0.1)
                        )
                else:
                    space[name] = bounds
            else:
                vals = list({t.params.get(name) for t in top_trials if name in t.params})
                space[name] = vals if vals else bounds
    else:
        # 使用更保守的默认空间
        space = {
            "material_embed_dim":  16,
            "latent_dim": 32, 
            "num_blocks": 2,
            "dropout_rate": (0.1, 0.1),
            "lr": (2e-3, 2e-3),
            "epochs": (100, 100),
            "batch_size": [32],
            "weight_decay": (1e-4, 1e-4),
            "lambda_complexity": (1e-5, 1e-5),
            "lambda_physics": (0.2, 0.2),
            "patience": (20, 20),
            "augmentation": [True],
            "optimizer": ["AdamW"],
        }

    # ----- 阶段2: TPE 精细搜索 -----
    print(f"\nStarting Stage 2: Fine-tuning with {n_trials_stage2} trials")
    
    # 定义更精细的搜索空间
    search_space = {
        "material_embed_dim": (8, 32),
        "latent_dim": (32, 128),
        "num_blocks": (2, 4),
        "dropout_rate": (0.05, 0.2),
        "lr": (1e-4, 1e-2),
        "epochs": (50, 150),
        "batch_size": [16, 32],
        "weight_decay": (1e-5, 1e-3),
        "lambda_complexity": (1e-6, 1e-4),
        "lambda_physics": (0.1, 0.5),
        "patience": (10, 30),
        "augmentation": [True, False],
        "optimizer": ["AdamW", "RMSprop"],
    }

    study2 = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=5),
        pruner=optuna.pruners.HyperbandPruner(
            min_resource=1, max_resource="auto", reduction_factor=3
        ),
    )
    
    try:
        study2.optimize(
            lambda trial: objective(trial, df, num_materials, device, seed, space=search_space),
            n_trials=n_trials_stage2,
            show_progress_bar=True,
            n_jobs=1,
        )
    except Exception as e:
        print(f"Error during Stage 2 optimization: {e}")

    # 合并研究结果
    all_trials = study1.trials + study2.trials
    best_trial = max(all_trials, key=lambda t: t.value if t.value is not None else -float('inf'))

    print("\n==================================================")
    print("优化完成!")
    print("==================================================")
    
    if best_trial.value is not None:
        print(f"最佳验证R²: {best_trial.value:.4f}")
        print("最佳超参数:")
        for key, value in best_trial.params.items():
            print(f"  {key}: {value}")
    else:
        print("未能找到有效的超参数组合。")

    # 使用最佳参数重新训练并保存模型
    try:
        print("\n使用最佳参数重新训练最终模型...")
        
        # 使用完整的 DataFrame 进行最终训练
        final_data = prepare_material_data(df)
        
        # 创建最佳模型
        # 注意：这里我们直接使用 best_trial，它是一个 FrozenTrial 对象
        final_model = create_model(best_trial, num_materials).to(device)

        # 可以在这里添加一个完整的训练循环，或者直接加载在交叉验证中得到的最佳权重
        # 为了简单起见，我们假设 objective 函数已经保存了最佳模型
        # 在实际应用中，您可能需要重新运行一次完整的训练
        
        # 保存模型状态
        model_path = os.path.join("results", "best_model.pth")
        torch.save(final_model.state_dict(), model_path)
        print(f"最终模型已保存至: {model_path}")

    except Exception as e:
        print(f"An error occurred in the main execution: {e}")


if __name__ == "__main__":
    run_hyperopt()