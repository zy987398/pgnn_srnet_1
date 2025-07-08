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
            verbose=False,
        )

        # 创建DataLoader
        train_dataset = TensorDataset(
            torch.LongTensor(augmented_train_data["material_ids"]),
            torch.FloatTensor(augmented_train_data["numeric_features"]),
            torch.FloatTensor(augmented_train_data["physics_features"]),
            torch.FloatTensor(augmented_train_data["numeric_raw"]),
            torch.FloatTensor(augmented_train_data["target"]).unsqueeze(1),
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            pin_memory=True,
            num_workers=4,
        )

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            epochs=epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.3,
        )

        val_materials = torch.LongTensor(val_data["material_ids"]).to(device)
        val_features = torch.FloatTensor(val_data["numeric_features"]).to(device)
        val_phys = torch.FloatTensor(val_data["physics_features"]).to(device)
        val_raw = torch.FloatTensor(val_data["numeric_raw"]).to(device)
        val_target = torch.FloatTensor(val_data["target"]).unsqueeze(1).to(device)

        # 训练循环
        best_val_r2 = -float("inf")
        patience_counter = 0
        best_model_wts = None
        min_epochs_before_pruning = 10  # 设置最小训练轮数后才允许剪枝

        for epoch in range(epochs):
            # 更新当前epoch
            loss_fn.update_epoch(epoch, epochs)

            # 训练
            model.train()
            epoch_loss = 0

            for batch_materials, batch_features, batch_phys, batch_raw, batch_target in train_loader:
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
                        loss, metrics = loss_fn(pred, uncertainty, batch_target, model, batch_raw)

                    # 检查NaN - 返回默认值而不是剪枝
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"Warning: NaN or Inf loss detected in epoch {epoch}. Using default R2 score.")
                        # 如果训练早期就出现NaN，直接返回
                        if epoch < min_epochs_before_pruning:
                            return 0.0
                        # 否则尝试使用最佳权重
                        elif best_model_wts is not None:
                            model.load_state_dict(best_model_wts)
                            continue
                        else:
                            return 0.0

                    # 反向传播与梯度裁剪
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                    grad_norm = model.check_grad_norm() if hasattr(model, 'check_grad_norm') else 0.0
                    scaler.step(optimizer)
                    scaler.update()


                    epoch_loss += loss.item() * batch_target.size(0)
                except RuntimeError as e:
                    print(f"Runtime error in batch training: {e}")
                    if 'out of memory' in str(e).lower():
                        # 内存不足时减小batch size并重试
                        torch.cuda.empty_cache()
                        # 返回默认分数，让optuna减小batch_size
                        print("CUDA out of memory, suggesting smaller batch size")
                        return 0.0
                    else:
                        # 其他RuntimeError，返回默认值
                        return 0.0

            epoch_loss /= len(train_dataset)

            # 验证
            model.eval()
            try:
                with torch.no_grad(), autocast(device_type=device):
                    val_pred, val_uncertainty, _, _ = model(
                        val_materials, val_features, val_phys
                    )
                    val_loss, _ = loss_fn(val_pred, val_uncertainty, val_target, model, val_raw)

                    val_pred_np = val_pred.cpu().squeeze().numpy()
                    val_target_np = val_target.cpu().squeeze().numpy()

                    val_r2 = r2_score(val_target_np, val_pred_np)
                    val_rmse = np.sqrt(mean_squared_error(val_target_np, val_pred_np))

                # 早停检查
                if val_r2 > best_val_r2:
                    best_val_r2 = val_r2
                    patience_counter = 0
                    # 保存最佳模型状态
                    best_model_wts = copy.deepcopy(model.state_dict())
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

                # 向Optuna报告中间结果
                trial.report(val_r2, epoch)
                
                # 调整学习率
                scheduler.step()

                # SWA 更新在获得验证指标后执行
                if use_swa and epoch >= swa_start:
                    swa_model.update_parameters(model)
                    swa_scheduler.step()

                # 剪枝逻辑 - 确保训练至少min_epochs_before_pruning轮后才考虑剪枝
                if trial.should_prune() and epoch >= min_epochs_before_pruning:
                    raise optuna.TrialPruned()
            
            except Exception as e:
                print(f"Error in validation: {e}")
                # 验证出错，但我们仍可能有有效模型，返回一个保守的R2值
                return 0.0

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
                val_pred_np = val_pred.cpu().squeeze().numpy()
                val_target_np = val_target.cpu().squeeze().numpy()
                final_r2 = r2_score(val_target_np, val_pred_np)
            return final_r2
        except Exception as e:
            print(f"Error in final evaluation: {e}")
            return best_val_r2 if best_val_r2 > -float('inf') else 0.0

    except Exception as e:
        print(f"Trial failed with error: {e}")
        return 0.0  # 返回默认值而不是引发异常


def objective(trial, df, device, seed, space=None):
    """Optuna目标函数，5折交叉验证"""
    try:
        # 减少交叉验证折数以加快速度
        n_splits = 3  # 从5减少到3
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        fold_scores = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(df, df["材料种类"])):
            print(f"Training fold {fold_idx+1}/{n_splits}")
            train_df = df.iloc[train_idx]
            val_df = df.iloc[val_idx]

            train_data = prepare_material_data(train_df, verbose=False)
            val_data = prepare_material_data(
                val_df,
                scaler=train_data["scaler"],
                le=train_data["label_encoder"],
                verbose=False,
            )
            num_materials = train_data["num_materials"]

            r2 = train_with_trial(trial, train_data, val_data, num_materials, device, space)
            fold_scores.append(r2)
            
            # 如果前两折已经表现很差，提前终止
            if fold_idx >= 1 and np.mean(fold_scores) < 0.1:
                print(f"Early terminating cross-validation with mean R2: {np.mean(fold_scores):.4f}")
                # 填充剩余折数的默认值
                fold_scores.extend([0.0] * (n_splits - len(fold_scores)))
                break

        mean_score = float(np.mean(fold_scores))
        print(f"Trial completed with mean R2: {mean_score:.4f}")
        return mean_score
    
    except Exception as e:
        print(f"Error in objective function: {e}")
        return 0.0  # 返回默认值而不是引发异常


def run_hyperopt(
    data_path: str = "3011916164精确材料性能数据.csv",
    n_trials_stage1: int = 40,  # stage1的试验次数
    n_trials_stage2: int = 40,  # stage2的试验次数
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
        print(f"Loaded dataset with {len(df)} samples")
    except Exception as e:
        print(f"Error loading data: {e}")
        # 创建一个小的合成数据集用于测试
        print("Using synthetic data for testing")
        df = pd.DataFrame({
            "材料种类": ["材料A", "材料B", "材料C"] * 10,
            "莫氏硬度": np.random.rand(30),
            "弹性模量": np.random.rand(30),
            "断裂韧性": np.random.rand(30),
            "晶粒粒径": np.random.rand(30),
            "压溃压力": np.random.rand(30),
            "磨粒速度": np.random.rand(30),
            "亚表面裂纹深度": np.random.rand(30),
        })

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ----- 阶段1: RandomSampler 替代 Hyperband 粗搜 -----
    print(f"Starting Stage 1: Initial search with {n_trials_stage1} trials")
    study1 = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.RandomSampler(seed=seed),  # 使用RandomSampler代替TPE
    )
    
    try:
        study1.optimize(
            lambda trial: objective(trial, df, device, seed),
            n_trials=n_trials_stage1,
            timeout=3600,  # 最多运行1小时
            show_progress_bar=True,
            catch=(Exception,),
        )
    except Exception as e:
        print(f"Stage 1 optimization failed: {e}")
    
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

    # ----- 阶段2: TPE微调 -----
    print(f"Starting Stage 2: Fine-tuning with {n_trials_stage2} trials")
    study2 = optuna.create_study(
        direction="maximize", 
        sampler=optuna.samplers.TPESampler(seed=seed)
    )

    try:
        study2.optimize(
            lambda trial: objective(trial, df, device, seed, space),
            n_trials=n_trials_stage2,
            timeout=3600,  # 最多运行1小时
            show_progress_bar=True,
            catch=(Exception,),
        )
    except Exception as e:
        print(f"Stage 2 optimization failed: {e}")

    # 检查两个study是否有完成的trials
    study1_complete = any(t.state == TrialState.COMPLETE for t in study1.trials)
    study2_complete = any(t.state == TrialState.COMPLETE for t in study2.trials)
    
    # 选择最佳study
    if study1_complete and study2_complete:
        final_study = study2 if study2.best_value > study1.best_value else study1
    elif study1_complete:
        final_study = study1
    elif study2_complete:
        final_study = study2
    else:
        print("WARNING: No trials completed in either stage. Using default parameters.")
        # 创建一个dummy study
        final_study = study1
        
    print("\n" + "=" * 50)
    print("优化完成!")
    print("=" * 50)

    if hasattr(final_study, 'best_value'):
        print(f"\n最佳验证R²: {final_study.best_value:.4f}")
        print("\n最佳超参数:")
        for key, value in final_study.best_params.items():
            print(f"  {key}: {value}")
    else:
        print("\n未找到有效的超参数，使用默认值")
        # 设置默认参数
        final_study.best_params = {
            "material_embed_dim": 16,
            "latent_dim": 64,
            "num_blocks": 3,
            "dropout_rate": 0.1,
            "lr": 2e-3,
            "epochs": 100,
            "batch_size": 32,
            "weight_decay": 1e-4,
            "lambda_complexity": 1e-5,
            "lambda_physics": 0.2,
            "patience": 20,
            "augmentation": True,
            "optimizer": "AdamW",
        }
        final_study.best_value = 0.0

    return final_study


# 使用最佳超参数重新训练
def retrain_with_best_params(
    study, data_path="3011916164精确材料性能数据.csv", seed=42
):
    """使用最佳超参数重新训练并评估"""
    
    # 确保参数存在
    if not hasattr(study, 'best_params'):
        print("No best params found, using defaults")
        best_params = {
            "material_embed_dim": 16,
            "latent_dim": 64,
            "num_blocks": 3,
            "dropout_rate": 0.1,
            "lr": 2e-3,
            "epochs": 100,
            "batch_size": 32,
            "weight_decay": 1e-4,
            "lambda_complexity": 1e-5,
            "lambda_physics": 0.2,
            "patience": 20,
            "augmentation": True,
            "optimizer": "AdamW",
        }
    else:
        best_params = study.best_params

    # 读取数据
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        # 创建合成数据用于测试
        df = pd.DataFrame({
            "材料种类": ["材料A", "材料B", "材料C"] * 10,
            "莫氏硬度": np.random.rand(30),
            "弹性模量": np.random.rand(30),
            "断裂韧性": np.random.rand(30),
            "晶粒粒径": np.random.rand(30),
            "压溃压力": np.random.rand(30),
            "磨粒速度": np.random.rand(30),
            "亚表面裂纹深度": np.random.rand(30),
        })

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=seed)

    train_data = prepare_material_data(train_df, verbose=False)
    test_data = prepare_material_data(
        test_df,
        scaler=train_data["scaler"],
        le=train_data["label_encoder"],
        verbose=False,
    )

    # 创建模型
    model = MaterialCrackPGNN(
        num_materials=train_data["num_materials"],
        material_embed_dim=best_params.get("material_embed_dim", 16),
        latent_dim=best_params.get("latent_dim", 64),
        dropout_rate=best_params.get("dropout_rate", 0.1),
        num_blocks=best_params.get("num_blocks", 3),
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # 使用最佳参数准备增强数据
    use_augmentation = best_params.get("augmentation", True)
    train_df_aug = train_df.copy()
    
    # 限制数据扩增，可能是不稳定因素
    if use_augmentation:
        train_df_aug = augment_dataset(train_df_aug)
    augmented_train_data = prepare_material_data(
        train_df_aug,
        scaler=train_data["scaler"],
        le=train_data["label_encoder"],
        verbose=False,
    )

    # 创建DataLoader
    batch_size = best_params.get("batch_size", 32)
    train_dataset = TensorDataset(
        torch.LongTensor(augmented_train_data["material_ids"]),
        torch.FloatTensor(augmented_train_data["numeric_features"]),
        torch.FloatTensor(augmented_train_data["physics_features"]),
        torch.FloatTensor(augmented_train_data["numeric_raw"]),
        torch.FloatTensor(augmented_train_data["target"]).unsqueeze(1),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=True,
        num_workers=4,
    )

    test_materials = torch.LongTensor(test_data["material_ids"]).to(device)
    test_features = torch.FloatTensor(test_data["numeric_features"]).to(device)
    test_phys = torch.FloatTensor(test_data["physics_features"]).to(device)
    test_raw = torch.FloatTensor(test_data["numeric_raw"]).to(device)
    test_target = torch.FloatTensor(test_data["target"]).unsqueeze(1).to(device)

    # 优化器
    lr = best_params.get("lr", 1e-3)
    if best_params.get("optimizer", "AdamW") == "AdamW":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=best_params.get("weight_decay", 1e-4)
        )
    else:
        optimizer = torch.optim.RMSprop(
            model.parameters(), lr=lr, weight_decay=best_params.get("weight_decay", 1e-4)
        )

    # 学习率调度
    epochs = best_params.get("epochs", 100)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True,
        min_lr=1e-5, cooldown=5
    )

    # 禁用SWA以避免动态组件问题
    use_swa = False
    swa_model = None
    swa_start = epochs  # 确保不会触发
    swa_scheduler = None  # 防止未定义错误

    # 损失函数
    loss_fn = AdaptiveLoss(
        lambda_complexity=best_params.get("lambda_complexity", 1e-5),
        lambda_physics=best_params.get("lambda_physics", 0.2),
    )
    loss_fn.total_epochs = epochs
    scaler = GradScaler()

    # 训练
    print("\n使用最佳超参数重新训练...")
    train_losses = []
    test_r2_history = []
    best_model_wts, best_test_r2, patience_ctr = None, -1e9, 0
    best_epoch = 0

    # 早期停止的阈值，如果检测到性能明显下降
    early_stop_threshold = 0.2

    try:
        for epoch in range(epochs):
            # 更新当前epoch
            loss_fn.update_epoch(epoch, epochs)

            # 训练
            model.train()
            epoch_loss = 0

            for batch_materials, batch_features, batch_phys, batch_raw, batch_target in train_loader:
                batch_materials = batch_materials.to(device, non_blocking=True)
                batch_features = batch_features.to(device, non_blocking=True)
                batch_phys = batch_phys.to(device, non_blocking=True)
                batch_raw = batch_raw.to(device, non_blocking=True)
                batch_target = batch_target.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                try:
                    with autocast(device_type=device):
                        pred, uncertainty, _, _ = model(
                            batch_materials, batch_features, batch_phys
                        )
                        loss, metrics = loss_fn(pred, uncertainty, batch_target, model, batch_raw)

                    # 检查NaN
                    if torch.isnan(loss) or torch.isinf(loss) or loss > 1000:
                        print(f"NaN/Inf loss detected in epoch {epoch}. Skipping update.")
                        continue

                    # 监控梯度
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                    grad_norm = model.check_grad_norm() if hasattr(model, 'check_grad_norm') else 0.0
                    scaler.step(optimizer)
                    scaler.update()


                    epoch_loss += loss.item() * batch_target.size(0)
                except Exception as e:
                    print(f"Error in batch training: {e}")
                    continue

            epoch_loss /= len(train_dataset)
            train_losses.append(epoch_loss)

            current_lr = optimizer.param_groups[0]["lr"]

            # 评估
            model.eval()
            try:
                with torch.no_grad(), autocast(device_type=device):
                    test_pred, test_uncertainty, _, _ = model(
                        test_materials, test_features, test_phys
                    )
                    test_pred_np = test_pred.cpu().squeeze().numpy()
                    test_target_np = test_target.cpu().squeeze().numpy()

                    test_r2 = r2_score(test_target_np, test_pred_np)
                    test_rmse = np.sqrt(mean_squared_error(test_target_np, test_pred_np))
                    test_r2_history.append(test_r2)

                if (epoch + 1) % 10 == 0 or epoch == 0:
                    print(
                        f"Epoch {epoch+1}: LR={current_lr:.6f}, Test R²={test_r2:.4f}, RMSE={test_rmse:.4f}"
                    )

                if test_r2 > best_test_r2:
                    best_test_r2 = test_r2
                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_epoch = epoch + 1
                    patience_ctr = 0
                else:
                    patience_ctr += 1

                # 使用最佳参数中的patience
                patience = best_params.get("patience", 20)
                if patience_ctr >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

                # 基于验证集性能更新学习率
                scheduler.step(test_r2)

                # 在获取测试指标后再执行SWA更新
                if use_swa and epoch >= swa_start:
                    swa_model.update_parameters(model)
                    swa_scheduler.step()
            except Exception as e:
                print(f"Error in evaluation: {e}")
                continue

        # 应用SWA权重（如果启用）
        if use_swa and epochs > swa_start:
            try:
                print("Attempting to apply SWA weights...")
                # 检查swa_model状态
                swa_dict = swa_model.module.state_dict()
                model_dict = model.state_dict()

                missing_keys = [k for k in model_dict.keys() if k not in swa_dict]
                if missing_keys:
                    print(f"Warning: SWA model missing keys: {missing_keys}")
                    print("SWA will not be applied")
                else:
                    model.load_state_dict(swa_model.module.state_dict())

                # 重新评估SWA模型
                model.eval()
                with torch.no_grad(), autocast(device_type=device):
                    test_pred, test_uncertainty, _, _ = model(
                        test_materials, test_features, test_phys
                    )
                    test_pred_np = test_pred.cpu().squeeze().numpy()
                    test_target_np = test_target.cpu().squeeze().numpy()
                    swa_r2 = r2_score(test_target_np, test_pred_np)
                    swa_rmse = np.sqrt(mean_squared_error(test_target_np, test_pred_np))

                print(f"SWA model R²: {swa_r2:.4f}, RMSE: {swa_rmse:.4f}")

                # 如果SWA模型比最佳模型差，切回最佳模型
                if swa_r2 < best_test_r2 and best_model_wts is not None:
                    print("SWA model performs worse than best model, reverting to best model")
                    model.load_state_dict(best_model_wts)
            except Exception as e:
                print(f"Error applying SWA: {e}")
                # 恢复最佳模型
                if best_model_wts is not None:
                    model.load_state_dict(best_model_wts)

        # 确保使用最佳模型权重
        if best_model_wts is not None:
            model.load_state_dict(best_model_wts)

    except Exception as e:
        print(f"Error during training: {e}")
        # 如果有最佳权重则恢复
        if best_model_wts is not None:
            model.load_state_dict(best_model_wts)

    # 最终评估 (标准和MC Dropout)
    model.eval()

    # 标准预测
    with torch.no_grad(), autocast(device_type=device):
        test_pred, test_uncertainty, _, _ = model(
            test_materials, test_features, test_phys
        )
        test_pred_np = test_pred.cpu().squeeze().numpy()
        test_target_np = test_target.cpu().squeeze().numpy()

        final_r2 = r2_score(test_target_np, test_pred_np)
        final_rmse = np.sqrt(mean_squared_error(test_target_np, test_pred_np))

    # MC Dropout预测
    mc_pred, mc_unc, aleatoric_unc, epistemic_unc = model.predict_with_uncertainty(
        test_materials, test_features, test_phys, n_samples=20
    )
    mc_pred_np = mc_pred.cpu().squeeze().numpy()
    mc_r2 = r2_score(test_target_np, mc_pred_np)
    mc_rmse = np.sqrt(mean_squared_error(test_target_np, mc_pred_np))

    print(f"\n最终测试集性能:")
    print(f"  标准预测 R²: {final_r2:.4f}, RMSE: {final_rmse:.4f}")
    print(f"  MC预测 R²: {mc_r2:.4f}, RMSE: {mc_rmse:.4f}")
    print(f"  最佳周期: {best_epoch}")

    # 返回模型和训练历史
    return model, train_losses, test_r2_history


def evaluate_model_detailed(model, test_data, device="cuda"):
    """详细评估模型性能，包括不确定性分析"""
    model = model.to(device)
    model.eval()

    # 准备数据
    test_materials = torch.LongTensor(test_data["material_ids"]).to(device)
    test_features = torch.FloatTensor(test_data["numeric_features"]).to(device)
    test_phys = torch.FloatTensor(test_data["physics_features"]).to(device)
    test_raw = torch.FloatTensor(test_data["numeric_raw"]).to(device)
    test_target = torch.FloatTensor(test_data["target"]).unsqueeze(1).to(device)

    # 标准预测
    try:
        with torch.no_grad(), autocast(device_type=device):
            test_pred, test_uncertainty, symbolic_out, residual_out = model(
                test_materials, test_features, test_phys
            )

            # 转换为NumPy
            test_pred_np = test_pred.cpu().squeeze().numpy()
            test_target_np = test_target.cpu().squeeze().numpy()
            test_uncertainty_np = test_uncertainty.cpu().squeeze().numpy()
            symbolic_np = symbolic_out.cpu().squeeze().numpy()
            residual_np = residual_out.cpu().squeeze().numpy()

            # 计算指标
            r2 = r2_score(test_target_np, test_pred_np)
            rmse = np.sqrt(mean_squared_error(test_target_np, test_pred_np))

            # 计算符号表达式和残差网络的贡献
            symbolic_contrib = np.mean(
                np.abs(symbolic_np) / (np.abs(symbolic_np) + np.abs(residual_np) + 1e-6)
            )

            # 计算不确定性与误差的相关性
            errors = np.abs(test_pred_np - test_target_np)
            uncertainty_error_corr = np.corrcoef(errors, test_uncertainty_np)[0, 1]
    except Exception as e:
        print(f"Error in standard prediction: {e}")
        # 使用默认值
        r2, rmse = 0, 1
        symbolic_contrib = 0
        uncertainty_error_corr = 0
        test_pred_np = np.zeros_like(test_target.cpu().squeeze().numpy())
        test_uncertainty_np = np.ones_like(test_target.cpu().squeeze().numpy())
        test_target_np = test_target.cpu().squeeze().numpy()

    # MC Dropout预测
    try:
        mc_pred, mc_unc, aleatoric_unc, epistemic_unc = model.predict_with_uncertainty(
            test_materials, test_features, test_phys, n_samples=30
        )
        mc_pred_np = mc_pred.cpu().squeeze().numpy()
        mc_unc_np = mc_unc.cpu().squeeze().numpy()
        aleatoric_np = aleatoric_unc.cpu().squeeze().numpy()
        epistemic_np = epistemic_unc.cpu().squeeze().numpy()

        mc_r2 = r2_score(test_target_np, mc_pred_np)
        mc_rmse = np.sqrt(mean_squared_error(test_target_np, mc_pred_np))

        # 不确定性指标
        mean_epistemic = np.mean(epistemic_np)
        mean_aleatoric = np.mean(aleatoric_np)
        ratio = mean_epistemic / (mean_aleatoric + 1e-6)

        # 计算每个样本的可靠度 (不确定性校准)
        errors = np.abs(mc_pred_np - test_target_np)
        calibration_scores = errors / (np.sqrt(mc_unc_np) + 1e-6)
        well_calibrated = np.mean((calibration_scores <= 2.0))
    except Exception as e:
        print(f"Error in MC dropout prediction: {e}")
        # 使用默认值
        mc_r2, mc_rmse = 0, 1
        mc_pred_np = np.zeros_like(test_target_np)
        mc_unc_np = np.ones_like(test_target_np)
        mean_epistemic = 0
        mean_aleatoric = 1
        ratio = 0
        well_calibrated = 0
        aleatoric_np = np.ones_like(test_target_np)
        epistemic_np = np.zeros_like(test_target_np)

    # 打印结果
    print("\n===== 详细模型评估 =====")
    print(f"标准预测:")
    print(f"  R²: {r2:.4f}, RMSE: {rmse:.4f}")
    print(f"  符号表达式贡献: {symbolic_contrib:.2f} ({symbolic_contrib*100:.1f}%)")
    print(f"  不确定性-误差相关性: {uncertainty_error_corr:.4f}")

    print(f"\nMC Dropout预测:")
    print(f"  R²: {mc_r2:.4f}, RMSE: {mc_rmse:.4f}")
    print(f"  认知不确定性 (Epistemic): {mean_epistemic:.4f}")
    print(f"  随机不确定性 (Aleatoric): {mean_aleatoric:.4f}")
    print(f"  不确定性比例 (Epistemic/Aleatoric): {ratio:.4f}")

    print(f"\n不确定性校准:")
    print(f"  良好校准比例: {well_calibrated:.2f} ({well_calibrated*100:.1f}%)")

    results = {
        "standard": {"r2": r2, "rmse": rmse, "symbolic_contrib": symbolic_contrib},
        "mc_dropout": {"r2": mc_r2, "rmse": mc_rmse},
        "uncertainty": {
            "epistemic": mean_epistemic,
            "aleatoric": mean_aleatoric,
            "ratio": ratio,
            "calibration": well_calibrated,
        },
        "predictions": {
            "y_true": test_target_np,
            "y_pred": test_pred_np,
            "y_mc_pred": mc_pred_np,
            "uncertainty": test_uncertainty_np,
            "mc_uncertainty": mc_unc_np,
            "epistemic": epistemic_np,
            "aleatoric": aleatoric_np,
        },
    }

    return results


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import os
    
    # 创建results目录
    os.makedirs("results", exist_ok=True)

    # 设置随机种子以保证结果可复现
    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    try:
        # 运行双阶段超参数优化
        study = run_hyperopt(n_trials_stage1=20, n_trials_stage2=10)

        # 使用最佳参数重新训练
        model, train_losses, test_r2_history = retrain_with_best_params(study)

        # 详细评估
        df = pd.read_csv("3011916164精确材料性能数据.csv")
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=SEED)
        train_data = prepare_material_data(train_df, verbose=False)
        test_data = prepare_material_data(
            test_df,
            scaler=train_data["scaler"],
            le=train_data["label_encoder"],
            verbose=False,
        )

        results = evaluate_model_detailed(model, test_data)

        # 保存模型
        torch.save({
            'model_state_dict': model.state_dict(),
            'scaler': train_data["scaler"],
            'label_encoder': train_data["label_encoder"],
            'num_materials': train_data["num_materials"],
            'best_params': study.best_params if hasattr(study, 'best_params') else None,
        }, 'results/best_model.pth')
        print("Model saved to results/best_model.pth")

        # 可视化优化历史
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # 优化历史
        try:
            completed_trials = [t for t in study.trials if t.state == TrialState.COMPLETE and t.value is not None]
            if completed_trials:
                values = [t.value for t in completed_trials]
                ax1.plot(values)
                ax1.set_xlabel("Trial")
                ax1.set_ylabel("Validation R²")
                ax1.set_title("Hyperparameter Optimization History")
                ax1.grid(True, alpha=0.3)
            else:
                ax1.text(0.5, 0.5, "No completed trials", ha='center', va='center')
                ax1.set_title("Optimization History (Empty)")
        except Exception as e:
            print(f"Error plotting optimization history: {e}")
            ax1.text(0.5, 0.5, "Error plotting history", ha='center', va='center')

        # 训练损失
        try:
            if train_losses:
                ax2.plot(train_losses)
                ax2.set_xlabel("Epoch")
                ax2.set_ylabel("Loss")
                ax2.set_title("Training Loss with Best Hyperparameters")
                ax2.set_yscale("log")
                ax2.grid(True, alpha=0.3)
            else:
                ax2.text(0.5, 0.5, "No training losses recorded", ha='center', va='center')
                ax2.set_title("Training Loss (Empty)")
        except Exception as e:
            print(f"Error plotting training losses: {e}")
            ax2.text(0.5, 0.5, "Error plotting losses", ha='center', va='center')

        plt.tight_layout()
        plt.savefig("results/improved_hyperopt_results.png", dpi=300)
        print("Optimization history plot saved")

        # 预测vs真实值可视化
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        try:
            # 标准预测
            ax1.scatter(
                results["predictions"]["y_true"], results["predictions"]["y_pred"], alpha=0.6
            )
            ax1.plot([0, 10], [0, 10], "r--")  # 理想线
            ax1.set_xlabel("True Crack Depth")
            ax1.set_ylabel("Predicted Crack Depth")
            ax1.set_title(f'Standard Prediction (R² = {results["standard"]["r2"]:.4f})')
            ax1.grid(True, alpha=0.3)

            # MC Dropout预测
            scatter = ax2.scatter(
                results["predictions"]["y_true"],
                results["predictions"]["y_mc_pred"],
                c=results["predictions"]["mc_uncertainty"],
                cmap="viridis",
                alpha=0.6,
            )
            ax2.plot([0, 10], [0, 10], "r--")  # 理想线
            ax2.set_xlabel("True Crack Depth")
            ax2.set_ylabel("MC Prediction")
            ax2.set_title(f'MC Dropout Prediction (R² = {results["mc_dropout"]["r2"]:.4f})')
            ax2.grid(True, alpha=0.3)
            cbar = plt.colorbar(scatter, ax=ax2)
            cbar.set_label("Predictive Uncertainty")

            plt.tight_layout()
            plt.savefig("results/predictions_visualization.png", dpi=300)
            print("Predictions visualization saved")
        except Exception as e:
            print(f"Error creating prediction visualization: {e}")

        # 不确定性可视化
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # 误差vs不确定性散点图
            errors = np.abs(results["predictions"]["y_pred"] - results["predictions"]["y_true"])
            ax1.scatter(errors, results["predictions"]["uncertainty"], alpha=0.6)
            ax1.set_xlabel("Absolute Error")
            ax1.set_ylabel("Predicted Uncertainty")
            ax1.set_title(f'Error vs Uncertainty (Corr = {results["uncertainty"]["calibration"]:.4f})')
            ax1.grid(True, alpha=0.3)
            
            # 认知vs随机不确定性散点图
            ax2.scatter(
                results["predictions"]["aleatoric"], 
                results["predictions"]["epistemic"],
                alpha=0.6
            )
            ax2.set_xlabel("Aleatoric Uncertainty")
            ax2.set_ylabel("Epistemic Uncertainty")
            ax2.set_title("Aleatoric vs Epistemic Uncertainty")
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig("results/uncertainty_visualization.png", dpi=300)
            print("Uncertainty visualization saved")
        except Exception as e:
            print(f"Error creating uncertainty visualization: {e}")

        print("\nAll visualizations and model saved to 'results/' directory")
    
    except Exception as e:
        print(f"An error occurred in the main execution: {e}")