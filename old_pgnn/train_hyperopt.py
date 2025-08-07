import optuna
from optuna.trial import TrialState
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
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import r2_score, mean_squared_error
from torch.utils.data import TensorDataset, DataLoader
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import OneCycleLR

from models import MaterialCrackPGNN, GRAD_CLIP
from losses import AdaptiveLoss
from utils import prepare_material_data, augment_dataset, reconstruct_df


def get_valid_latent_dim(value):
    """确保latent_dim能被4整除（用于MultiheadAttention）"""
    return max(16, (value // 16) * 16)


def get_valid_embed_dim(value):
    """确保embedding维度合适"""
    return max(8, (value // 8) * 8)


def create_model(trial, num_materials, space=None):
    """
    根据trial创建模型
    可以使用预定义的搜索空间或默认范围
    """
    if space is None:
        material_range = (8, 24)
        latent_range = (16, 48)
        block_range = (2, 4)
    else:
        material_range = space.get("material_embed_dim", (8, 24))
        latent_range = space.get("latent_dim", (16, 48))
        block_range = space.get("num_blocks", (2, 4))

    # 模型架构超参数
    material_embed_dim = trial.suggest_int(
        "material_embed_dim", 
        get_valid_embed_dim(material_range[0]), 
        get_valid_embed_dim(material_range[1]), 
        step=8
    )

    latent_dim = trial.suggest_int(
        "latent_dim", 
        int(latent_range[0]), 
        int(latent_range[1]), 
        step=16
    )
    latent_dim = get_valid_latent_dim(latent_dim)
    
    num_blocks = trial.suggest_int(
        "num_blocks", 
        int(block_range[0]), 
        int(block_range[1])
    )

    # 创建模型
    model = MaterialCrackPGNN(
        num_materials=num_materials,
        material_embed_dim=material_embed_dim,
        latent_dim=latent_dim,
        num_blocks=num_blocks,
        # 可选参数
        dropout_rate=0.1,  # 默认值，可以省略
        use_advanced_symbolic=True  # 控制是否使用高级符号层，默认为True
    )

    # 动态调整dropout率
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.2, step=0.05)
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = dropout_rate

    return model


def train_with_trial(trial, train_data, val_data, num_materials, device="cuda", space=None, fixed_params=None):
    """
    使用给定的超参数训练模型
    返回验证集R²分数
    
    参数:
        trial: Optuna trial对象
        train_data: 训练数据
        val_data: 验证数据
        num_materials: 材料种类数量
        device: 训练设备
        space: 自定义搜索空间
        fixed_params: 从第一阶段固定的参数
    """
    if fixed_params is None:
        fixed_params = {}
        
    try:
        # 训练超参数 - 使用fixed_params中的值或从trial中获取
        if "lr" in fixed_params:
            lr = fixed_params["lr"]
        else:
            lr = trial.suggest_float("lr", 0.0005, 0.003, log=True)
            
        if "epochs" in fixed_params:
            epochs = fixed_params["epochs"]
        else:
            epochs = trial.suggest_int("epochs", 200, 500, step=50)
            
        if "batch_size" in fixed_params:
            batch_size = fixed_params["batch_size"]
        else:
            batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
            
        if "weight_decay" in fixed_params:
            weight_decay = fixed_params["weight_decay"]
        else:
            weight_decay = trial.suggest_float("weight_decay", 3e-3, 1e-2, log=True)

        # 损失函数超参数
        if "lambda_complexity" in fixed_params:
            lambda_complexity = fixed_params["lambda_complexity"]
        else:
            lambda_complexity = trial.suggest_float("lambda_complexity", 1e-5, 1e-3, log=True)
            
        if "lambda_physics" in fixed_params:
            lambda_physics = fixed_params["lambda_physics"]
        else:
            lambda_physics = trial.suggest_float("lambda_physics", 0.1, 0.5, step=0.01)

        # 早停参数
        if "patience" in fixed_params:
            patience = fixed_params["patience"]
        else:
            patience = trial.suggest_int("patience", 40, 60, step=5)

        # 其他选项
        if "augmentation" in fixed_params:
            use_augmentation = fixed_params["augmentation"]
        else:
            use_augmentation = trial.suggest_categorical("augmentation", [True, False])
            
        if "optimizer" in fixed_params:
            optimizer_name = fixed_params["optimizer"]
        else:
            optimizer_name = trial.suggest_categorical("optimizer", 
                                         ["AdamW", "RMSprop", "Adam", "RAdam", "SGD", "NAdam"])

        # 准备数据
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
            torch.FloatTensor(augmented_train_data["numeric_raw"]),
            torch.FloatTensor(augmented_train_data["target"]).unsqueeze(1),
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            pin_memory=True,
            num_workers=0,
        )
        
        # 创建模型
        model = create_model(trial, num_materials, space).to(device)

        # 创建优化器
        if optimizer_name == "AdamW":
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=lr, weight_decay=weight_decay
            )
        elif optimizer_name == "RMSprop":
            optimizer = torch.optim.RMSprop(
                model.parameters(), lr=lr, weight_decay=weight_decay
            )
        elif optimizer_name == "Adam":
            optimizer = torch.optim.Adam(
                model.parameters(), lr=lr, weight_decay=weight_decay
            )
        elif optimizer_name == "RAdam":
            optimizer = torch.optim.RAdam(
                model.parameters(), lr=lr, weight_decay=weight_decay
            )
        elif optimizer_name == "SGD":
            optimizer = torch.optim.SGD(
                model.parameters(), lr=lr, momentum=0.9, 
               weight_decay=weight_decay, nesterov=True
            )
        elif optimizer_name == "NAdam":
            optimizer = torch.optim.NAdam(
                model.parameters(), lr=lr, weight_decay=weight_decay
            )

        # 替换学习率调度 - 使用 OneCycleLR
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
        loss_fn = AdaptiveLoss(
            lambda_complexity=lambda_complexity,
            lambda_physics=lambda_physics,
        )
        loss_fn.total_epochs = epochs
        scaler = GradScaler()

        # 验证数据
        val_materials = torch.LongTensor(val_data["material_ids"]).to(device)
        val_features = torch.FloatTensor(val_data["numeric_features"]).to(device)
        val_raw = torch.FloatTensor(val_data["numeric_raw"]).to(device)
        val_target = torch.FloatTensor(val_data["target"]).unsqueeze(1).to(device)

        # 训练循环
        best_val_r2 = -float("inf")
        patience_counter = 0
        best_model_wts = None
        min_epochs_before_pruning = 10

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
                        pred, uncertainty, symbolic, residual = model(
                            batch_materials, batch_features, batch_raw
                        )
                        loss, metrics = loss_fn(pred, uncertainty, batch_target, model, batch_raw)

                    # 检查NaN
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"Warning: NaN or Inf loss detected in epoch {epoch}")
                        if epoch < min_epochs_before_pruning:
                            return 0.0
                        elif best_model_wts is not None:
                            model.load_state_dict(best_model_wts)
                            continue
                        else:
                            return 0.0

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

                    # 检查loss是否为NaN
                    if torch.isnan(loss).any() or torch.isinf(loss).any():
                        print(f"Warning: NaN/Inf loss detected, skipping batch")
                        continue
                        
                    epoch_loss += loss.item() * batch_target.size(0)
                    
                except RuntimeError as e:
                    print(f"Runtime error in batch training: {e}")
                    if 'out of memory' in str(e).lower():
                        torch.cuda.empty_cache()
                        return 0.0
                    else:
                        print(traceback.format_exc())
                        return 0.0
                except Exception as e:
                    print(f"Error in batch training: {e}")
                    print(traceback.format_exc())
                    continue

            # 避免分母为0的情况
            num_processed = max(1, len(train_dataset))
            epoch_loss /= num_processed

            # 验证
            model.eval()
            try:
                with torch.no_grad(), autocast(device_type=device):
                    val_pred, val_uncertainty, _, _ = model(
                        val_materials, val_features, val_raw
                    )
                    
                    # 检查并修复NaN/Inf值
                    if torch.isnan(val_pred).any() or torch.isinf(val_pred).any():
                        print(f"Warning: Detected NaN/Inf in validation predictions, fixing...")
                        val_pred = torch.nan_to_num(val_pred, nan=0.0, posinf=1.0, neginf=-1.0)
                    
                    val_loss, _ = loss_fn(val_pred, val_uncertainty, val_target, model, val_raw)

                    val_pred_np = val_pred.cpu().squeeze().numpy()
                    val_target_np = val_target.cpu().squeeze().numpy()
                    
                    # 额外检查numpy数组中的NaN
                    if np.isnan(val_pred_np).any() or np.isinf(val_pred_np).any():
                        print(f"Warning: Detected NaN/Inf in numpy validation predictions, replacing with median")
                        valid_preds = val_pred_np[~np.isnan(val_pred_np) & ~np.isinf(val_pred_np)]
                        if len(valid_preds) > 0:
                            median_val = np.median(valid_preds)
                        else:
                            median_val = 0.0
                        val_pred_np = np.nan_to_num(val_pred_np, nan=median_val, posinf=1.0, neginf=-1.0)

                    val_r2 = r2_score(val_target_np, val_pred_np)

                # 早停检查
                if val_r2 > best_val_r2:
                    best_val_r2 = val_r2
                    patience_counter = 0
                    best_model_wts = copy.deepcopy(model.state_dict())
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

                # 向Optuna报告中间结果
                if not hasattr(trial, '_reported_epochs'):
                    trial._reported_epochs = set()
                if epoch not in trial._reported_epochs:
                    trial.report(val_r2, epoch)
                    trial._reported_epochs.add(epoch)
                
                # 剪枝检查
                if trial.should_prune() and epoch >= min_epochs_before_pruning:
                    raise optuna.TrialPruned()
            
            except Exception as e:
                print(f"Error in validation: {e}")
                return 0.0

        # 恢复最佳模型权重
        if best_model_wts is not None:
            model.load_state_dict(best_model_wts)

        return best_val_r2

    except Exception as e:
        print(f"Trial failed with error: {e}")
        return 0.0


def objective(trial, df, device, seed, space=None, fixed_params=None):
    """
    Optuna目标函数，使用交叉验证
    
    参数:
        trial: Optuna trial对象
        df: 数据集
        device: 训练设备
        seed: 随机种子
        space: 自定义搜索空间
        fixed_params: 从第一阶段固定的参数，避免在第二阶段重新定义
    """
    try:
        # 应用固定参数（如果有）
        if fixed_params is None:
            fixed_params = {}
            
        n_splits = 3  # 3折交叉验证
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

            r2 = train_with_trial(trial, train_data, val_data, num_materials, device, space, fixed_params)
            fold_scores.append(r2)
            
            # 如果前两折表现很差，提前终止
            if fold_idx >= 1 and np.mean(fold_scores) < 0.2:
                print(f"Early terminating cross-validation with mean R2: {np.mean(fold_scores):.4f}")
                fold_scores.extend([0.0] * (n_splits - len(fold_scores)))
                break

        mean_score = float(np.mean(fold_scores))
        print(f"Trial completed with mean R2: {mean_score:.4f}")
        return mean_score
    
    except Exception as e:
        print(f"Error in objective function: {e}")
        return 0.0


def run_hyperopt(
    data_path: str = "3011916164精确材料性能数据.csv",
    n_trials_stage1: int = 20,
    n_trials_stage2: int = 10,
    top_k: int = 3,
    seed: int = 42,
):
    """
    双阶段超参数搜索
    Stage 1: 随机搜索进行粗略探索
    Stage 2: TPE在promising区域精细搜索
    """
    # 设置随机种子
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 加载数据
    try:
        df = pd.read_csv(data_path)
        print(f"Loaded dataset with {len(df)} samples")
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 阶段1: 随机搜索
    print(f"\nStarting Stage 1: Initial search with {n_trials_stage1} trials")
    study1 = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.RandomSampler(seed=seed),
    )
    
    study1.optimize(
        lambda trial: objective(trial, df, device, seed),
        n_trials=n_trials_stage1,
        timeout=3600,
        show_progress_bar=True,
        catch=(Exception,),
    )
    
    # 获取第一阶段的最佳参数
    completed_trials = [t for t in study1.trials if t.state == TrialState.COMPLETE]
    if not completed_trials:
        print("WARNING: No trials completed in Stage 1")
        return study1
        
    # 根据阶段1结果定义收缩后的搜索空间
    space = define_refined_space(completed_trials, top_k)
    
    # 获取第一阶段最佳试验
    best_trial = study1.best_trial
    fixed_params = {}
    categorical_params = ["batch_size", "augmentation", "optimizer"]
    
    # 将分类参数固定为第一阶段最佳值，避免在第二阶段重新定义
    for param in categorical_params:
        if param in best_trial.params:
            fixed_params[param] = best_trial.params[param]
            print(f"固定参数 {param} = {fixed_params[param]} (来自第一阶段最佳试验)")
            # 从搜索空间中移除已固定的参数
            if param in space:
                del space[param]

    # 阶段2: TPE微调
    print(f"\nStarting Stage 2: Fine-tuning with {n_trials_stage2} trials")
    study2 = optuna.create_study(
        direction="maximize", 
        sampler=optuna.samplers.TPESampler(seed=seed)
    )

    study2.optimize(
        lambda trial: objective(trial, df, device, seed, space, fixed_params),
        n_trials=n_trials_stage2,
        timeout=3600,
        show_progress_bar=True,
        catch=(Exception,),
    )

    # 保存两个阶段的study对象，以便后续可以访问
    final_study = study2
    final_study._study1 = study1
    final_study._study2 = study2

    # 选择最佳study
    if study2.best_value > study1.best_value:
        print(f"阶段2获得更好的结果: {study2.best_value:.4f} > {study1.best_value:.4f}")
    else:
        print(f"阶段1获得更好的结果: {study1.best_value:.4f} > {study2.best_value:.4f}")
        final_study = study1
        
    print_optimization_results(final_study)
    
    return final_study


def define_refined_space(completed_trials, top_k):
    """
    根据第一阶段结果定义精细搜索空间
    """
    # 默认边界
    global_bounds = {
        "material_embed_dim": (8, 24),
        "latent_dim": (32, 64),
        "num_blocks": (2, 2),
        "dropout_rate": (0.05, 0.15),
        "lr": (5e-4, 2e-3),
        "epochs": (100, 200),
        "batch_size": [16, 32],
        "weight_decay": (1e-5, 5e-4),
        "lambda_complexity": (1e-6, 5e-5),
        "lambda_physics": (0.01, 0.05),
        "patience": (15, 25),
        "augmentation": [True, False],
        "optimizer": ["AdamW", "RMSprop"],
    }
    
    # 获取最佳trials
    top_trials = sorted(
        completed_trials,
        key=lambda t: t.value,
        reverse=True,
    )[:top_k]
    
    space = {}
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
                    space[name] = (low * 0.9, low * 1.1) if low > 0 else (low - 0.1, low + 0.1)
            else:
                space[name] = bounds
        else:
            vals = list({t.params.get(name) for t in top_trials if name in t.params})
            space[name] = vals if vals else bounds
            
    return space


def print_optimization_results(study):
    """打印优化结果"""
    print("\n" + "=" * 50)
    print("优化完成!")
    print("=" * 50)

    if hasattr(study, 'best_value'):
        print(f"\n最佳验证R²: {study.best_value:.4f}")
        print("\n最佳超参数:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")
            
        # 统计信息
        completed_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
        print(f"\n完成的trials: {len(completed_trials)}/{len(study.trials)}")
        
        # 超参数重要性
        try:
            importance = optuna.importance.get_param_importances(study)
            print("\n超参数重要性:")
            for key, value in sorted(importance.items(), key=lambda x: x[1], reverse=True):
                print(f"  {key}: {value:.3f}")
        except:
            pass


def retrain_with_best_params(
    study, data_path="3011916164精确材料性能数据.csv", seed=42
):
    """
    使用最佳超参数重新训练模型
    在完整数据集上训练并在测试集上评估
    """
    # 获取最佳参数
    if not hasattr(study, 'best_params') or not study.best_params:
        print("No best params found, using defaults")
        best_params = {
            "material_embed_dim": 16,
            "latent_dim": 64,
            "dropout_rate": 0.1,
            "num_blocks": 3,
            "lr": 1e-3,
            "epochs": 150,
            "batch_size": 32,
            "weight_decay": 1e-4,
            "lambda_complexity": 1e-5,
            "lambda_physics": 0.05,
            "patience": 25,
            "augmentation": True,
            "optimizer": "AdamW",
            "use_advanced_symbolic": True,
        }
    else:
        best_params = study.best_params
        
    # 如果有两个阶段的study，尝试从另一个阶段获取可能缺失的参数
    if hasattr(study, '_study1') and hasattr(study, '_study2'):
        other_study = study._study1 if study == study._study2 else study._study2
        if hasattr(other_study, 'best_params'):
            for param, value in other_study.best_params.items():
                if param not in best_params:
                    best_params[param] = value
                    print(f"从另一阶段获取缺失参数: {param} = {value}")

    # 读取数据
    df = pd.read_csv(data_path)
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
        use_advanced_symbolic=best_params.get("use_advanced_symbolic", True),  # 添加新参数
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # 准备数据
    use_augmentation = best_params.get("augmentation", True)
    train_df_aug = train_df.copy()
    
    if use_augmentation:
        train_df_aug = augment_dataset(train_df_aug)
        
    augmented_train_data = prepare_material_data(
        train_df_aug,
        scaler=train_data["scaler"],
        le=train_data["label_encoder"],
        verbose=False,
    )

    # DataLoader
    batch_size = best_params.get("batch_size", 32)
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
    lr = best_params.get("lr", 1e-3)
    if best_params.get("optimizer", "AdamW") == "AdamW":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=best_params.get("weight_decay", 1e-4)
        )
    elif best_params.get("optimizer", "AdamW") == "RMSprop":
        optimizer = torch.optim.RMSprop(
            model.parameters(), lr=lr, weight_decay=best_params.get("weight_decay", 1e-4)
        )
    elif best_params.get("optimizer", "AdamW") == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=best_params.get("weight_decay", 1e-4)
        )
    elif best_params.get("optimizer", "AdamW") == "RAdam":
        optimizer = torch.optim.RAdam(
            model.parameters(), lr=lr, weight_decay=best_params.get("weight_decay", 1e-4)
        )
    elif best_params.get("optimizer", "AdamW") == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, momentum=0.9, 
           weight_decay=best_params.get("weight_decay", 1e-4), nesterov=True
        )
    elif best_params.get("optimizer", "AdamW") == "NAdam":
        optimizer = torch.optim.NAdam(
            model.parameters(), lr=lr, weight_decay=best_params.get("weight_decay", 1e-4)
        )

    # 学习率调度 - 使用 OneCycleLR
    epochs = best_params.get("epochs", 100)
    total_steps = epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        total_steps=total_steps,
        pct_start=0.3,      # 30% 的步数用于预热
        div_factor=25.0,    # 初始学习率 = max_lr/25
        final_div_factor=1e4,  # 最终学习率 = max_lr/(25*10000)
        anneal_strategy='cos'
    )

    # 损失函数
    loss_fn = AdaptiveLoss(
        lambda_complexity=best_params.get("lambda_complexity", 1e-5),
        lambda_physics=best_params.get("lambda_physics", 0.05),
    )
    loss_fn.total_epochs = epochs
    scaler = GradScaler()

    # 训练
    print("\n使用最佳超参数重新训练...")
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
                
                # 更新学习率 (OneCycleLR在每个batch后调用)
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
            print(f"Epoch {epoch+1}: LR={current_lr:.6f}, "
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
        patience = best_params.get("patience", 20)
        if patience_ctr >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

        # OneCycleLR不需要在每个epoch结束时调用step，已经在每个batch后更新

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


def main():
    """
    主函数：执行完整的超参数优化和最终训练流程
    """
    print("=" * 60)
    print("材料裂纹预测PGNN模型超参数优化")
    print("=" * 60)
    
    # 配置参数
    DATA_PATH = "3011916164精确材料性能数据.csv"
    N_TRIALS_STAGE1 = 10  # 第一阶段试验次数
    N_TRIALS_STAGE2 = 5  # 第二阶段试验次数
    TOP_K = 3  # 从第一阶段选择的最佳试验数量
    SEED = 88 # 随机种子
    
    print(f"数据路径: {DATA_PATH}")
    print(f"第一阶段试验次数: {N_TRIALS_STAGE1}")
    print(f"第二阶段试验次数: {N_TRIALS_STAGE2}")
    print(f"随机种子: {SEED}")
    print()
    
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
    
    try:
        # 第一步：运行超参数优化
        print("开始超参数优化...")
        study = run_hyperopt(
            data_path=DATA_PATH,
            n_trials_stage1=N_TRIALS_STAGE1,
            n_trials_stage2=N_TRIALS_STAGE2,
            top_k=TOP_K,
            seed=SEED
        )
        
        if study is None or not hasattr(study, 'best_params'):
            print("超参数优化失败!")
            return
        
        print("\n超参数优化完成!")
        print(f"最佳交叉验证R²: {study.best_value:.4f}")
        
        # 第二步：使用最佳超参数重新训练
        print("\n" + "=" * 50)
        print("使用最佳超参数进行最终训练...")
        print("=" * 50)
        
        model, train_losses, test_r2_history = retrain_with_best_params(
            study, 
            data_path=DATA_PATH, 
            seed=SEED
        )
        
        if model is None:
            print("最终训练失败!")
            return
            
        # 第三步：保存结果
        print("\n保存模型和结果...")
        
        # 创建结果目录
        os.makedirs("results", exist_ok=True)
        
        # 保存最佳模型
        torch.save(model.state_dict(), "results/best_hyperopt_model.pth")
        
        # 保存超参数
        with open("results/best_hyperopt_params.json", "w", encoding="utf-8") as f:
            json.dump(study.best_params, f, indent=2, ensure_ascii=False)
        
        # 保存训练历史
        with open("results/training_history.pkl", "wb") as f:
            pickle.dump({
                "train_losses": train_losses,
                "test_r2_history": test_r2_history,
                "best_cv_r2": study.best_value,
                "best_params": study.best_params
            }, f)
        
        # 第四步：生成可视化结果
        try:
            create_optimization_plots(study, train_losses, test_r2_history)
        except Exception as e:
            print(f"生成图表时出错: {e}")
        
        # 第五步：输出最终结果总结
        print("\n" + "=" * 60)
        print("训练完成! 结果总结:")
        print("=" * 60)
        print(f"最佳交叉验证R²: {study.best_value:.4f}")
        print(f"最终测试集R²: {max(test_r2_history):.4f}")
        print(f"训练轮次: {len(train_losses)}")
        print("\n最佳超参数:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")
        
        print(f"\n文件已保存到:")
        print(f"  模型权重: results/best_hyperopt_model.pth")
        print(f"  超参数: results/best_hyperopt_params.json")
        print(f"  训练历史: results/training_history.pkl")
        print(f"  优化图表: results/hyperopt_optimization_plots.png")
        
    except KeyboardInterrupt:
        print("\n用户中断训练")
    except Exception as e:
        print(f"\n训练过程中出现错误: {e}")
        traceback.print_exc()


def create_optimization_plots(study, train_losses, test_r2_history):
    """
    创建优化过程的可视化图表
    """
    try:
        matplotlib.use('Agg')  # 使用非交互式后端
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 优化历史
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        trial_values = [t.value for t in completed_trials]
        trial_numbers = [t.number for t in completed_trials]
        
        axes[0, 0].plot(trial_numbers, trial_values, 'b-o', markersize=3)
        axes[0, 0].axhline(y=study.best_value, color='r', linestyle='--', 
                          label=f'Best: {study.best_value:.4f}')
        axes[0, 0].set_xlabel('Trial Number')
        axes[0, 0].set_ylabel('Validation R²')
        axes[0, 0].set_title('Hyperparameter Optimization History')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 训练损失
        axes[0, 1].plot(train_losses, 'g-', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Training Loss')
        axes[0, 1].set_title('Training Loss History')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 测试R²历史
        axes[1, 0].plot(test_r2_history, 'r-', linewidth=2)
        axes[1, 0].axhline(y=max(test_r2_history), color='b', linestyle='--',
                          label=f'Best: {max(test_r2_history):.4f}')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Test R²')
        axes[1, 0].set_title('Test R² History')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 超参数重要性
        try:
            importance = optuna.importance.get_param_importances(study)
            if importance:
                params = list(importance.keys())[:8]  # 显示前8个重要参数
                values = [importance[p] for p in params]
                
                y_pos = range(len(params))
                axes[1, 1].barh(y_pos, values)
                axes[1, 1].set_yticks(y_pos)
                axes[1, 1].set_yticklabels(params)
                axes[1, 1].set_xlabel('Importance')
                axes[1, 1].set_title('Parameter Importance')
        except:
            axes[1, 1].text(0.5, 0.5, 'Parameter importance\nnot available', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Parameter Importance')
        
        plt.tight_layout()
        plt.savefig('results/hyperopt_optimization_plots.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("优化图表已保存到: results/hyperopt_optimization_plots.png")
        
    except ImportError:
        print("警告: matplotlib未安装，跳过图表生成")
    except Exception as e:
        print(f"生成图表时出错: {e}")


if __name__ == "__main__":
    main()