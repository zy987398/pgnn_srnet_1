import os
import time
import warnings
import traceback
import copy
import joblib
import datetime
import optuna
from optuna.trial import TrialState
import sys
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import OneCycleLR
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 配置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi', 'FangSong']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['font.size'] = 12  # 设置字体大小

# 设置matplotlib后端，确保中文显示
import matplotlib
matplotlib.rcParams['font.family'] = 'sans-serif'

# 创建全局字体属性对象
chinese_font_prop = None
try:
    chinese_fonts = ['Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi', 'FangSong']
    for font_name in chinese_fonts:
        fonts = [f.name for f in matplotlib.font_manager.fontManager.ttflist]
        if font_name in fonts:
            chinese_font_prop = matplotlib.font_manager.FontProperties(family=font_name)
            print(f"使用中文字体: {font_name}")
            break
    if chinese_font_prop is None:
        chinese_font_prop = matplotlib.font_manager.FontProperties(family='DejaVu Sans')
        print("使用默认字体: DejaVu Sans")
except Exception as e:
    chinese_font_prop = matplotlib.font_manager.FontProperties(family='DejaVu Sans')
    print(f"字体设置出错，使用默认字体: {e}")

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import r2_score, mean_squared_error
from torch.utils.data import TensorDataset, DataLoader

# 导入自定义模块
from models import MaterialCrackPGNN
from losses import PhysicsInformedLoss
from utils import prepare_material_data, reconstruct_df, augment_dataset


# 日志重定向类：将所有print内容写入日志文件和控制台
class Logger(object):
    def __init__(self, log_file):
        self.terminal = sys.__stdout__
        self.log = open(log_file, "a", encoding="utf-8")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    def close(self):
        self.log.close()


# 忽略警告
warnings.filterwarnings("ignore")


def create_model(trial, num_materials, fixed_params=None):
    """根据trial创建模型，支持固定参数"""
    fixed_params = fixed_params or {}
    
    # 模型架构超参数
    if "material_embed_dim" in fixed_params:
        material_embed_dim = fixed_params["material_embed_dim"]
        print(f"  使用预设材料嵌入维度: {material_embed_dim}")
    else:
        material_embed_dim = trial.suggest_int(
            "material_embed_dim", 8, 24, step=8
        )  # 缩小范围
    
    if "latent_dim" in fixed_params:
        latent_dim = fixed_params["latent_dim"]
        print(f"  使用预设潜在维度: {latent_dim}")
    else:
        latent_dim = trial.suggest_int("latent_dim", 16, 48, step=16)  # 缩小范围

    # 符号层类型选择 - 始终从fixed_params获取，不再进行搜索
    symbolic_type = fixed_params.get("symbolic_type", "advanced")  # 默认使用高级符号层
    print(f"  使用符号层类型: {symbolic_type}")
    
    # 创建模型并修改部分架构
    model = MaterialCrackPGNN(
        num_materials=num_materials,
        material_embed_dim=material_embed_dim,
        latent_dim=latent_dim,
        symbolic_type=symbolic_type
    )

    # 动态调整dropout率
    if "dropout_rate" in fixed_params:
        dropout_rate = fixed_params["dropout_rate"]
        print(f"  使用预设Dropout率: {dropout_rate}")
    else:
        dropout_rate = trial.suggest_float(
            "dropout_rate", 0.3, 0.4, step=0.05
        )  # 更高的dropout范围
        
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = dropout_rate

    return model


def train_with_trial(trial, train_data, val_data, num_materials, device="cuda", 
                fixed_params=None):
    """使用给定的超参数训练模型，支持固定参数（用于第二阶段）"""
    try:
        # 记录开始时间
        start_time = time.time()
        trial_num = trial.number if hasattr(trial, 'number') else '?'
        
        # 初始化固定参数字典
        fixed_params = fixed_params or {}
        
        # 训练超参数 - 使用固定值或试验建议
        if "lr" in fixed_params:
            lr = fixed_params["lr"]
            print(f"  使用预设学习率: {lr:.1e}")
        else:
            lr = trial.suggest_float("lr", 0.0005, 0.003, log=True)  # 更窄的学习率范围
            
        if "epochs" in fixed_params:
            epochs = fixed_params["epochs"]
            print(f"  使用预设训练周期: {epochs}")
        else:
            epochs = trial.suggest_int("epochs", 200, 500, step=50)  # 减小上限
        
        # 批次大小 (类别参数)
        if "batch_size" in fixed_params:
            batch_size = fixed_params["batch_size"]
            print(f"  使用预设批次大小: {batch_size}")
        else:
            batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

        # 权重衰减
        if "weight_decay" in fixed_params:
            weight_decay = fixed_params["weight_decay"]
            print(f"  使用预设权重衰减: {weight_decay:.1e}")
        else:
            weight_decay = trial.suggest_float("weight_decay", 3e-3, 1e-2, log=True)  # 提高范围

        # 损失函数超参数
        if "lambda_complexity" in fixed_params:
            lambda_complexity = fixed_params["lambda_complexity"]
            print(f"  使用预设复杂度权重: {lambda_complexity:.1e}")
        else:
            lambda_complexity = trial.suggest_float("lambda_complexity", 1e-5, 1e-3, log=True)  # 提高符号表达式约束
            
        if "lambda_physics" in fixed_params:
            lambda_physics = fixed_params["lambda_physics"]
            print(f"  使用预设物理权重: {lambda_physics}")
        else:
            lambda_physics = trial.suggest_float("lambda_physics", 0.1, 0.5, step=0.05)  # 提高物理约束

        # 早停参数
        if "patience" in fixed_params:
            patience = fixed_params["patience"]
            print(f"  使用预设早停耐心值: {patience}")
        else:
            patience = trial.suggest_int("patience", 40, 60, step=5)

        # 数据增强 (类别参数)
        if "augmentation" in fixed_params:
            use_augmentation = fixed_params["augmentation"]
            print(f"  使用预设数据增强: {use_augmentation}")
        else:
            use_augmentation = trial.suggest_categorical("augmentation", [True, False])

        # 优化器选择 (类别参数)
        if "optimizer" in fixed_params:
            optimizer_name = fixed_params["optimizer"]
            print(f"  使用预设优化器: {optimizer_name}")
        else:
            optimizer_name = trial.suggest_categorical("optimizer", 
                                             ["AdamW", "RMSprop", "Adam", "RAdam", "SGD", "NAdam"])

        print(f"\nTrial {trial_num}: 开始训练 (LR={lr:.1e}, 批次大小={batch_size}, "
              f"权重衰减={weight_decay:.1e}, 优化器={optimizer_name}, 增强数据={use_augmentation})")
              
        # 创建模型 - 传递固定参数
        model = create_model(trial, num_materials, fixed_params).to(device)
        
        # 打印模型参数总数
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  模型参数总数: {total_params:,}")
    except Exception as e:
        print(f"Trial {trial_num} 初始化模型出错: {str(e)}")
        traceback.print_exc()
        return -1.0


    augmented_train_data = prepare_material_data(
        train_df,
        scaler=train_data["scaler"],
        le=train_data["label_encoder"],
        # augment参数在函数内部未实际使用，保持为False
        augment=False,
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
        drop_last=False,  # 改为False以使用所有数据
        pin_memory=True,  # 加速数据传输
        num_workers=0,  # 单进程加载避免额外开销
    )

    # 创建优化器
    if optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
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

    # 使用 OneCycleLR 学习率调度
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
        lambda_complexity=lambda_complexity,
        lambda_physics=lambda_physics,
        weight_decay=0.0,
    )
    loss_fn.total_epochs = epochs

    # 准备数据 - 可选数据增强
    train_df = reconstruct_df(train_data)
    original_size = len(train_df)
    
    # 显式应用数据增强
    if use_augmentation:
        train_df = augment_dataset(train_df)
        # 记录增强前后的大小变化，便于调试
        if len(train_df) > original_size:
            print(f"  Trial {trial.number}: 应用数据增强 {original_size} -> {len(train_df)} 样本")

    augmented_train_data = prepare_material_data(
        train_df,
        scaler=train_data["scaler"],
        le=train_data["label_encoder"],
        # augment参数在函数内部未实际使用，保持为False
        augment=False,
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
        drop_last=False,  # 改为False以使用所有数据
        pin_memory=True,  # 加速数据传输
        num_workers=0,  # 单进程加载避免额外开销
    )

    val_materials = torch.LongTensor(val_data["material_ids"]).to(device)
    val_raw = torch.FloatTensor(val_data["numeric_raw"]).to(device)
    val_target = torch.FloatTensor(val_data["target"]).unsqueeze(1).to(device)

    # 训练循环
    best_val_r2 = -float("inf")
    patience_counter = 0
    best_model_wts = None

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
                trial.report(-1.0, epoch)
                raise optuna.TrialPruned()

            loss.backward()
            # 更严格的梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            scheduler.step()  # OneCycleLR 需要在每个批次后更新，而不是每个epoch

            epoch_loss += loss.item() * batch_target.size(0)

        epoch_loss /= len(train_dataset)

        # 验证
        model.eval()
        with torch.no_grad():
            val_pred, val_uncertainty, _, _ = model(val_materials, val_raw)
            # val_loss, _ = loss_fn(val_pred, val_uncertainty, val_target, model, val_raw)

            val_pred_np = val_pred.cpu().squeeze().numpy()
            val_target_np = val_target.cpu().squeeze().numpy()

            val_r2 = r2_score(val_target_np, val_pred_np)
            # val_rmse = np.sqrt(mean_squared_error(val_target_np, val_pred_np))
            
            # 门控监控（每10个epoch打印一次）
            if epoch % 10 == 0 and hasattr(model, 'get_gate_statistics'):
                gate_stats = model.get_gate_statistics()
                print(f"Epoch {epoch}: Gate Mean: {gate_stats['mean']:.3f}, "
                      f"Std: {gate_stats['std']:.3f}, "
                      f"Symbolic Weight: {gate_stats['symbolic_weight']:.3f}")

        # 早停检查
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            patience_counter = 0
            # 保存最佳模型状态
            best_model_wts = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

        # 向Optuna报告中间结果
        trial.report(val_r2, epoch)

        # 如果trial应该被剪枝则停止
        if trial.should_prune():
            raise optuna.TrialPruned()

    # 恢复最佳模型状态进行最终评估
    try:
        if best_model_wts is not None:
            model.load_state_dict(best_model_wts)
            model = model.to(device)

            # 再次计算R²
            model.eval()
            with torch.no_grad():
                val_pred, _, _, _ = model(val_materials, val_raw)
                val_pred_np = val_pred.cpu().squeeze().numpy()
                final_r2 = r2_score(val_target_np, val_pred_np)
                
            # 输出训练时间和最终结果
            duration = time.time() - start_time
            print(f"  Trial {trial.number} 完成: 验证R² = {final_r2:.4f}, 用时 {duration:.1f}秒, "
                  f"优化器 = {optimizer_name}, 早停 = {patience}, 学习率 = {lr:.1e}")

            return final_r2
        
        print(f"  Trial {trial.number} 完成但无最佳模型状态, 返回最佳验证R² = {best_val_r2:.4f}")
        return best_val_r2
        
    except Exception as e:
        print(f"  Trial {trial.number} 评估时出错: {str(e)}")
        traceback.print_exc()
        # 如果已经有一个最佳值，返回它，否则返回一个很小的值
        return best_val_r2 if best_val_r2 > -float("inf") else -1.0


def objective(trial, df, device, seed, space=None, fixed_params=None):
    """Optuna目标函数，5折交叉验证，支持第二阶段精细搜索"""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    fold_scores = []
    
    # 初始化固定参数字典
    fixed_params = fixed_params or {}
    
    # 如果提供了搜索空间，根据搜索空间约束创建新的参数
    # 但不重新定义已经固定的参数
    if space is not None:
        # 应用搜索空间约束，只对非固定参数使用suggest API
        for param_name, bounds in space.items():
            # 跳过已经固定的参数
            if param_name in fixed_params:
                continue
                
            if isinstance(bounds, tuple) and len(bounds) == 2:
                if param_name == "lr" or param_name == "weight_decay" or param_name == "lambda_complexity" or param_name == "lambda_physics":
                    # 对数均匀分布
                    trial.suggest_float(param_name, bounds[0], bounds[1], log=True)
                elif param_name == "dropout_rate":
                    # 离散均匀分布
                    trial.suggest_float(param_name, bounds[0], bounds[1], step=0.05)
                elif param_name == "material_embed_dim":
                    # 整数分布，步长为8
                    trial.suggest_int(param_name, int(bounds[0]), int(bounds[1]), step=8)
                elif param_name == "latent_dim":
                    # 整数分布，步长为16
                    trial.suggest_int(param_name, int(bounds[0]), int(bounds[1]), step=16)
                elif param_name == "epochs":
                    # 整数分布，步长为50
                    trial.suggest_int(param_name, int(bounds[0]), int(bounds[1]), step=50)
                elif param_name == "patience":
                    # 整数分布，步长为5
                    trial.suggest_int(param_name, int(bounds[0]), int(bounds[1]), step=5)
                else:
                    # 均匀分布
                    trial.suggest_float(param_name, bounds[0], bounds[1])
            elif isinstance(bounds, list) and param_name not in fixed_params:
                # 只有非固定的类别参数才使用suggest
                trial.suggest_categorical(param_name, bounds)
    
    # 进行5折交叉验证
    for train_idx, val_idx in skf.split(df, df["材料种类"]):
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]

        train_data = prepare_material_data(train_df)
        val_data = prepare_material_data(
            val_df,
            scaler=train_data["scaler"],
            le=train_data["label_encoder"],
        )
        num_materials = train_data["num_materials"]

        # 使用更新后的train_with_trial函数，传递所有固定参数
        r2 = train_with_trial(
            trial, 
            train_data, 
            val_data, 
            num_materials, 
            device,
            fixed_params=fixed_params
        )
        fold_scores.append(r2)

    return float(np.mean(fold_scores))


def run_hyperopt(data_path="3011916164精确材料性能数据.csv", n_trials_stage1=30, n_trials_stage2=20, top_k=5, seed=66, symbolic_type="advanced"):
    """
    运行双阶段超参数优化
    
    参数:
        data_path: 数据文件路径
        n_trials_stage1: 第一阶段试验次数
        n_trials_stage2: 第二阶段试验次数
        top_k: 用于精炼搜索空间的最佳试验数量
        seed: 随机种子
        symbolic_type: 要使用的符号层类型 ("improved", "advanced", "lightweight")
    """
    # 设置随机种子
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 读取数据
    df = pd.read_csv(data_path)
    # num_materials = df["材料种类"].nunique()

    # 设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 设置固定的符号层类型
    print(f"使用指定的符号层类型: {symbolic_type}")

    # 阶段1: 随机搜索 - 广泛探索超参数空间
    print(f"\n开始第1阶段: 初始随机搜索 ({n_trials_stage1}次试验)")
    study1 = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.RandomSampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=30),
    )
    
    # 为所有试验固定符号层类型
    fixed_symbolic_params = {"symbolic_type": symbolic_type}
    
    study1.optimize(
        lambda trial: objective(trial, df, device, seed, fixed_params=fixed_symbolic_params),
        n_trials=n_trials_stage1,
        show_progress_bar=True,
        catch=(Exception,),
    )
    
    # 根据阶段1结果定义收缩后的搜索空间
    completed_trials = [t for t in study1.trials if t.state == TrialState.COMPLETE]
    
    # 打印阶段1结果
    print("\n" + "=" * 50)
    print("阶段1随机搜索完成!")
    print("=" * 50)
    
    # 检查是否有完成的试验
    if not completed_trials:
        print("\n警告: 阶段1没有成功完成的试验。使用默认参数继续...")
        # 使用默认参数定义搜索空间
        default_params = {
            "material_embed_dim": 16,
            "latent_dim": 32,
            "dropout_rate": 0.15,
            "lr": 0.001,
            "epochs": 300,
            "batch_size": 32,
            "weight_decay": 0.001,
            "lambda_complexity": 1e-4,
            "lambda_physics": 0.2,
            "patience": 30,
            "augmentation": True,
            "optimizer": "AdamW",
        }
        # 创建一个虚拟的完成试验
        dummy_trial = optuna.trial.create_trial(
            params=default_params,
            distributions={},
            value=0.0,  # 占位值
            state=TrialState.COMPLETE,
        )
        completed_trials = [dummy_trial]
        space = define_refined_space(completed_trials, 1)
    else:
        space = define_refined_space(completed_trials, top_k)
        print(f"\n阶段1最佳验证R²: {study1.best_value:.4f}")
        print("\n阶段1最佳超参数:")
        for key, value in study1.best_params.items():
            print(f"  {key}: {value}")
    
    print("\n精细搜索空间:")
    for key, value in space.items():
        print(f"  {key}: {value}")

    # 阶段2: TPE微调 - 在有希望的区域进行精细搜索
    # 从第一阶段获取所有最佳参数作为固定值
    fixed_params = {}
    if completed_trials:
        # 所有阶段1的最佳参数都固定在阶段2中
        for param_name, param_value in study1.best_params.items():
            fixed_params[param_name] = param_value
            print(f"第2阶段将使用固定 {param_name} = {param_value}")
    
    # 保存一个单独的固定参数字典供后续使用
    study1._fixed_params = fixed_params.copy() if fixed_params else {}
    
    print(f"\n开始第2阶段: 精细TPE搜索 ({n_trials_stage2}次试验)")
    study2 = optuna.create_study(
        direction="maximize", 
        sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=5),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=15),
    )

    study2.optimize(
        lambda trial: objective(trial, df, device, seed, space, fixed_params),
        n_trials=n_trials_stage2,
        show_progress_bar=True,
        catch=(Exception,),
    )

    # 选择最佳study
    completed_trials1 = len([t for t in study1.trials if t.state == TrialState.COMPLETE])
    completed_trials2 = len([t for t in study2.trials if t.state == TrialState.COMPLETE])
    
    # 检查两个阶段是否有完成的试验
    has_study1_result = completed_trials1 > 0
    has_study2_result = completed_trials2 > 0
    
    if has_study2_result and has_study1_result:
        if study2.best_value > study1.best_value:
            final_study = study2
            print("\n阶段2搜索找到了更好的参数!")
            # 将阶段1的固定参数存入study2
            if hasattr(study1, "_fixed_params"):
                final_study._fixed_params = study1._fixed_params
        else:
            final_study = study1
            print("\n阶段1搜索的参数更优!")
    elif has_study2_result:
        final_study = study2
        print("\n只有阶段2有完成的试验。使用阶段2最佳参数。")
    elif has_study1_result:
        final_study = study1
        print("\n只有阶段1有完成的试验。使用阶段1最佳参数。")
    else:
        # 如果两个阶段都没有完成的试验，创建一个默认study
        print("\n两个阶段都没有完成的试验。使用默认参数。")
        default_study = optuna.create_study(direction="maximize")
        default_trial = optuna.trial.create_trial(
            params={
                "material_embed_dim": 16,
                "latent_dim": 32,
                "dropout_rate": 0.15,
                "lr": 0.001,
                "epochs": 300,
                "batch_size": 32,
                "weight_decay": 0.001,
                "lambda_complexity": 1e-4,
                "lambda_physics": 0.2,
                "patience": 30,
                "augmentation": True,
                "optimizer": "AdamW",
            },
            distributions={},
            value=0.0,
            state=TrialState.COMPLETE,
        )
        default_study.add_trial(default_trial)
        final_study = default_study
    
    # 为了可视化，在final_study中保存两个阶段的study对象
    final_study._study1 = study1
    final_study._study2 = study2
    
    # 合并第一阶段和第二阶段的参数为完整参数集
    complete_params = {}
    
    # 首先添加固定参数（第一阶段的参数）
    if hasattr(study1, "_fixed_params"):
        complete_params.update(study1._fixed_params)
    elif hasattr(study1, "best_params") and study1.best_params:
        complete_params.update(study1.best_params)
    
    # 然后添加第二阶段可能搜索到的参数
    if final_study.best_params:
        complete_params.update(final_study.best_params)
    
    # 将完整参数保存到final_study中的一个特殊属性
    final_study._complete_params = complete_params
        
    # 打印最终结果
    print("\n" + "=" * 50)
    print("双阶段优化完成!")
    print("=" * 50)

    print(f"\n最终最佳验证R²: {final_study.best_value:.4f}")
    print("\n最终最佳超参数:")
    
    # 打印完整的参数集
    if hasattr(final_study, "_complete_params") and final_study._complete_params:
        for key, value in final_study._complete_params.items():
            print(f"  {key}: {value}")
    else:
        # 如果没有完整参数，则打印第二阶段的最佳参数（如果有）
        for key, value in final_study.best_params.items():
            print(f"  {key}: {value}")

    # 统计信息
    completed_trials1 = len([t for t in study1.trials if t.state == TrialState.COMPLETE])
    completed_trials2 = len([t for t in study2.trials if t.state == TrialState.COMPLETE])
    print(f"\n阶段1完成的trials: {completed_trials1}/{n_trials_stage1}")
    print(f"阶段2完成的trials: {completed_trials2}/{n_trials_stage2}")
    print(f"总完成trials: {completed_trials1 + completed_trials2}/{n_trials_stage1 + n_trials_stage2}")

    # 超参数重要性
    try:
        importance = optuna.importance.get_param_importances(final_study)
        print("\n超参数重要性:")
        for key, value in sorted(importance.items(), key=lambda x: x[1], reverse=True):
            print(f"  {key}: {value:.3f}")
    except Exception as e:
        print(f"\n计算超参数重要性时出错: {str(e)}")
    
    # 确保结果目录存在
    os.makedirs("results", exist_ok=True)
    
    # 保存study对象，以便后续可以恢复
    try:
        joblib.dump(final_study, "results/best_hyperopt_study.pkl")
        print("成功保存study对象到results/best_hyperopt_study.pkl")
    except Exception as e:
        print(f"保存study对象时出错: {str(e)}")

    return final_study


# 使用最佳超参数重新训练
def retrain_with_best_params(
    study, data_path="3011916164精确材料性能数据.csv", seed=66, symbolic_type="advanced"
):
    """
    使用最佳超参数重新训练并评估
    
    参数:
        study: Optuna study对象，包含最佳超参数
        data_path: 数据文件路径
        seed: 随机种子
        symbolic_type: 要使用的符号层类型 ("improved", "advanced", "lightweight")
    """

    # 读取数据
    df = pd.read_csv(data_path)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=seed)

    train_data = prepare_material_data(train_df)
    test_data = prepare_material_data(
        test_df,
        scaler=train_data["scaler"],
        le=train_data["label_encoder"],
    )

    # 创建模型
    best_params = study.best_params
    
    # 如果第二阶段全部使用固定参数，那么best_params可能是空的
    # 在这种情况下，我们从study.trials中获取最佳trial的值
    if not best_params and study.trials:
        best_trial = study.best_trial
        # 使用_values属性获取固定参数
        fixed_values = best_trial._values
        # 检查用户属性来获取架构参数
        if hasattr(best_trial, "user_attrs") and best_trial.user_attrs:
            for key, value in best_trial.user_attrs.items():
                if key not in best_params:
                    best_params[key] = value
        
        # 固定参数
        fixed_params = {
            "material_embed_dim": 8,
            "latent_dim": 64,
            "dropout_rate": 0.4
        }
        best_params.update(fixed_params)
    
    model = MaterialCrackPGNN(
        num_materials=train_data["num_materials"],
        material_embed_dim=best_params.get("material_embed_dim", 16),  # 提供默认值
        latent_dim=best_params.get("latent_dim", 32),  # 提供默认值
        dropout_rate=best_params.get("dropout_rate", 0.3),
        symbolic_type=symbolic_type  # 使用传入的符号层类型参数
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # 使用最佳参数准备增强数据
    use_augmentation = best_params.get("augmentation", True)  # 默认使用数据增强
    train_df_for_aug = train_df.copy()
    
    # 显式应用数据增强（如果启用）
    if use_augmentation:
        print("应用数据增强...")
        train_df_for_aug = augment_dataset(train_df_for_aug)
        print(f"数据集大小: {len(train_df)} -> {len(train_df_for_aug)} (增加 {len(train_df_for_aug)-len(train_df)} 个样本)")
    
    augmented_train_data = prepare_material_data(
        train_df_for_aug,  # 使用可能已增强的数据
        scaler=train_data["scaler"],
        le=train_data["label_encoder"],
        # augment参数在函数内部未使用，保持为False
        augment=False,
    )

    # 创建DataLoader
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
    lr = best_params.get("lr", 1.5e-3)  # 提供默认学习率
    optimizer_name = best_params.get("optimizer", "AdamW")
    if optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=best_params.get("weight_decay", 8.9e-3)
        )
    elif optimizer_name == "RMSprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=best_params.get("weight_decay", 8.9e-3))
    elif optimizer_name == "RAdam":
        optimizer = torch.optim.RAdam(model.parameters(), lr=lr, weight_decay=best_params.get("weight_decay", 8.9e-3))
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=best_params.get("weight_decay", 8.9e-3))

    # 学习率调度
    epochs = best_params.get("epochs", 400)  # 提供默认训练轮数
    
    # 使用 OneCycleLR 学习率调度
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
        lambda_complexity=best_params.get("lambda_complexity", 1.2e-5),  # 提供默认复杂度权重
        lambda_physics=best_params.get("lambda_physics", 0.2),  # 提供默认物理权重
        weight_decay=0.0,
    )
    loss_fn.total_epochs = epochs

    # 训练
    print("\n使用最佳超参数重新训练...")
    train_losses = []
    test_r2_history = []
    best_model_wts, best_test_r2, patience_ctr = None, -1e9, 0
    best_epoch = 0
    
    # 使用搜索到的耐心值
    patience = best_params.get("patience", 25)
    print(f"早停耐心值: {patience}")

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

            pred, uncertainty, _, _ = model(batch_materials, batch_raw)
            loss, metrics = loss_fn(pred, uncertainty, batch_target, model, batch_raw)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            scheduler.step()  # OneCycleLR 需要在每个批次后更新，而不是每个epoch

            epoch_loss += loss.item() * batch_target.size(0)

        epoch_loss /= len(train_dataset)
        train_losses.append(epoch_loss)

        current_lr = scheduler.get_last_lr()[0]

        # 评估
        model.eval()
        with torch.no_grad():
            test_pred, test_uncertainty, _, _ = model(test_materials, test_raw)
            test_pred_np = test_pred.cpu().squeeze().numpy()
            test_target_np = test_target.cpu().squeeze().numpy()

            test_r2 = r2_score(test_target_np, test_pred_np)
            test_rmse = np.sqrt(mean_squared_error(test_target_np, test_pred_np))
            test_r2_history.append(test_r2)

        if (epoch + 1) % 20 == 0:
            print(
                f"Epoch {epoch+1}: LR={current_lr:.6f}, Test R²={test_r2:.4f}, RMSE={test_rmse:.4f}"
            )
            # 门控监控
            if hasattr(model, 'get_gate_statistics'):
                gate_stats = model.get_gate_statistics()
                print(f"         Gate Mean: {gate_stats['mean']:.3f}, "
                      f"Std: {gate_stats['std']:.3f}, "
                      f"Symbolic Weight: {gate_stats['symbolic_weight']:.3f}")

        if test_r2 > best_test_r2:
            best_test_r2 = test_r2
            best_model_wts = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            patience_ctr = 0
        else:
            patience_ctr += 1

        if patience_ctr >= patience:
            print(f"Early-stop triggered at epoch {epoch+1} with patience {patience}.")
            break

    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)

    # 最终评估 (标准和MC Dropout)
    model.eval()

    # 标准预测
    with torch.no_grad():
        test_pred, test_uncertainty, _, _ = model(test_materials, test_raw)
        test_pred_np = test_pred.cpu().squeeze().numpy()
        test_target_np = test_target.cpu().squeeze().numpy()

        final_r2 = r2_score(test_target_np, test_pred_np)
        final_rmse = np.sqrt(mean_squared_error(test_target_np, test_pred_np))

    # MC Dropout预测
    mc_pred, mc_unc, alea_unc, epis_unc = model.predict_with_uncertainty(
        test_materials, test_raw, n_samples=20
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
    test_raw = torch.FloatTensor(test_data["numeric_raw"]).to(device)
    test_target = torch.FloatTensor(test_data["target"]).unsqueeze(1).to(device)

    # 标准预测
    with torch.no_grad():
        test_pred, test_uncertainty, symbolic_out, residual_out = model(
            test_materials, test_raw
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

    # MC Dropout预测
    mc_pred, mc_unc, aleatoric_unc, epistemic_unc = model.predict_with_uncertainty(
        test_materials, test_raw, n_samples=30
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

    # 计算每个样本的可靠度 (不确定性校准)
    errors = np.abs(mc_pred_np - test_target_np)
    calibration_scores = errors / (np.sqrt(mc_unc_np) + 1e-6)
    well_calibrated = np.mean((calibration_scores <= 2.0))

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


def define_refined_space(completed_trials, top_k):
    """根据第一阶段结果定义精细搜索空间"""
    # 默认边界 - 调整以提高泛化性能
    global_bounds = {
        "material_embed_dim": (8, 32),  # 降低上限以简化模型
        "latent_dim": (16, 64),         # 降低上限以简化模型
        "dropout_rate": (0.1, 0.3),     # 提高范围以增强正则化
        "lr": (5e-4, 3e-3),
        "epochs": (200, 500),
        "batch_size": [16, 32, 64],
        "weight_decay": (1e-4, 5e-3),   # 提高范围以增强正则化
        "lambda_complexity": (1e-5, 1e-3), # 提高范围以增强符号表达式约束
        "lambda_physics": (0.1, 0.5),      # 提高范围以增强物理约束
        "patience": (20, 40),
        "augmentation": [True, False],
        "optimizer": ["AdamW", "RMSprop", "Adam", "RAdam", "SGD", "NAdam"],
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


def log_message(message, log_file=None):
    """记录带时间戳的消息，可选写入日志文件"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"[{timestamp}] {message}"
    print(log_msg)
    
    if log_file:
        with open(log_file, 'a', encoding='utf-8') as f:  # 添加utf-8编码
            f.write(log_msg + "\n")


if __name__ == "__main__":
    # 确保结果目录存在
    os.makedirs("results", exist_ok=True)

    # 设置日志文件
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"results/hyperopt_run_{timestamp}.log"

    # 重定向所有print输出到日志文件和控制台
    logger = Logger(log_file)
    sys.stdout = logger
    sys.stderr = logger

    print(f"开始执行双阶段超参数优化 - {timestamp}")

    # 检查CUDA可用性
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        device_name = torch.cuda.get_device_name(0)
        print(f"CUDA可用: {device_name}")
        torch.backends.cudnn.benchmark = True
    else:
        print("CUDA不可用，将使用CPU（训练可能较慢）")

    # 设置不同调试级别的试验次数
    debug_mode = False  # 切换调试模式
    if debug_mode:
        n_trials_stage1 = 2
        n_trials_stage2 = 1
        top_k = 1
        print("[警告] 调试模式: 仅运行少量试验")
    else:
        n_trials_stage1 = 10
        n_trials_stage2 = 5
        top_k = 5
        print("正式模式: 运行完整优化过程")

    try:
        # 运行双阶段超参数优化
        print(f"开始双阶段超参数优化 (阶段1: {n_trials_stage1}次, 阶段2: {n_trials_stage2}次, 前{top_k}个最佳结果)")
        study = run_hyperopt(n_trials_stage1=n_trials_stage1, n_trials_stage2=n_trials_stage2, top_k=top_k)
    except Exception as e:
        print(f"超参数优化过程出错: {str(e)}")
        print(f"错误详情: {traceback.format_exc()}")
        # 尝试加载之前的结果（如果有）
        try:
            if os.path.exists("results/best_hyperopt_study.pkl"):
                study = joblib.load("results/best_hyperopt_study.pkl")
                print("已加载之前保存的study对象继续运行")
            else:
                raise FileNotFoundError("未找到已保存的study文件")
        except Exception as load_err:
            print(f"无法加载之前的study对象: {str(load_err)}，将创建默认参数")
            # 创建一个默认study用于继续运行
            study = optuna.create_study(direction="maximize")
            default_params = {
                "material_embed_dim": 16,
                "latent_dim": 32,
                "dropout_rate": 0.2,
                "lr": 1e-3,
                "epochs": 300,
                "batch_size": 32,
                "weight_decay": 1e-3,
                "lambda_complexity": 1e-4,
                "lambda_physics": 0.2,
                "patience": 30,
                "augmentation": True,
                "optimizer": "AdamW"
            }
            print("使用默认参数:")
            for k, v in default_params.items():
                print(f"  {k}: {v}")
            study.enqueue_trial(default_params)

    try:
        # 使用最佳参数重新训练
        print("\n开始使用最佳超参数重新训练模型...")
        # 获取所使用的符号层类型（如果有），否则使用 "lightweight"
        symbolic_type = study.best_params.get("symbolic_type", "advanced")
        print(f"使用符号层类型: {symbolic_type}")
        model, train_losses, test_r2_history = retrain_with_best_params(study, symbolic_type=symbolic_type)

        # 详细评估
        print("\n开始详细评估模型...")
        df = pd.read_csv("3011916164精确材料性能数据.csv")
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=66)
        train_data = prepare_material_data(train_df)
        test_data = prepare_material_data(
            test_df,
            scaler=train_data["scaler"],
            le=train_data["label_encoder"],
        )

        results = evaluate_model_detailed(model, test_data)

        # 保存模型
        try:
            torch.save(model.state_dict(), "results/select_model.pth")
            print("模型已保存至 results/select_model.pth")

            # 保存训练历史
            # 获取完整参数集
            if hasattr(study, "_complete_params") and study._complete_params:
                complete_params = study._complete_params
            else:
                # 尝试合并第一阶段参数
                complete_params = {}
                if hasattr(study, "_study1") and hasattr(study._study1, "best_params"):
                    complete_params.update(study._study1.best_params)
                if study.best_params:
                    complete_params.update(study.best_params)
                if not complete_params:
                    complete_params = study.best_params or {}
            
            history_dict = {
                "train_losses": train_losses,
                "test_r2_history": test_r2_history,
                "best_params": complete_params,  # 使用完整参数
                "best_value": study.best_value,
            }
            joblib.dump(history_dict, "results/training_history.pkl")
            print("训练历史已保存至 results/training_history.pkl")

        except Exception as save_err:
            print(f"保存模型时出错: {str(save_err)}")

    except Exception as e:
        print(f"训练或评估过程中出错: {str(e)}")
        print(f"错误详情: {traceback.format_exc()}")
        model = None
        train_losses = []
        test_r2_history = []
        results = None

    try:
        print("\n生成可视化图表...")
        # 可视化优化历史 - 双阶段
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # 优化历史 - 分阶段显示
        if hasattr(study, "_study1") and hasattr(study, "_study2"):
            # 如果study对象有保存两个阶段的study
            values1 = [t.value for t in study._study1.trials if t.value is not None]
            values2 = [t.value for t in study._study2.trials if t.value is not None]

            if values1:  # 只有当有值时才绘制阶段1
                ax1.plot(range(len(values1)), values1, 'b-', alpha=0.7, label='阶段1 - 随机搜索')

            if values2:  # 只有当有值时才绘制阶段2
                ax1.plot(range(len(values1), len(values1) + len(values2)), values2, 'r-', alpha=0.7, label='阶段2 - TPE精细搜索')

            if values1 or values2:  # 只有当至少有一个阶段有值时才添加分隔线和图例
                if values1:  # 只有当阶段1有值时才添加分隔线
                    ax1.axvline(x=len(values1), color='k', linestyle='--', alpha=0.5)
                ax1.legend(prop=chinese_font_prop)
        else:
            # 兼容旧版本
            values = [t.value for t in study.trials if t.value is not None]
            if values:  # 只有当有值时才绘制
                ax1.plot(values)
            else:
                ax1.text(0.5, 0.5, '没有有效的优化结果', horizontalalignment='center', verticalalignment='center')

        ax1.set_xlabel("Trial")
        ax1.set_ylabel("Validation R²")
        ax1.set_title("双阶段超参数优化历史", fontproperties=chinese_font_prop)
        ax1.grid(True, alpha=0.3)
        
        # 为图例设置中文字体
        legend = ax1.get_legend()
        if legend:
            for text in legend.get_texts():
                text.set_fontproperties(chinese_font_prop)

        # 训练损失
        if train_losses:
            ax2.plot(train_losses)
            ax2.set_yscale("log")
        else:
            ax2.text(0.5, 0.5, '没有训练损失数据', horizontalalignment='center', verticalalignment='center')

        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.set_title("最佳超参数训练损失", fontproperties=chinese_font_prop)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("results/improved_hyperopt_results.png", dpi=300)
        plt.close(fig)

        if results:  # 只有在有结果时才可视化预测
            # 预测vs真实值可视化
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            # 标准预测
            ax1.scatter(
                results["predictions"]["y_true"], results["predictions"]["y_pred"], alpha=0.6
            )
            ax1.plot([0, 10], [0, 10], "r--")  # 理想线
            ax1.set_xlabel("实际裂纹深度", fontproperties=chinese_font_prop)
            ax1.set_ylabel("预测裂纹深度", fontproperties=chinese_font_prop)
            ax1.set_title(f'标准预测 (R² = {results["standard"]["r2"]:.4f})', fontproperties=chinese_font_prop)
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
            ax2.set_xlabel("实际裂纹深度", fontproperties=chinese_font_prop)
            ax2.set_ylabel("MC 预测", fontproperties=chinese_font_prop)
            ax2.set_title(f'MC Dropout 预测 (R² = {results["mc_dropout"]["r2"]:.4f})', fontproperties=chinese_font_prop)
            ax2.grid(True, alpha=0.3)
            cbar = plt.colorbar(scatter, ax=ax2)
            cbar.set_label("预测不确定性", fontproperties=chinese_font_prop)

            plt.tight_layout()
            plt.savefig("results/predictions_visualization.png", dpi=300)
            plt.close(fig)

            # 不确定性可视化
            try:
                fig, ax = plt.subplots(figsize=(10, 6))

                # 按误差大小排序
                errors = np.abs(results["predictions"]["y_mc_pred"] - results["predictions"]["y_true"])
                sorted_indices = np.argsort(errors)

                ax.errorbar(
                    range(len(sorted_indices)),
                    results["predictions"]["y_mc_pred"][sorted_indices],
                    yerr=np.sqrt(results["predictions"]["mc_uncertainty"][sorted_indices]),
                    fmt='o',
                    alpha=0.6,
                    ecolor='r',
                    capsize=2
                )
                ax.scatter(
                    range(len(sorted_indices)),
                    results["predictions"]["y_true"][sorted_indices],
                    marker='x',
                    color='k',
                    label='实际值'
                )

                ax.set_xlabel('样本索引 (按误差排序)', fontproperties=chinese_font_prop)
                ax.set_ylabel('裂纹深度', fontproperties=chinese_font_prop)
                ax.set_title('预测值及不确定性区间', fontproperties=chinese_font_prop)
                ax.legend(prop=chinese_font_prop)
                ax.grid(True, alpha=0.3)

                plt.tight_layout()
                plt.savefig("results/uncertainty_visualization.png", dpi=300)
                plt.close(fig)
            except Exception as e:
                print(f"绘制不确定性可视化时出错: {str(e)}")

        print("可视化图表已保存到results文件夹")

    except Exception as e:
        print(f"生成可视化过程出错: {str(e)}")
        print(f"错误详情: {traceback.format_exc()}")

    print("\n优化和训练过程完成!")

    # 运行结束时间
    end_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"运行结束时间: {end_timestamp}")

    # 关闭日志文件
    logger.close()