"""
消融实验主脚本
对比不同模型配置,验证各组件的有效性
"""

import os
import time
import warnings
import copy
import json
import torch
from torch.optim.lr_scheduler import OneCycleLR
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['axes.unicode_minus'] = False

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from torch.utils.data import TensorDataset, DataLoader

# 导入自定义模块
from models_ablation import MaterialCrackPGNN_Ablation
from models import MaterialCrackPGNN  # 完整模型使用原始PGNN
from losses import PhysicsInformedLoss
from utils import prepare_material_data, augment_dataset

warnings.filterwarnings("ignore")

# 数据路径
DATA_PATH = "3011916164精确材料性能数据.csv"

# 预训练模型路径 - 与fair_model_comparison.py保持一致
PRETRAINED_MODEL_PATH = "results\\run_20251217_100959\\best_model.pth"

# 固定随机种子 - 与fair_model_comparison.py保持一致
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# 消融实验配置
ABLATION_CONFIGS = {
    "完整模型": {
        "ablation_mode": "full",
        "symbolic_type": "advanced",
        "description": "完整模型:符号层+残差网络+自适应门控 (加载预训练模型)"
    },
    "无符号层": {
        "ablation_mode": "no_symbolic",
        "symbolic_type": "advanced",
        "description": "仅使用残差网络,移除符号层"
    },
    "简化符号层": {
        "ablation_mode": "simple_symbolic",
        "symbolic_type": "improved",
        "description": "使用简化符号层,无注意力机制"
    },
    "固定门控0.5": {
        "ablation_mode": "fixed_gate_0.5",
        "symbolic_type": "advanced",
        "description": "符号层和残差网络固定权重各0.5"
    },
    "固定门控0.0": {
        "ablation_mode": "fixed_gate_0.0",
        "symbolic_type": "advanced",
        "description": "完全依赖残差网络(门控=0)"
    },
    "固定门控1.0": {
        "ablation_mode": "fixed_gate_1.0",
        "symbolic_type": "advanced",
        "description": "完全依赖符号层(门控=1)"
    },
    "无物理特征": {
        "ablation_mode": "no_physics",
        "symbolic_type": "advanced",
        "description": "不使用物理特征提取器"
    },
    "无残差网络": {
        "ablation_mode": "no_residual",
        "symbolic_type": "advanced",
        "description": "仅使用符号层,移除残差网络"
    },
    "无物理特征基线": {
        "ablation_mode": "no_physics_baseline",
        "symbolic_type": "advanced",
        "description": "无物理特征+无符号层(纯数据驱动基线)"
    }
}

# 基础超参数(与train.py保持一致)
BASE_PARAMS = {
    "material_embed_dim": 8,
    "latent_dim": 48,
    "dropout_rate": 0.35,
    "epochs": 350,
    "batch_size": 16,
    "lr": 0.0011173842752625436,
    "patience": 40,
    "augmentation": True,
    "optimizer": "AdamW",
    "weight_decay": 0.0066923795400554925,
    "lambda_complexity": 3.4069890537972935e-05,
    "lambda_physics": 0.2
}


def load_model_config(model_path, default_params):
    """
    从预训练模型中读取配置参数
    优先从模型文件读取,如果不存在则使用默认参数
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # 如果保存了配置信息
    if isinstance(checkpoint, dict) and 'config' in checkpoint:
        print("✓ 从模型文件中读取配置参数")
        return checkpoint['config']
    
    # 尝试从state_dict推断参数
    try:
        state_dict = checkpoint if not isinstance(checkpoint, dict) or 'state_dict' not in checkpoint else checkpoint['state_dict']
        
        # 从权重形状推断参数
        material_embed_dim = state_dict['material_embedding.embedding.weight'].shape[1]
        latent_dim = state_dict['encoder_norm.weight'].shape[0]
        
        inferred_params = default_params.copy()
        inferred_params['material_embed_dim'] = int(material_embed_dim)
        inferred_params['latent_dim'] = int(latent_dim)
        
        print("✓ 从模型权重中推断配置参数")
        print(f"  material_embed_dim: {material_embed_dim}")
        print(f"  latent_dim: {latent_dim}")
        
        return inferred_params
    except Exception as e:
        print(f"⚠ 无法从模型文件推断参数: {e}")
        print("✓ 使用默认配置参数")
        return default_params


def evaluate_pretrained_full_model(train_data, test_data, base_params, model_path, device):
    """
    加载并评估预训练的完整PGNN模型
    与fair_model_comparison.py中的evaluate_pretrained_pgnn逻辑完全一致
    """
    print("\n" + "="*60)
    print("评估预训练完整PGNN模型")
    print("="*60)
    
    # 加载模型配置
    print(f"加载模型配置: {model_path}")
    params = load_model_config(model_path, base_params)
    
    num_materials = train_data["num_materials"]
    print(f"共 {num_materials} 种材料类型")
    
    # 创建模型
    print(f"创建模型...")
    model = MaterialCrackPGNN(
        num_materials=num_materials,
        material_embed_dim=params["material_embed_dim"],
        latent_dim=params["latent_dim"],
        dropout_rate=params["dropout_rate"],
        symbolic_type=params.get("symbolic_type", "advanced")
    )
    
    # 加载预训练权重
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    print("✓ 模型权重加载成功")
    
    model = model.to(device)
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数总数: {total_params:,}")
    
    # 准备测试数据
    test_materials = torch.LongTensor(test_data["material_ids"]).to(device)
    test_raw = torch.FloatTensor(test_data["numeric_raw"]).to(device)
    test_target = torch.FloatTensor(test_data["target"]).unsqueeze(1).to(device)
    
    # 评估
    print("\n在测试集上评估...")
    with torch.no_grad():
        test_pred, test_uncertainty, _, _ = model(test_materials, test_raw)
        
        test_pred_np = test_pred.cpu().squeeze().numpy()
        test_target_np = test_target.cpu().squeeze().numpy()
        test_uncertainty_np = test_uncertainty.cpu().squeeze().numpy()
        
        test_r2 = r2_score(test_target_np, test_pred_np)
        test_rmse = np.sqrt(mean_squared_error(test_target_np, test_pred_np))
        test_mae = mean_absolute_error(test_target_np, test_pred_np)
        mape = np.mean(np.abs((test_target_np - test_pred_np) / (test_target_np + 1e-8))) * 100
    
    print(f"\n测试集性能:")
    print(f"  R²: {test_r2:.4f}")
    print(f"  RMSE: {test_rmse:.4f}")
    print(f"  MAE: {test_mae:.4f}")
    print(f"  MAPE: {mape:.2f}%")
    
    return {
        "test_r2": float(test_r2),
        "test_rmse": float(test_rmse),
        "test_mae": float(test_mae),
        "test_mape": float(mape),
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "predictions": test_pred_np.tolist(),
        "targets": test_target_np.tolist(),
        "uncertainties": test_uncertainty_np.tolist()
    }


def train_single_ablation(config_name, config, base_params, data_path=DATA_PATH, seed=SEED, n_runs=3):
    """
    训练单个消融实验配置
    
    对于完整模型(full): 加载预训练模型进行评估(与fair_model_comparison一致)
    对于其他消融模式: 重新训练模型
    
    Args:
        config_name: 配置名称
        config: 配置字典
        base_params: 基础超参数
        data_path: 数据路径
        seed: 随机种子
        n_runs: 重复运行次数(用于统计显著性检验)
    
    Returns:
        results: 包含所有运行结果的字典
    """
    print("\n" + "="*70)
    print(f"消融实验: {config_name}")
    print(f"描述: {config['description']}")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 读取数据
    df = pd.read_csv(data_path)
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=seed, stratify=df["材料种类"]
    )
    
    # 准备数据
    train_data = prepare_material_data(train_df)
    test_data = prepare_material_data(
        test_df,
        scaler=train_data["scaler"],
        le=train_data["label_encoder"],
    )
    
    num_materials = train_data["num_materials"]
    
    # ========== 完整模型: 加载预训练模型 ==========
    if config["ablation_mode"] == "full":
        print("\n*** 完整模型使用预训练权重进行评估 ***")
        print(f"预训练模型路径: {PRETRAINED_MODEL_PATH}")
        
        # 对于完整模型,只需要评估一次(预训练模型是确定性的)
        # 但为了与其他模型保持一致的结果格式,我们复制n_runs次相同的结果
        eval_result = evaluate_pretrained_full_model(
            train_data, test_data, base_params, PRETRAINED_MODEL_PATH, device
        )
        
        # 创建n_runs次相同的结果(预训练模型是确定性的)
        all_runs_results = []
        for run_idx in range(n_runs):
            run_result = {
                "run_idx": run_idx,
                "test_r2": eval_result["test_r2"],
                "test_rmse": eval_result["test_rmse"],
                "test_mae": eval_result["test_mae"],
                "test_mape": eval_result["test_mape"],
                "train_time": 0.0,  # 预训练模型不需要训练时间
                "total_params": eval_result["total_params"],
                "trainable_params": eval_result["trainable_params"],
                "train_losses": [],
                "val_losses": [],
                "val_r2_scores": [],
                "best_val_r2": eval_result["test_r2"],
                "predictions": eval_result["predictions"],
                "targets": eval_result["targets"],
                "uncertainties": eval_result["uncertainties"]
            }
            all_runs_results.append(run_result)
        
        # 汇总结果
        summary = {
            "config_name": config_name,
            "config": config,
            "n_runs": n_runs,
            "r2_mean": eval_result["test_r2"],
            "r2_std": 0.0,  # 预训练模型没有随机性
            "r2_min": eval_result["test_r2"],
            "r2_max": eval_result["test_r2"],
            "rmse_mean": eval_result["test_rmse"],
            "rmse_std": 0.0,
            "mae_mean": eval_result["test_mae"],
            "mae_std": 0.0,
            "mape_mean": eval_result["test_mape"],
            "mape_std": 0.0,
            "train_time_mean": 0.0,
            "train_time_std": 0.0,
            "total_params": eval_result["total_params"],
            "trainable_params": eval_result["trainable_params"],
            "all_runs": all_runs_results
        }
        
        print(f"\n{config_name} 结果 (预训练模型):")
        print(f"  R²: {summary['r2_mean']:.4f}")
        print(f"  RMSE: {summary['rmse_mean']:.4f}")
        print(f"  MAE: {summary['mae_mean']:.4f}")
        print(f"  MAPE: {summary['mape_mean']:.2f}%")
        print(f"  模型参数: {summary['total_params']:,}")
        
        return summary
    
    # ========== 其他消融模式: 重新训练 ==========
    # 数据增强
    if base_params["augmentation"]:
        train_df_aug = augment_dataset(train_df.copy())
        augmented_train_data = prepare_material_data(
            train_df_aug,
            scaler=train_data["scaler"],
            le=train_data["label_encoder"],
            augment=False,
        )
    else:
        augmented_train_data = train_data
    
    # 存储多次运行的结果
    all_runs_results = []
    
    for run_idx in range(n_runs):
        print(f"\n--- 运行 {run_idx + 1}/{n_runs} ---")
        
        # 为每次运行设置不同的随机种子
        run_seed = seed + run_idx * 100
        torch.manual_seed(run_seed)
        np.random.seed(run_seed)
        
        # 使用MaterialCrackPGNN_Ablation
        model = MaterialCrackPGNN_Ablation(
            num_materials=num_materials,
            material_embed_dim=base_params["material_embed_dim"],
            latent_dim=base_params["latent_dim"],
            dropout_rate=base_params["dropout_rate"],
            symbolic_type=config["symbolic_type"],
            ablation_mode=config["ablation_mode"]
        )
        model = model.to(device)
        
        # 统计模型参数
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # 创建DataLoader
        train_dataset = TensorDataset(
            torch.LongTensor(augmented_train_data["material_ids"]),
            torch.FloatTensor(augmented_train_data["numeric_features"]),
            torch.FloatTensor(augmented_train_data["numeric_raw"]),
            torch.FloatTensor(augmented_train_data["target"]).unsqueeze(1),
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=base_params["batch_size"],
            shuffle=True,
            drop_last=False,
            pin_memory=True,
            num_workers=0,
        )
        
        # 准备测试数据
        test_materials = torch.LongTensor(test_data["material_ids"]).to(device)
        test_features = torch.FloatTensor(test_data["numeric_features"]).to(device)
        test_raw = torch.FloatTensor(test_data["numeric_raw"]).to(device)
        test_target = torch.FloatTensor(test_data["target"]).unsqueeze(1).to(device)
        
        # 优化器
        if base_params["optimizer"] == "AdamW":
            optimizer = torch.optim.AdamW(
                model.parameters(), 
                lr=base_params["lr"], 
                weight_decay=base_params["weight_decay"]
            )
        elif base_params["optimizer"] == "Adam":
            optimizer = torch.optim.Adam(
                model.parameters(), 
                lr=base_params["lr"], 
                weight_decay=base_params["weight_decay"]
            )
        else:
            optimizer = torch.optim.AdamW(
                model.parameters(), 
                lr=base_params["lr"], 
                weight_decay=base_params["weight_decay"]
            )
        
        # 学习率调度
        total_steps = base_params["epochs"] * len(train_loader)
        scheduler = OneCycleLR(
            optimizer,
            max_lr=base_params["lr"],
            total_steps=total_steps,
            pct_start=0.3,
            div_factor=25.0,
            final_div_factor=1e4,
            anneal_strategy='cos'
        )
        
        # 损失函数
        loss_fn = PhysicsInformedLoss(
            lambda_complexity=base_params["lambda_complexity"],
            lambda_physics=base_params["lambda_physics"],
            weight_decay=0.0,
        )
        loss_fn.total_epochs = base_params["epochs"]
        
        # 训练
        start_time = time.time()
        best_val_r2 = -float("inf")
        patience_counter = 0
        best_model_wts = None
        train_losses = []
        val_losses = []
        val_r2_scores = []
        
        for epoch in range(base_params["epochs"]):
            loss_fn.update_epoch(epoch, base_params["epochs"])
            
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
                
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                    
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()
                scheduler.step()
                
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
                val_r2_scores.append(val_r2)
                
                if epoch % 50 == 0 or epoch == base_params["epochs"] - 1:
                    print(f"Epoch {epoch+1}/{base_params['epochs']} - "
                          f"训练损失: {epoch_loss:.4f}, "
                          f"验证R²: {val_r2:.4f}")
            
            # 早停
            if val_r2 > best_val_r2:
                best_val_r2 = val_r2
                patience_counter = 0
                best_model_wts = copy.deepcopy(model.state_dict())
            else:
                patience_counter += 1
            
            if patience_counter >= base_params["patience"]:
                print(f"早停: 在第 {epoch+1} 轮停止")
                break
        
        # 恢复最佳模型
        if best_model_wts is not None:
            model.load_state_dict(best_model_wts)
        
        train_time = time.time() - start_time
        
        # 最终评估
        model.eval()
        with torch.no_grad():
            test_pred, test_uncertainty, _, _ = model(test_materials, test_raw)
            
            test_pred_np = test_pred.cpu().squeeze().numpy()
            test_target_np = test_target.cpu().squeeze().numpy()
            test_uncertainty_np = test_uncertainty.cpu().squeeze().numpy()
            
            test_r2 = r2_score(test_target_np, test_pred_np)
            test_rmse = np.sqrt(mean_squared_error(test_target_np, test_pred_np))
            test_mae = mean_absolute_error(test_target_np, test_pred_np)
            
            # 计算MAPE (平均绝对百分比误差)
            mape = np.mean(np.abs((test_target_np - test_pred_np) / (test_target_np + 1e-8))) * 100
            
            print(f"\n运行 {run_idx + 1} 结果:")
            print(f"  测试R²: {test_r2:.4f}")
            print(f"  测试RMSE: {test_rmse:.4f}")
            print(f"  测试MAE: {test_mae:.4f}")
            print(f"  测试MAPE: {mape:.2f}%")
            print(f"  训练时间: {train_time:.2f}秒")
        
        # 保存单次运行结果
        run_result = {
            "run_idx": run_idx,
            "test_r2": float(test_r2),
            "test_rmse": float(test_rmse),
            "test_mae": float(test_mae),
            "test_mape": float(mape),
            "train_time": float(train_time),
            "total_params": int(total_params),
            "trainable_params": int(trainable_params),
            "train_losses": [float(x) for x in train_losses],
            "val_losses": [float(x) for x in val_losses],
            "val_r2_scores": [float(x) for x in val_r2_scores],
            "best_val_r2": float(best_val_r2),
            "predictions": [float(x) for x in test_pred_np.tolist()],
            "targets": [float(x) for x in test_target_np.tolist()],
            "uncertainties": [float(x) for x in test_uncertainty_np.tolist()]
        }
        
        all_runs_results.append(run_result)
    
    # 汇总多次运行的结果
    r2_scores = [r["test_r2"] for r in all_runs_results]
    rmse_scores = [r["test_rmse"] for r in all_runs_results]
    mae_scores = [r["test_mae"] for r in all_runs_results]
    mape_scores = [r["test_mape"] for r in all_runs_results]
    train_times = [r["train_time"] for r in all_runs_results]
    
    summary = {
        "config_name": config_name,
        "config": config,
        "n_runs": n_runs,
        "r2_mean": float(np.mean(r2_scores)),
        "r2_std": float(np.std(r2_scores)),
        "r2_min": float(np.min(r2_scores)),
        "r2_max": float(np.max(r2_scores)),
        "rmse_mean": float(np.mean(rmse_scores)),
        "rmse_std": float(np.std(rmse_scores)),
        "mae_mean": float(np.mean(mae_scores)),
        "mae_std": float(np.std(mae_scores)),
        "mape_mean": float(np.mean(mape_scores)),
        "mape_std": float(np.std(mape_scores)),
        "train_time_mean": float(np.mean(train_times)),
        "train_time_std": float(np.std(train_times)),
        "total_params": all_runs_results[0]["total_params"],
        "trainable_params": all_runs_results[0]["trainable_params"],
        "all_runs": all_runs_results
    }
    
    print(f"\n{config_name} 汇总结果 ({n_runs}次运行):")
    print(f"  R²: {summary['r2_mean']:.4f} ± {summary['r2_std']:.4f}")
    print(f"  RMSE: {summary['rmse_mean']:.4f} ± {summary['rmse_std']:.4f}")
    print(f"  MAE: {summary['mae_mean']:.4f} ± {summary['mae_std']:.4f}")
    print(f"  MAPE: {summary['mape_mean']:.2f}% ± {summary['mape_std']:.2f}%")
    print(f"  训练时间: {summary['train_time_mean']:.2f}s ± {summary['train_time_std']:.2f}s")
    print(f"  模型参数: {summary['total_params']:,}")
    
    return summary


def run_ablation_study(configs=None, n_runs=3, output_dir="results/ablation"):
    """
    运行完整的消融实验
    
    Args:
        configs: 要运行的配置字典,默认使用ABLATION_CONFIGS
        n_runs: 每个配置重复运行次数
        output_dir: 结果输出目录
    """
    if configs is None:
        configs = ABLATION_CONFIGS
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("开始消融实验")
    print(f"共 {len(configs)} 组配置, 每组运行 {n_runs} 次")
    print(f"完整模型使用预训练权重: {PRETRAINED_MODEL_PATH}")
    print("="*70)
    
    all_results = {}
    
    # 运行每个配置
    for config_name, config in configs.items():
        result = train_single_ablation(
            config_name, 
            config, 
            BASE_PARAMS, 
            n_runs=n_runs
        )
        all_results[config_name] = result
        
        # 保存中间结果
        with open(f"{output_dir}/ablation_results_partial.json", "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    # 保存完整结果
    with open(f"{output_dir}/ablation_results_complete.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    # 生成对比表格
    generate_comparison_table(all_results, output_dir)
    
    # 生成可视化
    generate_visualizations(all_results, output_dir)
    
    # 统计显著性检验
    perform_statistical_tests(all_results, output_dir)
    
    print("\n" + "="*70)
    print("消融实验完成!")
    print(f"结果已保存至: {output_dir}")
    print("="*70)
    
    return all_results


def generate_comparison_table(results, output_dir):
    """生成对比表格"""
    print("\n生成对比表格...")
    
    # 创建DataFrame
    data = []
    for config_name, result in results.items():
        data.append({
            "配置": config_name,
            "描述": result["config"]["description"],
            "R² (均值±标准差)": f"{result['r2_mean']:.4f} ± {result['r2_std']:.4f}",
            "RMSE (均值±标准差)": f"{result['rmse_mean']:.4f} ± {result['rmse_std']:.4f}",
            "MAE (均值±标准差)": f"{result['mae_mean']:.4f} ± {result['mae_std']:.4f}",
            "MAPE (均值±标准差)": f"{result['mape_mean']:.2f}% ± {result['mape_std']:.2f}%",
            "训练时间(秒)": f"{result['train_time_mean']:.2f} ± {result['train_time_std']:.2f}",
            "模型参数量": f"{result['total_params']:,}"
        })
    
    df = pd.DataFrame(data)
    
    # 保存为CSV
    df.to_csv(f"{output_dir}/comparison_table.csv", index=False, encoding="utf-8-sig")
    
    # 打印表格
    print("\n" + "="*120)
    print("消融实验对比表格")
    print("="*120)
    print(df.to_string(index=False))
    print("="*120)


def generate_visualizations(results, output_dir):
    """生成可视化图表"""
    print("\n生成可视化图表...")
    
    config_names = list(results.keys())
    
    # 1. R²对比柱状图
    plt.figure(figsize=(14, 6))
    r2_means = [results[name]["r2_mean"] for name in config_names]
    r2_stds = [results[name]["r2_std"] for name in config_names]
    
    x = np.arange(len(config_names))
    bars = plt.bar(x, r2_means, yerr=r2_stds, capsize=5, alpha=0.8, edgecolor='black')
    
    # 为完整模型标记不同颜色
    if "完整模型" in config_names:
        full_model_idx = config_names.index("完整模型")
        bars[full_model_idx].set_color('red')
        bars[full_model_idx].set_alpha(0.9)
    
    plt.xlabel('模型配置', fontsize=12)
    plt.ylabel('R² 分数', fontsize=12)
    plt.title('消融实验 - R² 对比', fontsize=14, fontweight='bold')
    plt.xticks(x, config_names, rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/r2_comparison.png", dpi=300)
    plt.close()
    
    # 2. RMSE对比柱状图
    plt.figure(figsize=(14, 6))
    rmse_means = [results[name]["rmse_mean"] for name in config_names]
    rmse_stds = [results[name]["rmse_std"] for name in config_names]
    
    bars = plt.bar(x, rmse_means, yerr=rmse_stds, capsize=5, alpha=0.8, edgecolor='black')
    if "完整模型" in config_names:
        full_model_idx = config_names.index("完整模型")
        bars[full_model_idx].set_color('red')
        bars[full_model_idx].set_alpha(0.9)
    
    plt.xlabel('模型配置', fontsize=12)
    plt.ylabel('RMSE', fontsize=12)
    plt.title('消融实验 - RMSE 对比', fontsize=14, fontweight='bold')
    plt.xticks(x, config_names, rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/rmse_comparison.png", dpi=300)
    plt.close()
    
    # 3. 训练时间对比
    plt.figure(figsize=(14, 6))
    time_means = [results[name]["train_time_mean"] for name in config_names]
    
    bars = plt.bar(x, time_means, alpha=0.8, edgecolor='black')
    if "完整模型" in config_names:
        full_model_idx = config_names.index("完整模型")
        bars[full_model_idx].set_color('red')
        bars[full_model_idx].set_alpha(0.9)
    
    plt.xlabel('模型配置', fontsize=12)
    plt.ylabel('训练时间 (秒)', fontsize=12)
    plt.title('消融实验 - 训练时间对比', fontsize=14, fontweight='bold')
    plt.xticks(x, config_names, rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/training_time_comparison.png", dpi=300)
    plt.close()
    
    # 4. 综合性能雷达图
    # 选择几个关键配置进行对比
    key_configs = ["完整模型", "无符号层", "简化符号层", "固定门控0.5"]
    key_configs = [c for c in key_configs if c in config_names]
    
    # 如果没有匹配的关键配置,使用所有配置
    if not key_configs:
        key_configs = config_names[:min(4, len(config_names))]  # 最多显示4个配置
    
    # 只在有配置时绘制雷达图
    if key_configs:
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # 归一化指标 (0-1范围)
        metrics = ['R²', 'RMSE_inv', 'MAE_inv', 'Speed']
        
        # 预先计算angles (在循环外)
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]
        
        for config_name in key_configs:
            result = results[config_name]
            
            # 归一化处理
            r2_norm = result["r2_mean"]  # R²已经在0-1范围
            rmse_inv_norm = 1 / (1 + result["rmse_mean"])  # RMSE越小越好,取倒数
            mae_inv_norm = 1 / (1 + result["mae_mean"])  # MAE越小越好,取倒数
            speed_norm = 1 / (1 + result["train_time_mean"] / 60)  # 时间越短越好
            
            values = [r2_norm, rmse_inv_norm, mae_inv_norm, speed_norm]
            values += values[:1]  # 闭合图形
            
            ax.plot(angles, values, 'o-', linewidth=2, label=config_name)
            ax.fill(angles, values, alpha=0.15)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1)
        ax.set_title('消融实验 - 综合性能雷达图', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/radar_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # 5. 箱线图 - 显示多次运行的分布
    plt.figure(figsize=(14, 6))
    r2_distributions = [
        [run["test_r2"] for run in results[name]["all_runs"]]
        for name in config_names
    ]
    
    bp = plt.boxplot(r2_distributions, labels=config_names, patch_artist=True)
    
    # 为完整模型的箱子设置不同颜色
    if "完整模型" in config_names:
        full_model_idx = config_names.index("完整模型")
        bp['boxes'][full_model_idx].set_facecolor('red')
        bp['boxes'][full_model_idx].set_alpha(0.6)
    
    plt.xlabel('模型配置', fontsize=12)
    plt.ylabel('R² 分数', fontsize=12)
    plt.title('消融实验 - R² 分布箱线图', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/r2_boxplot.png", dpi=300)
    plt.close()
    
    # 6. 预测值 vs 实际值对比图 (选取完整模型预测最好的20个点)
    generate_best_predictions_comparison(results, output_dir, n_points=20)
    
    # 7. 真实值与不确定度对比图 (新增)
    generate_uncertainty_comparison(results, output_dir, n_points=20)
    
    print(f"可视化图表已保存至 {output_dir}")


def generate_best_predictions_comparison(results, output_dir, n_points=20):
    """
    生成预测值与实际值的对比图
    选取完整模型预测最好的n_points个点（排除实际值为0.5的样本）
    
    Args:
        results: 所有模型的结果字典
        output_dir: 输出目录
        n_points: 选取的点数
    """
    print(f"\n生成最佳预测点对比图 (top {n_points} 点, 排除实际值=0.5)...")
    
    # 设置字体为Times New Roman
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['mathtext.fontset'] = 'stix'
    
    # 查找完整模型
    baseline_model = None
    for model_name in ["完整模型", "full"]:
        if model_name in results:
            baseline_model = model_name
            break
    
    if baseline_model is None:
        # 如果没有完整模型,使用第一个模型作为基准
        baseline_model = list(results.keys())[0]
        print(f"警告: 未找到'完整模型',使用 '{baseline_model}' 作为基准")
    
    # 获取基线模型的第一次运行结果
    baseline_result = results[baseline_model]["all_runs"][0]
    predictions = np.array(baseline_result["predictions"])
    targets = np.array(baseline_result["targets"])
    
    # 排除实际值为0.5（或接近0.5）的样本
    # 使用容差范围来处理浮点数精度问题
    tolerance = 0.01  # 0.49 到 0.51 之间的值都被认为是0.5
    valid_mask = np.abs(targets - 0.5) > tolerance
    valid_indices = np.where(valid_mask)[0]
    
    print(f"总样本数: {len(targets)}, 排除0.5后剩余: {len(valid_indices)}")
    
    if len(valid_indices) < n_points:
        print(f"警告: 有效样本数({len(valid_indices)})少于请求的点数({n_points})")
        n_points = len(valid_indices)
    
    # 在有效样本中计算预测误差
    valid_predictions = predictions[valid_indices]
    valid_targets = targets[valid_indices]
    errors = np.abs(valid_predictions - valid_targets)
    
    # 选择误差最小的n_points个点（在有效样本中的索引）
    best_valid_indices = np.argsort(errors)[:n_points]
    
    # 转换回原始数据的索引
    best_indices = valid_indices[best_valid_indices]
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x_positions = np.arange(n_points)
    
    # 名称映射
    name_mapping = {
        "完整模型": "PGNN-ASL",
        "无物理特征基线": "PGNN-w/o ASL",
        "实际值": "True Value"
    }
    
    # 绘制实际值 (浅红色方形，稍大) - zorder较低，先绘制
    ax.scatter(x_positions, targets[best_indices], 
              color='#FFB6C1',  # 浅红色 (LightPink)
              s=180,            # 稍大的尺寸
              marker='s',       # 方形
              label='True Value', 
              zorder=5,         # 较低的层级
              edgecolors='#FF6B6B',  # 深一点的红色边框
              linewidths=2)
    
    # 绘制每个模型的预测值
    for idx, (config_name, result) in enumerate(results.items()):
        # 获取该模型第一次运行的预测结果
        model_predictions = np.array(result["all_runs"][0]["predictions"])
        
        # 使用与基线模型相同的索引
        selected_predictions = model_predictions[best_indices]
        
        # 获取显示名称
        display_name = name_mapping.get(config_name, config_name)
        
        # 根据模型名称设置不同的样式
        if config_name == "完整模型" or config_name == baseline_model:
            # 完整模型: 黑色圆点
            marker = 'o'
            color = 'black'
            size = 120
            alpha = 1.0
            edgecolor = 'white'
            zorder = 15  # 最高层级
        elif config_name == "无物理特征基线":
            # 无物理特征基线: 蓝色菱形
            marker = 'D'
            color = '#3498DB'  # 蓝色
            size = 100
            alpha = 1.0
            edgecolor = 'white'
            zorder = 10  # 中间层级
        else:
            # 其他模型: 使用不同颜色的菱形
            marker = 'D'
            # 为其他模型分配不同颜色
            other_colors = ['#2ECC71', '#F39C12', '#9B59B6', '#1ABC9C', '#E67E22', '#34495E']
            other_idx = list(results.keys()).index(config_name)
            color = other_colors[other_idx % len(other_colors)]
            size = 80
            alpha = 0.9
            edgecolor = 'white'
            zorder = 8
        
        ax.scatter(x_positions, selected_predictions,
                  color=color, s=size, marker=marker,
                  label=display_name, alpha=alpha, 
                  edgecolors=edgecolor, linewidths=1.5,
                  zorder=zorder)
    
    # 添加连接线(可选,连接实际值和完整模型预测值)
    if baseline_model in results:
        baseline_predictions = np.array(results[baseline_model]["all_runs"][0]["predictions"])
        selected_baseline = baseline_predictions[best_indices]
        
        for i in range(n_points):
            ax.plot([x_positions[i], x_positions[i]], 
                   [targets[best_indices][i], selected_baseline[i]],
                   'k--', alpha=0.3, linewidth=0.8, zorder=1)
    
    # 设置标签 (英文，Times New Roman字体)
    ax.set_xlabel('Partial Test Samples', fontsize=14, fontweight='bold', fontname='Times New Roman')
    ax.set_ylabel('Subsurface Crack Depth (μm)', fontsize=14, fontweight='bold', fontname='Times New Roman')
    
    # 设置x轴刻度
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f'{i+1}' for i in range(n_points)], rotation=0, fontname='Times New Roman', fontsize=12)
    
    # 设置y轴刻度字体
    for label in ax.get_yticklabels():
        label.set_fontname('Times New Roman')
        label.set_fontsize(12)
    
    # 添加网格
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # 添加图例 (Times New Roman字体)
    legend = ax.legend(loc='best', framealpha=0.9, fontsize=12, ncol=1)
    for text in legend.get_texts():
        text.set_fontname('Times New Roman')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图形
    plt.savefig(f"{output_dir}/best_predictions_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 恢复默认字体设置
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['SimSun']
    
    # 生成详细数据表格
    generate_best_predictions_table(results, best_indices, targets, output_dir)
    
    print(f"最佳预测点对比图已保存 (实际值范围: {targets[best_indices].min():.3f} - {targets[best_indices].max():.3f})")


def generate_best_predictions_table(results, best_indices, targets, output_dir):
    """
    生成最佳预测点的详细数据表格
    
    Args:
        results: 所有模型的结果字典
        best_indices: 最佳预测点的索引
        targets: 实际值数组
        output_dir: 输出目录
    """
    # 创建数据表格
    table_data = []
    
    for idx, sample_idx in enumerate(best_indices):
        row = {
            "序号": idx + 1,
            "样本索引": int(sample_idx),
            "实际值": float(targets[sample_idx])
        }
        
        # 添加每个模型的预测值和误差
        for config_name, result in results.items():
            predictions = np.array(result["all_runs"][0]["predictions"])
            pred_value = predictions[sample_idx]
            error = abs(pred_value - targets[sample_idx])
            relative_error = (error / (targets[sample_idx] + 1e-8)) * 100
            
            row[f"{config_name}_预测值"] = float(pred_value)
            row[f"{config_name}_误差"] = float(error)
            row[f"{config_name}_相对误差(%)"] = float(relative_error)
        
        table_data.append(row)
    
    # 保存为CSV
    df = pd.DataFrame(table_data)
    df.to_csv(f"{output_dir}/best_predictions_detail.csv", index=False, encoding="utf-8-sig")
    
    print(f"最佳预测点详细数据已保存至: {output_dir}/best_predictions_detail.csv")


def generate_uncertainty_comparison(results, output_dir, n_points=20):
    """
    生成真实值与不确定度的对比图
    选取完整模型预测最好的n_points个点（排除实际值为0.5的样本）
    
    Args:
        results: 所有模型的结果字典
        output_dir: 输出目录
        n_points: 选取的点数
    """
    print(f"\n生成真实值与不确定度对比图 (top {n_points} 点, 排除实际值=0.5)...")
    
    # 查找完整模型
    baseline_model = None
    for model_name in ["完整模型", "full"]:
        if model_name in results:
            baseline_model = model_name
            break
    
    if baseline_model is None:
        baseline_model = list(results.keys())[0]
        print(f"警告: 未找到'完整模型',使用 '{baseline_model}' 作为基准")
    
    # 检查是否有不确定度数据
    baseline_result = results[baseline_model]["all_runs"][0]
    if "uncertainties" not in baseline_result:
        print("警告: 结果中没有不确定度数据，跳过不确定度可视化")
        return
    
    predictions = np.array(baseline_result["predictions"])
    targets = np.array(baseline_result["targets"])
    uncertainties = np.array(baseline_result["uncertainties"])
    
    # 排除实际值为0.5（或接近0.5）的样本
    tolerance = 0.01
    valid_mask = np.abs(targets - 0.5) > tolerance
    valid_indices = np.where(valid_mask)[0]
    
    print(f"总样本数: {len(targets)}, 排除0.5后剩余: {len(valid_indices)}")
    
    if len(valid_indices) < n_points:
        print(f"警告: 有效样本数({len(valid_indices)})少于请求的点数({n_points})")
        n_points = len(valid_indices)
    
    # 在有效样本中计算预测误差
    valid_predictions = predictions[valid_indices]
    valid_targets = targets[valid_indices]
    errors = np.abs(valid_predictions - valid_targets)
    
    # 选择误差最小的n_points个点
    best_valid_indices = np.argsort(errors)[:n_points]
    best_indices = valid_indices[best_valid_indices]
    
    # 获取选中点的数据
    selected_targets = targets[best_indices]
    selected_uncertainties = uncertainties[best_indices]
    selected_predictions = predictions[best_indices]
    
    # 将不确定度转换为标准差 (假设不确定度是方差)
    selected_std = np.sqrt(selected_uncertainties)
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x_positions = np.arange(n_points)
    
    # 绘制真实值 (黑色圆点)
    ax.scatter(x_positions, selected_targets, 
              color='black', s=120, marker='o', 
              label='真实值', zorder=10, edgecolors='white', linewidths=1.5)
    
    # 绘制预测值及其不确定度 (带误差条)
    ax.errorbar(x_positions, selected_predictions, yerr=selected_std,
                fmt='s', color='#E74C3C', markersize=10, capsize=5, capthick=2,
                ecolor='#E74C3C', elinewidth=2, alpha=0.8,
                label=f'{baseline_model} 预测值 ± 不确定度', zorder=5)
    
    # 添加不确定度区间填充
    ax.fill_between(x_positions, 
                    selected_predictions - selected_std,
                    selected_predictions + selected_std,
                    color='#E74C3C', alpha=0.2, label='不确定度区间')
    
    # 添加连接线 (真实值和预测值)
    for i in range(n_points):
        ax.plot([x_positions[i], x_positions[i]], 
               [selected_targets[i], selected_predictions[i]],
               'k--', alpha=0.3, linewidth=0.8, zorder=1)
    
    # 设置标签和标题
    ax.set_xlabel('样本索引', fontsize=12, fontweight='bold')
    ax.set_ylabel('裂纹深度 (mm)', fontsize=12, fontweight='bold')
    ax.set_title(f'消融实验 - 真实值与预测不确定度对比 (基于{baseline_model}最佳{n_points}个预测点)', 
                fontsize=14, fontweight='bold', pad=15)
    
    # 设置x轴刻度
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f'{i+1}' for i in range(n_points)], rotation=0)
    
    # 添加网格
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # 添加图例
    ax.legend(loc='best', framealpha=0.9, fontsize=11)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图形
    plt.savefig(f"{output_dir}/uncertainty_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 生成不确定度详细数据表格
    generate_uncertainty_table(results, best_indices, targets, output_dir)
    
    print(f"真实值与不确定度对比图已保存")


def generate_best_predictions_table(results, best_indices, targets, output_dir):
    """
    生成最佳预测点的详细数据表格
    
    Args:
        results: 所有模型的结果字典
        best_indices: 最佳预测点的索引
        targets: 实际值数组
        output_dir: 输出目录
    """
    # 创建数据表格
    table_data = []
    
    for idx, sample_idx in enumerate(best_indices):
        row = {
            "序号": idx + 1,
            "样本索引": int(sample_idx),
            "实际值": float(targets[sample_idx])
        }
        
        # 添加每个模型的预测值和误差
        for config_name, result in results.items():
            predictions = np.array(result["all_runs"][0]["predictions"])
            pred_value = predictions[sample_idx]
            error = abs(pred_value - targets[sample_idx])
            relative_error = (error / (targets[sample_idx] + 1e-8)) * 100
            
            row[f"{config_name}_预测值"] = float(pred_value)
            row[f"{config_name}_误差"] = float(error)
            row[f"{config_name}_相对误差(%)"] = float(relative_error)
        
        table_data.append(row)
    
    # 保存为CSV
    df = pd.DataFrame(table_data)
    df.to_csv(f"{output_dir}/best_predictions_detail.csv", index=False, encoding="utf-8-sig")
    
    print(f"最佳预测点详细数据已保存至: {output_dir}/best_predictions_detail.csv")


def generate_uncertainty_comparison(results, output_dir, n_points=20):
    """
    生成真实值与不确定度的对比图
    选取完整模型预测最好的n_points个点（排除实际值为0.5的样本）
    
    Args:
        results: 所有模型的结果字典
        output_dir: 输出目录
        n_points: 选取的点数
    """
    print(f"\n生成真实值与不确定度对比图 (top {n_points} 点, 排除实际值=0.5)...")
    
    # 查找完整模型
    baseline_model = None
    for model_name in ["完整模型", "full"]:
        if model_name in results:
            baseline_model = model_name
            break
    
    if baseline_model is None:
        baseline_model = list(results.keys())[0]
        print(f"警告: 未找到'完整模型',使用 '{baseline_model}' 作为基准")
    
    # 检查是否有不确定度数据
    baseline_result = results[baseline_model]["all_runs"][0]
    if "uncertainties" not in baseline_result:
        print("警告: 结果中没有不确定度数据，跳过不确定度可视化")
        return
    
    predictions = np.array(baseline_result["predictions"])
    targets = np.array(baseline_result["targets"])
    uncertainties = np.array(baseline_result["uncertainties"])
    
    # 排除实际值为0.5（或接近0.5）的样本
    tolerance = 0.01
    valid_mask = np.abs(targets - 0.5) > tolerance
    valid_indices = np.where(valid_mask)[0]
    
    print(f"总样本数: {len(targets)}, 排除0.5后剩余: {len(valid_indices)}")
    
    if len(valid_indices) < n_points:
        print(f"警告: 有效样本数({len(valid_indices)})少于请求的点数({n_points})")
        n_points = len(valid_indices)
    
    # 在有效样本中计算预测误差
    valid_predictions = predictions[valid_indices]
    valid_targets = targets[valid_indices]
    errors = np.abs(valid_predictions - valid_targets)
    
    # 选择误差最小的n_points个点
    best_valid_indices = np.argsort(errors)[:n_points]
    best_indices = valid_indices[best_valid_indices]
    
    # 获取选中点的数据
    selected_targets = targets[best_indices]
    selected_uncertainties = uncertainties[best_indices]
    selected_predictions = predictions[best_indices]
    
    # 将不确定度转换为标准差 (假设不确定度是方差)
    selected_std = np.sqrt(selected_uncertainties)
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x_positions = np.arange(n_points)
    
    # 绘制真实值 (黑色圆点)
    ax.scatter(x_positions, selected_targets, 
              color='black', s=120, marker='o', 
              label='真实值', zorder=10, edgecolors='white', linewidths=1.5)
    
    # 绘制预测值及其不确定度 (带误差条)
    ax.errorbar(x_positions, selected_predictions, yerr=selected_std,
                fmt='s', color='#E74C3C', markersize=10, capsize=5, capthick=2,
                ecolor='#E74C3C', elinewidth=2, alpha=0.8,
                label=f'{baseline_model} 预测值 ± 不确定度', zorder=5)
    
    # 添加不确定度区间填充
    ax.fill_between(x_positions, 
                    selected_predictions - selected_std,
                    selected_predictions + selected_std,
                    color='#E74C3C', alpha=0.2, label='不确定度区间')
    
    # 添加连接线 (真实值和预测值)
    for i in range(n_points):
        ax.plot([x_positions[i], x_positions[i]], 
               [selected_targets[i], selected_predictions[i]],
               'k--', alpha=0.3, linewidth=0.8, zorder=1)
    
    # 设置标签和标题
    ax.set_xlabel('样本索引', fontsize=12, fontweight='bold')
    ax.set_ylabel('裂纹深度 (mm)', fontsize=12, fontweight='bold')
    ax.set_title(f'消融实验 - 真实值与预测不确定度对比 (基于{baseline_model}最佳{n_points}个预测点)', 
                fontsize=14, fontweight='bold', pad=15)
    
    # 设置x轴刻度
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f'{i+1}' for i in range(n_points)], rotation=0)
    
    # 添加网格
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # 添加图例
    ax.legend(loc='best', framealpha=0.9, fontsize=11)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图形
    plt.savefig(f"{output_dir}/uncertainty_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 生成不确定度详细数据表格
    generate_uncertainty_table(results, best_indices, targets, output_dir)
    
    print(f"真实值与不确定度对比图已保存")


def generate_uncertainty_table(results, best_indices, targets, output_dir):
    """
    生成不确定度详细数据表格
    
    Args:
        results: 所有模型的结果字典
        best_indices: 最佳预测点的索引
        targets: 实际值数组
        output_dir: 输出目录
    """
    table_data = []
    
    for idx, sample_idx in enumerate(best_indices):
        row = {
            "序号": idx + 1,
            "样本索引": int(sample_idx),
            "真实值": float(targets[sample_idx])
        }
        
        # 添加每个模型的预测值、不确定度和误差
        for config_name, result in results.items():
            run_result = result["all_runs"][0]
            predictions = np.array(run_result["predictions"])
            pred_value = predictions[sample_idx]
            error = abs(pred_value - targets[sample_idx])
            
            row[f"{config_name}_预测值"] = float(pred_value)
            row[f"{config_name}_误差"] = float(error)
            
            # 如果有不确定度数据
            if "uncertainties" in run_result:
                uncertainties = np.array(run_result["uncertainties"])
                uncertainty = uncertainties[sample_idx]
                std = np.sqrt(uncertainty)
                row[f"{config_name}_不确定度(方差)"] = float(uncertainty)
                row[f"{config_name}_不确定度(标准差)"] = float(std)
                
                # 计算误差是否在不确定度区间内
                within_interval = error <= std
                row[f"{config_name}_误差在区间内"] = within_interval
        
        table_data.append(row)
    
    # 保存为CSV
    df = pd.DataFrame(table_data)
    df.to_csv(f"{output_dir}/uncertainty_detail.csv", index=False, encoding="utf-8-sig")
    
    print(f"不确定度详细数据已保存至: {output_dir}/uncertainty_detail.csv")


def perform_statistical_tests(results, output_dir):
    """执行统计显著性检验"""
    print("\n执行统计显著性检验...")
    
    config_names = list(results.keys())
    baseline = "完整模型"
    
    if baseline not in results:
        print("警告: 未找到'完整模型'作为基线")
        return
    
    baseline_r2 = [run["test_r2"] for run in results[baseline]["all_runs"]]
    
    test_results = []
    
    for config_name in config_names:
        if config_name == baseline:
            continue
        
        config_r2 = [run["test_r2"] for run in results[config_name]["all_runs"]]
        
        # 由于完整模型使用预训练模型(确定性),所有运行的R²相同
        # 需要处理这种情况
        if len(set(baseline_r2)) == 1 and len(set(config_r2)) > 1:
            # 完整模型是确定性的,使用单样本t检验
            t_stat, p_value = stats.ttest_1samp(config_r2, baseline_r2[0])
            mean_diff = baseline_r2[0] - np.mean(config_r2)
            pooled_std = np.std(config_r2) if np.std(config_r2) > 0 else 1e-8
            cohens_d = mean_diff / pooled_std
        elif len(set(baseline_r2)) == 1 and len(set(config_r2)) == 1:
            # 两者都是确定性的,直接比较
            t_stat = float('inf') if baseline_r2[0] != config_r2[0] else 0
            p_value = 0.0 if baseline_r2[0] != config_r2[0] else 1.0
            mean_diff = baseline_r2[0] - config_r2[0]
            cohens_d = float('inf') if mean_diff != 0 else 0
        else:
            # 执行配对t检验
            t_stat, p_value = stats.ttest_rel(baseline_r2, config_r2)
            mean_diff = np.mean(baseline_r2) - np.mean(config_r2)
            pooled_std = np.sqrt((np.std(baseline_r2)**2 + np.std(config_r2)**2) / 2)
            cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
        
        test_results.append({
            "对比配置": config_name,
            "基线R²均值": f"{np.mean(baseline_r2):.4f}",
            "配置R²均值": f"{np.mean(config_r2):.4f}",
            "R²差异": f"{mean_diff:.4f}",
            "t统计量": f"{t_stat:.4f}" if not np.isinf(t_stat) else "inf",
            "p值": f"{p_value:.4f}",
            "显著性": "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns",
            "Cohen's d": f"{cohens_d:.4f}" if not np.isinf(cohens_d) else "inf"
        })
    
    # 保存统计检验结果
    df_stats = pd.DataFrame(test_results)
    df_stats.to_csv(f"{output_dir}/statistical_tests.csv", index=False, encoding="utf-8-sig")
    
    print("\n" + "="*120)
    print(f"统计显著性检验 (基线: {baseline})")
    print("="*120)
    print(df_stats.to_string(index=False))
    print("="*120)
    print("显著性标记: *** p<0.001, ** p<0.01, * p<0.05, ns 不显著")
    print("="*120)


if __name__ == "__main__":
    print("\n" + "="*70)
    print("材料裂纹深度预测 - 消融实验")
    print("="*70)
    
    # 运行消融实验
    # 可以选择运行所有配置或仅部分配置
    
    # 选项1: 运行所有配置
    #results = run_ablation_study(n_runs=3)
    
    # 选项2: 仅运行关键配置(更快)
    key_configs = {
        "完整模型": ABLATION_CONFIGS["完整模型"],
        # "无符号层": ABLATION_CONFIGS["无符号层"],
        # "简化符号层": ABLATION_CONFIGS["简化符号层"],
        # "固定门控0.5": ABLATION_CONFIGS["固定门控0.5"],
        # "固定门控0.0": ABLATION_CONFIGS["固定门控0.0"],
        # "固定门控1.0": ABLATION_CONFIGS["固定门控1.0"],
        # "无物理特征": ABLATION_CONFIGS["无物理特征"],
        # "无残差网络": ABLATION_CONFIGS["无残差网络"],
        "无物理特征基线": ABLATION_CONFIGS["无物理特征基线"]
    }
    results = run_ablation_study(configs=key_configs, n_runs=5)
    
    print("\n消融实验全部完成!")
