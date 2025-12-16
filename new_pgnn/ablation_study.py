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
from losses import PhysicsInformedLoss
from utils import prepare_material_data, augment_dataset

warnings.filterwarnings("ignore")

# 数据路径
DATA_PATH = "3011916164精确材料性能数据.csv"

# 固定随机种子
SEED = 66
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# 消融实验配置
ABLATION_CONFIGS = {
    "完整模型": {
        "ablation_mode": "full",
        "symbolic_type": "advanced",
        "description": "完整模型:符号层+残差网络+自适应门控"
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


def train_single_ablation(config_name, config, base_params, data_path=DATA_PATH, seed=SEED, n_runs=3):
    """
    训练单个消融实验配置
    
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
        
        # 创建模型
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
            "targets": [float(x) for x in test_target_np.tolist()]
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
    
    print(f"可视化图表已保存至 {output_dir}")


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
        
        # 执行配对t检验
        t_stat, p_value = stats.ttest_rel(baseline_r2, config_r2)
        
        # 计算效应量 (Cohen's d)
        mean_diff = np.mean(baseline_r2) - np.mean(config_r2)
        pooled_std = np.sqrt((np.std(baseline_r2)**2 + np.std(config_r2)**2) / 2)
        cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
        
        test_results.append({
            "对比配置": config_name,
            "基线R²均值": f"{np.mean(baseline_r2):.4f}",
            "配置R²均值": f"{np.mean(config_r2):.4f}",
            "R²差异": f"{mean_diff:.4f}",
            "t统计量": f"{t_stat:.4f}",
            "p值": f"{p_value:.4f}",
            "显著性": "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns",
            "Cohen's d": f"{cohens_d:.4f}"
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
        # "完整模型": ABLATION_CONFIGS["完整模型"],
        # "无符号层": ABLATION_CONFIGS["无符号层"],
        # "简化符号层": ABLATION_CONFIGS["简化符号层"],
        # "固定门控0.5": ABLATION_CONFIGS["固定门控0.5"],
        # "固定门控0.0": ABLATION_CONFIGS["固定门控0.0"],
        # "固定门控1.0": ABLATION_CONFIGS["固定门控1.0"],
        "无物理特征": ABLATION_CONFIGS["无物理特征"],
        # "无残差网络": ABLATION_CONFIGS["无残差网络"],
        "无物理特征基线": ABLATION_CONFIGS["无物理特征基线"]
    }
    results = run_ablation_study(configs=key_configs, n_runs=5)
    
    print("\n消融实验全部完成!")
