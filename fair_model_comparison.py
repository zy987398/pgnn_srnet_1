# -*- coding: utf-8 -*-
"""
公平的模型对比实验 - 传统ML vs PGNN
确保所有模型在相同的数据划分上进行评估
PGNN使用new_pgnn中的模型和超参数
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import glob
import json

# 传统ML模型
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet
from sklearn.neural_network import MLPRegressor

# XGBoost (可选)
try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("警告: 未安装xgboost,将跳过XGBoost模型")

# 添加new_pgnn到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'new_pgnn'))

# PGNN imports
from new_pgnn.models import MaterialCrackPGNN
from new_pgnn.utils import prepare_material_data


# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 固定随机种子 - 与new_pgnn/train.py保持一致
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ========== 实验配置开关 ==========
# 是否为传统ML模型使用物理特征工程
# 注意：PGNN始终使用物理特征（通过prepare_material_data），此开关仅控制传统模型
USE_PHYSICS_FEATURES_FOR_ML = False  # True: 传统模型使用物理特征, False: 传统模型仅使用原始特征

# 数据路径
DATA_PATH = "3011916164精确材料性能数据.csv"

# 结果保存路径
RESULTS_DIR = os.path.join("results", "compare")

# PGNN超参数 - 从new_pgnn/train.py复制
PGNN_PARAMS = {
    # 模型架构参数
    "material_embed_dim": 8,
    "latent_dim": 48,
    "dropout_rate": 0.35,
    "symbolic_type": "advanced",
    
    # 训练参数
    "epochs": 350,
    "batch_size": 16,
    "lr": 0.0011173842752625436,
    "patience": 55,
    "augmentation": False,
    "optimizer": "AdamW",
    "weight_decay": 0.0066923795400554925,
    "lambda_complexity": 3.4069890537972935e-05,
    "lambda_physics": 0.2
}

# 预训练模型路径
PRETRAINED_MODEL_PATH = "results\\run_20251217_100959\\best_model.pth"

# ---------------------------------------------------------------------------
# 传统ML特征工程
# ---------------------------------------------------------------------------

def create_basic_features(df):
    """
    创建基础特征（仅使用原始数值特征）
    用于公平对比
    """
    numeric_cols = ["莫氏硬度", "弹性模量", "断裂韧性", "晶粒粒径", "压溃压力", "磨粒速度"]
    X = df[numeric_cols].values
    feature_names = numeric_cols.copy()
    
    return X, feature_names


def create_physics_features(df):
    """
    创建物理意义的特征（不使用目标变量）
    这些特征是为PGNN设计的，也可选择性应用于传统模型
    """
    features, feature_names = [], []
    numeric_cols = ["莫氏硬度", "弹性模量", "断裂韧性", "晶粒粒径", "压溃压力", "磨粒速度"]
    
    # 1. 原始特征
    features.append(df[numeric_cols].values)
    feature_names.extend(numeric_cols)
    
    # 添加小的eps避免除零
    eps = 1e-6
    
    # 2. 物理特征
    # 脆性指数
    brittleness = df["莫氏硬度"] / (df["断裂韧性"] + eps)
    features.append(brittleness.values.reshape(-1, 1))
    feature_names.append("脆性指数")
    
    # Griffith能量释放率
    energy_release = (df["压溃压力"] ** 2) * np.sqrt(df["晶粒粒径"]) / (df["弹性模量"] + eps)
    features.append(energy_release.values.reshape(-1, 1))
    feature_names.append("能量释放率")
    
    # 应力强度因子比
    k_ratio = df["压溃压力"] * np.sqrt(df["晶粒粒径"]) / (df["断裂韧性"] + eps)
    features.append(k_ratio.values.reshape(-1, 1))
    feature_names.append("应力强度因子比")
    
    # Hertz接触应力
    contact_stress = np.power(
        df["压溃压力"] * df["弹性模量"] ** 2 / (df["晶粒粒径"] ** 2 + eps), 1 / 3
    )
    features.append(contact_stress.values.reshape(-1, 1))
    feature_names.append("接触应力")
    
    # 能量输入
    energy_input = df["压溃压力"] * df["磨粒速度"]
    features.append(energy_input.values.reshape(-1, 1))
    feature_names.append("能量输入")
    
    # 硬度-弹性模量比
    h_e_ratio = df["莫氏硬度"] / (df["弹性模量"] + eps)
    features.append(h_e_ratio.values.reshape(-1, 1))
    feature_names.append("硬度弹性模量比")
    
    # Hall-Petch项
    hall_petch = 1 / np.sqrt(df["晶粒粒径"] + eps)
    features.append(hall_petch.values.reshape(-1, 1))
    feature_names.append("Hall-Petch")
    
    # 载荷-强度比
    load_strength_ratio = df["压溃压力"] / (df["莫氏硬度"] + eps)
    features.append(load_strength_ratio.values.reshape(-1, 1))
    feature_names.append("载荷强度比")
    
    # 功率-强度比
    power_strength_ratio = (df["压溃压力"] * df["磨粒速度"]) / (df["莫氏硬度"] + eps)
    features.append(power_strength_ratio.values.reshape(-1, 1))
    feature_names.append("功率强度比")
    
    # 延-脆临界深度相关
    e_over_h = df["弹性模量"] / (df["莫氏硬度"] + eps)
    k_over_h = df["断裂韧性"] / (df["莫氏硬度"] + eps)
    critical_depth_factor = 0.15 * e_over_h * (k_over_h ** 2)
    features.append(critical_depth_factor.values.reshape(-1, 1))
    feature_names.append("临界深度因子")
    
    # 弹性应变能密度
    strain_energy = 0.5 * (df["压溃压力"] ** 2) / (df["弹性模量"] + eps)
    features.append(strain_energy.values.reshape(-1, 1))
    feature_names.append("应变能密度")
    
    # 韧性-模量比
    toughness_modulus = df["断裂韧性"] / (df["弹性模量"] + eps)
    features.append(toughness_modulus.values.reshape(-1, 1))
    feature_names.append("韧性模量比")
    
    X = np.hstack(features)
    
    # 处理NaN和Inf
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return X, feature_names

# ---------------------------------------------------------------------------
# 传统ML模型训练
# ---------------------------------------------------------------------------

def train_traditional_models(X_train, X_test, y_train, y_test):
    """训练和评估传统ML模型"""
    
    models = {
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=15, min_samples_split=5, 
            min_samples_leaf=2, random_state=SEED, n_jobs=-1
        ),
        "XGBoost": XGBRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.1, 
            subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.1, 
            subsample=0.8, random_state=SEED
        ),
        "SVR": SVR(kernel="rbf", C=10, gamma="scale", epsilon=0.1),
        # "MLP": MLPRegressor(
        #     hidden_layer_sizes=(100, 50), activation="relu", 
        #     max_iter=1000, random_state=SEED, early_stopping=True
        # ),
        "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=SEED),
    }
    
    results = {}
    for name, model in models.items():
        print(f"训练 {name}...")
        model.fit(X_train, y_train)
        y_pred_test = model.predict(X_test)
        
        results[name] = {
            "model": model,
            "predictions": y_pred_test,
            "test_r2": r2_score(y_test, y_pred_test),
            "test_mse": mean_squared_error(y_test, y_pred_test),
            "test_mae": mean_absolute_error(y_test, y_pred_test),
        }
        print(f"  R²: {results[name]['test_r2']:.4f}, RMSE: {np.sqrt(results[name]['test_mse']):.4f}")
    
    return results

# ---------------------------------------------------------------------------
# PGNN模型评估（使用预训练模型）
# ---------------------------------------------------------------------------

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


def evaluate_pretrained_pgnn(train_df, test_df, default_params, model_path, device):
    """加载并评估预训练的PGNN模型"""
    
    print("\n" + "="*60)
    print("评估预训练PGNN模型")
    print("="*60)
    
    # 加载模型配置
    print(f"加载模型配置: {model_path}")
    params = load_model_config(model_path, default_params)
    
    # 准备数据
    print("准备数据...")
    train_data = prepare_material_data(train_df)
    test_data = prepare_material_data(
        test_df,
        scaler=train_data["scaler"],
        le=train_data["label_encoder"],
    )
    
    num_materials = train_data["num_materials"]
    print(f"共 {num_materials} 种材料类型")
    
    # 创建模型
    print(f"创建模型...")
    model = MaterialCrackPGNN(
        num_materials=num_materials,
        material_embed_dim=params["material_embed_dim"],
        latent_dim=params["latent_dim"],
        dropout_rate=params["dropout_rate"],
        symbolic_type=params["symbolic_type"]
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
        
        test_r2 = r2_score(test_target_np, test_pred_np)
        test_rmse = np.sqrt(mean_squared_error(test_target_np, test_pred_np))
        test_mae = mean_absolute_error(test_target_np, test_pred_np)
    
    print(f"\n测试集性能:")
    print(f"  R²: {test_r2:.4f}")
    print(f"  RMSE: {test_rmse:.4f}")
    print(f"  MAE: {test_mae:.4f}")
    
    return {
        "model": model,
        "predictions": test_pred_np,
        "test_r2": test_r2,
        "test_mse": test_rmse ** 2,
        "test_mae": test_mae,
    }

# ---------------------------------------------------------------------------
# 可视化
# ---------------------------------------------------------------------------

def plot_comparison(results, y_test, save_dir=RESULTS_DIR):
    """绘制模型对比图"""
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 获取运行ID
    scatter_files = glob.glob(os.path.join(save_dir, "fair_comparison_scatter_*.png"))
    run_id = max([int(f.split("_")[-1].split(".")[0]) for f in scatter_files] + [0]) + 1
    
    # 1. 散点图网格
    n = len(results)
    rows = (n + 2) // 3
    cols = 3
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten() if n > 1 else [axes]
    
    for ax, (name, res) in zip(axes, results.items()):
        y_pred = res["predictions"]
        ax.scatter(y_test, y_pred, alpha=0.5, s=20, edgecolors='k', linewidths=0.5)
        
        # 绘制理想线
        lims = [y_test.min(), y_test.max()]
        ax.plot(lims, lims, "r--", lw=2, label="理想预测")
        
        # 标题和标签
        rmse = np.sqrt(res["test_mse"])
        ax.set_title(f"{name}\nR²={res['test_r2']:.4f}, RMSE={rmse:.3f}μm", 
                     fontsize=10, fontweight='bold')
        ax.set_xlabel("实际裂纹深度 (μm)", fontsize=9)
        ax.set_ylabel("预测裂纹深度 (μm)", fontsize=9)
        ax.grid(alpha=0.3, linestyle='--')
        ax.legend(fontsize=8)
    
    # 隐藏多余的子图
    for ax in axes[len(results):]:
        ax.axis("off")
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"fair_comparison_scatter_{run_id}.png"), dpi=300, bbox_inches='tight')
    print(f"散点图已保存: fair_comparison_scatter_{run_id}.png")
    plt.close()
    
    # 2. 性能对比柱状图
    model_names = list(results.keys())
    r2_scores = [res["test_r2"] for res in results.values()]
    rmse_scores = [np.sqrt(res["test_mse"]) for res in results.values()]
    mae_scores = [res["test_mae"] for res in results.values()]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # R² 对比
    colors = ['#1f77b4' if 'PGNN' not in name else '#d62728' for name in model_names]
    axes[0].bar(range(len(model_names)), r2_scores, color=colors, edgecolor='black', linewidth=1.2)
    axes[0].set_ylabel("R² 分数", fontsize=11, fontweight='bold')
    axes[0].set_title("R² 性能对比", fontsize=12, fontweight='bold')
    axes[0].set_ylim(0, 1)
    axes[0].set_xticks(range(len(model_names)))
    axes[0].set_xticklabels(model_names, rotation=45, ha='right')
    axes[0].grid(axis='y', alpha=0.3, linestyle='--')
    for i, v in enumerate(r2_scores):
        axes[0].text(i, v + 0.02, f"{v:.4f}", ha="center", fontsize=9, fontweight='bold')
    
    # RMSE 对比
    axes[1].bar(range(len(model_names)), rmse_scores, color=colors, edgecolor='black', linewidth=1.2)
    axes[1].set_ylabel("RMSE (μm)", fontsize=11, fontweight='bold')
    axes[1].set_title("RMSE 性能对比", fontsize=12, fontweight='bold')
    axes[1].set_xticks(range(len(model_names)))
    axes[1].set_xticklabels(model_names, rotation=45, ha='right')
    axes[1].grid(axis='y', alpha=0.3, linestyle='--')
    for i, v in enumerate(rmse_scores):
        axes[1].text(i, v + 0.05, f"{v:.3f}", ha="center", fontsize=9, fontweight='bold')
    
    # MAE 对比
    axes[2].bar(range(len(model_names)), mae_scores, color=colors, edgecolor='black', linewidth=1.2)
    axes[2].set_ylabel("MAE (μm)", fontsize=11, fontweight='bold')
    axes[2].set_title("MAE 性能对比", fontsize=12, fontweight='bold')
    axes[2].set_xticks(range(len(model_names)))
    axes[2].set_xticklabels(model_names, rotation=45, ha='right')
    axes[2].grid(axis='y', alpha=0.3, linestyle='--')
    for i, v in enumerate(mae_scores):
        axes[2].text(i, v + 0.05, f"{v:.3f}", ha="center", fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"fair_comparison_metrics_{run_id}.png"), dpi=300, bbox_inches='tight')
    print(f"性能对比图已保存: fair_comparison_metrics_{run_id}.png")
    plt.close()

# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def main():
    print("="*80)
    print("公平的模型对比实验 - 传统ML vs PGNN")
    print("="*80)
    print(f"\n数据文件: {DATA_PATH}")
    print(f"结果保存路径: {RESULTS_DIR}")
    print(f"随机种子: {SEED}")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建结果保存目录
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # 数据划分 - 与new_pgnn/train.py保持一致
    print("\n划分数据集...")
    df = pd.read_csv(DATA_PATH)
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=SEED, stratify=df["材料种类"]
    )
    print(f"训练集: {len(train_df)} 样本")
    print(f"测试集: {len(test_df)} 样本")
    
    # 保存材料分布统计
    print("\n材料分布:")
    print("训练集:", train_df["材料种类"].value_counts().to_dict())
    print("测试集:", test_df["材料种类"].value_counts().to_dict())
    
    # ========== 传统ML模型 ==========
    print("\n" + "="*60)
    print("训练传统机器学习模型")
    print("="*60)
    
    # 根据配置选择特征创建方式
    if USE_PHYSICS_FEATURES_FOR_ML:
        print("⚠ 传统模型使用物理特征（原始特征 + 物理衍生特征）")
        print("  注意：这可能给传统模型带来额外优势")
        X_train, feat_names = create_physics_features(train_df)
        X_test, _ = create_physics_features(test_df)
    else:
        print("✓ 传统模型仅使用原始特征（公平对比）")
        print("  PGNN使用内置物理特征，传统模型使用原始特征")
        X_train, feat_names = create_basic_features(train_df)
        X_test, _ = create_basic_features(test_df)
    
    y_train = train_df["亚表面裂纹深度"].values
    y_test = test_df["亚表面裂纹深度"].values
    
    print(f"特征维度: {X_train.shape[1]}")
    print(f"特征列表: {feat_names}")
    
    # 标准化
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 训练传统模型
    traditional_results = train_traditional_models(X_train_scaled, X_test_scaled, y_train, y_test)
    
    # ========== PGNN模型 ==========
    print("\n" + "="*60)
    print("评估PGNN模型")
    print("="*60)
    print(f"使用预训练模型: {PRETRAINED_MODEL_PATH}")
    print("注意：PGNN通过prepare_material_data内部使用物理特征")
    
    pgnn_result = evaluate_pretrained_pgnn(train_df, test_df, PGNN_PARAMS, PRETRAINED_MODEL_PATH, device)
    
    # 合并结果
    all_results = traditional_results.copy()
    all_results["PGNN"] = pgnn_result
    
    # ========== 结果汇总 ==========
    print("\n" + "="*80)
    print("最终结果汇总 (按R²降序排列)")
    print("="*80)
    print(f"{'模型名称':<25} | {'R²':>8} | {'RMSE':>8} | {'MAE':>8}")
    print("-" * 80)
    
    for name, res in sorted(all_results.items(), key=lambda x: x[1]["test_r2"], reverse=True):
        print(f"{name:<25} | {res['test_r2']:>8.4f} | {np.sqrt(res['test_mse']):>8.4f} | {res['test_mae']:>8.4f}")
    
    # 保存结果到JSON
    results_summary = {
        "config": {
            "use_physics_features_for_ml": USE_PHYSICS_FEATURES_FOR_ML,
            "random_seed": SEED,
            "test_size": 0.2,
            "num_features": X_train.shape[1]
        },
        "results": {
            name: {
                "R2": float(res["test_r2"]),
                "RMSE": float(np.sqrt(res["test_mse"])),
                "MAE": float(res["test_mae"])
            }
            for name, res in all_results.items()
        }
    }
    
    results_file = os.path.join(RESULTS_DIR, f"fair_comparison_results_{'physics' if USE_PHYSICS_FEATURES_FOR_ML else 'basic'}.json")
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=4, ensure_ascii=False)
    print(f"\n结果已保存到: {results_file}")
    
    # ========== 可视化 ==========
    print("\n生成对比图...")
    plot_comparison(all_results, y_test)
    
    # ========== 性能提升分析 ==========
    print("\n" + "="*80)
    print("PGNN vs 传统模型性能提升分析")
    print("="*80)
    print(f"实验设置: 传统模型{'使用' if USE_PHYSICS_FEATURES_FOR_ML else '不使用'}物理特征")
    
    pgnn_r2 = all_results["PGNN"]["test_r2"]
    best_traditional = max(
        [(name, res["test_r2"]) for name, res in all_results.items() if name != "PGNN"],
        key=lambda x: x[1]
    )
    
    print(f"PGNN R²: {pgnn_r2:.4f}")
    print(f"最佳传统模型: {best_traditional[0]} (R² = {best_traditional[1]:.4f})")
    
    if pgnn_r2 > best_traditional[1]:
        improvement = (pgnn_r2 - best_traditional[1]) / best_traditional[1] * 100
        print(f"✓ PGNN相对提升: {improvement:.2f}%")
        print(f"✓ PGNN性能优于所有传统模型")
        if not USE_PHYSICS_FEATURES_FOR_ML:
            print(f"  （传统模型未使用物理特征，对比更公平）")
    else:
        decline = (best_traditional[1] - pgnn_r2) / best_traditional[1] * 100
        print(f"✗ PGNN相对下降: {decline:.2f}%")
        print(f"✗ PGNN性能低于最佳传统模型")
        if USE_PHYSICS_FEATURES_FOR_ML:
            print(f"  （传统模型也使用了物理特征）")
    
    print("\n" + "="*80)
    print("实验完成！")
    print("="*80)


if __name__ == "__main__":
    main()
