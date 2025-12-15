# -*- coding: utf-8 -*-
"""
Comparison script for traditional ML models vs PGNN model
Uses the trained model from new_pgnn/model_pre.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import glob

# Traditional ML models
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.linear_model import ElasticNet
from sklearn.neural_network import MLPRegressor

# 添加new_pgnn到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'new_pgnn'))

# PGNN imports - 从new_pgnn文件夹导入
from new_pgnn.models import MaterialCrackPGNN
from new_pgnn.model_pre import CrackDepthPredictor
from new_pgnn.utils import prepare_material_data

# 设置中文字体（用于matplotlib）
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 固定随机种子
SEED = 66
torch.manual_seed(SEED)
np.random.seed(SEED)

# 模型路径配置
MODEL_PATH = "results/select_model.pth"
TRAIN_DATA_PATH = "new_pgnn/3011916164精确材料性能数据.csv"

# ---------------------------------------------------------------------------
# Traditional ML helpers
# ---------------------------------------------------------------------------

def evaluate_traditional_models(X_train, X_test, y_train, y_test):
    """Train & evaluate classical regressors"""
    
    models = {
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=15, min_samples_split=5, random_state=42
        ),
        "XGBoost": XGBRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42
        ),
        "SVR": SVR(kernel="rbf", C=10, gamma="scale"),
        "MLP": MLPRegressor(
            hidden_layer_sizes=(100, 50), activation="relu", max_iter=1000, random_state=42
        ),
        "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),
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
    
    return results

def create_ensemble_features(df):
    """Create physics-aware features"""
    
    features, feature_names = [], []
    numeric_cols = ["莫氏硬度", "弹性模量", "断裂韧性", "晶粒粒径", "压溃压力", "磨粒速度"]
    
    # Raw features
    features.append(df[numeric_cols].values)
    feature_names.extend(numeric_cols)
    
    # Physics features
    features.append((df["莫氏硬度"] / df["断裂韧性"]).values.reshape(-1, 1))
    feature_names.append("脆性指数")
    
    energy_release = (df["压溃压力"] ** 2) * np.sqrt(df["晶粒粒径"]) / df["弹性模量"]
    features.append(energy_release.values.reshape(-1, 1))
    feature_names.append("能量释放率")
    
    k_ratio = df["压溃压力"] * np.sqrt(df["晶粒粒径"]) / df["断裂韧性"]
    features.append(k_ratio.values.reshape(-1, 1))
    feature_names.append("应力强度因子比")
    
    contact_stress = np.power(
        df["压溃压力"] * df["弹性模量"] ** 2 / df["晶粒粒径"] ** 2, 1 / 3
    )
    features.append(contact_stress.values.reshape(-1, 1))
    feature_names.append("接触应力")
    
    features.append((df["压溃压力"] * df["磨粒速度"]).values.reshape(-1, 1))
    feature_names.append("能量输入")
    
    features.append((df["莫氏硬度"] / df["弹性模量"]).values.reshape(-1, 1))
    feature_names.append("硬度弹性模量比")
    
    features.append((1 / np.sqrt(df["晶粒粒径"])).values.reshape(-1, 1))
    feature_names.append("Hall-Petch")
    
    # 移除数据泄露特征：材料平均裂纹深度（使用了目标变量）
    # material_mean = df.groupby("材料种类")["亚表面裂纹深度"].transform("mean")
    # features.append(material_mean.values.reshape(-1, 1))
    # feature_names.append("材料平均裂纹深度")
    
    X = np.hstack(features)
    return X, feature_names

def plot_model_comparison(results, y_test):
    """Create comparison plots"""
    
    save_dir = "results"
    os.makedirs(save_dir, exist_ok=True)
    
    # Get run ID
    scatter_files = glob.glob(os.path.join(save_dir, "scatter_grid_*.png"))
    run_id = max([int(f.split("_")[-1].split(".")[0]) for f in scatter_files] + [0]) + 1
    
    # Scatter grid
    n = len(results)
    rows, cols = (n + 2) // 3, 3
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten()
    
    for ax, (name, res) in zip(axes, results.items()):
        y_pred = res["predictions"]
        ax.scatter(y_test, y_pred, alpha=0.5, s=18)
        lims = [y_test.min(), y_test.max()]
        ax.plot(lims, lims, "r--", lw=1)
        rmse = np.sqrt(res["test_mse"])
        ax.set_title(f"{name}\nR²={res['test_r2']:.3f}, RMSE={rmse:.3f}")
        ax.set_xlabel("Actual (μm)")
        ax.set_ylabel("Predicted (μm)")
        ax.grid(alpha=0.3)
    
    for ax in axes[len(results):]:
        ax.axis("off")
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"scatter_grid_{run_id}.png"), dpi=300)
    plt.show()
    
    # Bar charts
    model_names = list(results.keys())
    r2_scores = [res["test_r2"] for res in results.values()]
    rmse_scores = [np.sqrt(res["test_mse"]) for res in results.values()]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.bar(model_names, r2_scores, edgecolor="navy")
    ax1.set_ylabel("R²")
    ax1.set_title("R² Comparison")
    ax1.set_ylim(0, 1)
    for i, v in enumerate(r2_scores):
        ax1.text(i, v + 0.01, f"{v:.3f}", ha="center")
    
    ax2.bar(model_names, rmse_scores, edgecolor="darkred")
    ax2.set_ylabel("RMSE (μm)")
    ax2.set_title("RMSE Comparison")
    for i, v in enumerate(rmse_scores):
        ax2.text(i, v + 0.05, f"{v:.3f}", ha="center")
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"bar_charts_{run_id}.png"), dpi=300)
    plt.show()

# ---------------------------------------------------------------------------
# Main experiment with pre-trained PGNN model
# ---------------------------------------------------------------------------

def main():
    # 数据路径
    CSV_PATH = "3011916164精确材料性能数据.csv"
    train_data_path = TRAIN_DATA_PATH  # 使用全局配置
    
    print("="*60)
    print("模型对比实验 - 传统ML模型 vs PGNN预训练模型")
    print("="*60)
    print(f"\n数据文件: {CSV_PATH}")
    print(f"PGNN模型: {MODEL_PATH}")
    print(f"训练数据: {train_data_path}")
    
    print("\n读取数据...")
    df = pd.read_csv(CSV_PATH)
    
    # 首先进行数据划分 - 使用stratify确保材料种类分布一致
    df_train, df_test = train_test_split(
        df, test_size=0.2, random_state=SEED, stratify=df["材料种类"]
    )
    
    print(f"训练集大小: {len(df_train)} 样本")
    print(f"测试集大小: {len(df_test)} 样本")
    
    # Traditional ML features - 从划分好的数据创建特征
    X_train, _ = create_ensemble_features(df_train)
    X_test, feat_names = create_ensemble_features(df_test)
    y_train_all = df_train["亚表面裂纹深度"].values
    y_test = df_test["亚表面裂纹深度"].values
    
    # 标准化特征
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    print("\n训练传统机器学习模型...")
    baseline_results = evaluate_traditional_models(X_train_s, X_test_s, y_train_all, y_test)
    
    # PGNN - 加载预训练模型
    print("\n加载预训练的PGNN模型...")
    
    # 检查模型文件是否存在
    if not os.path.exists(MODEL_PATH):
        print(f"错误: 模型文件不存在: {MODEL_PATH}")
        print("请先训练模型或检查路径配置")
        return
    
    if not os.path.exists(train_data_path):
        print(f"错误: 训练数据文件不存在: {train_data_path}")
        print("尝试使用当前目录的数据文件...")
        train_data_path = CSV_PATH
    
    try:
        # 创建预测器（会自动加载模型）
        predictor = CrackDepthPredictor(
            model_path=MODEL_PATH,
            train_data_path=train_data_path,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        print(f"模型加载成功！")
        print(f"使用设备: {predictor.device}")
        
        # 准备测试数据
        # 创建临时CSV文件用于预测
        temp_test_file = "temp_test_data.csv"
        df_test.to_csv(temp_test_file, index=False)
        
        # 使用预测器进行预测
        result_df = predictor.predict(
            input_data_path=temp_test_file,
            output_path=None,  # 不保存文件
            use_mc_dropout=False
        )
        
        # 获取预测结果
        preds = result_df["预测裂纹深度"].values
        
        # 删除临时文件
        if os.path.exists(temp_test_file):
            os.remove(temp_test_file)
        
        # 计算评估指标
        pgnn_r2 = r2_score(y_test, preds)
        pgnn_rmse = np.sqrt(mean_squared_error(y_test, preds))
        pgnn_mae = mean_absolute_error(y_test, preds)
        
        print(f"\nPGNN结果 => R²={pgnn_r2:.3f}, RMSE={pgnn_rmse:.3f}, MAE={pgnn_mae:.3f}")
        
        baseline_results["PGNN (Pre-trained)"] = {
            "model": predictor.model,
            "predictions": preds,
            "test_r2": pgnn_r2,
            "test_mse": pgnn_rmse ** 2,
            "test_mae": pgnn_mae,
        }
        
    except Exception as e:
        print(f"\n加载或评估PGNN模型时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\n将仅显示传统ML模型的对比结果")
        # 继续执行，只是不包含PGNN结果
    
    # Plot comparisons
    if len(baseline_results) > 0:
        print("\n生成对比图...")
        plot_model_comparison(baseline_results, y_test)
    
    # Print summary
    print("\n" + "="*60)
    print("最终结果汇总")
    print("="*60)
    for name, res in sorted(baseline_results.items(), key=lambda x: x[1]["test_r2"], reverse=True):
        print(f"{name:30s} | R²: {res['test_r2']:.4f} | RMSE: {np.sqrt(res['test_mse']):.4f} | MAE: {res['test_mae']:.4f}")
    
    print("\n" + "="*60)
    print("实验完成！")
    print("="*60)

if __name__ == "__main__":
    main()
