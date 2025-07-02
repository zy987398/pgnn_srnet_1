# -*- coding: utf-8 -*-
"""
Self‑contained script for training the improved Physics‑Guided Neural Network (PGNN)
side‑by‑side with several traditional machine‑learning baselines and visualising the
results in a single set of plots.

Usage
-----
$ pip install torch==2.2.1 scikit-learn pandas matplotlib seaborn xgboost
$ python run_pgnn_vs_baselines.py

Assumptions
-----------
*   `material_crack_pgnn.py` (your PGNN implementation) is in the same folder.
*   `3011916164精确材料性能数据.csv` (your data set) is in the same folder.

The script will
1. build handcrafted physics‑aware features;
2. train six classical regressors (RF, XGB, GBDT, SVR, MLP, ElasticNet);
3. train the improved PGNN you provided;
4. print R² / RMSE / MAE for every model; and
5. pop up two comparison figures (scatter grids + bar charts).
"""

import warnings
warnings.filterwarnings("ignore")  # optional: silence sklearn / pandas msgs

import numpy as np
import pandas as pd
import torch
import torch
torch.autograd.set_detect_anomaly(True)           # 反向阶段任何 NaN 立刻报栈
torch.backends.cuda.matmul.allow_tf32 = False     # 保守起见，禁用 TF32
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob

# --- PGNN ---
from material_crack_pgnn import (
    MaterialCrackPGNN,
    prepare_material_data,
    train_material_model,
)

# ---------------------------------------------------------------------------
# Traditional ML helpers (taken from your original `model_comparison.py`)
# ---------------------------------------------------------------------------
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.linear_model import ElasticNet
from sklearn.neural_network import MLPRegressor


def evaluate_traditional_models(X_train, X_test, y_train, y_test):
    """Train & evaluate a suite of classical regressors."""

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
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        results[name] = {
            "model": model,
            "train_mse": mean_squared_error(y_train, y_pred_train),
            "test_mse": mean_squared_error(y_test, y_pred_test),
            "train_r2": r2_score(y_train, y_pred_train),
            "test_r2": r2_score(y_test, y_pred_test),
            "test_mae": mean_absolute_error(y_test, y_pred_test),
            "predictions": y_pred_test,
        }
    return results


def create_ensemble_features(df):
    """Physics‑aware feature engineering to augment raw numeric inputs."""

    features, feature_names = [], []
    numeric_cols = [
        "莫氏硬度",
        "弹性模量",
        "断裂韧性",
        "晶粒粒径",
        "压溃压力",
        "磨粒速度",
    ]
    # 0. raw numeric cols
    features.append(df[numeric_cols].values)
    feature_names.extend(numeric_cols)

    # 1. 脆性指数
    features.append((df["莫氏硬度"] / df["断裂韧性"]).values.reshape(-1, 1))
    feature_names.append("脆性指数")

    # 2. 能量释放率代理
    energy_release = (df["压溃压力"] ** 2) * np.sqrt(df["晶粒粒径"]) / df["弹性模量"]
    features.append(energy_release.values.reshape(-1, 1))
    feature_names.append("能量释放率")

    # 3. 应力强度因子比
    k_ratio = df["压溃压力"] * np.sqrt(df["晶粒粒径"]) / df["断裂韧性"]
    features.append(k_ratio.values.reshape(-1, 1))
    feature_names.append("应力强度因子比")

    # 4. 接触应力
    contact_stress = np.power(
        df["压溃压力"] * df["弹性模量"] ** 2 / df["晶粒粒径"] ** 2, 1 / 3
    )
    features.append(contact_stress.values.reshape(-1, 1))
    feature_names.append("接触应力")

    # 5. 能量输入
    energy_input = df["压溃压力"] * df["磨粒速度"]
    features.append(energy_input.values.reshape(-1, 1))
    feature_names.append("能量输入")

    # 6. 硬度/弹性模量比
    features.append((df["莫氏硬度"] / df["弹性模量"]).values.reshape(-1, 1))
    feature_names.append("硬度弹性模量比")

    # 7. Hall‑Petch
    features.append((1 / np.sqrt(df["晶粒粒径"])).values.reshape(-1, 1))
    feature_names.append("Hall‑Petch")

    # 8. 材料平均裂纹深度 (target encoding idea)
    material_mean = df.groupby("材料种类")["亚表面裂纹深度"].transform("mean")
    features.append(material_mean.values.reshape(-1, 1))
    feature_names.append("材料平均裂纹深度")

    X = np.hstack(features)
    material_enc = LabelEncoder().fit_transform(df["材料种类"])
    return X, feature_names, material_enc


def plot_model_comparison(results, y_test):
    """Scatter grids + R²/RMSE bar charts."""

    # 新增：创建 results 文件夹
    save_dir = "results"
    os.makedirs(save_dir, exist_ok=True)

    # 新增：查找已有编号，生成新编号
    scatter_files = glob.glob(os.path.join(save_dir, "scatter_grid_*.png"))
    bar_files = glob.glob(os.path.join(save_dir, "bar_charts_*.png"))
    run_id = max(
        [int(os.path.splitext(os.path.basename(f))[0].split("_")[-1]) for f in scatter_files + bar_files] + [0]
    ) + 1

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
        ax.fill_between(lims, [l - rmse for l in lims], [l + rmse for l in lims], alpha=0.2)
        ax.set_title(f"{name}\nR²={res['test_r2']:.3f}, RMSE={rmse:.3f}")
        ax.set_xlabel("Actual (μm)")
        ax.set_ylabel("Predicted (μm)")
        ax.grid(alpha=0.3)

    # remove unused subplots
    for ax in axes[len(results) :]:
        ax.axis("off")

    plt.tight_layout()
    scatter_path = os.path.join(save_dir, f"scatter_grid_{run_id}.png")
    plt.savefig(scatter_path, dpi=300, bbox_inches="tight")
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

    plt.xticks(rotation=45)
    plt.tight_layout()
    bar_path = os.path.join(save_dir, f"bar_charts_{run_id}.png")
    plt.savefig(bar_path, dpi=300, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main():

    CSV_PATH = "3011916164精确材料性能数据.csv"
    SEED = 42

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("Reading data ...")
    df = pd.read_csv(CSV_PATH)

    # ---------------- traditional features ----------------
    X_all, feat_names, _ = create_ensemble_features(df)
    y_all = df["亚表面裂纹深度"].values

    X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
        X_all, y_all, df, test_size=0.2, random_state=SEED
    )

    scaler_tab = StandardScaler().fit(X_train)
    X_train_s, X_test_s = scaler_tab.transform(X_train), scaler_tab.transform(X_test)

    print("Training classical models ...")
    baseline_results = evaluate_traditional_models(
        X_train_s, X_test_s, y_train, y_test
    )

    # ------------------- PGNN ---------------------------
    num_cols = [
        "莫氏硬度",
        "弹性模量",
        "断裂韧性",
        "晶粒粒径",
        "压溃压力",
        "磨粒速度",
    ]

    # --- Remove double scaling: PGNN will scale internally ---

    train_dict = prepare_material_data(df_train)
    test_dict = prepare_material_data(df_test)

    model = MaterialCrackPGNN(
    num_materials      = train_dict["num_materials"],  # 必填
    material_embed_dim = 32,    # 你想要的嵌入维度
    latent_dim         = 128    # 你想要的潜在/隐藏维度
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Training PGNN ...")
    train_losses, val_losses = train_material_model(
        model,
        train_data=train_dict,
        val_data=test_dict,
        epochs=800,
        lr=5e-4,
        device=device,
    )

    # 新增：画训练损失曲线并自动编号保存
    save_dir = "results"
    os.makedirs(save_dir, exist_ok=True)
    loss_files = glob.glob(os.path.join(save_dir, "pgnn_loss_curve_*.png"))
    loss_run_id = max(
        [int(os.path.splitext(os.path.basename(f))[0].split("_")[-1]) for f in loss_files] + [0]
    ) + 1

    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.yscale("log")
    plt.legend()
    plt.title("PGNN Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    loss_path = os.path.join(save_dir, f"pgnn_loss_curve_{loss_run_id}.png")
    plt.savefig(loss_path, dpi=300)
    plt.show()

    model.eval()
    with torch.no_grad():
        mat_ids = torch.LongTensor(test_dict["material_ids"]).to(device)
        num_feats = torch.FloatTensor(test_dict["numeric_features"]).to(device)
        preds, *_ = model(mat_ids, num_feats)
        preds = preds.cpu().squeeze().numpy()

    pgnn_r2 = r2_score(y_test, preds)
    pgnn_rmse = np.sqrt(mean_squared_error(y_test, preds))
    pgnn_mae = mean_absolute_error(y_test, preds)

    print(f"PGNN   => R²={pgnn_r2:.3f},  RMSE={pgnn_rmse:.3f},  MAE={pgnn_mae:.3f}")

    baseline_results["PGNN"] = {
        "model": model,
        "predictions": preds,
        "test_r2": pgnn_r2,
        "test_mse": pgnn_rmse ** 2,
        "test_mae": pgnn_mae,
    }

    # ---------------- Visual comparison ------------------
    plot_model_comparison(baseline_results, y_test)


if __name__ == "__main__":
    main()
