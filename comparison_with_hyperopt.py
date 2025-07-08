# -*- coding: utf-8 -*-
"""
Enhanced comparison script with hyperparameter optimization for PGNN
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import optuna
from optuna.trial import TrialState
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import json

# Traditional ML models
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.linear_model import ElasticNet
from sklearn.neural_network import MLPRegressor

# PGNN imports
from improved_material_crack_pgnn import (
    MaterialCrackPGNN,
    PhysicsInformedLoss,
    prepare_material_data,
)

# Enable anomaly detection
torch.autograd.set_detect_anomaly(True)
torch.backends.cuda.matmul.allow_tf32 = False

# ---------------------------------------------------------------------------
# Hyperparameter optimization functions
# ---------------------------------------------------------------------------

def create_pgnn_with_hyperparams(trial, num_materials):
    """Create PGNN model with trial hyperparameters"""
    material_embed_dim = trial.suggest_int("material_embed_dim", 8, 64, step=8)
    latent_dim = trial.suggest_int("latent_dim", 16, 128, step=16)
    
    model = MaterialCrackPGNN(
        num_materials=num_materials,
        material_embed_dim=material_embed_dim,
        latent_dim=latent_dim
    )
    
    # Adjust dropout
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5, step=0.05)
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = dropout_rate
    
    return model

def train_pgnn_with_trial(trial, train_data, val_data, num_materials, device='cuda'):
    """Train PGNN with hyperparameters from trial"""
    
    # Training hyperparameters
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    epochs = trial.suggest_int("epochs", 200, 800, step=100)
    
    # Loss hyperparameters
    lambda_complexity = trial.suggest_float("lambda_complexity", 1e-5, 1e-2, log=True)
    lambda_physics = trial.suggest_float("lambda_physics", 0, 0.5, step=0.05)
    
    # Optimizer
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW", "RMSprop"])
    weight_decay = trial.suggest_float("weight_decay", 0, 0.01) if optimizer_name == "AdamW" else 0
    
    # Create model
    model = create_pgnn_with_hyperparams(trial, num_materials).to(device)
    
    # Optimizer
    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=20, factor=0.5, min_lr=1e-6
    )
    
    # Loss function
    loss_fn = PhysicsInformedLoss(
        lambda_complexity=lambda_complexity,
        lambda_physics=lambda_physics
    )
    
    # Prepare data
    train_materials = torch.LongTensor(train_data['material_ids']).to(device)
    train_features = torch.FloatTensor(train_data['numeric_features']).to(device)
    train_target = torch.FloatTensor(train_data['target']).unsqueeze(1).to(device)
    
    val_materials = torch.LongTensor(val_data['material_ids']).to(device)
    val_features = torch.FloatTensor(val_data['numeric_features']).to(device)
    val_target = torch.FloatTensor(val_data['target']).unsqueeze(1).to(device)
    
    # Training
    best_val_r2 = -float('inf')
    patience = 50
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        pred, uncertainty, _, _ = model(train_materials, train_features)
        loss, _ = loss_fn(pred, uncertainty, train_target, model, train_features)
        
        if torch.isnan(loss):
            return -float('inf')
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_pred, _, _, _ = model(val_materials, val_features)
            val_r2 = r2_score(
                val_target.cpu().squeeze().numpy(),
                val_pred.cpu().squeeze().numpy()
            )
        
        scheduler.step(-val_r2)  # Minimize negative R2
        
        # Early stopping
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            break
        
        # Report to Optuna
        trial.report(val_r2, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return best_val_r2

def optimize_hyperparameters(train_df, val_df, n_trials=30, seed=42):
    """Run hyperparameter optimization"""
    
    train_data = prepare_material_data(train_df)
    val_data = prepare_material_data(val_df)
    num_materials = train_data['num_materials']
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=30),
        sampler=optuna.samplers.TPESampler(seed=seed)
    )
    
    objective = lambda trial: train_pgnn_with_trial(
        trial, train_data, val_data, num_materials, device
    )
    
    print("\n" + "="*60)
    print("Starting hyperparameter optimization...")
    print("="*60)
    
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"\nBest validation R²: {study.best_value:.4f}")
    print("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    return study.best_params

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
    
    material_mean = df.groupby("材料种类")["亚表面裂纹深度"].transform("mean")
    features.append(material_mean.values.reshape(-1, 1))
    feature_names.append("材料平均裂纹深度")
    
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
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"bar_charts_{run_id}.png"), dpi=300)
    plt.show()

# ---------------------------------------------------------------------------
# Main experiment with hyperopt
# ---------------------------------------------------------------------------

from sklearn.model_selection import StratifiedKFold, GroupKFold

def main():
    CSV_PATH = "3011916164精确材料性能数据.csv"
    SEED = 42
    USE_HYPEROPT = True  # Toggle hyperparameter optimization
    N_TRIALS = 30  # Number of hyperopt trials
    
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    print("Reading data...")
    df = pd.read_csv(CSV_PATH)
    
    # Split data using StratifiedKFold or GroupKFold
    group_kfold = GroupKFold(n_splits=5)
    groups = df["材料种类"].values
    
    for train_idx, val_idx in group_kfold.split(df, groups=groups):
        df_train, df_val = df.iloc[train_idx], df.iloc[val_idx]
        break  # 只取一次分割
    
    df_test = df.sample(frac=0.2, random_state=SEED)
    df_train = df.drop(df_test.index)
    
    # Traditional ML features
    X_all, feat_names = create_ensemble_features(df)
    y_all = df["亚表面裂纹深度"].values
    
    X_train_all, X_test, y_train_all, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=SEED
    )
    
    scaler = StandardScaler().fit(X_train_all)
    X_train_s = scaler.transform(X_train_all)
    X_test_s = scaler.transform(X_test)
    
    print("\nTraining classical models...")
    baseline_results = evaluate_traditional_models(X_train_s, X_test_s, y_train_all, y_test)
    
    # PGNN with hyperopt
    if USE_HYPEROPT:
        # Hyperparameter optimization
        best_params = optimize_hyperparameters(df_train, df_val, n_trials=N_TRIALS, seed=SEED)
        
        # Save best params
        save_dir = "results"
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "best_hyperparams.json"), "w") as f:
            json.dump(best_params, f, indent=2)
    else:
        # Default params
        best_params = {
            "material_embed_dim": 32,
            "latent_dim": 64,
            "dropout_rate": 0.2,
            "lr": 5e-4,
            "epochs": 800,
            "lambda_complexity": 0,
            "lambda_physics": 0.1,
            "optimizer": "Adam",
        }
    
    # Train PGNN with best params
    print(f"\nTraining PGNN with {'optimized' if USE_HYPEROPT else 'default'} parameters...")
    
    # Prepare data
    train_data = prepare_material_data(df_train if USE_HYPEROPT else df.iloc[df_train.index])
    test_data = prepare_material_data(df_test)
    
    # Adjust sample weights for deep cracks
    train_data['sample_weights'] = np.where(train_data['target'] > 4, 5, 1)  # 深裂纹权重增加
    
    # Create model
    model = MaterialCrackPGNN(
        num_materials=train_data["num_materials"],
        material_embed_dim=best_params["material_embed_dim"],
        latent_dim=best_params["latent_dim"]
    )
    
    # Set dropout
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = best_params["dropout_rate"]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    # Optimizer
    if best_params["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=best_params["lr"])
    elif best_params["optimizer"] == "AdamW":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=best_params["lr"],
            weight_decay=best_params.get("weight_decay", 0)
        )
    else:
        optimizer = torch.optim.RMSprop(model.parameters(), lr=best_params["lr"])
    
    # Loss function
    loss_fn = PhysicsInformedLoss(
        lambda_complexity=best_params.get("lambda_complexity", 0),
        lambda_physics=best_params.get("lambda_physics", 0.1)
    )
    
    # Training data
    train_materials = torch.LongTensor(train_data['material_ids']).to(device)
    train_features = torch.FloatTensor(train_data['numeric_features']).to(device)
    train_target = torch.FloatTensor(train_data['target']).unsqueeze(1).to(device)
    train_weights = torch.FloatTensor(train_data['sample_weights']).to(device)  # 使用样本权重
    
    test_materials = torch.LongTensor(test_data['material_ids']).to(device)
    test_features = torch.FloatTensor(test_data['numeric_features']).to(device)
    test_target = torch.FloatTensor(test_data['target']).unsqueeze(1).to(device)
    
    # Train
    train_losses = []
    val_losses = []
    
    for epoch in range(best_params["epochs"]):
        model.train()
        optimizer.zero_grad()
        
        pred, uncertainty, _, _ = model(train_materials, train_features)
        loss, _ = loss_fn(pred, uncertainty, train_target, model, train_features)
        loss = (loss * train_weights).mean()  # 加权损失
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        train_losses.append(loss.item())
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{best_params['epochs']}: Loss={loss.item():.4f}")
    
    # Evaluate PGNN
    model.eval()
    with torch.no_grad():
        preds, *_ = model(test_materials, test_features)
        preds = preds.cpu().squeeze().numpy()
    
    pgnn_r2 = r2_score(test_target.cpu().squeeze().numpy(), preds)
    pgnn_rmse = np.sqrt(mean_squared_error(test_target.cpu().squeeze().numpy(), preds))
    pgnn_mae = mean_absolute_error(test_target.cpu().squeeze().numpy(), preds)
    
    print(f"\nPGNN => R²={pgnn_r2:.3f}, RMSE={pgnn_rmse:.3f}, MAE={pgnn_mae:.3f}")
    
    baseline_results["PGNN (Optimized)" if USE_HYPEROPT else "PGNN"] = {
        "model": model,
        "predictions": preds,
        "test_r2": pgnn_r2,
        "test_mse": pgnn_rmse ** 2,
        "test_mae": pgnn_mae,
    }
    
    # Plot comparisons
    plot_model_comparison(baseline_results, test_target.cpu().squeeze().numpy())
    
    # Plot training curve
    plt.figure()
    plt.plot(train_losses)
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("PGNN Training Loss")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join("results", "pgnn_training_loss.png"), dpi=300)
    plt.show()
    
    # Print summary
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    for name, res in sorted(baseline_results.items(), key=lambda x: x[1]["test_r2"], reverse=True):
        print(f"{name:20s} | R²: {res['test_r2']:.4f} | RMSE: {np.sqrt(res['test_mse']):.4f}")

if __name__ == "__main__":
    main()
