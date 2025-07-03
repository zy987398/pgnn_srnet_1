import optuna
from optuna.trial import TrialState
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import copy
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.metrics import r2_score, mean_squared_error
import warnings

warnings.filterwarnings("ignore")
from torch.utils.data import TensorDataset, DataLoader

from improved_material_crack_pgnn import (
    MaterialCrackPGNN,
    PhysicsInformedLoss,
    prepare_material_data,
)


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


def augment_dataset(orig_df, noise_scale=0.05):
    """在数值列上添加噪声的数据增强"""
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


def create_model(trial, num_materials, num_mean, num_std):
    """根据trial创建模型"""
    # 模型架构超参数
    material_embed_dim = trial.suggest_int(
        "material_embed_dim", 8, 32, step=8
    )  # 缩小范围
    latent_dim = trial.suggest_int("latent_dim", 16, 64, step=16)  # 缩小范围

    # 创建模型并修改部分架构
    model = MaterialCrackPGNN(
        num_materials=num_materials,
        material_embed_dim=material_embed_dim,
        latent_dim=latent_dim,
        num_mean=num_mean,
        num_std=num_std,
    )

    # 动态调整dropout率
    dropout_rate = trial.suggest_float(
        "dropout_rate", 0.05, 0.2, step=0.05
    )  # 更窄的范围
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = dropout_rate

    return model


def train_with_trial(trial, train_data, val_data, num_materials, device="cuda"):
    """使用给定的超参数训练模型"""

    # 训练超参数 - 缩小学习率范围
    lr = trial.suggest_float("lr", 0.0005, 0.003, log=True)  # 更窄的范围
    epochs = trial.suggest_int("epochs", 200, 500, step=50)  # 减小上限
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

    # 权重衰减 - 新增
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)

    # 损失函数超参数
    lambda_complexity = trial.suggest_float("lambda_complexity", 1e-6, 1e-4, log=True)
    lambda_physics = trial.suggest_float("lambda_physics", 0.05, 0.20, step=0.01)

    # 早停参数 - 更敏感
    patience = trial.suggest_int("patience", 20, 40, step=5)

    # 数据增强
    use_augmentation = trial.suggest_categorical("augmentation", [True, False])

    # 优化器选择
    optimizer_name = trial.suggest_categorical("optimizer", ["AdamW", "RMSprop"])

    # 创建模型
    model = create_model(trial, num_materials, train_data["num_mean"], train_data["num_std"]).to(device)

    # 创建优化器
    if optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
    else:
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)

    # 损失函数
    loss_fn = PhysicsInformedLoss(
        lambda_complexity=lambda_complexity,
        lambda_physics=lambda_physics,
        weight_decay=0.0,
    )
    loss_fn.total_epochs = epochs

    # 准备数据 - 可选数据增强
    train_df = reconstruct_df(train_data)
    if use_augmentation:
        train_df = augment_dataset(train_df)

    augmented_train_data = prepare_material_data(
        train_df,
        scaler=train_data["scaler"],
        le=train_data["label_encoder"],
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

    # 学习率预热与调度 (在DataLoader创建后确定steps_per_epoch)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
    )

    val_materials = torch.LongTensor(val_data["material_ids"]).to(device)
    val_features = torch.FloatTensor(val_data["numeric_features"]).to(device)
    val_raw = torch.FloatTensor(val_data["numeric_raw"]).to(device)
    val_target = torch.FloatTensor(val_data["target"]).unsqueeze(1).to(device)

    # 训练循环
    best_val_r2 = -float("inf")
    patience_counter = 0
    best_model_wts = None

    for epoch in range(epochs):
        loss_fn.update_epoch(epoch, epochs)
        model.set_epoch(epoch, epochs)

        # 训练
        model.train()
        epoch_loss = 0

        for batch_materials, batch_features, batch_raw, batch_target in train_loader:
            batch_materials = batch_materials.to(device, non_blocking=True)
            batch_features = batch_features.to(device, non_blocking=True)
            batch_raw = batch_raw.to(device, non_blocking=True)
            batch_target = batch_target.to(device, non_blocking=True)

            optimizer.zero_grad()

            pred1, unc1, _, _ = model(batch_materials, batch_features)
            pred2, unc2, _, _ = model(batch_materials, batch_features)
            loss1, metrics = loss_fn(pred1, unc1, batch_target, model, batch_raw)
            loss2, _ = loss_fn(pred2, unc2, batch_target, model, batch_raw)
            kl = F.kl_div(pred1.log(), pred2, reduction='batchmean') + F.kl_div(pred2.log(), pred1, reduction='batchmean')
            loss = (loss1 + loss2) / 2 + 0.5 * kl

            # 检查NaN
            if torch.isnan(loss) or torch.isinf(loss):
                trial.report(-1.0, epoch)
                raise optuna.TrialPruned()

            loss.backward()
            # 更严格的梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item() * batch_target.size(0)

        epoch_loss /= len(train_dataset)

        # 验证
        model.eval()
        with torch.no_grad():
            val_pred, val_uncertainty, _, _ = model(val_materials, val_features)
            val_loss, _ = loss_fn(val_pred, val_uncertainty, val_target, model, val_raw)

            val_pred_np = np.expm1(val_pred.cpu().squeeze().numpy())
            val_target_np = np.expm1(val_target.cpu().squeeze().numpy())

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
            break

        # 向Optuna报告中间结果
        trial.report(val_r2, epoch)

        # 如果trial应该被剪枝则停止
        if trial.should_prune():
            raise optuna.TrialPruned()

    # 恢复最佳模型状态进行最终评估
    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)
        model = model.to(device)

        # 再次计算R²
        model.eval()
        with torch.no_grad():
            val_pred, _, _, _ = model(val_materials, val_features)
            val_pred_np = np.expm1(val_pred.cpu().squeeze().numpy())
            val_target_np = np.expm1(val_target.cpu().squeeze().numpy())
            final_r2 = r2_score(val_target_np, val_pred_np)

        return final_r2

    return best_val_r2


def objective(trial, df, device, seed):
    """Optuna目标函数，5折交叉验证"""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    fold_scores = []
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

        r2 = train_with_trial(trial, train_data, val_data, num_materials, device)
        fold_scores.append(r2)

    return float(np.mean(fold_scores))


def run_hyperopt(data_path="3011916164精确材料性能数据.csv", n_trials=100, seed=42):
    """运行超参数优化"""

    # 设置随机种子
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 读取数据
    df = pd.read_csv(data_path)

    num_materials = df["材料种类"].nunique()

    # 设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 创建study - 使用更好的搜索策略
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=30),
        sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=5),
    )

    # 优化
    study.optimize(
        lambda trial: objective(trial, df, device, seed),
        n_trials=n_trials,
        show_progress_bar=True,
        catch=(Exception,),
    )

    # 打印结果
    print("\n" + "=" * 50)
    print("优化完成!")
    print("=" * 50)

    print(f"\n最佳验证R²: {study.best_value:.4f}")
    print("\n最佳超参数:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # 统计信息
    completed_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
    print(f"\n完成的trials: {len(completed_trials)}/{n_trials}")

    # 超参数重要性
    try:
        importance = optuna.importance.get_param_importances(study)
        print("\n超参数重要性:")
        for key, value in sorted(importance.items(), key=lambda x: x[1], reverse=True):
            print(f"  {key}: {value:.3f}")
    except:
        pass

    return study


# 使用最佳超参数重新训练
def retrain_with_best_params(
    study, data_path="3011916164精确材料性能数据.csv", seed=42
):
    """使用最佳超参数在5折交叉验证上训练模型并返回集成"""

    df = pd.read_csv(data_path)
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    best_params = study.best_params
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ensemble_models = []
    fold = 0
    for train_idx, val_idx in kf.split(df):
        fold += 1
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]

        train_data = prepare_material_data(train_df)
        val_data = prepare_material_data(
            val_df,
            scaler=train_data["scaler"],
            le=train_data["label_encoder"],
        )

        use_augmentation = best_params.get("augmentation", False)
        train_df_aug = train_df.copy()
        if use_augmentation:
            train_df_aug = augment_dataset(train_df_aug)
        augmented_train_data = prepare_material_data(
            train_df_aug,
            scaler=train_data["scaler"],
            le=train_data["label_encoder"],
            augment=False,
        )

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

        val_materials = torch.LongTensor(val_data["material_ids"]).to(device)
        val_features = torch.FloatTensor(val_data["numeric_features"]).to(device)
        val_raw = torch.FloatTensor(val_data["numeric_raw"]).to(device)
        val_target = torch.FloatTensor(val_data["target"]).unsqueeze(1).to(device)

        model = MaterialCrackPGNN(
            num_materials=train_data["num_materials"],
            material_embed_dim=best_params["material_embed_dim"],
            latent_dim=best_params["latent_dim"],
            dropout_rate=best_params["dropout_rate"],
            num_mean=train_data["num_mean"],
            num_std=train_data["num_std"],
        ).to(device)

        lr = best_params["lr"]
        if best_params.get("optimizer", "AdamW") == "AdamW":
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=lr, weight_decay=best_params.get("weight_decay", 0)
            )
        else:
            optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)

        epochs = best_params["epochs"]
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            epochs=epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.3,
        )

        loss_fn = PhysicsInformedLoss(
            lambda_complexity=best_params["lambda_complexity"],
            lambda_physics=best_params["lambda_physics"],
            weight_decay=0.0,
        )
        loss_fn.total_epochs = epochs

        best_r2, patience_ctr = -1e9, 0
        best_wts = None

        for epoch in range(epochs):
            loss_fn.update_epoch(epoch, epochs)
            model.set_epoch(epoch, epochs)
            model.train()
            epoch_loss = 0

            for batch_materials, batch_features, batch_raw, batch_target in train_loader:
                batch_materials = batch_materials.to(device, non_blocking=True)
                batch_features = batch_features.to(device, non_blocking=True)
                batch_raw = batch_raw.to(device, non_blocking=True)
                batch_target = batch_target.to(device, non_blocking=True)

                optimizer.zero_grad()

                pred1, unc1, _, _ = model(batch_materials, batch_features)
                pred2, unc2, _, _ = model(batch_materials, batch_features)
                loss1, _ = loss_fn(pred1, unc1, batch_target, model, batch_raw)
                loss2, _ = loss_fn(pred2, unc2, batch_target, model, batch_raw)
                kl = F.kl_div(pred1.log(), pred2, reduction="batchmean") + F.kl_div(
                    pred2.log(), pred1, reduction="batchmean"
                )
                loss = (loss1 + loss2) / 2 + 0.5 * kl

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item() * batch_target.size(0)

            epoch_loss /= len(train_dataset)

            model.eval()
            with torch.no_grad():
                val_pred, _, _, _ = model(val_materials, val_features)
                val_pred_np = np.expm1(val_pred.cpu().squeeze().numpy())
                val_target_np = np.expm1(val_target.cpu().squeeze().numpy())
                val_r2 = r2_score(val_target_np, val_pred_np)

            if val_r2 > best_r2:
                best_r2 = val_r2
                best_wts = copy.deepcopy(model.state_dict())
                patience_ctr = 0
            else:
                patience_ctr += 1

            if patience_ctr >= 25:
                break

        if best_wts is not None:
            model.load_state_dict(best_wts)

        ensemble_models.append(copy.deepcopy(model))

    return ensemble_models, None, None


def evaluate_model_detailed(model, test_data, device="cuda"):
    """详细评估模型性能，包括不确定性分析

    Args:
        model: 单个模型或模型列表(用于集成)
        test_data: 预处理后的测试数据
    """

    is_ensemble = isinstance(model, list)
    if is_ensemble:
        models = [m.to(device) for m in model]
        for m in models:
            m.eval()
    else:
        model = model.to(device)
        model.eval()

    # 准备数据
    test_materials = torch.LongTensor(test_data["material_ids"]).to(device)
    test_features = torch.FloatTensor(test_data["numeric_features"]).to(device)
    test_raw = torch.FloatTensor(test_data["numeric_raw"]).to(device)
    test_target = torch.FloatTensor(test_data["target"]).unsqueeze(1).to(device)

    # 标准预测
    with torch.no_grad():
        if is_ensemble:
            preds = []
            uncs = []
            syms = []
            ress = []
            for m in models:
                p, u, s, r = m(test_materials, test_features)
                preds.append(p)
                uncs.append(u)
                syms.append(s)
                ress.append(r)
            test_pred = torch.stack(preds).mean(0)
            test_uncertainty = torch.stack(uncs).mean(0)
            symbolic_out = torch.stack(syms).mean(0)
            residual_out = torch.stack(ress).mean(0)
        else:
            test_pred, test_uncertainty, symbolic_out, residual_out = model(
                test_materials, test_features
            )

        # 转换为NumPy
        test_pred_np = np.expm1(test_pred.cpu().squeeze().numpy())
        test_target_np = np.expm1(test_target.cpu().squeeze().numpy())
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
    if is_ensemble:
        mc_preds = []
        mc_uncs = []
        ale_uncs = []
        epi_uncs = []
        for m in models:
            p, u, a, e = m.predict_with_uncertainty(
                test_materials, test_features, n_samples=30
            )
            mc_preds.append(p)
            mc_uncs.append(u)
            ale_uncs.append(a)
            epi_uncs.append(e)
        mc_pred = torch.stack(mc_preds).mean(0)
        mc_unc = torch.stack(mc_uncs).mean(0)
        aleatoric_unc = torch.stack(ale_uncs).mean(0)
        epistemic_unc = torch.stack(epi_uncs).mean(0)
    else:
        mc_pred, mc_unc, aleatoric_unc, epistemic_unc = model.predict_with_uncertainty(
            test_materials, test_features, n_samples=30
        )
    mc_pred_np = np.expm1(mc_pred.cpu().squeeze().numpy())
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


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # 运行超参数优化
    study = run_hyperopt(n_trials=50)  # 可以增加trials获得更好结果

    # 使用最佳参数重新训练
    ensemble_models, train_losses, test_r2_history = retrain_with_best_params(study)
    train_losses = train_losses or []

    # 详细评估
    df = pd.read_csv("3011916164精确材料性能数据.csv")
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_data = prepare_material_data(train_df)
    test_data = prepare_material_data(
        test_df,
        scaler=train_data["scaler"],
        le=train_data["label_encoder"],
    )

    results = evaluate_model_detailed(ensemble_models, test_data)

    # 可视化优化历史
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 优化历史
    ax1.plot([t.value for t in study.trials if t.value is not None])
    ax1.set_xlabel("Trial")
    ax1.set_ylabel("Validation R²")
    ax1.set_title("Hyperparameter Optimization History")
    ax1.grid(True, alpha=0.3)

    # 训练损失
    ax2.plot(train_losses)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.set_title("Training Loss with Best Hyperparameters")
    ax2.set_yscale("log")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/improved_hyperopt_results.png", dpi=300)

    # 预测vs真实值可视化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

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

    plt.show()
