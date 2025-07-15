"""
模型训练相关函数
包含基础训练循环和评估函数
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from models import GRAD_CLIP
from losses import PhysicsInformedLoss


def train_material_model(
    model, train_data, val_data, epochs=100, lr=5e-4, device="cuda"
):
    """
    训练材料裂纹预测模型
    
    参数:
        model: 模型实例
        train_data: 训练数据字典
        val_data: 验证数据字典
        epochs: 训练轮数
        lr: 学习率
        device: 计算设备
    
    返回:
        训练后的模型，训练损失列表，验证损失列表
    """
    model = model.to(device)
    
    # 优化器和调度器
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # 损失函数
    loss_fn = PhysicsInformedLoss()
    
    # 准备数据
    train_materials = torch.LongTensor(train_data["material_ids"]).to(device)
    train_scaled = torch.FloatTensor(train_data["numeric_features"]).to(device)
    train_raw = torch.FloatTensor(train_data["numeric_raw"]).to(device)
    train_target = torch.FloatTensor(train_data["target"]).unsqueeze(1).to(device)
    
    val_materials = torch.LongTensor(val_data["material_ids"]).to(device)
    val_scaled = torch.FloatTensor(val_data["numeric_features"]).to(device)
    val_raw = torch.FloatTensor(val_data["numeric_raw"]).to(device)
    val_target = torch.FloatTensor(val_data["target"]).unsqueeze(1).to(device)
    
    train_losses = []
    val_losses = []
    
    print(f"Starting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        # 更新损失函数的epoch信息（用于动态权重）
        loss_fn.update_epoch(epoch, epochs)
        
        # 训练阶段
        model.train()
        optimizer.zero_grad()
        
        # 前向传播
        pred, uncertainty, _, _ = model(train_materials, train_scaled, train_raw)
        loss, metrics = loss_fn(pred, uncertainty, train_target, model, train_raw)
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        
        optimizer.step()
        train_losses.append(loss.item())
        
        # 验证阶段（每10个epoch验证一次）
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_pred, val_uncertainty, _, _ = model(val_materials, val_scaled, val_raw)
                val_loss, _ = loss_fn(val_pred, val_uncertainty, val_target, model, val_raw)
                val_losses.append(val_loss.item())
                
                # 计算R²分数
                val_r2 = r2_score(
                    val_target.cpu().numpy(), 
                    val_pred.cpu().numpy()
                )
                
                print(f"Epoch {epoch}: Train Loss = {loss.item():.4f}, "
                      f"Val Loss = {val_loss.item():.4f}, Val R² = {val_r2:.4f}")
                print(f"  Metrics: NLL = {metrics['nll']:.4f}, "
                      f"Physics = {metrics['physics']:.4f} "
                      f"(weight = {metrics['physics_weight']:.4f})")
        
        # 更新学习率
        scheduler.step()
        
    return model, train_losses, val_losses


def evaluate_model_detailed(model, test_data, device="cuda"):
    """
    详细评估模型性能，包括不确定性分析
    
    参数:
        model: 训练好的模型
        test_data: 测试数据字典
        device: 计算设备
    
    返回:
        包含各种评估指标的结果字典
    """
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
            test_materials, test_features, test_raw
        )

        # 转换为NumPy
        test_pred_np = test_pred.cpu().squeeze().numpy()
        test_target_np = test_target.cpu().squeeze().numpy()
        test_uncertainty_np = test_uncertainty.cpu().squeeze().numpy()
        symbolic_np = symbolic_out.cpu().squeeze().numpy()
        residual_np = residual_out.cpu().squeeze().numpy()

        # 计算基础指标
        r2 = r2_score(test_target_np, test_pred_np)
        rmse = np.sqrt(mean_squared_error(test_target_np, test_pred_np))
        mae = np.mean(np.abs(test_target_np - test_pred_np))

        # 计算符号表达式和残差网络的贡献
        symbolic_contrib = np.mean(
            np.abs(symbolic_np) / (np.abs(symbolic_np) + np.abs(residual_np) + 1e-6)
        )

        # 计算不确定性与误差的相关性
        errors = np.abs(test_pred_np - test_target_np)
        uncertainty_error_corr = np.corrcoef(errors, test_uncertainty_np)[0, 1]

    # MC Dropout预测（带不确定性）
    mc_pred, mc_unc, aleatoric_unc, epistemic_unc = model.predict_with_uncertainty(
        test_materials, test_features, test_raw, n_samples=30
    )
    
    mc_pred_np = mc_pred.cpu().squeeze().numpy()
    mc_unc_np = mc_unc.cpu().squeeze().numpy()
    aleatoric_np = aleatoric_unc.cpu().squeeze().numpy()
    epistemic_np = epistemic_unc.cpu().squeeze().numpy()

    # MC Dropout指标
    mc_r2 = r2_score(test_target_np, mc_pred_np)
    mc_rmse = np.sqrt(mean_squared_error(test_target_np, mc_pred_np))

    # 不确定性分析
    mean_epistemic = np.mean(epistemic_np)
    mean_aleatoric = np.mean(aleatoric_np)
    ratio = mean_epistemic / (mean_aleatoric + 1e-6)

    # 不确定性校准
    errors = np.abs(mc_pred_np - test_target_np)
    calibration_scores = errors / (np.sqrt(mc_unc_np) + 1e-6)
    well_calibrated = np.mean((calibration_scores <= 2.0))

    # 打印详细结果
    print("\n===== 详细模型评估 =====")
    print(f"标准预测:")
    print(f"  R²: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    print(f"  符号表达式贡献: {symbolic_contrib:.2f} ({symbolic_contrib*100:.1f}%)")
    print(f"  不确定性-误差相关性: {uncertainty_error_corr:.4f}")

    print(f"\nMC Dropout预测:")
    print(f"  R²: {mc_r2:.4f}, RMSE: {mc_rmse:.4f}")
    print(f"  认知不确定性 (Epistemic): {mean_epistemic:.4f}")
    print(f"  随机不确定性 (Aleatoric): {mean_aleatoric:.4f}")
    print(f"  不确定性比例 (Epistemic/Aleatoric): {ratio:.4f}")

    print(f"\n不确定性校准:")
    print(f"  良好校准比例: {well_calibrated:.2f} ({well_calibrated*100:.1f}%)")

    # 返回结果字典
    results = {
        "standard": {
            "r2": r2, 
            "rmse": rmse, 
            "mae": mae,
            "symbolic_contrib": symbolic_contrib,
            "uncertainty_error_corr": uncertainty_error_corr
        },
        "mc_dropout": {
            "r2": mc_r2, 
            "rmse": mc_rmse
        },
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