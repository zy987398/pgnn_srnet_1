import torch
import torch.nn as nn
import torch.nn.functional as F

# Enable cuDNN autotuner for fixed-size inputs
torch.backends.cudnn.benchmark = True
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Dict, List, Tuple, Optional
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader


def safe_expm1(x: torch.Tensor) -> torch.Tensor:
    """GPU-safe expm1 with clipping."""
    return torch.clamp(x, -10, 10).expm1()


class MaterialEmbedding(nn.Module):
    """材料种类嵌入层"""

    def __init__(self, num_materials, embedding_dim=16):
        super().__init__()
        self.embedding = nn.Embedding(num_materials, embedding_dim)

    def forward(self, material_ids):
        return self.embedding(material_ids)


class PhysicsInformedFeatureExtractor(nn.Module):
    """物理信息特征提取器"""

    def __init__(self, input_dim=6):
        super().__init__()
        self.input_dim = input_dim

    def forward(self, x):
        """
        x: [batch_size, 6] - [硬度, 弹性模量, 断裂韧性, 晶粒粒径, 压溃压力, 磨粒速度]
        """
        features = [x]

        # 基于断裂力学的特征工程
        hardness = x[:, 0:1]
        elastic_modulus = x[:, 1:2]
        fracture_toughness = x[:, 2:3]
        grain_size = x[:, 3:4]
        pressure = x[:, 4:5]
        velocity = x[:, 5:6]

        # 1. Griffith能量释放率相关特征
        # Griffith G = (P² √a)/E
        sqrt_g = torch.sqrt(torch.abs(grain_size) + 1e-6)
        energy_release_rate = (pressure**2) * sqrt_g / (elastic_modulus.abs() + 1e-6)
        features.append(energy_release_rate)

        # K_ratio = P √a / K_IC
        k_ratio = pressure * sqrt_g / (fracture_toughness.abs() + 1e-6)
        features.append(k_ratio)

        # Hertz contact stress ∝ (P·E²/a²)^{1/3}
        expr = pressure * elastic_modulus**2 / (grain_size**2 + 1e-6)
        contact_stress = torch.sign(expr) * torch.pow(expr.abs() + 1e-6, 1 / 3)
        features.append(contact_stress)

        # Hall-Petch
        hall_petch = 1 / torch.sqrt(torch.abs(grain_size) + 1e-6)
        features.append(hall_petch)

        # 3. 材料脆性指数
        # B = H/K_IC (Lawn's brittleness index)
        brittleness = hardness / (fracture_toughness + 1e-6)
        features.append(brittleness)

        # 5. 能量输入
        energy_input = pressure * velocity
        features.append(energy_input)

        # 6. 韧性模量比
        toughness_modulus_ratio = fracture_toughness / (elastic_modulus + 1e-6)
        features.append(toughness_modulus_ratio)

        # 改进：添加更多交互特征
        # 硬度与弹性模量的比值 (H/E) - 与塑性变形相关
        he_ratio = hardness / (elastic_modulus + 1e-6)
        features.append(he_ratio)

        # 能量释放率与断裂韧性比值 - 裂纹扩展趋势
        fracture_tendency = energy_release_rate / (fracture_toughness + 1e-6)
        features.append(fracture_tendency)

        return torch.cat(features, dim=1)


class ResidualBlock(nn.Module):
    """残差连接块，提高训练稳定性"""

    def __init__(self, dim, dropout_rate=0.1):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * 2)
        self.act = nn.GELU()  # 替换ReLU为GELU获得更平滑的梯度
        self.layer_norm2 = nn.LayerNorm(dim * 2)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(dim * 2, dim)

    def forward(self, x):
        residual = x
        out = self.layer_norm1(x)
        out = self.fc1(out)
        out = self.act(out)
        out = self.layer_norm2(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out + residual  # 残差连接


class ImprovedSymbolicLayer(nn.Module):
    """改进的符号表达式层，专门针对裂纹深度预测"""

    def __init__(self, latent_dim=32):
        super().__init__()

        # 预定义的物理操作符
        self.operators = ["+", "-", "*", "sqrt", "1/x", "^1.5"]
        self.num_operators = len(self.operators)

        # 物理变量选择网络
        self.variable_selector = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.GELU(),  # 改进：使用GELU激活函数
            nn.LayerNorm(64),  # 改进：LayerNorm替换BatchNorm
            nn.Linear(64, 15),  # 原始+衍生特征 (增加了2个交互特征)
        )

        # 操作符选择网络
        self.operator_selector = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.GELU(),
            nn.LayerNorm(32),
            nn.Linear(32, self.num_operators),
        )

        # 系数预测网络
        self.coefficient_net = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.GELU(),
            nn.LayerNorm(32),
            nn.Linear(32, self.num_operators),  # 与操作符数量一致
        )

    def forward(self, z, physics_features, epoch: int = 0, total_epochs: int = 1):
        """根据当前训练进度生成符号表达式

        Args:
            z: 潜在表示
            physics_features: 物理特征
            epoch: 当前epoch
            total_epochs: 总epoch数
        """
        # Gumbel-Softmax退火温度
        tau = max(0.3, 3.0 * (1 - epoch / float(total_epochs)))

        # 选择变量
        var_weights = F.gumbel_softmax(self.variable_selector(z), tau=tau)

        # 选择操作符
        op_weights = F.gumbel_softmax(self.operator_selector(z), tau=tau)

        # 预测系数
        coeffs = self.coefficient_net(z)

        # 构建符号表达式（加权组合）
        # 使用软选择机制
        selected_features = torch.matmul(
            var_weights.unsqueeze(1), physics_features.unsqueeze(2)
        ).squeeze(-1)

        # 应用不同的操作符（软操作）
        expressions = []

        # 线性项 (+)
        linear = coeffs[:, 0:1] * selected_features
        expressions.append(linear * op_weights[:, 0:1])

        # 反向项 (-)
        negative = coeffs[:, 1:2] * (-selected_features)
        expressions.append(negative * op_weights[:, 1:2])

        # 乘法或平方 (*)
        square = coeffs[:, 2:3] * (selected_features ** 2)
        expressions.append(square * op_weights[:, 2:3])

        # 平方根项 (sqrt)
        sqrt_term = coeffs[:, 3:4] * torch.sqrt(torch.abs(selected_features) + 1e-6)
        expressions.append(sqrt_term * op_weights[:, 3:4])

        # 倒数项 (1/x)
        inv_term = coeffs[:, 4:5] * (1.0 / (selected_features + 1e-3))
        expressions.append(inv_term * op_weights[:, 4:5])

        # 1.5次方项 (^1.5)
        power_term = coeffs[:, 5:6] * torch.sign(selected_features) * torch.pow(torch.abs(selected_features) + 1e-6, 1.5)
        expressions.append(power_term * op_weights[:, 5:6])

        # 组合所有项
        symbolic_output = sum(expressions)

        # 计算复杂度（用于正则化）
        self.complexity = -torch.sum(
            var_weights * torch.log(var_weights + 1e-8), dim=-1
        ).mean()

        return symbolic_output


class MaterialCrackPGNN(nn.Module):
    """专门用于材料裂纹深度预测的PGNN"""

    def __init__(
        self,
        num_materials,
        material_embed_dim=16,
        latent_dim=32,
        dropout_rate=0.1,
        num_mean=None,
        num_std=None,
    ):
        super().__init__()

        # 材料嵌入
        self.material_embedding = MaterialEmbedding(num_materials, material_embed_dim)

        # 物理特征提取
        self.physics_extractor = PhysicsInformedFeatureExtractor()

        if num_mean is not None:
            self.register_buffer("num_mean", torch.tensor(num_mean, dtype=torch.float32))
            self.register_buffer("num_std", torch.tensor(num_std, dtype=torch.float32))
        else:
            self.num_mean = None
            self.num_std = None

        # 改进：计算特征总维度 (包括新增的交互特征)
        physics_dim = 6 + 7 + 2  # 原始特征 + 原有衍生特征 + 新增交互特征
        input_dim = material_embed_dim + physics_dim

        # 编码器 (使用残差连接)
        self.input_layer = nn.Linear(input_dim, latent_dim)
        self.encoder_norm = nn.LayerNorm(latent_dim)

        # 添加残差块
        self.residual_blocks = nn.ModuleList(
            [ResidualBlock(latent_dim, dropout_rate) for _ in range(2)]
        )

        # 符号表达式层
        self.symbolic_layer = ImprovedSymbolicLayer(latent_dim)
        # Compile symbolic layer for fusion and speed
        try:
            self.symbolic_layer = torch.jit.script(self.symbolic_layer)
        except Exception:
            pass

        # FiLM 条件化层
        self.film = nn.Linear(material_embed_dim, latent_dim * 2)

        # 残差网络（捕获非线性效应）
        self.residual_net = nn.Sequential(
            nn.Linear(latent_dim + 1, 64),
            nn.GELU(),
            nn.LayerNorm(64),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

        # 不确定性预测 - 分离认知不确定性和随机不确定性
        self.aleatoric_net = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),  # 输出log_var (随机不确定性)
        )

        # 输出层（确保正值）
        self.output_activation = nn.Softplus()

        # MC Dropout的开关
        self.mc_dropout_enabled = False

        # 训练过程中更新的epoch信息
        self.current_epoch = 0
        self.total_epochs = 1

    def enable_mc_dropout(self, enable=True):
        """启用/禁用MC Dropout"""
        self.mc_dropout_enabled = enable

    def set_epoch(self, epoch, total_epochs=None):
        self.current_epoch = epoch
        if total_epochs is not None:
            self.total_epochs = total_epochs

    def unscaled(self, x):
        if self.num_mean is not None and self.num_std is not None:
            return x * self.num_std + self.num_mean
        return x

    def forward(self, material_ids, numeric_features):
        """
        material_ids: [batch_size] - 材料类别ID
        numeric_features: [batch_size, 6] - 数值特征
        """
        # 材料嵌入
        material_embed = self.material_embedding(material_ids)

        # 提取物理特征 (反标准化后使用物理公式)
        physics_features = self.physics_extractor(self.unscaled(numeric_features))

        # 组合所有特征
        combined_features = torch.cat([material_embed, physics_features], dim=1)

        # 编码到潜在空间 (使用残差结构)
        z = self.input_layer(combined_features)
        z = self.encoder_norm(z)

        # 通过残差块
        for block in self.residual_blocks:
            z = block(z)
            # 在推理时如果启用MC Dropout，保持dropout激活
            if self.mc_dropout_enabled and not self.training:
                z = F.dropout(z, p=0.1, training=True)

        # 符号表达式输出
        symbolic_output = self.symbolic_layer(z, physics_features, self.current_epoch, self.total_epochs)

        # 残差网络输入：符号层输出与潜在表示，切断梯度
        sym_detached = symbolic_output.detach()
        gamma_beta = self.film(material_embed)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        z_norm = F.layer_norm(z, z.shape[-1:])
        res_in = gamma * z_norm + beta
        residual_input = torch.cat([res_in, sym_detached], dim=-1)
        residual_output = self.residual_net(residual_input)

        # 组合预测（非负）
        raw_out = symbolic_output + residual_output
        raw_out = raw_out.clamp(-10, 10)
        prediction = F.softplus(raw_out, beta=5) + 1e-3

        # 不确定性估计 - 随机不确定性
        log_var = self.aleatoric_net(z).clamp(-10, 6)
        uncertainty = torch.exp(log_var) + 1e-6

        return prediction, uncertainty, symbolic_output, residual_output

    def predict_with_uncertainty(self, material_ids, numeric_features, n_samples=10):
        """使用MC Dropout进行带不确定性的预测"""
        self.enable_mc_dropout(True)  # 启用MC Dropout
        self.eval()  # 设置为评估模式

        predictions = []
        uncertainties = []

        with torch.no_grad():
            for _ in range(n_samples):
                pred, uncertainty, _, _ = self(material_ids, numeric_features)
                predictions.append(pred)
                uncertainties.append(uncertainty)

        # 恢复正常模式
        self.enable_mc_dropout(False)

        # 计算均值和不确定性
        mean_pred = torch.mean(torch.stack(predictions), dim=0)
        # 总不确定性 = 认知不确定性 + 随机不确定性
        aleatoric_uncertainty = torch.mean(
            torch.stack(uncertainties), dim=0
        )  # 随机不确定性
        epistemic_uncertainty = torch.var(
            torch.stack(predictions), dim=0
        )  # 认知不确定性
        total_uncertainty = aleatoric_uncertainty + epistemic_uncertainty

        return (
            mean_pred,
            total_uncertainty,
            aleatoric_uncertainty,
            epistemic_uncertainty,
        )


class PhysicsInformedLoss(nn.Module):
    """改进的物理信息损失函数"""

    def __init__(self, lambda_complexity=1e-3, lambda_physics=1e-2, weight_decay=0.0):
        super().__init__()
        self.lambda_complexity = lambda_complexity
        self.lambda_physics = lambda_physics
        self.current_epoch = 0
        self.total_epochs = 1  # 会在训练中更新

    def update_epoch(self, current, total):
        """更新当前epoch和总epoch数"""
        self.current_epoch = current
        self.total_epochs = total

    def get_physics_weight(self):
        """使用余弦退火计算物理约束权重"""
        if self.current_epoch < 10:  # 前10个epoch权重为0，让模型先学习数据
            return 0.0

        # 余弦退火，逐渐增加到最大值，然后保持
        progress = min(1.0, (self.current_epoch - 10) / (self.total_epochs * 0.3))
        weight = self.lambda_physics * (1 - np.cos(progress * np.pi)) / 2
        return weight

    def forward(self, pred, uncertainty, target, model, numeric_features):
        batch_size = pred.shape[0]

        # 1. NLL损失（异方差高斯）
        nll_loss = 0.5 * (
            torch.log(2 * np.pi * uncertainty) + (target - pred) ** 2 / uncertainty
        )
        nll_loss = nll_loss.mean()

        # 2. 复杂度正则化
        complexity_loss = (
            model.symbolic_layer.complexity
            if hasattr(model.symbolic_layer, "complexity")
            else 0
        )

        # 3. 物理约束损失 - 使用余弦退火的权重
        physics_loss = self.compute_physics_constraints(pred, numeric_features)
        physics_weight = self.get_physics_weight()

        # 总损失
        total_loss = (
            nll_loss
            + self.lambda_complexity * complexity_loss
            + physics_weight * physics_loss
        )

        return total_loss, {
            "nll": nll_loss.item(),
            "complexity": (
                complexity_loss.item()
                if torch.is_tensor(complexity_loss)
                else complexity_loss
            ),
            "physics": physics_loss.item(),
            "physics_weight": physics_weight,
            "total": total_loss.item(),
        }

    def compute_physics_constraints(self, pred, features):
        """向量化的物理约束计算"""
        pred_real = safe_expm1(pred.squeeze())

        grain_size = features[:, 3]
        pressure = features[:, 4]
        hardness = features[:, 0]
        fracture_toughness = features[:, 2]

        constraint1 = torch.relu(pred_real - 2.5 * grain_size)
        pressure_factor = torch.sigmoid(-(pressure - 0.2) * 10)
        constraint2 = pressure_factor * pred_real
        brittleness = hardness / (fracture_toughness + 1e-6)
        high_brittleness = torch.sigmoid((brittleness - brittleness.mean()) * 5)
        baseline = safe_expm1(pred.mean().detach()) * 0.5
        constraint3 = high_brittleness * torch.relu(pred_real - baseline)

        constraints = torch.stack([constraint1, constraint2, 0.5 * constraint3])
        return constraints.mean()


# 数据预处理与增强
def prepare_material_data(
    df,
    scaler: Optional[StandardScaler] = None,
    le: Optional[LabelEncoder] = None,
    augment: bool = False,
):
    """准备材料数据，可选数据增强"""
    # 编码材料种类
    if le is None:
        le = LabelEncoder()
        material_ids = le.fit_transform(df["材料种类"])
    else:
        material_ids = le.transform(df["材料种类"])

    # 提取数值特征
    numeric_cols = [
        "莫氏硬度",
        "弹性模量",
        "断裂韧性",
        "晶粒粒径",
        "压溃压力",
        "磨粒速度",
    ]
    numeric_raw = df[numeric_cols].values.astype(np.float32)

    # 目标变量
    target_raw = df["亚表面裂纹深度"].values.astype(np.float32)
    target = np.log1p(target_raw)

    # 使用StandardScaler
    if scaler is None:
        scaler = StandardScaler().fit(numeric_raw)
    numeric_scaled = scaler.transform(numeric_raw)
    num_mean = scaler.mean_.astype(np.float32)
    num_std = scaler.scale_.astype(np.float32)

    # 数据增强 - 只在训练时进行
    if augment:
        # 1. 样本数量
        n_samples = len(material_ids)
        n_augment = n_samples // 4  # 增加25%的样本

        # 2. 随机选择要增强的样本索引
        aug_indices = np.random.choice(n_samples, n_augment, replace=False)

        # 3. 为每个特征添加微小噪声
        aug_material_ids = material_ids[aug_indices].copy()
        aug_numeric_scaled = numeric_scaled[aug_indices].copy()
        aug_numeric_raw = numeric_raw[aug_indices].copy()
        aug_target_raw = target_raw[aug_indices].copy()

        # 4. 添加高斯噪声
        noise_scale = 0.05  # 5%的噪声
        aug_numeric_scaled += np.random.normal(0, noise_scale, aug_numeric_scaled.shape)

        # 为原始特征也添加相应的噪声（确保一致性）
        aug_numeric_raw *= 1 + np.random.normal(0, noise_scale, aug_numeric_raw.shape)

        # 保持裂纹与晶粒尺寸的比例一致
        factor = 1 + np.random.normal(0, noise_scale, size=(n_augment, 1))
        aug_numeric_raw[:, 3:4] *= factor
        aug_target_raw *= factor.squeeze()

        aug_target = np.log1p(aug_target_raw)

        # 5. 合并原始数据和增强数据
        material_ids = np.concatenate([material_ids, aug_material_ids])
        numeric_scaled = np.concatenate([numeric_scaled, aug_numeric_scaled])
        numeric_raw = np.concatenate([numeric_raw, aug_numeric_raw])
        target_raw = np.concatenate([target_raw, aug_target_raw])
        target = np.concatenate([target, aug_target])

    return {
        "material_ids": material_ids,
        "numeric_features": numeric_scaled,
        "numeric_raw": numeric_raw,
        "target": target,
        "target_raw": target_raw,
        "num_mean": num_mean,
        "num_std": num_std,
        "label_encoder": le,
        "scaler": scaler,
        "num_materials": len(le.classes_),
    }


# 训练函数
def train_material_model(
    model, train_data, val_data, epochs=200, lr=1e-3, weight_decay=0.0001, device="cuda"
):
    """训练材料裂纹预测模型"""
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=0
    )  # 不使用AdamW的weight_decay

    train_dataset = TensorDataset(
        torch.LongTensor(train_data["material_ids"]),
        torch.FloatTensor(train_data["numeric_features"]),
        torch.FloatTensor(train_data["numeric_raw"]),
        torch.FloatTensor(train_data["target"]).unsqueeze(1),
    )
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
    )

    # 改进：使用自定义物理损失函数
    loss_fn = PhysicsInformedLoss(weight_decay=weight_decay)

    # 准备验证数据
    val_materials = torch.LongTensor(val_data["material_ids"]).to(device)
    val_features = torch.FloatTensor(val_data["numeric_features"]).to(device)
    val_raw = torch.FloatTensor(val_data["numeric_raw"]).to(device)
    val_target = torch.FloatTensor(val_data["target"]).unsqueeze(1).to(device)

    train_losses = []
    val_losses = []

    # 更新总epoch数
    loss_fn.total_epochs = epochs

    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(epochs):
        loss_fn.update_epoch(epoch, epochs)
        model.set_epoch(epoch, epochs)

        model.train()
        epoch_loss = 0.0

        for batch_mats, batch_feats, batch_raw, batch_tgt in train_loader:
            batch_mats = batch_mats.to(device)
            batch_feats = batch_feats.to(device)
            batch_raw = batch_raw.to(device)
            batch_tgt = batch_tgt.to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                pred1, unc1, _, _ = model(batch_mats, batch_feats)
                pred2, unc2, _, _ = model(batch_mats, batch_feats)
                loss1, metrics = loss_fn(pred1, unc1, batch_tgt, model, batch_raw)
                loss2, _ = loss_fn(pred2, unc2, batch_tgt, model, batch_raw)
                consistency = F.mse_loss(pred1, pred2)
                loss = (loss1 + loss2) / 2 + 0.5 * consistency

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item() * batch_tgt.size(0)

        epoch_loss /= len(train_dataset)
        train_losses.append(epoch_loss)

        # 验证
        model.eval()
        with torch.no_grad(), torch.cuda.amp.autocast():
            val_pred, val_uncertainty, _, _ = model(val_materials, val_features)
            val_loss, _ = loss_fn(val_pred, val_uncertainty, val_target, model, val_raw)
            val_losses.append(val_loss.item())

        if (epoch + 1) % 20 == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch+1}/{epochs}, LR: {current_lr:.6f}")
            print(
                f'  Train Loss: {loss.item():.4f} (NLL: {metrics["nll"]:.4f}, Physics: {metrics["physics"]:.4f}, Weight: {metrics["physics_weight"]:.4f})'
            )
            print(f"  Val Loss: {val_loss.item():.4f}")

    return train_losses, val_losses
