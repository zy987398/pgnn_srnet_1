import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from typing import Dict, List, Tuple, Optional
import pandas as pd


class MaterialEmbedding(nn.Module):
    """材料种类嵌入层"""

    def __init__(self, num_materials, embedding_dim=16):
        super().__init__()
        self.embedding = nn.Embedding(num_materials, embedding_dim)

    def forward(self, material_ids):
        return self.embedding(material_ids)


class FeatureInteractionModule(nn.Module):
    """计算特征间二阶交互"""

    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.interaction_net = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, dim = x.shape
        x_expand1 = x.unsqueeze(2).expand(-1, -1, dim)
        x_expand2 = x.unsqueeze(1).expand(-1, dim, -1)
        interactions = torch.cat([x_expand1, x_expand2], dim=-1)
        interactions = self.interaction_net(interactions.view(batch_size, -1))
        return x + interactions


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

        # ---- 物理特征增强 ----
        # 应变能密度
        strain_energy = 0.5 * pressure**2 / (elastic_modulus + 1e-6)
        features.append(strain_energy)

        # 临界裂纹长度 (Irwin 准则)
        critical_crack = (fracture_toughness / (pressure + 1e-6)) ** 2
        features.append(critical_crack)

        # 塑性区尺寸估计
        plastic_zone = (fracture_toughness / (hardness + 1e-6)) ** 2
        features.append(plastic_zone)

        # 疲劳参数 (Paris law inspired)
        fatigue_param = velocity * pressure / (fracture_toughness + 1e-6)
        features.append(fatigue_param)

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


class AdaptiveSymbolicLayer(nn.Module):
    """具有注意力和门控机制的符号层"""

    def __init__(self, latent_dim: int = 32):
        super().__init__()

        self.attention = nn.MultiheadAttention(latent_dim, num_heads=4, batch_first=True)
        self.operators = ["*", "/", "^2", "^3", "sqrt", "log", "exp", "sin"]
        self.num_operators = len(self.operators)

        self.variable_selector = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.GELU(),
            nn.LayerNorm(64),
            nn.Linear(64, 19),
        )

        self.operator_selector = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.GELU(),
            nn.LayerNorm(32),
            nn.Linear(32, self.num_operators),
        )

        self.coefficient_net = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.GELU(),
            nn.LayerNorm(32),
            nn.Linear(32, 4),
        )

        self.gate = nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.Sigmoid())

    def forward(self, z: torch.Tensor, physics_features: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attention(z.unsqueeze(1), z.unsqueeze(1), z.unsqueeze(1))
        z = attn_out.squeeze(1)

        var_weights = F.softmax(self.variable_selector(z), dim=-1)
        op_weights = F.softmax(self.operator_selector(z), dim=-1)
        coeffs = self.coefficient_net(z)

        selected_features = torch.matmul(var_weights.unsqueeze(1), physics_features.unsqueeze(2)).squeeze(-1)

        expressions = []
        expressions.append(coeffs[:, 0:1] * selected_features)
        expressions.append(coeffs[:, 1:2] * (selected_features ** 2))
        expressions.append(coeffs[:, 2:3] * (selected_features ** 3))
        expressions.append(coeffs[:, 3:4] * torch.sqrt(torch.abs(selected_features) + 1e-6))

        ops = [selected_features,
               1.0 / (selected_features + 1e-6),
               selected_features ** 2,
               selected_features ** 3,
               torch.sqrt(torch.abs(selected_features) + 1e-6),
               torch.log(torch.abs(selected_features) + 1e-6),
               torch.exp(selected_features.clamp(max=20)),
               torch.sin(selected_features)]

        op_comb = sum(w.unsqueeze(1) * op for w, op in zip(op_weights.split(1, dim=1), ops))

        symbolic_output = sum(expressions) * self.gate(z) + op_comb

        self.complexity = -torch.sum(var_weights * torch.log(var_weights + 1e-8), dim=-1).mean()

        return symbolic_output


class MaterialCrackPGNN(nn.Module):
    """专门用于材料裂纹深度预测的PGNN"""

    def __init__(
        self, num_materials, material_embed_dim=16, latent_dim=32, dropout_rate=0.1
    ):
        super().__init__()

        # 材料嵌入
        self.material_embedding = MaterialEmbedding(num_materials, material_embed_dim)

        # 物理特征提取
        self.physics_extractor = PhysicsInformedFeatureExtractor()
        physics_dim = 6 + 7 + 2 + 4  # 原始特征 + 衍生特征 + 交互 + 物理增强
        self.feature_interaction = FeatureInteractionModule(physics_dim)
        # 改进：计算特征总维度
        input_dim = material_embed_dim + physics_dim

        # 编码器 (使用残差连接)
        self.input_layer = nn.Linear(input_dim, latent_dim)
        self.encoder_norm = nn.LayerNorm(latent_dim)

        # 添加残差块
        dropout_rates = [0.05, 0.1, 0.15, 0.2]
        self.residual_blocks = nn.ModuleList(
            [ResidualBlock(latent_dim, dropout_rates[min(i, 3)]) for i in range(4)]
        )

        # 符号表达式层
        self.symbolic_layer = AdaptiveSymbolicLayer(latent_dim)

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

    def enable_mc_dropout(self, enable=True):
        """启用/禁用MC Dropout"""
        self.mc_dropout_enabled = enable

    def forward(self, material_ids, numeric_features):
        """
        material_ids: [batch_size] - 材料类别ID
        numeric_features: [batch_size, 6] - 数值特征
        """
        # 材料嵌入
        material_embed = self.material_embedding(material_ids)

        # 提取物理特征
        physics_features = self.physics_extractor(numeric_features)
        physics_features = self.feature_interaction(physics_features)

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
        symbolic_output = self.symbolic_layer(z, physics_features)

        # 残差网络输入：符号层输出与潜在表示，切断梯度
        sym_detached = symbolic_output.detach()
        residual_input = torch.cat([z, sym_detached], dim=-1)
        residual_output = self.residual_net(residual_input)

        # 组合预测（非负）
        prediction = F.softplus(symbolic_output + residual_output, beta=5) + 1e-3

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
        """改进的物理约束计算"""
        # 约束1：裂纹深度应该与晶粒尺寸有关（不能超过某个比例）
        # 改进：使用更合理的上限系数
        grain_size = features[:, 3]
        constraint1 = F.relu(
            pred.squeeze() - 3 * grain_size
        )  # 从5降低到3，更严格的约束

        # 约束2：零压力应该产生很小的裂纹
        # 改进：使用更平滑的过渡函数
        pressure = features[:, 4]
        # 使用sigmoid在低压力区域逐渐增加惩罚
        pressure_factor = torch.sigmoid(-(pressure - 0.2) * 10)  # 在0.2附近快速过渡
        constraint2 = pressure_factor * pred.squeeze()

        # 约束3：高硬度材料的裂纹应该更小（新增）
        hardness = features[:, 0]
        fracture_toughness = features[:, 2]
        # 高硬度低韧性材料应该有较小裂纹
        brittleness = hardness / (fracture_toughness + 1e-6)
        high_brittleness = torch.sigmoid((brittleness - brittleness.mean()) * 5)
        constraint3 = high_brittleness * F.relu(pred.squeeze() - pred.mean() * 0.5)

        # 加权平均所有约束
        return constraint1.mean() + constraint2.mean() + 0.5 * constraint3.mean()


class AdaptiveLoss(PhysicsInformedLoss):
    """动态权重的物理损失函数"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_weights = nn.Parameter(torch.ones(3))

    def forward(self, pred, uncertainty, target, model, numeric_features):
        nll_loss = 0.5 * (
            torch.log(2 * np.pi * uncertainty) + (target - pred) ** 2 / uncertainty
        ).mean()
        complexity_loss = (
            model.symbolic_layer.complexity if hasattr(model.symbolic_layer, "complexity") else 0
        )
        physics_loss = self.compute_physics_constraints(pred, numeric_features)
        weights = F.softmax(self.loss_weights, dim=0)
        total_loss = (
            weights[0] * nll_loss + weights[1] * complexity_loss + weights[2] * physics_loss
        )

        return total_loss, {
            "nll": nll_loss.item(),
            "complexity": complexity_loss.item() if torch.is_tensor(complexity_loss) else complexity_loss,
            "physics": physics_loss.item(),
            "physics_weight": float(weights[2].item()),
            "total": total_loss.item(),
        }


def smart_augment(df: pd.DataFrame, model_predictions=None):
    """根据预测误差进行mixup的数据增强"""
    df = df.copy()
    features = df[[
        "莫氏硬度",
        "弹性模量",
        "断裂韧性",
        "晶粒粒径",
        "压溃压力",
        "磨粒速度",
    ]].values
    targets = df["亚表面裂纹深度"].values

    if model_predictions is not None:
        errors = np.abs(targets - model_predictions)
        weights = errors / (errors.sum() + 1e-6)
    else:
        weights = np.ones(len(df)) / len(df)

    shuffled_idx = np.random.permutation(len(df))
    alpha = 0.2
    lam = np.random.beta(alpha, alpha)
    mixed_features = lam * features + (1 - lam) * features[shuffled_idx]
    mixed_targets = lam * targets + (1 - lam) * targets[shuffled_idx]

    aug_df = pd.DataFrame(mixed_features, columns=[
        "莫氏硬度",
        "弹性模量",
        "断裂韧性",
        "晶粒粒径",
        "压溃压力",
        "磨粒速度",
    ])
    aug_df["材料种类"] = df["材料种类"].values
    aug_df["亚表面裂纹深度"] = mixed_targets

    return pd.concat([df, aug_df], ignore_index=True)


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
    target = df["亚表面裂纹深度"].values

    # 使用StandardScaler
    if scaler is None:
        scaler = StandardScaler().fit(numeric_raw)
    numeric_scaled = scaler.transform(numeric_raw)

    # 数据增强 - 只在训练时进行
    if augment:
        aug_df = smart_augment(df)
        return prepare_material_data(
            aug_df, scaler=scaler, le=le, augment=False
        )

    return {
        "material_ids": material_ids,
        "numeric_features": numeric_scaled,
        "numeric_raw": numeric_raw,
        "target": target,
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

    # 改进：使用带预热的学习率调度器
    warmup_epochs = 10

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # 线性预热
            return epoch / warmup_epochs
        else:
            # 余弦退火
            progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ---- Stochastic Weight Averaging (SWA) ----
    from torch.optim.swa_utils import AveragedModel, SWALR
    swa_model = AveragedModel(model)
    swa_start = int(epochs * 0.8)
    swa_scheduler = SWALR(optimizer, swa_lr=lr * 0.1)

    # 改进：使用自定义物理损失函数
    loss_fn = AdaptiveLoss(weight_decay=weight_decay)

    from torch.cuda.amp import autocast, GradScaler
    scaler = GradScaler()

    # 准备数据
    train_materials = torch.LongTensor(train_data["material_ids"]).to(device)
    train_features = torch.FloatTensor(train_data["numeric_features"]).to(device)
    train_raw = torch.FloatTensor(train_data["numeric_raw"]).to(device)
    train_target = torch.FloatTensor(train_data["target"]).unsqueeze(1).to(device)

    val_materials = torch.LongTensor(val_data["material_ids"]).to(device)
    val_features = torch.FloatTensor(val_data["numeric_features"]).to(device)
    val_raw = torch.FloatTensor(val_data["numeric_raw"]).to(device)
    val_target = torch.FloatTensor(val_data["target"]).unsqueeze(1).to(device)

    train_losses = []
    val_losses = []

    # 更新总epoch数
    loss_fn.total_epochs = epochs

    for epoch in range(epochs):
        # 更新当前epoch
        loss_fn.update_epoch(epoch, epochs)

        # 训练
        model.train()
        optimizer.zero_grad()

        with autocast():
            pred, uncertainty, symbolic, residual = model(train_materials, train_features)
            loss, metrics = loss_fn(pred, uncertainty, train_target, model, train_raw)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        scaler.step(optimizer)
        scaler.update()

        # Stochastic Weight Averaging update
        if epoch >= swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()

        train_losses.append(loss.item())

        # 验证
        model.eval()
        with torch.no_grad(), autocast():
            val_pred, val_uncertainty, _, _ = model(val_materials, val_features)
            val_loss, _ = loss_fn(val_pred, val_uncertainty, val_target, model, val_raw)
            val_losses.append(val_loss.item())


        if (epoch + 1) % 20 == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch+1}/{epochs}, LR: {current_lr:.6f}")
            print(
                f'  Train Loss: {loss.item():.4f} (NLL: {metrics["nll"]:.4f}, Physics: {metrics["physics"]:.4f}, Weight: {metrics["physics_weight"]:.4f})'
            )
            print(f"  Val Loss: {val_loss.item():.4f}")

    # If SWA was applied, copy averaged weights back to the model
    if epochs > swa_start:
        model.load_state_dict(swa_model.module.state_dict())

    return train_losses, val_losses