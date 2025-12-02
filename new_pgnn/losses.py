import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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
        # 为不确定性（方差）设置一个安全的最小阈值，防止除零
        min_uncertainty_floor = 1e-4  
        safe_uncertainty = torch.clamp(uncertainty, min=min_uncertainty_floor)
        nll_loss = 0.5 * (
            torch.log(2 * np.pi * safe_uncertainty) + (target - pred) ** 2 / safe_uncertainty
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

        # 4. 新增：门控正则化（鼓励合理的融合行为）
        gate_reg_loss = torch.tensor(0.0, device=pred.device)

        # 总损失
        total_loss = (
            nll_loss
            + self.lambda_complexity * complexity_loss
            + physics_weight * physics_loss
            + gate_reg_loss
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
            "gate_reg": gate_reg_loss.item() if torch.is_tensor(gate_reg_loss) else gate_reg_loss,
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