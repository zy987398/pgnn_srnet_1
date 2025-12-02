"""
改进的符号表达式层，专门针对裂纹深度预测
包含更多物理操作符和变量选择机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ImprovedSymbolicLayer(nn.Module):

    def __init__(self, latent_dim=32):
        super().__init__()

        # 预定义的物理操作符
        self.operators = ["*", "/", "^", "sqrt", "log"]
        self.num_operators = len(self.operators)

        # 物理变量选择网络
        self.variable_selector = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.GELU(),  # 改进：使用GELU激活函数
            nn.LayerNorm(64),  # 改进：LayerNorm替换BatchNorm
            nn.Linear(64, 18),  # 适配新的物理特征维度: 6个原始特征 + 12个物理特征
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
            nn.Linear(32, 4),  # 预测4个系数
        )

    def forward(self, z, physics_features):
        """
        z: 潜在表示
        physics_features: 物理特征
        """
        batch_size = z.shape[0]

        # 选择变量
        var_weights = F.softmax(self.variable_selector(z), dim=-1)

        # 选择操作符
        op_weights = F.softmax(self.operator_selector(z), dim=-1)

        # 预测系数
        coeffs = self.coefficient_net(z)

        # 构建符号表达式（加权组合）
        # 使用软选择机制
        selected_features = torch.matmul(
            var_weights.unsqueeze(1), physics_features.unsqueeze(2)
        ).squeeze(-1)

        # 应用不同的操作符（软操作）
        expressions = []

        # 线性项
        linear = coeffs[:, 0:1] * selected_features
        expressions.append(linear * op_weights[:, 0:1])

        # 平方项
        square = coeffs[:, 1:2] * (selected_features**2)
        expressions.append(square * op_weights[:, 1:2])

        # 平方根项
        sqrt_term = coeffs[:, 2:3] * torch.sqrt(torch.abs(selected_features) + 1e-6)
        expressions.append(sqrt_term * op_weights[:, 2:3])

        # 对数项
        log_term = coeffs[:, 3:4] * torch.log(torch.abs(selected_features) + 1)
        expressions.append(log_term * op_weights[:, 3:4])

        # 组合所有项
        symbolic_output = sum(expressions)

        # 计算复杂度（用于正则化）
        self.complexity = -torch.sum(
            var_weights * torch.log(var_weights + 1e-8), dim=-1
        ).mean()

        return symbolic_output