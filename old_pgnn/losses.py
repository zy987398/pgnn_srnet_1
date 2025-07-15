"""
损失函数定义
包含物理信息损失和自适应损失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PhysicsInformedLoss(nn.Module):
    """
    物理信息损失函数
    结合数据驱动损失和物理约束
    """
    def __init__(self, lambda_complexity=1e-4, lambda_physics=1e-2):
        super().__init__()
        self.lambda_complexity = lambda_complexity
        self.lambda_physics = lambda_physics
        self.current_epoch = 0
        self.total_epochs = 100
    
    def update_epoch(self, current, total):
        """更新当前训练进度，用于动态调整权重"""
        self.current_epoch = current
        self.total_epochs = total
    
    def forward(self, pred, uncertainty, target, model, numeric_raw):
        """
        计算总损失
        pred: 模型预测
        uncertainty: 不确定性估计
        target: 真实值
        model: 模型实例（用于获取复杂度）
        numeric_raw: 原始尺度数据（用于物理约束）
        """
        # 1. NLL损失（使用Huber损失增强鲁棒性）
        uncertainty = uncertainty.clamp(min=0.01, max=10.0)
        residual = (target - pred).abs()
        huber_delta = 1.0
        huber_loss = torch.where(
            residual < huber_delta,
            0.5 * residual ** 2,
            huber_delta * (residual - 0.5 * huber_delta)
        )
        nll_loss = (0.5 * torch.log(2 * np.pi * uncertainty) + 
                   huber_loss / uncertainty).mean()
        
        # 复杂度损失
        complexity_loss = 0.0
        if hasattr(model, 'symbolic_layer'):
            if hasattr(model.symbolic_layer, 'compute_complexity'):
                # 优先使用带梯度的复杂度计算方法
                complexity_loss = model.symbolic_layer.compute_complexity()
            elif hasattr(model.symbolic_layer, 'complexity'):
                # 向后兼容，但这种方式不会传播梯度
                complexity_loss = model.symbolic_layer.complexity
                
            # 确保 complexity_loss 是张量
            if not isinstance(complexity_loss, torch.Tensor):
                complexity_loss = torch.tensor(complexity_loss, 
                                            device=pred.device,
                                            dtype=pred.dtype,
                                            requires_grad=True)
        
        # 3. 物理约束（使用原始尺度数据）
        physics_loss = self.compute_physics_constraints(pred, numeric_raw)
        
        # 4. 动态权重调整（前30个epoch不使用物理约束）
        warm_progress = max(0.0, self.current_epoch - 30) / 20
        physics_weight = self.lambda_physics * min(0.3, warm_progress)
        
        # 5. 总损失
        total_loss = (nll_loss + 
                     self.lambda_complexity * complexity_loss + 
                     physics_weight * physics_loss)
        
        return total_loss, {
            "nll": nll_loss.item(),
            "complexity": complexity_loss if isinstance(complexity_loss, float) else complexity_loss.item(),
            "physics": physics_loss.item(),
            "physics_weight": physics_weight,
            "total": total_loss.item()
        }
    
    def compute_physics_constraints(self, pred, features):
        """
        计算物理约束损失
        基于材料科学原理的约束条件
        """
        # 提取特征
        grain_size = features[:, 3]  # mm
        pressure = features[:, 4]     # GPa
        hardness = features[:, 0]     # GPa
        fracture_toughness = features[:, 2]  # MPa·m^0.5
        
        # 约束1：裂纹深度不应超过晶粒尺寸的合理倍数
        max_crack_ratio = 5.0
        constraint1 = F.relu(pred.squeeze() - max_crack_ratio * grain_size)
        
        # 约束2：低压力下裂纹应该较小
        pressure_threshold = 0.5  # GPa
        low_pressure_mask = (pressure < pressure_threshold).float()
        constraint2 = low_pressure_mask * pred.squeeze()
        
        # 约束3：高脆性材料更容易产生裂纹
        brittleness = hardness / (fracture_toughness + 1e-6)
        brittleness_mean = brittleness.mean()
        
        # 安全计算标准差，处理批次大小为1的情况
        if brittleness.size(0) <= 1:
            # 批次大小为1时不能计算样本标准差，使用简单阈值
            high_brittleness_mask = (brittleness > brittleness_mean).float()
        else:
            # 批次大小大于1时，正常计算标准差
            brittleness_std = brittleness.std()
            # 如果标准差接近于0，使用简单阈值而不是标准化
            if brittleness_std < 1e-4:
                high_brittleness_mask = (brittleness > brittleness_mean).float()
            else:
                brittleness_normalized = (brittleness - brittleness_mean) / (brittleness_std + 1e-6)
                high_brittleness_mask = (brittleness_normalized > 1.0).float()
        expected_crack_depth = pred.mean()
        constraint3 = high_brittleness_mask * F.relu(expected_crack_depth - pred.squeeze())
        
        # 组合约束
        total_constraint = (constraint1.mean() + 
                          0.5 * constraint2.mean() + 
                          0.3 * constraint3.mean())
        
        return total_constraint


class AdaptiveLoss(PhysicsInformedLoss):
    """
    自适应损失函数
    使用可学习的权重动态平衡各项损失
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 可学习的损失权重
        self.loss_weights = nn.Parameter(torch.ones(3))
    
    def safety_check(self, loss_tensor):
        """检查损失值是否有数值问题"""
        if torch.isnan(loss_tensor) or torch.isinf(loss_tensor):
            return torch.tensor(1.0, device=loss_tensor.device)
        return loss_tensor

    def forward(self, pred, uncertainty, target, model, numeric_features):
        """
        计算自适应加权的总损失
        """
        # NLL损失
        nll_loss = 0.5 * (
            torch.log(2 * np.pi * uncertainty) + 
            (target - pred) ** 2 / uncertainty
        ).mean()
        
        # 复杂度损失
        complexity_loss = 0.0
        if hasattr(model, 'symbolic_layer'):
            if hasattr(model.symbolic_layer, 'compute_complexity'):
                # 优先使用带梯度的复杂度计算方法
                complexity_loss = model.symbolic_layer.compute_complexity()
            elif hasattr(model.symbolic_layer, 'complexity'):
                # 向后兼容，但这种方式不会传播梯度
                complexity_loss = model.symbolic_layer.complexity
                
            # 确保 complexity_loss 是张量
            if not isinstance(complexity_loss, torch.Tensor):
                complexity_loss = torch.tensor(complexity_loss, 
                                            device=pred.device,
                                            dtype=pred.dtype,
                                            requires_grad=True)
        
        # 数值安全检查
        nll_loss = self.safety_check(nll_loss)
        if isinstance(complexity_loss, torch.Tensor):
            complexity_loss = self.safety_check(complexity_loss)
            
        # 物理损失
        physics_loss = self.compute_physics_constraints(pred, numeric_features)
        physics_loss = self.safety_check(physics_loss)
        
        # 动态权重（使用softmax确保权重和为1）
        weights = F.softmax(self.loss_weights, dim=0)
        
        # 总损失
        total_loss = (
            weights[0] * nll_loss + 
            weights[1] * complexity_loss + 
            weights[2] * physics_loss
        )
        
        # 限制总损失范围，防止梯度爆炸
        if total_loss > 1000:
            print(f"Warning: Large loss detected ({total_loss.item():.2f}), clipping to 1000")
            total_loss = torch.clamp(total_loss, max=1000)
            
        # 确保返回合法损失值
        total_loss = self.safety_check(total_loss)
        
        return total_loss, {
            "nll": nll_loss.item(),
            "complexity": complexity_loss.item() if torch.is_tensor(complexity_loss) else complexity_loss,
            "physics": physics_loss.item(),
            "physics_weight": float(weights[2].item()),
            "total": total_loss.item(),
        }