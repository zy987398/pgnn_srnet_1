"""
轻量级注意力符号层模块
结合了注意力机制与简化的符号表达式生成
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import safe_sqrt, safe_log


class LightweightAttentionSymbolicLayer(nn.Module):
    """
    轻量级注意力符号层
    1. 使用单头注意力机制而非复杂的Transformer架构
    2. 简化的操作符集合和符号表达式生成
    3. 优化的错误处理和数值稳定性设计
    4. 保留可解释性和物理约束
    """
    def __init__(self, latent_dim=32):
        super().__init__()
        
        # 预定义的物理操作符，减少数量以提高效率
        self.operators = ["线性", "平方", "平方根", "对数", "倒数"]
        self.num_operators = len(self.operators)
        
        # 特征注意力投影
        self.query_proj = nn.Linear(latent_dim, 32)
        self.key_proj = nn.Linear(1, 32)  # 投影单个物理特征
        
        # 物理变量选择网络 - 简化设计，减少层数
        self.variable_selector = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.GELU(),
            nn.LayerNorm(32),
            nn.Linear(32, 18)  # 6个原始特征 + 12个物理特征
        )
        
        # 操作符选择网络 - 简化设计，减少层数
        self.operator_selector = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.GELU(),
            nn.Linear(16, self.num_operators)
        )
        
        # 系数预测网络 - 匹配操作符数量，每个操作符一个系数
        self.coefficient_net = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.GELU(),
            nn.Linear(16, self.num_operators),  # 每个操作符一个系数
            nn.Tanh()  # 限制系数范围在[-1,1]
        )
        
        # 用于可解释性分析的属性
        self.last_attn_weights = None
        self.last_var_weights = None
        self.last_op_weights = None
        self.last_coeffs = None
        self.complexity = 0.0
        
    def forward(self, z, physics_features):
        """
        前向传播
        z: [batch_size, latent_dim] - 潜在表示
        physics_features: [batch_size, 18] - 物理特征 (6个原始 + 12个派生)
        """
        batch_size = z.shape[0]
        
        # 1. 注意力机制 - 改进的轻量级单头注意力计算
        # 计算查询向量
        query = self.query_proj(z)  # [batch_size, 32]
        
        # 为每个物理特征单独计算键向量
        # 为物理特征创建投影
        keys = []
        for i in range(physics_features.size(1)):  # 对每个物理特征
            feature = physics_features[:, i:i+1]  # [batch_size, 1]
            key = self.key_proj(feature)           # [batch_size, 32]
            keys.append(key)
        
        # 计算注意力分数
        attn_scores = torch.zeros(batch_size, physics_features.size(1), device=z.device)
        
        for i in range(physics_features.size(1)):
            # 计算点积相似度
            similarity = (query * keys[i]).sum(dim=1)  # [batch_size]
            attn_scores[:, i] = similarity
        
        # 应用softmax并保存注意力权重 - 使用简化的异常处理
        try:
            attn_weights = F.softmax(attn_scores, dim=-1)  # [batch_size, 18]
        except Exception:
            # 使用均匀注意力作为后备，使用inplace操作减少内存分配
            attn_weights = torch.ones(batch_size, physics_features.size(1), device=z.device)
            attn_weights.div_(physics_features.size(1))  # inplace除法
        
        # 存储注意力权重
        self.last_attn_weights = attn_weights
    
        # 2. 特征选择与权重计算 - 合并连续操作减少中间计算
        var_logits = self.variable_selector(z)
        var_weights = F.softmax(var_logits, dim=-1)
        self.last_var_weights = var_weights
        
        # 3. 操作符选择 - 同样合并操作
        op_weights = F.softmax(self.operator_selector(z), dim=-1)
        self.last_op_weights = op_weights
        
        # 4. 系数预测 - 缩放到合理范围并确保维度正确
        coeffs = self.coefficient_net(z)
        
        # 确保系数维度与操作符数量匹配
        if coeffs.shape[1] != self.num_operators:
            print(f"Warning: Coefficient shape mismatch. Expected {self.num_operators}, got {coeffs.shape[1]}")
            # 构造正确维度的系数
            temp_coeffs = torch.zeros(batch_size, self.num_operators, device=z.device)
            # 复制现有系数
            temp_coeffs[:, :min(coeffs.shape[1], self.num_operators)] = coeffs[:, :min(coeffs.shape[1], self.num_operators)]
            coeffs = temp_coeffs
            
        # 应用合理的缩放
        coeffs = coeffs * 1.0  # 保持在[-1,1]范围内，避免过大的系数
        self.last_coeffs = coeffs
        
        # 5. 使用KL散度计算复杂度 - 速度更快
        uniform = torch.ones_like(var_weights) / var_weights.size(-1)
        self.complexity = F.kl_div(
            F.log_softmax(var_logits, dim=-1), 
            uniform, 
            reduction='batchmean'
        )
        
        # 6. 构建符号表达式 - 优化的向量化实现
        # 使用变量选择器加权物理特征并降维
        selected_features = (var_weights * physics_features).sum(dim=1, keepdim=True)
        
        # 预先计算共用的中间值，避免重复计算
        abs_features = torch.abs(selected_features)
        
        # 检查操作符权重维度是否与操作符数量匹配
        if op_weights.shape[1] != self.num_operators:
            print(f"Warning: Operator weights dimension mismatch. Expected {self.num_operators}, got {op_weights.shape[1]}")
            # 如果不匹配，重新生成合适维度的均匀权重
            op_weights = torch.ones(batch_size, self.num_operators, device=z.device) / self.num_operators
        
        # 直接使用向量化操作构建表达式，避免列表和循环
        try:
            # 初始化输出为零
            symbolic_output = torch.zeros(batch_size, 1, device=z.device)
            
            # 现在coeffs的形状应该是[batch_size, num_operators]，每个操作符有自己的系数
            
            # 线性项
            symbolic_output.add_(coeffs[:, 0:1] * selected_features * op_weights[:, 0:1])
            
            # 平方项
            square_term = torch.sign(selected_features) * torch.clamp(selected_features.pow(2), -10.0, 10.0)
            symbolic_output.add_(coeffs[:, 1:2] * square_term * op_weights[:, 1:2])
            
            # 平方根项
            sqrt_term = safe_sqrt(abs_features)
            symbolic_output.add_(coeffs[:, 2:3] * sqrt_term * op_weights[:, 2:3])
            
            # 对数项
            log_term = safe_log(abs_features + 1.0)
            symbolic_output.add_(coeffs[:, 3:4] * log_term * op_weights[:, 3:4])
            
            # 倒数项
            recip_term = torch.clamp(1.0 / (abs_features + 1e-3), -10.0, 10.0)
            symbolic_output.add_(coeffs[:, 4:5] * recip_term * op_weights[:, 4:5])
            
            # 确保数值稳定性 - 使用tanh限制输出范围
            symbolic_output = torch.tanh(symbolic_output)
            
        except Exception as e:
            # 增加错误诊断信息
            print(f"Error in symbolic layer: {e}")
            print(f"coeffs shape: {coeffs.shape}, op_weights shape: {op_weights.shape}")
            # 提供安全的后备值
            symbolic_output = torch.zeros(batch_size, 1, device=z.device)
    
        # 确保输出形状正确
        if symbolic_output.shape != (batch_size, 1):
            print(f"Warning: Reshaping symbolic output from {symbolic_output.shape} to {(batch_size, 1)}")
            symbolic_output = symbolic_output.reshape(batch_size, 1)
            
        return symbolic_output
    
    def get_interpretation(self):
        """返回符号层的可解释性信息 - 优化版"""
        # 使用更统一的方式处理可能的None值
        def safe_detach_cpu(tensor):
            if tensor is not None and isinstance(tensor, torch.Tensor):
                return tensor.detach().cpu()
            return None
        
        # 一次性构建解释字典
        interpretation = {
            "attention_weights": safe_detach_cpu(self.last_attn_weights),
            "variable_weights": safe_detach_cpu(self.last_var_weights),
            "operator_weights": safe_detach_cpu(self.last_op_weights),
            "coefficients": safe_detach_cpu(self.last_coeffs),
            "complexity": float(self.complexity) if isinstance(self.complexity, (torch.Tensor, float)) else 0.0,
            "operators": self.operators
        }
        return interpretation
    
    def compute_complexity(self):
        """计算并返回带有梯度的复杂度张量，用于损失函数优化"""
        if not hasattr(self, 'last_var_weights') or self.last_var_weights is None:
            return torch.tensor(0.5, device='cuda' if torch.cuda.is_available() else 'cpu', requires_grad=True)
        
        # 使用预先计算的复杂度值（避免重复计算）
        if hasattr(self, 'complexity') and isinstance(self.complexity, torch.Tensor) and self.complexity.requires_grad:
            return self.complexity
        
        # 如果没有可用的预计算值，使用快速KL散度计算
        uniform = torch.ones_like(self.last_var_weights) / self.last_var_weights.size(-1)
        return F.kl_div(
            torch.log(self.last_var_weights + 1e-6), 
            uniform, 
            reduction='batchmean',
            log_target=False
        )
    
    def visualize_attention(self, feature_names=None, figsize=(8, 3)):
        """可视化注意力权重 - 优化版"""
        # 延迟导入matplotlib，只在需要时加载
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("需要安装matplotlib来可视化注意力权重")
            return None
        
        if self.last_attn_weights is None:
            print("尚无注意力权重，请先运行前向传播")
            return None
        
        # 默认物理特征名称 - 使用更短的判断方式
        feature_names = feature_names or [
            "莫氏硬度", "弹性模量", "断裂韧性", "晶粒粒径", "压溃压力", "磨粒速度",
            "G", "K/Kc", "Hall-Petch", "屏蔽因子", "ρ₁", "ρ₂", 
            "d_c", "a_c/d", "r_p/d", "脆性指数", "K_IC/E", "应变能"
        ][:self.last_attn_weights.shape[1]]
        
        # 预计算numpy数组，避免多次转换
        weights = self.last_attn_weights.detach().cpu().numpy().mean(axis=0)
        
        # 创建图形和坐标轴对象，便于后续操作
        fig, ax = plt.subplots(figsize=figsize)
        ax.bar(range(len(feature_names)), weights, color='royalblue')
        ax.set_xticks(range(len(feature_names)))
        ax.set_xticklabels(feature_names, rotation=45, ha='right')
        ax.set_ylabel('注意力权重')
        ax.set_title('特征注意力分布')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        fig.tight_layout()
        
        return plt
