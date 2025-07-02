import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict
import math

class Encoder(nn.Module):
    """编码器：将原始特征压缩到潜在表示"""
    def __init__(self, input_dim=60, hidden_dims=[128, 64], latent_dim=16):
        super().__init__()
        
        dims = [input_dim] + hidden_dims + [latent_dim]
        self.layers = nn.ModuleList()
        
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:  # 除了最后一层外都加ReLU
                self.layers.append(nn.ReLU())
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class SymbolicExpressionLayer(nn.Module):
    """可微符号表达式层 - 简化版实现"""
    def __init__(self, latent_dim=16, num_operators=4, num_variables=10):
        super().__init__()
        
        # 操作符选择网络
        self.operator_net = nn.Linear(latent_dim, num_operators)
        # 变量选择网络
        self.variable_net = nn.Linear(latent_dim, num_variables)
        # 常数预测网络
        self.constant_net = nn.Linear(latent_dim, num_variables)
        
        # 预定义的物理变量嵌入（如log(a_p)等）
        self.physics_embed = nn.Parameter(torch.randn(num_variables, latent_dim))
        
        # Soft operators (可微分的操作符)
        self.operators = ['+', '-', '*', '/']
        
    def forward(self, z, phi=None):
        """
        z: 潜在变量 [batch_size, latent_dim]
        phi: 物理嵌入 (可选)
        """
        batch_size = z.shape[0]
        
        # 选择操作符（soft selection）
        op_logits = self.operator_net(z)  # [batch_size, num_operators]
        op_weights = F.softmax(op_logits, dim=-1)
        
        # 选择变量
        var_logits = self.variable_net(z)  # [batch_size, num_variables]
        var_weights = F.softmax(var_logits, dim=-1)
        
        # 预测常数
        constants = self.constant_net(z)  # [batch_size, num_variables]
        
        # 构建表达式（这里使用简化的加权和）
        # 实际实现中应该使用更复杂的树结构
        expression = torch.sum(var_weights * constants, dim=-1)
        
        # 表达式复杂度（用于正则化）
        self.complexity = self._compute_complexity(op_weights, var_weights)
        
        return expression.unsqueeze(-1)
    
    def _compute_complexity(self, op_weights, var_weights):
        """计算表达式复杂度"""
        # 使用熵作为复杂度度量
        op_entropy = -torch.sum(op_weights * torch.log(op_weights + 1e-8), dim=-1)
        var_entropy = -torch.sum(var_weights * torch.log(var_weights + 1e-8), dim=-1)
        return torch.mean(op_entropy + var_entropy)

class ResidualBranch(nn.Module):
    """残差分支：捕获符号表达式未能解释的部分"""
    def __init__(self, latent_dim=16, hidden_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, z):
        return self.mlp(z)

class UncertaintyHead(nn.Module):
    """不确定性头：预测 aleatoric 不确定性"""
    def __init__(self, latent_dim=16):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 1)
        self.softplus = nn.Softplus()
    
    def forward(self, z):
        return self.softplus(self.fc(z))

class PGNNSymbolic(nn.Module):
    """完整的PGNN符号回归模型"""
    def __init__(self, input_dim=60, latent_dim=16):
        super().__init__()
        
        # 编码器
        self.encoder = Encoder(input_dim=input_dim, latent_dim=latent_dim)
        
        # 符号表达式层
        self.symbolic_layer = SymbolicExpressionLayer(latent_dim=latent_dim)
        
        # 残差分支
        self.residual_branch = ResidualBranch(latent_dim=latent_dim)
        
        # 不确定性头
        self.uncertainty_head = UncertaintyHead(latent_dim=latent_dim)
        
    def forward(self, x):
        """
        x: 输入特征 [batch_size, input_dim]
        返回: (预测值, 不确定性, 符号表达式, 残差)
        """
        # 编码
        z = self.encoder(x)
        
        # 符号表达式
        c = self.symbolic_layer(z)
        
        # 残差
        r = self.residual_branch(z)
        
        # 最终预测
        y_pred = c + r
        
        # 不确定性
        sigma_squared = self.uncertainty_head(z)
        
        return y_pred, sigma_squared, c, r

class PGNNLoss(nn.Module):
    """PGNN损失函数：NLL + 复杂度正则化"""
    def __init__(self, lambda_simp=1e-3, beta=0.5):
        super().__init__()
        self.lambda_simp = lambda_simp
        self.beta = beta
        
    def forward(self, y_pred, sigma_squared, y_true, model):
        """
        计算总损失
        y_pred: 预测值 [batch_size, 1]
        sigma_squared: 预测的方差 [batch_size, 1]
        y_true: 真实值 [batch_size, 1]
        model: 模型实例（用于获取复杂度）
        """
        batch_size = y_pred.shape[0]
        
        # NLL损失（高斯分布）
        nll_loss = 0.5 * (torch.log(2 * math.pi * sigma_squared) + 
                         (y_true - y_pred) ** 2 / sigma_squared)
        nll_loss = torch.mean(nll_loss)
        
        # 表达式复杂度
        expr_complexity = model.symbolic_layer.complexity if hasattr(model.symbolic_layer, 'complexity') else 0
        
        # β项（用于平衡复杂度和重构）
        beta_term = self.beta * torch.sum(torch.abs(self.beta - self.beta**2))
        
        # 总损失
        total_loss = nll_loss + self.lambda_simp * expr_complexity + beta_term
        
        return total_loss, {
            'nll_loss': nll_loss.item(),
            'complexity': expr_complexity.item() if torch.is_tensor(expr_complexity) else expr_complexity,
            'total_loss': total_loss.item()
        }

# 训练函数
def train_pgnn(model, train_loader, val_loader, epochs=100, lr=1e-3, device='cuda'):
    """训练PGNN模型"""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = PGNNLoss()
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss_epoch = 0
        train_metrics = {'nll_loss': 0, 'complexity': 0}
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            
            # 前向传播
            y_pred, sigma_squared, c, r = model(x)
            
            # 计算损失
            loss, metrics = loss_fn(y_pred, sigma_squared, y, model)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            train_loss_epoch += loss.item()
            for k, v in metrics.items():
                if k != 'total_loss':
                    train_metrics[k] += v
        
        # 平均训练损失
        train_loss_epoch /= len(train_loader)
        for k in train_metrics:
            train_metrics[k] /= len(train_loader)
        train_losses.append(train_loss_epoch)
        
        # 验证阶段
        model.eval()
        val_loss_epoch = 0
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                
                y_pred, sigma_squared, c, r = model(x)
                loss, _ = loss_fn(y_pred, sigma_squared, y, model)
                
                val_loss_epoch += loss.item()
        
        val_loss_epoch /= len(val_loader)
        val_losses.append(val_loss_epoch)
        
        # 打印进度
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}]')
            print(f'  Train Loss: {train_loss_epoch:.4f} (NLL: {train_metrics["nll_loss"]:.4f}, Complexity: {train_metrics["complexity"]:.4f})')
            print(f'  Val Loss: {val_loss_epoch:.4f}')
    
    return train_losses, val_losses

# 工具函数
def create_physics_features(x, feature_names=None):
    """创建物理特征（如log变换等）"""
    features = [x]
    
    # 对数变换（避免log(0)）
    log_x = torch.log(torch.abs(x) + 1e-8)
    features.append(log_x)
    
    # 平方
    features.append(x ** 2)
    
    # 倒数（避免除零）
    inv_x = 1 / (x + 1e-8)
    features.append(inv_x)
    
    return torch.cat(features, dim=-1)

def extract_symbolic_expression(model, sample_input):
    """从模型中提取符号表达式的可解释形式"""
    model.eval()
    with torch.no_grad():
        z = model.encoder(sample_input)
        
        # 获取操作符和变量权重
        op_logits = model.symbolic_layer.operator_net(z)
        var_logits = model.symbolic_layer.variable_net(z)
        constants = model.symbolic_layer.constant_net(z)
        
        op_weights = F.softmax(op_logits, dim=-1)
        var_weights = F.softmax(var_logits, dim=-1)
        
        # 找到最可能的操作符和变量
        top_op = torch.argmax(op_weights, dim=-1)
        top_vars = torch.topk(var_weights, k=3, dim=-1).indices
        
        return {
            'operators': model.symbolic_layer.operators,
            'selected_op': top_op.item(),
            'top_variables': top_vars.cpu().numpy(),
            'constants': constants.cpu().numpy()
        }

# 示例使用
if __name__ == "__main__":
    # 创建模型
    model = PGNNSymbolic(input_dim=60, latent_dim=16)
    
    # 创建示例数据
    batch_size = 32
    x = torch.randn(batch_size, 60)
    y = torch.randn(batch_size, 1)
    
    # 前向传播测试
    y_pred, sigma_squared, c, r = model(x)
    print(f"预测形状: {y_pred.shape}")
    print(f"不确定性形状: {sigma_squared.shape}")
    print(f"符号表达式形状: {c.shape}")
    print(f"残差形状: {r.shape}")
    
    # 损失计算测试
    loss_fn = PGNNLoss()
    loss, metrics = loss_fn(y_pred, sigma_squared, y, model)
    print(f"\n损失值: {loss.item():.4f}")
    print(f"损失组成: {metrics}")