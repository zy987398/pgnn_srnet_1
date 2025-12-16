"""
消融实验模型 - 支持多种消融模式
用于验证符号层、门控机制等组件的有效性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt

from utils import (safe_division, safe_log, safe_sqrt, safe_power, 
                  adaptive_scale,)
from light_symbolic_layer import LightweightAttentionSymbolicLayer
from ImprovedSymbolicLayer import ImprovedSymbolicLayer
from models import (MaterialEmbedding, PhysicsInformedFeatureExtractor, 
                   ResidualBlock, AdaptiveSymbolicLayer)


class MaterialCrackPGNN_Ablation(nn.Module):
    """
    消融实验版本的PGNN模型
    
    支持的消融模式:
    - "full": 完整模型 (符号层 + 残差网络 + 自适应门控)
    - "no_symbolic": 无符号层,仅使用残差网络
    - "simple_symbolic": 使用简化符号层,无注意力机制
    - "fixed_gate_0.5": 固定门控权重为0.5
    - "fixed_gate_0.0": 固定门控权重为0.0 (纯残差)
    - "fixed_gate_1.0": 固定门控权重为1.0 (纯符号层)
    - "no_physics": 不使用物理特征提取器
    - "no_residual": 仅使用符号层,无残差网络
    - "no_physics_baseline": 无物理特征+无符号层(纯数据驱动基线)
    """

    def __init__(
        self, 
        num_materials, 
        material_embed_dim=16, 
        latent_dim=32, 
        dropout_rate=0.1,
        symbolic_type="advanced",
        ablation_mode="full"
    ):
        super().__init__()
        
        self.ablation_mode = ablation_mode
        self.symbolic_type = symbolic_type
        
        print(f"初始化消融模型 - 模式: {ablation_mode}")

        # 材料嵌入
        self.material_embedding = MaterialEmbedding(num_materials, material_embed_dim)

        # 物理特征提取
        if ablation_mode in ["no_physics", "no_physics_baseline"]:
            # 不使用物理特征提取,直接使用原始特征
            self.physics_extractor = None
            physics_dim = 6  # 仅原始特征
        else:
            self.physics_extractor = PhysicsInformedFeatureExtractor()
            physics_dim = 18  # 6个原始 + 12个物理特征

        # 计算输入维度
        input_dim = material_embed_dim + physics_dim

        # 编码器
        self.input_layer = nn.Linear(input_dim, latent_dim)
        self.encoder_norm = nn.LayerNorm(latent_dim)

        # 残差块
        self.residual_blocks = nn.ModuleList(
            [ResidualBlock(latent_dim, dropout_rate) for _ in range(2)]
        )

        # 符号层 - 根据消融模式决定是否创建
        self.symbolic_layer = None
        if ablation_mode not in ["no_symbolic", "fixed_gate_0.0", "no_physics_baseline"]:
            if ablation_mode == "simple_symbolic":
                # 使用简化符号层
                self.symbolic_layer = ImprovedSymbolicLayer(latent_dim)
                self.use_advanced_symbolic = False
            else:
                # 使用高级符号层
                if symbolic_type == "advanced":
                    self.symbolic_layer = AdaptiveSymbolicLayer(latent_dim=latent_dim)
                elif symbolic_type == "lightweight":
                    self.symbolic_layer = LightweightAttentionSymbolicLayer(latent_dim=latent_dim)
                else:
                    self.symbolic_layer = ImprovedSymbolicLayer(latent_dim)
                self.use_advanced_symbolic = (symbolic_type in ["advanced", "lightweight"])

        # 残差网络 - 根据消融模式决定输入维度
        if ablation_mode == "no_residual":
            self.residual_net = None
        else:
            # 根据是否有符号层决定输入维度
            if self.symbolic_layer is not None:
                residual_input_dim = latent_dim + 1  # z + symbolic_output
            else:
                residual_input_dim = latent_dim  # 仅z
                
            self.residual_net = nn.Sequential(
                nn.Linear(residual_input_dim, 64),
                nn.GELU(),
                nn.LayerNorm(64),
                nn.Dropout(dropout_rate),
                nn.Linear(64, 32),
                nn.GELU(),
                nn.Linear(32, 1),
            )

        # 符号输出对齐网络 - 仅在有符号层时创建
        if self.symbolic_layer is not None:
            self.sym_scale_net = nn.Sequential(
                nn.Linear(latent_dim, 16),
                nn.GELU(),
                nn.Linear(16, 2)
            )
        else:
            self.sym_scale_net = None
        
        # 融合门控网络 - 根据消融模式决定
        if ablation_mode.startswith("fixed_gate"):
            # 固定门控模式
            gate_value = float(ablation_mode.split("_")[-1])
            self.fusion_gate = None
            self.fixed_gate_value = gate_value
            print(f"使用固定门控值: {gate_value}")
        elif ablation_mode in ["no_symbolic", "no_residual", "no_physics_baseline"]:
            # 无需门控
            self.fusion_gate = None
            self.fixed_gate_value = None
        else:
            # 自适应门控
            self.fusion_gate = nn.Sequential(
                nn.Linear(latent_dim, 16),
                nn.GELU(),
                nn.Linear(16, 1),
                nn.Sigmoid()
            )
            self.fixed_gate_value = None
        
        self.last_gate = None

        # 不确定性预测
        self.aleatoric_net = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

        # 输出层
        self.output_activation = nn.Softplus()
        self.mc_dropout_enabled = False

    def enable_mc_dropout(self, enable=True):
        """启用/禁用MC Dropout"""
        self.mc_dropout_enabled = enable

    def forward(self, material_ids, numeric_features):
        """
        前向传播
        material_ids: [batch_size] - 材料类别ID
        numeric_features: [batch_size, 6] - 数值特征(原始尺度)
        """
        # 材料嵌入
        material_embed = self.material_embedding(material_ids)

        # 提取物理特征
        if self.physics_extractor is not None:
            physics_features = self.physics_extractor(numeric_features)
        else:
            # 不使用物理特征提取器,直接使用原始特征
            physics_features = numeric_features

        # 组合所有特征
        combined_features = torch.cat([material_embed, physics_features], dim=1)

        # 编码到潜在空间
        z = self.input_layer(combined_features)
        z = self.encoder_norm(z)

        # 通过残差块
        for block in self.residual_blocks:
            z = block(z)
            if self.mc_dropout_enabled and not self.training:
                pass

        # 符号表达式输出
        symbolic_output = None
        symbolic_aligned = None
        
        if self.symbolic_layer is not None:
            try:
                if self.use_advanced_symbolic:
                    symbolic_output = self.symbolic_layer(z, numeric_features)
                else:
                    symbolic_output = self.symbolic_layer(z)
                    
                # 符号输出对齐
                if self.sym_scale_net is not None:
                    scale_bias = self.sym_scale_net(z)
                    scale_raw = scale_bias[:, 0:1]
                    bias_raw = scale_bias[:, 1:2]
                    scale = torch.exp(scale_raw * 0.5)
                    bias = torch.tanh(bias_raw) * 0.5
                    symbolic_aligned = scale * symbolic_output + bias
                else:
                    symbolic_aligned = symbolic_output
                    
            except Exception as e:
                print(f"Error in symbolic layer: {e}")
                symbolic_output = torch.zeros(z.size(0), 1, device=z.device)
                symbolic_aligned = symbolic_output

        # 残差网络输出
        residual_output = None
        
        if self.residual_net is not None:
            if symbolic_output is not None:
                residual_input = torch.cat([z, symbolic_output], dim=-1)
            else:
                residual_input = z
            residual_output = self.residual_net(residual_input)

        # 融合预测结果
        if self.ablation_mode in ["no_symbolic", "fixed_gate_0.0", "no_physics_baseline"]:
            # 仅使用残差网络
            prediction = residual_output
            gate = torch.zeros(z.size(0), 1, device=z.device)
            
        elif self.ablation_mode == "no_residual" or self.ablation_mode == "fixed_gate_1.0":
            # 仅使用符号层
            prediction = symbolic_aligned
            gate = torch.ones(z.size(0), 1, device=z.device)
            
        elif self.ablation_mode.startswith("fixed_gate"):
            # 固定门控值
            gate = torch.full((z.size(0), 1), self.fixed_gate_value, device=z.device)
            prediction = gate * symbolic_aligned + (1 - gate) * residual_output
            
        else:
            # 自适应门控(完整模型)
            gate = self.fusion_gate(z)
            prediction = gate * symbolic_aligned + (1 - gate) * residual_output
        
        self.last_gate = gate.detach().clone()

        # 应用激活函数确保非负
        prediction = F.softplus(prediction, beta=5) + 1e-3

        # 不确定性估计
        log_var = self.aleatoric_net(z).clamp(-10, 6)
        uncertainty = torch.exp(log_var) + 1e-6

        # 处理None值
        if symbolic_output is None:
            symbolic_output = torch.zeros_like(prediction)
        if residual_output is None:
            residual_output = torch.zeros_like(prediction)

        return prediction, uncertainty, symbolic_output, residual_output

    def predict_with_uncertainty(self, material_ids, numeric_features, n_samples=10):
        """使用MC Dropout进行带不确定性的预测"""
        self.enable_mc_dropout(True)
        self.eval()

        predictions = []
        uncertainties = []

        with torch.no_grad():
            for _ in range(n_samples):
                pred, unc, _, _ = self.forward(material_ids, numeric_features)
                predictions.append(pred)
                uncertainties.append(unc)

        self.enable_mc_dropout(False)

        mean_pred = torch.mean(torch.stack(predictions), dim=0)
        aleatoric_uncertainty = torch.mean(torch.stack(uncertainties), dim=0)
        epistemic_uncertainty = torch.var(torch.stack(predictions), dim=0)
        total_uncertainty = aleatoric_uncertainty + epistemic_uncertainty

        return (
            mean_pred,
            total_uncertainty,
            aleatoric_uncertainty,
            epistemic_uncertainty,
        )
        
    def get_symbolic_interpretation(self):
        """获取符号层的解释性信息"""
        interpretation = {"ablation_mode": self.ablation_mode}
        
        if hasattr(self, 'last_gate') and self.last_gate is not None:
            interpretation["gate_value"] = float(self.last_gate.mean().item())
            
        if self.symbolic_layer is not None and hasattr(self.symbolic_layer, 'get_interpretation'):
            symbolic_interpretation = self.symbolic_layer.get_interpretation()
            interpretation.update(symbolic_interpretation)
        else:
            interpretation["symbolic_message"] = f"消融模式 {self.ablation_mode} 无符号层解释"
            
        return interpretation
    
    def get_gate_statistics(self):
        """获取门控统计信息"""
        if self.last_gate is None:
            return {
                'mean': 0.5,
                'std': 0.0,
                'min': 0.5,
                'max': 0.5,
                'symbolic_weight': 0.5,
                'residual_weight': 0.5,
                'mode': self.ablation_mode
            }
        
        gate = self.last_gate
        mean_gate = float(gate.mean().item())
        std_gate = float(gate.std().item())
        min_gate = float(gate.min().item())
        max_gate = float(gate.max().item())
        
        return {
            'mean': mean_gate,
            'std': std_gate,
            'min': min_gate,
            'max': max_gate,
            'symbolic_weight': mean_gate,
            'residual_weight': 1.0 - mean_gate,
            'mode': self.ablation_mode
        }
    
    def get_fusion_interpretation(self):
        """分析融合门的行为"""
        stats = self.get_gate_statistics()
        mode = stats['mode']
        mean_gate = stats['mean']
        
        if mode == "no_symbolic":
            return f"消融模式: 无符号层 (纯残差网络)"
        elif mode == "no_physics_baseline":
            return f"消融模式: 无物理特征+无符号层 (纯数据驱动基线)"
        elif mode == "no_residual":
            return f"消融模式: 无残差网络 (纯符号层)"
        elif mode.startswith("fixed_gate"):
            return f"消融模式: 固定门控 (符号层权重={mean_gate:.2f})"
        elif mode == "simple_symbolic":
            return f"消融模式: 简化符号层 (无注意力机制), 平均门控={mean_gate:.2f}"
        elif mode == "no_physics":
            return f"消融模式: 无物理特征提取, 平均门控={mean_gate:.2f}"
        else:
            if mean_gate > 0.7:
                return f"完整模型 - 符号层主导, 平均信任度: {mean_gate:.2f}"
            elif mean_gate > 0.5:
                return f"完整模型 - 偏向符号层, 平均信任度: {mean_gate:.2f}"
            elif mean_gate > 0.3:
                return f"完整模型 - 平衡状态, 平均信任度: {mean_gate:.2f}"
            else:
                return f"完整模型 - 残差主导, 平均信任度: {mean_gate:.2f}"
