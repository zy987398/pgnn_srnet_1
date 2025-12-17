import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import pandas as pd
from matplotlib import font_manager
import os
import json

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi', 'FangSong']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12

# 设置matplotlib后端
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = 'sans-serif'

# 创建全局字体属性对象
chinese_font_prop = None
try:
    chinese_fonts = ['Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi', 'FangSong']
    for font_name in chinese_fonts:
        fonts = [f.name for f in font_manager.fontManager.ttflist]
        if font_name in fonts:
            chinese_font_prop = font_manager.FontProperties(family=font_name)
            print(f"使用中文字体: {font_name}")
            break
    if chinese_font_prop is None:
        chinese_font_prop = font_manager.FontProperties(family='DejaVu Sans')
        print("使用默认字体: DejaVu Sans")
except Exception as e:
    chinese_font_prop = font_manager.FontProperties(family='DejaVu Sans')
    print(f"字体设置出错，使用默认字体: {e}")


def get_chinese_font():
    """获取可用的中文字体"""
    global chinese_font_prop
    return chinese_font_prop.get_name() if chinese_font_prop else 'DejaVu Sans'


def apply_chinese_font_to_axes(ax):
    """为指定的axes应用中文字体"""
    global chinese_font_prop
    if chinese_font_prop is None:
        return
    
    # 强制设置轴标签字体
    xlabel = ax.get_xlabel()
    if xlabel:
        ax.set_xlabel(xlabel, fontproperties=chinese_font_prop)
    
    ylabel = ax.get_ylabel()
    if ylabel:
        ax.set_ylabel(ylabel, fontproperties=chinese_font_prop)
    
    title = ax.get_title()
    if title:
        ax.set_title(title, fontproperties=chinese_font_prop)
    
    # 强制设置刻度标签字体
    for label in ax.get_xticklabels():
        label.set_fontproperties(chinese_font_prop)
    for label in ax.get_yticklabels():
        label.set_fontproperties(chinese_font_prop)
    
    # 设置图例字体
    legend = ax.get_legend()
    if legend:
        for text in legend.get_texts():
            text.set_fontproperties(chinese_font_prop)
    
    # 强制刷新显示
    if hasattr(ax.figure, 'canvas'):
        ax.figure.canvas.draw_idle()


class InterpretabilityAnalyzer:
    """
    模型可解释性分析器
    提取并分析AdaptiveSymbolicLayer内部的推理参数
    """
    
    def __init__(self, model, feature_names=None):
        self.model = model
        self.feature_names = feature_names or self._get_default_feature_names()
        self.device = next(model.parameters()).device
        
    def _get_default_feature_names(self):
        """默认物理特征名称"""
        return [
            # 原始特征 (6个)
            "莫氏硬度 H", "弹性模量 E", "断裂韧性 KIC", 
            "晶粒粒径 d", "压溃压力 σc", "磨粒速度 V",
            # 物理特征 (12个)
            "Griffith能量G", "应力强度比 K/Kc", "Hall-Petch项 d^(-1/2)",
            "裂纹屏蔽因子 χ", "载荷强度比 ρ₁", "功率强度比 ρ₂",
            "临界深度 dc", "临界裂纹比 ac/d", "塑性区比 rp/d",
            "脆性指数 B", "韧性模量比 KIC/E", "应变能密度 U"
        ]
    
    def analyze_mechanism_attention(self, data_loader, material_type_filter=None, save_dir="results"):
        """
        4.4.2.1 物理机理注意力分布分析
        
        参数:
            data_loader: 数据加载器
            material_type_filter: 材料类型过滤 (如 "脆性材料")
            save_dir: 保存目录路径
        
        返回:
            attention_analysis: 注意力权重分析结果
        """
        print("\n" + "="*70)
        print("4.4.2.1 物理机理注意力分布 (Mechanism Attention Map)")
        print("="*70)
        
        self.model.eval()
        all_var_weights = []
        all_material_ids = []
        
        with torch.no_grad():
            for batch in data_loader:
                material_ids, numeric_features, numeric_raw, _ = batch
                material_ids = material_ids.to(self.device)
                numeric_raw = numeric_raw.to(self.device)
                
                # 前向传播以触发符号层
                _ = self.model(material_ids, numeric_raw)
                
                # 提取变量权重
                interp = self.model.get_symbolic_interpretation()
                if 'variable_weights' in interp and interp['variable_weights'] is not None:
                    var_weights = interp['variable_weights'].cpu()
                    all_var_weights.append(var_weights)
                    all_material_ids.extend(material_ids.cpu().numpy())
        
        if len(all_var_weights) == 0:
            print("警告: 无法提取变量权重")
            return None
        
        # 合并所有批次的权重
        all_var_weights = torch.cat(all_var_weights, dim=0).numpy()  # [N, 18]
        
        # 计算重要性: Importance(φᵢ) = [w_var]ᵢ (已经是softmax后的值)
        mean_importance = all_var_weights.mean(axis=0)
        std_importance = all_var_weights.std(axis=0)
        
        print("\n物理特征重要性排名:")
        print("-" * 70)
        sorted_indices = np.argsort(mean_importance)[::-1]
        
        for rank, idx in enumerate(sorted_indices, 1):
            feature_name = self.feature_names[idx]
            importance = mean_importance[idx]
            std = std_importance[idx]
            bar = '█' * int(importance * 100)
            print(f"{rank:2d}. {feature_name:25s}: {importance:.4f} ± {std:.4f} {bar}")
        
        # 识别主导物理机制
        self._identify_dominant_mechanisms(mean_importance, sorted_indices)
        
        # 可视化 - 使用save_dir
        save_path = os.path.join(save_dir, "attention_heatmap.png")
        self._plot_attention_heatmap(all_var_weights, all_material_ids, save_path=save_path)
        
        return {
            'mean_importance': mean_importance,
            'std_importance': std_importance,
            'all_weights': all_var_weights,
            'material_ids': all_material_ids
        }
    
    def _identify_dominant_mechanisms(self, importance, sorted_indices):
        """识别主导物理机制"""
        print("\n主导物理机制识别:")
        print("-" * 70)
        
        # 定义物理机制组
        mechanisms = {
            'Griffith能量准则': [6],  # G
            'Hall-Petch效应': [8],    # d^(-1/2)
            '应力强度理论': [7, 9],   # K/Kc, χ
            '塑性变形': [14],         # rp/d
            '脆性断裂': [15],         # B
            '材料基础属性': [0, 1, 2, 3]  # H, E, KIC, d
        }
        
        mechanism_scores = {}
        for mech_name, indices in mechanisms.items():
            score = sum(importance[i] for i in indices if i < len(importance))
            mechanism_scores[mech_name] = score
        
        sorted_mechs = sorted(mechanism_scores.items(), key=lambda x: x[1], reverse=True)
        
        for mech_name, score in sorted_mechs:
            percentage = score * 100
            print(f"  • {mech_name:20s}: {score:.4f} ({percentage:.1f}%)")
        
        # 判断材料类型特征
        print("\n物理定律验证:")
        if importance[6] > 0.15:  # Griffith能量项
            print("  ✓ 模型成功捕捉能量驱动的断裂机制 (Griffith准则)")
        if importance[8] > 0.10:  # Hall-Petch
            print("  ✓ 模型识别出晶粒尺寸效应 (Hall-Petch关系)")
        if importance[15] > 0.10:  # 脆性指数
            print("  ✓ 模型区分了脆性/韧性断裂模式")
    
    def _plot_attention_heatmap(self, weights, material_ids, save_path):
        """绘制注意力热图"""
        # 确保保存目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # 左图: 所有样本的注意力热图
        ax1 = axes[0]
        im1 = ax1.imshow(weights.T, aspect='auto', cmap='YlOrRd', interpolation='nearest')
        ax1.set_xlabel('样本索引', fontsize=12, fontproperties=chinese_font_prop)
        ax1.set_ylabel('物理特征', fontsize=12, fontproperties=chinese_font_prop)
        ax1.set_title('物理机理注意力分布热图', fontsize=14, fontweight='bold', fontproperties=chinese_font_prop)
        ax1.set_yticks(range(len(self.feature_names)))
        ax1.set_yticklabels(self.feature_names, fontsize=9, fontproperties=chinese_font_prop)
        
        # 为colorbar添加中文标签
        cbar1 = plt.colorbar(im1, ax=ax1)
        cbar1.set_label('注意力权重', fontproperties=chinese_font_prop)
        
        # 应用中文字体到所有刻度标签
        for label in ax1.get_xticklabels():
            label.set_fontproperties(chinese_font_prop)
        
        # 右图: 平均注意力柱状图
        ax2 = axes[1]
        mean_weights = weights.mean(axis=0)
        std_weights = weights.std(axis=0)
        indices = np.argsort(mean_weights)[::-1][:15]  # 显示前15个
        
        y_pos = np.arange(len(indices))
        ax2.barh(y_pos, mean_weights[indices], xerr=std_weights[indices], 
                 color='steelblue', alpha=0.8, capsize=3)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels([self.feature_names[i] for i in indices], fontsize=10, fontproperties=chinese_font_prop)
        ax2.set_xlabel('平均注意力权重', fontsize=12, fontproperties=chinese_font_prop)
        ax2.set_title('Top-15 物理特征重要性', fontsize=14, fontweight='bold', fontproperties=chinese_font_prop)
        ax2.grid(axis='x', alpha=0.3)
        
        # 应用中文字体到所有刻度标签
        for label in ax2.get_xticklabels():
            label.set_fontproperties(chinese_font_prop)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\n注意力热图已保存: {save_path}")
    
    def analyze_symbolic_topology(self, data_loader, save_dir="results"):
        """
        4.4.2.2 符号算子拓扑分析
        
        参数:
            save_dir: 保存目录路径
        
        返回:
            topology_analysis: 算子使用频率和拓扑结构
        """
        print("\n" + "="*70)
        print("4.4.2.2 符号算子拓扑 (Symbolic Operator Topology)")
        print("="*70)
        
        self.model.eval()
        all_op_weights = []
        
        with torch.no_grad():
            for batch in data_loader:
                material_ids, numeric_features, numeric_raw, _ = batch
                material_ids = material_ids.to(self.device)
                numeric_raw = numeric_raw.to(self.device)
                
                _ = self.model(material_ids, numeric_raw)
                
                interp = self.model.get_symbolic_interpretation()
                if 'operator_weights' in interp and interp['operator_weights'] is not None:
                    op_weights = interp['operator_weights'].cpu()
                    all_op_weights.append(op_weights)
        
        if len(all_op_weights) == 0:
            print("警告: 无法提取算子权重")
            return None
        
        all_op_weights = torch.cat(all_op_weights, dim=0).numpy()
        operators = self.model.get_symbolic_interpretation()['operators']
        
        # 计算平均激活概率
        mean_op_weights = all_op_weights.mean(axis=0)
        std_op_weights = all_op_weights.std(axis=0)
        
        # Topology = argmax([w_op]_k)
        dominant_operator_idx = np.argmax(mean_op_weights)
        dominant_operator = operators[dominant_operator_idx]
        
        print("\n算子激活频率分布:")
        print("-" * 70)
        sorted_op_indices = np.argsort(mean_op_weights)[::-1]
        
        for rank, idx in enumerate(sorted_op_indices, 1):
            op_name = operators[idx]
            weight = mean_op_weights[idx]
            std = std_op_weights[idx]
            bar = '█' * int(weight * 50)
            marker = " ← 主导算子" if idx == dominant_operator_idx else ""
            print(f"{rank}. {op_name:8s}: {weight:.4f} ± {std:.4f} {bar}{marker}")
        
        # 物理意义解释
        self._interpret_operator_topology(mean_op_weights, operators)
        
        # 可视化 - 使用save_dir
        save_path = os.path.join(save_dir, "operator_topology.png")
        self._plot_operator_topology(all_op_weights, operators, save_path=save_path)
        
        return {
            'mean_weights': mean_op_weights,
            'std_weights': std_op_weights,
            'dominant_operator': dominant_operator,
            'operators': operators,
            'all_weights': all_op_weights
        }
    
    def _interpret_operator_topology(self, weights, operators):
        """解释算子拓扑的物理意义"""
        print("\n物理耦合形式解释:")
        print("-" * 70)
        
        interpretations = {
            'sqrt': "应力奇异性 (1/√r) - 裂纹尖端场",
            '^2': "能量密度 (σ²/E) - 弹性能量",
            '^3': "高阶非线性效应",
            '*': "线性耦合 - 多物理场相乘",
            '/': "无量纲化 - 物理比值",
            'log': "尺度效应 - 对数衰减",
            'exp': "指数增长/衰减",
            'sin': "周期性效应"
        }
        
        for op, weight in zip(operators, weights):
            if weight > 0.05:  # 只显示显著的算子
                interp = interpretations.get(op, "未知物理意义")
                print(f"  • {op:8s} (权重={weight:.3f}): {interp}")
        
        # 特殊警示
        if weights[operators.index('sqrt')] > 0.15:
            print("\n  ✓ 高频激活√算子 → 模型捕捉到应力奇异性特征")
        if weights[operators.index('^2')] > 0.15:
            print("  ✓ 高频激活^2算子 → 模型识别能量密度关系")
    
    def _plot_operator_topology(self, weights, operators, save_path):
        """绘制算子拓扑图"""
        # 确保保存目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 左图: 算子权重箱线图
        ax1 = axes[0]
        bp = ax1.boxplot(weights, labels=operators, patch_artist=True, 
                         notch=True, showmeans=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        ax1.set_ylabel('算子权重', fontsize=12, fontproperties=chinese_font_prop)
        ax1.set_title('算子激活分布 (箱线图)', fontsize=14, fontweight='bold', fontproperties=chinese_font_prop)
        ax1.grid(axis='y', alpha=0.3)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # 应用中文字体到刻度标签
        for label in ax1.get_yticklabels():
            label.set_fontproperties(chinese_font_prop)
        
        # 右图: 算子共现矩阵
        ax2 = axes[1]
        # 计算算子之间的相关性
        correlation = np.corrcoef(weights.T)
        im = ax2.imshow(correlation, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
        ax2.set_xticks(range(len(operators)))
        ax2.set_yticks(range(len(operators)))
        ax2.set_xticklabels(operators, rotation=45, ha='right')
        ax2.set_yticklabels(operators)
        ax2.set_title('算子共现相关性矩阵', fontsize=14, fontweight='bold', fontproperties=chinese_font_prop)
        
        # 添加数值标注
        for i in range(len(operators)):
            for j in range(len(operators)):
                text = ax2.text(j, i, f'{correlation[i, j]:.2f}',
                               ha="center", va="center", color="black", fontsize=8)
        
        # 为colorbar添加中文标签
        cbar2 = plt.colorbar(im, ax=ax2)
        cbar2.set_label('相关系数', fontproperties=chinese_font_prop)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\n算子拓扑图已保存: {save_path}")
    
    def analyze_coefficient_sensitivity(self, data_loader, varying_feature_idx=0, save_dir="results"):
        """
        4.4.2.3 唯象系数敏感度分析
        
        参数:
            data_loader: 数据加载器
            varying_feature_idx: 要分析的变化特征索引 (默认0=莫氏硬度H)
            save_dir: 保存目录路径
        
        返回:
            sensitivity_analysis: 系数随材料属性的演化轨迹
        """
        print("\n" + "="*70)
        print("4.4.2.3 唯象系数敏感度 (Phenomenological Coefficient Sensitivity)")
        print("="*70)
        
        self.model.eval()
        all_coeffs = []
        all_features = []
        
        with torch.no_grad():
            for batch in data_loader:
                material_ids, numeric_features, numeric_raw, _ = batch
                material_ids = material_ids.to(self.device)
                numeric_raw = numeric_raw.to(self.device)
                
                _ = self.model(material_ids, numeric_raw)
                
                interp = self.model.get_symbolic_interpretation()
                if 'coefficients' in interp and interp['coefficients'] is not None:
                    coeffs = interp['coefficients'].cpu()
                    all_coeffs.append(coeffs)
                    all_features.append(numeric_raw.cpu())
        
        if len(all_coeffs) == 0:
            print("警告: 无法提取系数")
            return None
        
        all_coeffs = torch.cat(all_coeffs, dim=0).numpy()  # [N, 4]
        all_features = torch.cat(all_features, dim=0).numpy()  # [N, 6]
        
        # c = [c₁, c₂, c₃, c₄]^T 对应 [线性, 平方, 立方, 平方根]
        coeff_names = ['c₁ (线性项)', 'c₂ (平方项)', 'c₃ (立方项)', 'c₄ (平方根项)']
        
        print(f"\n系数统计 (基于{len(all_coeffs)}个样本):")
        print("-" * 70)
        for i, name in enumerate(coeff_names):
            mean_c = all_coeffs[:, i].mean()
            std_c = all_coeffs[:, i].std()
            min_c = all_coeffs[:, i].min()
            max_c = all_coeffs[:, i].max()
            print(f"{name:15s}: μ={mean_c:8.4f}, σ={std_c:7.4f}, "
                  f"范围=[{min_c:7.4f}, {max_c:7.4f}]")
        
        # 分析系数随材料属性的演化
        varying_feature = all_features[:, varying_feature_idx]
        feature_name = self.feature_names[varying_feature_idx]
        
        print(f"\n系数随 {feature_name} 的演化趋势:")
        print("-" * 70)
        
        for i, coeff_name in enumerate(coeff_names):
            correlation = np.corrcoef(varying_feature, all_coeffs[:, i])[0, 1]
            trend = "正相关" if correlation > 0.1 else ("负相关" if correlation < -0.1 else "无关")
            print(f"{coeff_name:15s}: 相关系数 r={correlation:6.3f} ({trend})")
        
        # 可视化 - 使用save_dir
        save_path = os.path.join(save_dir, "coefficient_sensitivity.png")
        self._plot_coefficient_evolution(all_coeffs, varying_feature, feature_name, coeff_names, save_path=save_path)
        
        # 发现潜在的无量纲物理数
        self._discover_dimensionless_numbers(all_coeffs, all_features)
        
        return {
            'coefficients': all_coeffs,
            'features': all_features,
            'varying_feature': varying_feature,
            'coeff_names': coeff_names
        }
    
    def _plot_coefficient_evolution(self, coeffs, feature, feature_name, coeff_names, save_path):
        """绘制系数演化图"""
        # 确保保存目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for i, (ax, coeff_name) in enumerate(zip(axes, coeff_names)):
            # 散点图 + 拟合曲线
            ax.scatter(feature, coeffs[:, i], alpha=0.5, s=20, color='steelblue')
            
            # 多项式拟合
            z = np.polyfit(feature, coeffs[:, i], 2)
            p = np.poly1d(z)
            x_fit = np.linspace(feature.min(), feature.max(), 100)
            ax.plot(x_fit, p(x_fit), 'r-', linewidth=2, label='二次拟合')
            
            ax.set_xlabel(feature_name, fontsize=11, fontproperties=chinese_font_prop)
            ax.set_ylabel(coeff_name, fontsize=11, fontproperties=chinese_font_prop)
            ax.set_title(f'{coeff_name} vs {feature_name}', fontsize=12, fontweight='bold', fontproperties=chinese_font_prop)
            ax.grid(alpha=0.3)
            ax.legend(prop=chinese_font_prop)
            
            # 应用中文字体到刻度标签
            for label in ax.get_xticklabels():
                label.set_fontproperties(chinese_font_prop)
            for label in ax.get_yticklabels():
                label.set_fontproperties(chinese_font_prop)
            
            # 显示拟合方程
            eq_text = f'y = {z[0]:.2e}x² + {z[1]:.2e}x + {z[2]:.2e}'
            ax.text(0.05, 0.95, eq_text, transform=ax.transAxes, 
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\n系数敏感度图已保存: {save_path}")
    
    def _discover_dimensionless_numbers(self, coeffs, features):
        """发现潜在的无量纲物理数"""
        print("\n潜在无量纲物理数发现:")
        print("-" * 70)
        
        # 计算系数比值
        ratios = {
            'c₂/c₁² (非线性度)': coeffs[:, 1] / (coeffs[:, 0]**2 + 1e-8),
            'c₃/c₁³ (高阶效应)': coeffs[:, 3] / (coeffs[:, 0]**3 + 1e-8),
            'c₄/√c₁ (根号修正)': coeffs[:, 3] / (np.sqrt(np.abs(coeffs[:, 0])) + 1e-8)
        }
        
        for ratio_name, ratio_values in ratios.items():
            mean_ratio = np.nanmean(ratio_values[np.isfinite(ratio_values)])
            std_ratio = np.nanstd(ratio_values[np.isfinite(ratio_values)])
            
            # 判断是否接近常数 (潜在的无量纲数)
            cv = std_ratio / (abs(mean_ratio) + 1e-8)  # 变异系数
            
            if cv < 0.3:  # 变异系数小于30%
                print(f"  ✓ 发现准常数: {ratio_name}")
                print(f"    值: {mean_ratio:.4f} ± {std_ratio:.4f} (CV={cv:.2%})")
            else:
                print(f"  • {ratio_name}: {mean_ratio:.4f} ± {std_ratio:.4f} (CV={cv:.2%})")
    
    def generate_full_report(self, data_loader, save_dir="results"):
        """生成完整的可解释性分析报告"""
        os.makedirs(save_dir, exist_ok=True)
        
        print("\n" + "="*70)
        print("生成完整可解释性分析报告")
        print("="*70)
        
        # 1. 物理机理注意力分析 - 传递save_dir
        attention_result = self.analyze_mechanism_attention(data_loader, save_dir=save_dir)
        
        # 2. 符号算子拓扑分析 - 传递save_dir
        topology_result = self.analyze_symbolic_topology(data_loader, save_dir=save_dir)
        
        # 3. 唯象系数敏感度分析 - 传递save_dir
        sensitivity_result = self.analyze_coefficient_sensitivity(
            data_loader, 
            varying_feature_idx=0, 
            save_dir=save_dir
        )
        
        # 保存JSON报告
        report = {
            'mechanism_attention': {
                'mean_importance': attention_result['mean_importance'].tolist(),
                'feature_names': self.feature_names,
                'top_features': [
                    {
                        'name': self.feature_names[i],
                        'importance': float(attention_result['mean_importance'][i])
                    }
                    for i in np.argsort(attention_result['mean_importance'])[::-1][:10]
                ]
            },
            'operator_topology': {
                'operator_weights': topology_result['mean_weights'].tolist(),
                'operators': topology_result['operators'],
                'dominant_operator': topology_result['dominant_operator']
            },
            'coefficient_sensitivity': {
                'coefficient_stats': [
                    {
                        'name': sensitivity_result['coeff_names'][i],
                        'mean': float(sensitivity_result['coefficients'][:, i].mean()),
                        'std': float(sensitivity_result['coefficients'][:, i].std())
                    }
                    for i in range(4)
                ]
            }
        }
        
        report_path = os.path.join(save_dir, 'interpretability_full_report.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=4, ensure_ascii=False)
        
        print(f"\n完整报告已保存: {report_path}")
        
        return report


# ========== 使用示例 ==========
def run_interpretability_analysis(model, test_data, device, batch_size=32, save_dir="results"):
    """
    运行完整的可解释性分析
    
    参数:
        model: 训练好的模型
        test_data: 测试数据字典
        device: 设备
        batch_size: 批次大小
        save_dir: 保存目录路径
    """
    from torch.utils.data import TensorDataset, DataLoader
    
    # 准备数据加载器
    test_dataset = TensorDataset(
        torch.LongTensor(test_data["material_ids"]),
        torch.FloatTensor(test_data["numeric_features"]),
        torch.FloatTensor(test_data["numeric_raw"]),
        torch.FloatTensor(test_data["target"]).unsqueeze(1),
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 创建分析器
    analyzer = InterpretabilityAnalyzer(model)
    
    # 生成完整报告 - 传递save_dir
    report = analyzer.generate_full_report(test_loader, save_dir=save_dir)
    
    return report