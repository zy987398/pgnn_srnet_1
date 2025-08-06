"""
材料性能数据分析
对材料物理特性和生成的物理特征进行可视化分析
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.feature_selection import mutual_info_regression
import torch
from scipy.stats import pearsonr, spearmanr
from old_pgnn.utils import prepare_material_data
import os

# 设置更好的可视化样式
plt.style.use('seaborn-v0_8-whitegrid')
sns.set(font_scale=1.2)
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.family'] = 'SimHei'  # 支持中文显示
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 创建结果目录
if not os.path.exists('analysis_results'):
    os.makedirs('analysis_results')

def load_and_preprocess_data(file_path='3011916164精确材料性能数据.csv'):
    """加载和预处理数据"""
    print(f"正在加载数据: {file_path}")
    df = pd.read_csv(file_path)
    
    # 显示基本信息
    print("\n数据基本信息:")
    print(f"样本数量: {len(df)}")
    print(f"特征数量: {len(df.columns)}")
    print(f"材料种类数量: {df['材料种类'].nunique()}")
    
    # 使用项目中的预处理方法
    data_dict = prepare_material_data(df, verbose=True)
    
    # 返回原始数据和预处理后的数据
    return df, data_dict

def generate_physics_features(data_dict):
    """生成物理特征"""
    # 转换为PyTorch张量
    numeric_raw = torch.tensor(data_dict['numeric_raw'], dtype=torch.float32)
    
    # 创建特征名称列表
    orig_feature_names = ['莫氏硬度', '弹性模量', '断裂韧性', '晶粒粒径', '压溃压力', '磨粒速度']
    physics_feature_names = [
        'Griffith能量释放率', 
        '应力强度比',
        'Hall-Petch系数',
        '裂纹屏蔽因子',
        '载荷-强度比',
        '功率-强度比',
        '延-脆转临界深度',
        '临界裂纹长度比',
        '塑性区尺寸比',
        '脆性指数',
        '初-剖比',
        '弹性应变能密度'
    ]
    
    # 手动计算物理特征
    hardness = numeric_raw[:, 0:1]          # 莫氏硬度
    elastic_modulus = numeric_raw[:, 1:2]   # 弹性模量
    fracture_tough = numeric_raw[:, 2:3]    # 断裂韧性
    grain_size = numeric_raw[:, 3:4]        # 晶粒粒径
    stress = numeric_raw[:, 4:5]            # 压溃压力
    velocity = numeric_raw[:, 5:6]          # 磨粒速度
    
    eps = 1e-6  # 避免除零
    π = 3.14159
    
    # 计算12个物理特征
    physics_features = []
    
    # 1. Griffith 能量释放率 G = σ² π d / E [J·m⁻²]
    stress_squared = torch.clamp(stress**2, max=1e12)
    G = stress_squared * π * grain_size / (elastic_modulus + eps)
    G_scaled = torch.tanh(torch.clamp(G / 1e3, -10, 10))
    physics_features.append(G_scaled.numpy())
    
    # 2. 应力强度比 K/Kc = σ √(π d) / K_IC [—]
    sqrt_term = torch.sqrt(torch.clamp(π * grain_size, min=eps, max=1.0))
    k_ratio = stress * sqrt_term / (fracture_tough + eps)
    physics_features.append(torch.tanh(torch.clamp(k_ratio, -10, 10)).numpy())
    
    # 3. Hall-Petch 尺度项 d^{-1/2} [m^{-½}]
    hall_petch = torch.rsqrt(torch.clamp(grain_size + eps, min=eps, max=1.0))
    hall_petch = torch.clamp(hall_petch, max=1e6)
    physics_features.append(torch.tanh(torch.clamp(hall_petch / 1e3, -10, 10)).numpy())
    
    # 4. 裂纹屏蔽因子 χ = (K_IC / E) √d [m]
    k_ic_over_e = fracture_tough / (elastic_modulus + eps)
    sqrt_grain = torch.sqrt(torch.clamp(grain_size, min=eps, max=1.0))
    shielding = k_ic_over_e * sqrt_grain
    shielding = torch.clamp(shielding, max=1e-2)
    physics_features.append(torch.tanh(torch.clamp(shielding / 1e-4, -10, 10)).numpy())
    
    # 5. 载荷-强度比 ρ₁ = σ / H [—]
    rho1 = stress / (hardness + eps)
    # 使用Min-Max缩放代替tanh，保留更多信息
    rho1_min = torch.min(rho1)
    rho1_max = torch.max(rho1)
    rho1_range = rho1_max - rho1_min + eps
    rho1_norm = (rho1 - rho1_min) / rho1_range
    print(f"载荷-强度比原始值范围: {rho1_min.item():.6f} - {rho1_max.item():.6f}, 变异系数: {torch.std(rho1).item() / (torch.mean(rho1).item() + eps):.6f}")
    physics_features.append(rho1_norm.numpy())
    
    # 6. 功率-强度比 ρ₂ = σ V / H [m·s⁻¹]
    rho2 = stress * velocity / (hardness + eps)
    # 使用Min-Max缩放
    rho2_min = torch.min(rho2)
    rho2_max = torch.max(rho2)
    rho2_range = rho2_max - rho2_min + eps
    rho2_norm = (rho2 - rho2_min) / rho2_range
    print(f"功率-强度比原始值范围: {rho2_min.item():.6f} - {rho2_max.item():.6f}, 变异系数: {torch.std(rho2).item() / (torch.mean(rho2).item() + eps):.6f}")
    physics_features.append(rho2_norm.numpy())
    
    # 7. 延-脆转临界深度 d_c = η (E/H) (K_IC/H)² [m] (η≈0.15)
    eta = 0.15
    e_over_h = elastic_modulus / (hardness + eps)
    k_over_h = fracture_tough / (hardness + eps)
    k_over_h_squared = k_over_h**2  # 移除过度截断
    d_c = eta * e_over_h * k_over_h_squared
    # 使用标准化缩放而非tanh
    d_c_mean = torch.mean(d_c)
    d_c_std = torch.std(d_c) + eps
    d_c_norm = (d_c - d_c_mean) / d_c_std * 0.3 + 0.5  # 缩放到大约[0.2-0.8]范围
    print(f"延-脆转临界深度原始值范围: {torch.min(d_c).item():.6e} - {torch.max(d_c).item():.6e}, 变异系数: {torch.std(d_c).item() / (torch.mean(d_c).item() + eps):.6f}")
    physics_features.append(torch.clamp(d_c_norm, 0, 1).numpy())
    
    # 8. 临界裂纹长度比 a_c / d, a_c = (K_IC/σ)² / π [—]
    k_over_s = fracture_tough / (stress + eps)
    k_over_s_squared = k_over_s**2  # 移除过度截断
    a_c = k_over_s_squared / π
    ac_over_d = a_c / (grain_size + eps)
    # 对数缩放后再标准化
    ac_log = torch.log1p(torch.clamp(ac_over_d, min=eps))
    ac_log_mean = torch.mean(ac_log)
    ac_log_std = torch.std(ac_log) + eps
    ac_norm = (ac_log - ac_log_mean) / ac_log_std * 0.3 + 0.5
    print(f"临界裂纹长度比原始值范围: {torch.min(ac_over_d).item():.6e} - {torch.max(ac_over_d).item():.6e}, 变异系数: {torch.std(ac_over_d).item() / (torch.mean(ac_over_d).item() + eps):.6f}")
    physics_features.append(torch.clamp(ac_norm, 0, 1).numpy())
    
    # 9. 塑性区尺寸比 r_p / d, r_p = (K_IC / H)² / (2π) [—]
    r_p = k_over_h_squared / (2.0 * π)
    rp_over_d = r_p / (grain_size + eps)
    # 对数缩放后再使用Min-Max归一化
    rp_log = torch.log1p(torch.clamp(rp_over_d, min=eps))
    rp_log_min = torch.min(rp_log)
    rp_log_max = torch.max(rp_log)
    rp_log_range = rp_log_max - rp_log_min + eps
    rp_norm = (rp_log - rp_log_min) / rp_log_range
    print(f"塑性区尺寸比原始值范围: {torch.min(rp_over_d).item():.6e} - {torch.max(rp_over_d).item():.6e}, 变异系数: {torch.std(rp_over_d).item() / (torch.mean(rp_over_d).item() + eps):.6f}")
    physics_features.append(rp_norm.numpy())
    
    # 10. 脆性指数 B = H / K_IC [√m⁻¹]
    B = hardness / (fracture_tough + eps)
    # 使用Min-Max缩放代替tanh，避免信息损失
    B_min = torch.min(B)
    B_max = torch.max(B)
    B_range = B_max - B_min + eps
    B_norm = (B - B_min) / B_range
    print(f"脆性指数原始值范围: {B_min.item():.6f} - {B_max.item():.6f}, 变异系数: {torch.std(B).item() / (torch.mean(B).item() + eps):.6f}")
    physics_features.append(B_norm.numpy())
    
    # 11. 初-剖比 K_IC / E [√m]
    ki_over_e = fracture_tough / (elastic_modulus + eps)
    # 使用标准化归一化
    ki_over_e_mean = torch.mean(ki_over_e)
    ki_over_e_std = torch.std(ki_over_e) + eps
    ki_over_e_norm = (ki_over_e - ki_over_e_mean) / ki_over_e_std * 0.3 + 0.5  # 缩放到大约[0.2-0.8]范围
    print(f"初-剖比原始值范围: {torch.min(ki_over_e).item():.6e} - {torch.max(ki_over_e).item():.6e}, 变异系数: {torch.std(ki_over_e).item() / (torch.mean(ki_over_e).item() + eps):.6f}")
    physics_features.append(torch.clamp(ki_over_e_norm, 0, 1).numpy())
    
    # 12. 弹性应变能密度 U = 0.5 σ² / E [J·m⁻³]
    U = 0.5 * stress_squared / (elastic_modulus + eps)
    # 对数缩放后使用Min-Max归一化
    U_log = torch.log1p(torch.clamp(U, min=eps))
    U_log_min = torch.min(U_log)
    U_log_max = torch.max(U_log)
    U_log_range = U_log_max - U_log_min + eps
    U_norm = (U_log - U_log_min) / U_log_range
    print(f"弹性应变能密度原始值范围: {torch.min(U).item():.6e} - {torch.max(U).item():.6e}, 变异系数: {torch.std(U).item() / (torch.mean(U).item() + eps):.6f}")
    physics_features.append(U_norm.numpy())
    
    # 将所有特征拼接起来
    physics_array = np.hstack(physics_features)
    
    # 创建包含所有特征的DataFrame
    features_df = pd.DataFrame(
        np.hstack([data_dict['numeric_raw'], physics_array]),
        columns=orig_feature_names + physics_feature_names
    )
    
    # 添加材料种类和目标变量
    features_df['材料种类'] = data_dict['label_encoder'].inverse_transform(data_dict['material_ids'])
    features_df['亚表面裂纹深度'] = data_dict['target']
    
    return features_df

def basic_statistics_analysis(df, features_df):
    """基本统计分析"""
    print("\n=== 基本统计描述 ===")
    print(df.describe())
    
    # 保存详细统计信息到CSV
    stats_df = features_df.describe().T
    
    # 只对数值列计算统计数据
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_df = features_df[numeric_cols]
    
    stats_df['变异系数'] = stats_df['std'] / stats_df['mean']
    stats_df['中位数-均值比'] = numeric_df.median() / numeric_df.mean()
    stats_df.to_csv('analysis_results/特征统计描述.csv', encoding='utf-8-sig')
    print("详细统计信息已保存到 'analysis_results/特征统计描述.csv'")
    
    # 各材料种类的样本数量
    material_counts = df['材料种类'].value_counts()
    plt.figure(figsize=(12, 6))
    material_counts.plot(kind='bar')
    plt.title('各材料种类样本数量')
    plt.xlabel('材料种类')
    plt.ylabel('样本数量')
    plt.tight_layout()
    plt.savefig('analysis_results/材料种类分布.png', dpi=300)
    
    # 目标变量分布
    plt.figure(figsize=(10, 6))
    sns.histplot(df['亚表面裂纹深度'], kde=True)
    plt.title('亚表面裂纹深度分布')
    plt.xlabel('亚表面裂纹深度')
    plt.ylabel('频率')
    plt.tight_layout()
    plt.savefig('analysis_results/目标变量分布.png', dpi=300)
    
def correlation_analysis(features_df):
    """相关性分析"""
    # 计算所有特征与目标变量的相关系数
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
    
    # 计算Pearson相关系数
    pearson_corrs = []
    spearman_corrs = []
    mi_scores = []
    
    target = features_df['亚表面裂纹深度']
    for col in numeric_cols:
        if col != '亚表面裂纹深度':
            try:
                # 检查是否为常数列
                if features_df[col].nunique() <= 1:
                    print(f"警告: '{col}'是常数特征，跳过相关系数计算")
                    p_corr = s_corr = 0.0
                    mi = 0.0
                else:
                    p_corr, _ = pearsonr(features_df[col], target)
                    s_corr, _ = spearmanr(features_df[col], target)
                    
                    # 计算互信息分数
                    mi = mutual_info_regression(
                        features_df[[col]], target, random_state=42
                    )[0]
                
                pearson_corrs.append((col, p_corr))
                spearman_corrs.append((col, s_corr))
                mi_scores.append((col, mi))
            except Exception as e:
                print(f"计算'{col}'相关性时出错: {e}")
                pearson_corrs.append((col, 0.0))
                spearman_corrs.append((col, 0.0))
                mi_scores.append((col, 0.0))
    
    # 按绝对值排序
    pearson_corrs.sort(key=lambda x: abs(x[1]), reverse=True)
    spearman_corrs.sort(key=lambda x: abs(x[1]), reverse=True)
    mi_scores.sort(key=lambda x: x[1], reverse=True)
    
    # 创建相关性DataFrame
    corr_df = pd.DataFrame({
        '特征': [x[0] for x in pearson_corrs],
        'Pearson相关系数': [x[1] for x in pearson_corrs],
        'Spearman等级相关': [spearman_corrs[i][1] for i in range(len(spearman_corrs))],
        '互信息分数': [mi_scores[i][1] for i in range(len(mi_scores))]
    })
    
    # 保存相关性结果
    corr_df.to_csv('analysis_results/特征相关性分析.csv', index=False, encoding='utf-8-sig')
    print("\n特征与目标变量的相关性分析结果已保存到 'analysis_results/特征相关性分析.csv'")
    
    # 过滤掉常数列，以避免热力图中的警告
    numeric_cols_filtered = [col for col in numeric_cols if features_df[col].nunique() > 1]
    
    # 绘制热力图
    plt.figure(figsize=(16, 12))
    if len(numeric_cols_filtered) > 1:  # 确保至少有两个非常数列
        try:
            corr_matrix = features_df[numeric_cols_filtered].corr()
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            cmap = sns.diverging_palette(230, 20, as_cmap=True)
            
            sns.heatmap(
                corr_matrix, 
                mask=mask, 
                cmap=cmap, 
                vmax=1, 
                vmin=-1, 
                center=0,
                annot=True, 
                fmt='.2f', 
                square=True, 
                linewidths=.5
            )
        except Exception as e:
            print(f"生成热力图时出错: {e}")
            plt.text(0.5, 0.5, f"热力图生成失败: {e}", 
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=16)
    else:
        print("警告: 没有足够的非常数特征来生成热力图")
        plt.text(0.5, 0.5, "没有足够的有效特征生成热力图", 
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=20)
    
    plt.title('特征相关性热力图')
    plt.tight_layout()
    plt.savefig('analysis_results/特征相关性热力图.png', dpi=300)
    
    # 可视化与目标变量最相关的特征
    if not corr_df.empty:
        top_n = min(6, len(corr_df))
        if top_n > 0:
            top_features = corr_df['特征'][:top_n].tolist()
            
            # 计算所需的行列数
            num_cols = min(3, top_n)
            num_rows = (top_n + num_cols - 1) // num_cols  # 向上取整
            
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(16, 5*num_rows))
            if top_n > 1:  # 如果有多个子图，flatten才有效
                axes = axes.flatten()
            else:
                axes = [axes]  # 将单个axes包装成列表以便统一处理
            
            for i, feature in enumerate(top_features):
                try:
                    ax = axes[i]
                    
                    # 检查是否为常数列
                    if features_df[feature].nunique() <= 1:
                        ax.text(0.5, 0.5, f"{feature} 是常数列，无法绘制散点图", 
                                horizontalalignment='center',
                                verticalalignment='center')
                        continue
                        
                    sns.scatterplot(
                        x=features_df[feature], 
                        y=features_df['亚表面裂纹深度'],
                        hue=features_df['材料种类'] if i < 3 else None,  # 只在前3个图中使用材料种类作为颜色
                        alpha=0.7,
                        ax=ax
                    )
                    
                    # 添加趋势线
                    try:
                        sns.regplot(
                            x=features_df[feature], 
                            y=features_df['亚表面裂纹深度'],
                            scatter=False,
                            ax=ax,
                            color='red'
                        )
                    except Exception as e:
                        print(f"为'{feature}'添加趋势线失败: {e}")
                    
                    ax.set_title(f'{feature} vs 亚表面裂纹深度')
                    
                    # 计算并显示相关系数
                    p_corr_rows = corr_df[corr_df['特征'] == feature]['Pearson相关系数']
                    if len(p_corr_rows) > 0:
                        p_corr = p_corr_rows.values[0]
                        ax.annotate(f'r = {p_corr:.3f}', xy=(0.05, 0.95), xycoords='axes fraction')
                    
                except Exception as e:
                    print(f"绘制'{feature}'散点图时出错: {e}")
                    axes[i].text(0.5, 0.5, f"绘图失败: {e}", 
                                horizontalalignment='center',
                                verticalalignment='center')
            
            # 隐藏未使用的子图
            if top_n < num_rows * num_cols and top_n > 1:
                for j in range(top_n, len(axes)):
                    axes[j].set_visible(False)
            
            plt.tight_layout()
            plt.savefig('analysis_results/重要特征相关性散点图.png', dpi=300)
        else:
            print("警告: 没有特征可用于散点图")
    else:
        print("警告: 相关性DataFrame为空，无法绘制散点图")
    
def feature_dimensionality_reduction(features_df):
    """特征降维分析"""
    # 准备数据
    try:
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
        
        # 检查是否存在'亚表面裂纹深度'列
        if '亚表面裂纹深度' in numeric_cols:
            numeric_cols.remove('亚表面裂纹深度')  # 移除目标变量
        
        if len(numeric_cols) < 2:
            print("警告: 数值特征不足以进行降维分析，至少需要2个特征")
            return
            
        # 过滤掉常数列
        non_constant_cols = []
        for col in numeric_cols:
            if features_df[col].nunique() > 1:
                non_constant_cols.append(col)
            else:
                print(f"警告: '{col}'是常数特征，将被排除在降维分析之外")
        
        if len(non_constant_cols) < 2:
            print("警告: 非常数特征不足以进行降维分析，至少需要2个非常数特征")
            return
            
        numeric_cols = non_constant_cols
        
        X = features_df[numeric_cols].values
        y = features_df['亚表面裂纹深度'].values
        materials = features_df['材料种类'].values
        
        # 检查是否有NaN或inf值
        if np.isnan(X).any() or np.isinf(X).any():
            print("警告: 输入数据包含NaN或inf值，将进行替换")
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 标准化
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        
        # PCA降维
        pca = PCA(n_components=min(2, len(numeric_cols)))
        X_pca = pca.fit_transform(X_scaled)
        
        # t-SNE降维 - perplexity必须小于样本数
        perplexity = min(30, max(5, len(X)//5))  # 自适应perplexity
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        X_tsne = tsne.fit_transform(X_scaled)
    except Exception as e:
        print(f"准备降维数据时出错: {e}")
        return
    
    # 创建可视化
    try:
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        
        # PCA可视化 - 按裂纹深度着色
        scatter1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.8, s=60)
        axes[0].set_title(f'PCA投影 (解释方差: {pca.explained_variance_ratio_.sum():.2%})')
        axes[0].set_xlabel(f'主成分1 ({pca.explained_variance_ratio_[0]:.2%})')
        axes[0].set_ylabel(f'主成分2 ({pca.explained_variance_ratio_[1]:.2%})')
        cbar1 = plt.colorbar(scatter1, ax=axes[0])
        cbar1.set_label('亚表面裂纹深度')
        
        # t-SNE可视化 - 按裂纹深度着色
        scatter2 = axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', alpha=0.8, s=60)
        axes[1].set_title('t-SNE投影')
        axes[1].set_xlabel('t-SNE维度1')
        axes[1].set_ylabel('t-SNE维度2')
        cbar2 = plt.colorbar(scatter2, ax=axes[1])
        cbar2.set_label('亚表面裂纹深度')
        
        plt.tight_layout()
        plt.savefig('analysis_results/特征降维可视化_按裂纹深度.png', dpi=300)
        
        # 再创建一个图，按材料种类着色
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        
        # 创建材料种类的颜色映射
        try:
            unique_materials = np.unique(materials)
            material_to_int = {m: i for i, m in enumerate(unique_materials)}
            color_indices = np.array([material_to_int[m] for m in materials])
            
            # PCA可视化 - 按材料种类着色
            scatter1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=color_indices, cmap='tab20', alpha=0.8, s=60)
            axes[0].set_title(f'PCA投影 - 材料种类分布')
            axes[0].set_xlabel(f'主成分1 ({pca.explained_variance_ratio_[0]:.2%})')
            axes[0].set_ylabel(f'主成分2 ({pca.explained_variance_ratio_[1]:.2%})')
            
            # 为每种材料添加标签，但限制数量
            max_labels = min(10, len(unique_materials))
            for i, material in enumerate(unique_materials):
                if i < max_labels:  # 限制显示的材料数量，避免过于拥挤
                    mask = materials == material
                    if np.sum(mask) > 0:  # 确保至少有一个样本
                        centroid_x = X_pca[mask, 0].mean()
                        centroid_y = X_pca[mask, 1].mean()
                        axes[0].text(centroid_x, centroid_y, material, fontsize=9)
        except Exception as e:
            print(f"绘制PCA材料分布图时出错: {e}")
            axes[0].text(0.5, 0.5, "绘图失败", ha='center', va='center')
    except Exception as e:
        print(f"绘制降维可视化图时出错: {e}")
        return
    
    # t-SNE可视化 - 按材料种类着色
    try:
        scatter2 = axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=color_indices, cmap='tab20', alpha=0.8, s=60)
        axes[1].set_title('t-SNE投影 - 材料种类分布')
        axes[1].set_xlabel('t-SNE维度1')
        axes[1].set_ylabel('t-SNE维度2')
        
        # 为每种材料添加标签
        max_labels = min(10, len(unique_materials))
        for i, material in enumerate(unique_materials):
            if i < max_labels:  # 限制显示的材料数量
                mask = materials == material
                if np.sum(mask) > 0:  # 确保至少有一个样本
                    centroid_x = X_tsne[mask, 0].mean()
                    centroid_y = X_tsne[mask, 1].mean()
                    axes[1].text(centroid_x, centroid_y, material, fontsize=9)
                    
        plt.tight_layout()
        plt.savefig('analysis_results/特征降维可视化_按材料种类.png', dpi=300)
        
        # 计算PCA组件与原始特征的相关性
        pca_components = pca.components_
        component_df = pd.DataFrame(pca_components.T, columns=['PC1', 'PC2'], index=numeric_cols)
        component_df = component_df.abs().sort_values(by='PC1', ascending=False)
        
        # 保存PCA组件
        component_df.to_csv('analysis_results/PCA组件特征权重.csv', encoding='utf-8-sig')
        print("\nPCA组件与原始特征的关系已保存到 'analysis_results/PCA组件特征权重.csv'")
    except Exception as e:
        print(f"绘制t-SNE图或保存结果时出错: {e}")
    
    # 可视化主成分与特征的关系
    plt.figure(figsize=(10, 14))
    
    # 绘制PC1的特征权重
    component_df = component_df.sort_values(by='PC1', ascending=True)
    plt.subplot(1, 2, 1)
    bars = plt.barh(component_df.index, component_df['PC1'], color='skyblue')
    plt.title('PC1特征权重')
    plt.xlabel('绝对权重')
    
    # 绘制PC2的特征权重
    component_df = component_df.sort_values(by='PC2', ascending=True)
    plt.subplot(1, 2, 2)
    plt.barh(component_df.index, component_df['PC2'], color='lightgreen')
    plt.title('PC2特征权重')
    plt.xlabel('绝对权重')
    
    plt.tight_layout()
    plt.savefig('analysis_results/PCA组件特征权重可视化.png', dpi=300)

def material_group_analysis(features_df):
    """材料分组分析"""
    # 按材料种类分组计算统计量
    material_stats = features_df.groupby('材料种类')['亚表面裂纹深度'].agg([
        'count', 'mean', 'std', 'min', 'max'
    ]).sort_values('mean', ascending=False)
    
    # 计算95%置信区间
    material_stats['ci_95'] = 1.96 * material_stats['std'] / np.sqrt(material_stats['count'])
    
    # 保存材料统计信息
    material_stats.to_csv('analysis_results/材料分组统计.csv', encoding='utf-8-sig')
    print("\n材料分组统计信息已保存到 'analysis_results/材料分组统计.csv'")
    
    # 可视化不同材料的裂纹深度分布
    plt.figure(figsize=(14, 8))
    
    # 获取前15种材料(按样本数量)
    top_materials = features_df['材料种类'].value_counts().nlargest(15).index.tolist()
    
    # 绘制箱线图
    sns.boxplot(
        x='材料种类', 
        y='亚表面裂纹深度',
        data=features_df[features_df['材料种类'].isin(top_materials)]
    )
    plt.title('不同材料种类的亚表面裂纹深度分布')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('analysis_results/材料种类裂纹深度箱线图.png', dpi=300)
    
    # 按材料种类的特征平均值
    plt.figure(figsize=(16, 12))
    
    # 选择要显示的特征
    selected_features = [
        '莫氏硬度', '弹性模量', '断裂韧性', '晶粒粒径', '压溃压力', '磨粒速度',
        'Griffith能量释放率', '应力强度比', 'Hall-Petch系数', '脆性指数', '弹性应变能密度'
    ]
    
    # 计算每种材料的平均特征值
    material_avg = features_df.groupby('材料种类')[selected_features].mean()
    
    # 对材料按亚表面裂纹深度的平均值排序
    material_crack_avg = features_df.groupby('材料种类')['亚表面裂纹深度'].mean().sort_values(ascending=False)
    material_avg = material_avg.loc[material_crack_avg.index[:10]]  # 取前10种材料
    
    # 绘制热力图
    sns.heatmap(
        material_avg, 
        cmap='YlGnBu', 
        annot=True, 
        fmt='.2f', 
        linewidths=.5,
        cbar_kws={'label': '平均值'}
    )
    plt.title('材料种类的平均特征值(前10种材料)')
    plt.tight_layout()
    plt.savefig('analysis_results/材料特征平均值热力图.png', dpi=300)

def physics_features_analysis(features_df):
    """物理特征分析"""
    # 原始特征与生成的物理特征比较
    orig_features = ['莫氏硬度', '弹性模量', '断裂韧性', '晶粒粒径', '压溃压力', '磨粒速度']
    physics_features = [col for col in features_df.columns if col not in orig_features + ['材料种类', '亚表面裂纹深度']]
    
    # 计算每个特征与目标的相关性
    target = features_df['亚表面裂纹深度']
    
    orig_corrs = []
    for feat in orig_features:
        try:
            # 检查是否为常数数组
            if features_df[feat].nunique() <= 1:
                print(f"警告: '{feat}'是常数特征，跳过相关系数计算")
                orig_corrs.append((feat, 0.0))  # 用0表示无相关性
            else:
                corr, _ = pearsonr(features_df[feat], target)
                orig_corrs.append((feat, corr))
        except Exception as e:
            print(f"计算'{feat}'相关性时出错: {e}")
            orig_corrs.append((feat, 0.0))
    
    physics_corrs = []
    for feat in physics_features:
        try:
            # 检查是否为常数数组
            if features_df[feat].nunique() <= 1:
                print(f"警告: '{feat}'是常数特征，跳过相关系数计算")
                physics_corrs.append((feat, 0.0))  # 用0表示无相关性
            else:
                corr, _ = pearsonr(features_df[feat], target)
                physics_corrs.append((feat, corr))
        except Exception as e:
            print(f"计算'{feat}'相关性时出错: {e}")
            physics_corrs.append((feat, 0.0))
    
    # 按绝对值排序
    orig_corrs.sort(key=lambda x: abs(x[1]), reverse=True)
    physics_corrs.sort(key=lambda x: abs(x[1]), reverse=True)
    
    # 可视化原始特征和物理特征的相关性比较
    plt.figure(figsize=(14, 10))
    
    # 原始特征相关性
    plt.subplot(2, 1, 1)
    y_pos = np.arange(len(orig_corrs))
    colors = ['green' if c > 0 else 'red' for _, c in orig_corrs]
    plt.barh(y_pos, [abs(c) for _, c in orig_corrs], color=colors, alpha=0.7)
    plt.yticks(y_pos, [f"{name} ({corr:.3f})" for name, corr in orig_corrs])
    plt.title('原始特征与亚表面裂纹深度的相关性')
    plt.xlabel('|相关系数|')
    
    # 物理特征相关性
    plt.subplot(2, 1, 2)
    y_pos = np.arange(len(physics_corrs))
    colors = ['green' if c > 0 else 'red' for _, c in physics_corrs]
    plt.barh(y_pos, [abs(c) for _, c in physics_corrs], color=colors, alpha=0.7)
    plt.yticks(y_pos, [f"{name} ({corr:.3f})" for name, corr in physics_corrs])
    plt.title('物理生成特征与亚表面裂纹深度的相关性')
    plt.xlabel('|相关系数|')
    
    plt.tight_layout()
    plt.savefig('analysis_results/原始vs物理特征相关性比较.png', dpi=300)
    
    # 创建物理特征对比表 - 确保两个列表长度一致
    min_len = min(len(orig_corrs), len(physics_corrs))
    
    # 如果物理特征比原始特征多，填充原始特征
    if len(orig_corrs) < len(physics_corrs):
        for i in range(len(orig_corrs), len(physics_corrs)):
            orig_corrs.append(("", 0.0))
    # 如果原始特征比物理特征多，填充物理特征
    elif len(physics_corrs) < len(orig_corrs):
        for i in range(len(physics_corrs), len(orig_corrs)):
            physics_corrs.append(("", 0.0))
    
    corr_comparison = pd.DataFrame({
        '原始特征': [name for name, _ in orig_corrs],
        '原始特征相关系数': [corr for _, corr in orig_corrs],
        '物理特征': [name for name, _ in physics_corrs],
        '物理特征相关系数': [corr for _, corr in physics_corrs]
    })
    
    corr_comparison.to_csv('analysis_results/特征相关性对比表.csv', index=False, encoding='utf-8-sig')
    print("\n原始特征与物理特征的相关性对比已保存到 'analysis_results/特征相关性对比表.csv'")
    
    # 物理特征与原始特征的关系可视化
    plt.figure(figsize=(18, 15))
    
    # 选择物理特征和相应的原始特征组合
    feature_pairs = [
        ('脆性指数', '莫氏硬度'),
        ('脆性指数', '断裂韧性'),
        ('功率-强度比', '磨粒速度'),
        ('Griffith能量释放率', '压溃压力'),
        ('Griffith能量释放率', '弹性模量'),
        ('应力强度比', '断裂韧性'),
        ('Hall-Petch系数', '晶粒粒径'),
        ('塑性区尺寸比', '晶粒粒径')
    ]
    
    for i, (phys_feat, orig_feat) in enumerate(feature_pairs):
        if i >= 8:  # 只显示8个图
            break
            
        plt.subplot(3, 3, i+1)
        
        # 散点图
        scatter = plt.scatter(
            features_df[orig_feat], 
            features_df[phys_feat],
            c=features_df['亚表面裂纹深度'],
            cmap='viridis', 
            alpha=0.7,
            s=50
        )
        
        # 回归线
        m, b = np.polyfit(features_df[orig_feat], features_df[phys_feat], 1)
        plt.plot(features_df[orig_feat], m*features_df[orig_feat] + b, color='red', linewidth=2)
        
        # 计算相关系数
        corr, _ = pearsonr(features_df[orig_feat], features_df[phys_feat])
        
        plt.title(f'{phys_feat} vs {orig_feat}\nr = {corr:.3f}')
        plt.xlabel(orig_feat)
        plt.ylabel(phys_feat)
        
    plt.colorbar(scatter, ax=plt.gca(), label='亚表面裂纹深度')
    plt.tight_layout()
    plt.savefig('analysis_results/物理特征与原始特征关系.png', dpi=300)

def multidimensional_analysis(features_df):
    """多维特征分析"""
    try:
        # 获取实际存在的特征列
        available_features = features_df.columns.tolist()
        
        # 排除非特征列
        non_feature_cols = ['材料种类', '亚表面裂纹深度']
        for col in non_feature_cols:
            if col in available_features:
                available_features.remove(col)
        
        # 选择最重要的原始特征和物理特征
        orig_features = ['莫氏硬度', '弹性模量', '断裂韧性', '晶粒粒径', '压溃压力', '磨粒速度']
        
        # 从可用的物理特征中选择至多3个（优先选择与目标相关性高的）
        phys_features = [col for col in available_features if col not in orig_features]
        
        # 检查哪些特征实际存在
        selected_features = [feat for feat in orig_features + phys_features[:3] if feat in available_features]
        
        if len(selected_features) < 2:
            print("警告: 没有足够的特征进行多维分析")
            return
            
        print(f"选择以下特征进行多维分析: {selected_features}")
        
        # 创建特征对网格
        plt.figure(figsize=(16, 16))
        g = sns.pairplot(
            features_df[selected_features + ['亚表面裂纹深度', '材料种类']],
            hue='材料种类',
            diag_kind='kde',
            plot_kws={'alpha': 0.6, 's': 30},
            height=2.5,
            corner=True,  # 只显示下三角矩阵
            diag_kws={'alpha': 0.5}
        )
        
        g.fig.suptitle('多维特征关系图', y=1.02, fontsize=16)
        plt.tight_layout()
        plt.savefig('analysis_results/多维特征关系图.png', dpi=300)
        
        # 3D可视化
        from mpl_toolkits.mplot3d import Axes3D
        
        # 选择三个特征用于3D可视化
        # 优先使用原始特征，因为物理特征中可能有常数特征
        vis_features = []
        for feat in selected_features:
            if features_df[feat].nunique() > 1:  # 确保不是常数特征
                vis_features.append(feat)
                if len(vis_features) >= 3:
                    break
        
        if len(vis_features) < 3:
            print("警告: 没有足够的非常数特征用于3D可视化")
            return
        
        # 选择三个特征
        x_feature = vis_features[0]
        y_feature = vis_features[1]
        z_feature = vis_features[2]
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(
            features_df[x_feature],
            features_df[y_feature],
            features_df[z_feature],
            c=features_df['亚表面裂纹深度'],
            cmap='viridis',
            s=50,
            alpha=0.7
        )
    except Exception as e:
        print(f"多维特征分析出错: {e}")
        return
    
        ax.set_xlabel(x_feature)
        ax.set_ylabel(y_feature)
        ax.set_zlabel(z_feature)
        ax.set_title(f'三维特征空间可视化: {x_feature}, {y_feature}, {z_feature}')
        
        cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
        cbar.set_label('亚表面裂纹深度')
        
        plt.tight_layout()
        plt.savefig('analysis_results/三维特征空间可视化.png', dpi=300)
        print(f"\n完成三维特征空间可视化，使用特征: {x_feature}, {y_feature}, {z_feature}")

def create_output_dir():
    """创建输出目录"""
    import os
    
    # 创建分析结果目录
    output_dir = 'analysis_results'
    if not os.path.exists(output_dir):
        print(f"创建分析结果目录: {output_dir}")
        os.makedirs(output_dir)
    else:
        print(f"分析结果将保存到: {output_dir}")

def main():
    """主函数"""
    print("===== 材料性能数据分析 =====")
    
    # 0. 创建输出目录
    create_output_dir()
    
    # 1. 加载和预处理数据
    df, data_dict = load_and_preprocess_data()
    
    # 2. 生成物理特征
    features_df = generate_physics_features(data_dict)
    
    # 3. 基本统计分析
    print("\n执行基本统计分析...")
    basic_statistics_analysis(df, features_df)
    
    # 4. 相关性分析
    print("\n执行相关性分析...")
    correlation_analysis(features_df)
    
    # 5. 特征降维分析
    print("\n执行特征降维分析...")
    feature_dimensionality_reduction(features_df)
    
    # 6. 材料分组分析
    print("\n执行材料分组分析...")
    material_group_analysis(features_df)
    
    # 7. 物理特征分析
    print("\n执行物理特征分析...")
    physics_features_analysis(features_df)
    
    # 8. 多维特征分析
    print("\n执行多维特征分析...")
    multidimensional_analysis(features_df)
    
    print("\n分析完成！结果已保存到 'analysis_results' 文件夹")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print(f"分析过程中出错: {e}")
        traceback.print_exc()
        print("尝试继续执行可能会成功的部分分析...")
