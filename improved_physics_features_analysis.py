"""
改进材料物理特征分析
对improved_material_crack_pgnn.py中的物理特征进行可视化分析和评估
"""
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import pearsonr, spearmanr
import warnings
from typing import Dict, List, Tuple, Optional

# 导入改进的PGNN模型和数据处理函数
from improved_material_crack_pgnn import PhysicsInformedFeatureExtractor
from old_pgnn.utils import prepare_material_data

# 忽略警告
warnings.filterwarnings('ignore')

# 设置更好的可视化样式
plt.style.use('seaborn-v0_8-whitegrid')
sns.set(font_scale=1.2)
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.family'] = 'SimHei'  # 支持中文显示
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 创建结果目录
if not os.path.exists('physics_analysis_results'):
    os.makedirs('physics_analysis_results')

def load_data(file_path='3011916164精确材料性能数据.csv'):
    """加载原始数据"""
    print(f"正在加载数据: {file_path}")
    df = pd.read_csv(file_path)
    
    # 显示基本信息
    print("\n数据基本信息:")
    print(f"样本数量: {len(df)}")
    print(f"特征数量: {len(df.columns)}")
    print(f"材料种类数量: {df['材料种类'].nunique()}")
    
    return df

def extract_physics_features(df):
    """使用PhysicsInformedFeatureExtractor提取物理特征"""
    # 1. 准备数据
    data_dict = prepare_material_data(df)
    
    # 2. 转换为PyTorch张量
    numeric_raw = torch.tensor(data_dict['numeric_raw'], dtype=torch.float32)
    
    # 3. 初始化特征提取器
    feature_extractor = PhysicsInformedFeatureExtractor()
    
    # 4. 提取物理特征
    with torch.no_grad():
        physics_features = feature_extractor(numeric_raw)
    
    # 5. 转换回NumPy数组
    physics_np = physics_features.numpy()
    
    # 6. 原始特征的列名
    orig_feature_names = ['莫氏硬度', '弹性模量', '断裂韧性', '晶粒粒径', '压溃压力', '磨粒速度']
    
    # 检查物理特征的实际维度
    num_physics_features = physics_np.shape[1]
    print(f"\n物理特征维度: {num_physics_features}")
    
    # 构建物理特征名称
    # 基于阅读PhysicsInformedFeatureExtractor的实现，我们知道它会生成15个特征
    # 第一个是原始特征集，然后是9个基于物理原理的特征
    physics_feature_names = []
    
    # 根据PhysicsInformedFeatureExtractor类的forward方法，准备一个完整的特征名称列表
    complete_physics_names = [
        '原始特征集',         # features.append(x)
        'Griffith能量释放率',  # energy_release_rate
        '应力强度比',         # k_ratio
        'Hertz接触应力',      # contact_stress
        'Hall-Petch系数',     # hall_petch
        '材料脆性指数',       # brittleness
        '能量输入',           # energy_input
        '韧性模量比',         # toughness_modulus_ratio
        '硬度弹性模量比',     # he_ratio
        '裂纹扩展趋势',       # fracture_tendency
        '压力-硬度比',         # 额外的物理特征
        '速度-韧性比',
        '屈服强度指数',
        '塑性变形指数',
        '材料断裂趋势'
    ]
    
    # 确保名称列表与特征数量匹配
    if num_physics_features <= len(complete_physics_names):
        # 如果物理特征数量不超过预定义的名称列表，使用前num_physics_features个名称
        physics_feature_names = complete_physics_names[:num_physics_features]
    else:
        # 如果物理特征数量超过预定义的名称，使用所有预定义名称，然后添加通用名称
        physics_feature_names = complete_physics_names.copy()
        for i in range(num_physics_features - len(complete_physics_names)):
            physics_feature_names.append(f'物理特征{len(complete_physics_names) + i + 1}')
                
    print(f"使用的物理特征名称: {physics_feature_names}")
    
    # 7. 创建包含所有特征的DataFrame
    features_df = pd.DataFrame(
        np.hstack([data_dict['numeric_raw'], physics_np]),
        columns=orig_feature_names + physics_feature_names
    )
    
    # 8. 添加材料种类和目标变量
    features_df['材料种类'] = data_dict['label_encoder'].inverse_transform(data_dict['material_ids'])
    features_df['亚表面裂纹深度'] = data_dict['target']
    
    print("\n物理特征提取完成:")
    print(f"生成的特征数量: {len(physics_feature_names)}")
    print(f"总特征数量: {len(orig_feature_names) + len(physics_feature_names)}")
    
    return features_df, data_dict

def analyze_feature_statistics(features_df):
    """分析特征统计数据"""
    print("\n=== 物理特征基本统计分析 ===")
    
    # 1. 计算基本统计量
    stats_df = features_df.describe().T
    stats_df['变异系数'] = stats_df['std'] / stats_df['mean']
    
    # 过滤出原始特征和物理特征
    orig_features = ['莫氏硬度', '弹性模量', '断裂韧性', '晶粒粒径', '压溃压力', '磨粒速度']
    physics_features = [col for col in features_df.columns if col not in orig_features 
                       and col not in ['材料种类', '亚表面裂纹深度']]
    
    # 2. 保存统计结果
    stats_df.to_csv('physics_analysis_results/物理特征统计描述.csv', encoding='utf-8-sig')
    print("详细统计信息已保存到 'physics_analysis_results/物理特征统计描述.csv'")
    
    # 3. 可视化特征分布
    plt.figure(figsize=(16, 10))
    
    # 每行展示3个特征
    num_features = len(physics_features)
    num_rows = (num_features + 2) // 3
    
    for i, feature in enumerate(physics_features):
        plt.subplot(num_rows, 3, i + 1)
        
        # KDE图
        try:
            sns.kdeplot(features_df[feature], fill=True, color='skyblue')
            plt.axvline(features_df[feature].mean(), color='r', linestyle='--', label=f'均值: {features_df[feature].mean():.3f}')
            plt.axvline(features_df[feature].median(), color='g', linestyle=':', label=f'中位数: {features_df[feature].median():.3f}')
            plt.title(f'{feature}分布')
            plt.legend()
        except Exception as e:
            print(f"绘制'{feature}'分布时出错: {e}")
    
    plt.tight_layout()
    plt.savefig('physics_analysis_results/物理特征分布.png', dpi=300)
    
    # 4. 绘制物理特征热力图
    plt.figure(figsize=(14, 12))
    
    # 只使用物理特征部分计算相关性矩阵
    physics_corr = features_df[physics_features].corr()
    
    # 绘制热力图
    mask = np.triu(np.ones_like(physics_corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    sns.heatmap(
        physics_corr, 
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
    
    plt.title('物理特征相关性热力图')
    plt.tight_layout()
    plt.savefig('physics_analysis_results/物理特征相关性热力图.png', dpi=300)
    
    return physics_features

def analyze_feature_importance(features_df, physics_features):
    """分析物理特征与目标变量的关系和重要性"""
    print("\n=== 物理特征重要性分析 ===")
    
    # 准备原始特征和物理特征列表
    orig_features = ['莫氏硬度', '弹性模量', '断裂韧性', '晶粒粒径', '压溃压力', '磨粒速度']
    
    # 1. 计算所有特征与目标变量的相关系数
    target = features_df['亚表面裂纹深度']
    
    # 存储相关性指标
    feature_importance = {
        '特征': [],
        'Pearson相关系数': [],
        'Spearman等级相关': [],
        '互信息分数': [],
        '特征类型': []
    }
    
    # 计算原始特征的相关性
    for feature in orig_features:
        try:
            p_corr, _ = pearsonr(features_df[feature], target)
            s_corr, _ = spearmanr(features_df[feature], target)
            mi = mutual_info_regression(features_df[[feature]], target, random_state=42)[0]
            
            feature_importance['特征'].append(feature)
            feature_importance['Pearson相关系数'].append(p_corr)
            feature_importance['Spearman等级相关'].append(s_corr)
            feature_importance['互信息分数'].append(mi)
            feature_importance['特征类型'].append('原始特征')
        except Exception as e:
            print(f"计算'{feature}'相关性时出错: {e}")
    
    # 计算物理特征的相关性
    for feature in physics_features:
        try:
            p_corr, _ = pearsonr(features_df[feature], target)
            s_corr, _ = spearmanr(features_df[feature], target)
            mi = mutual_info_regression(features_df[[feature]], target, random_state=42)[0]
            
            feature_importance['特征'].append(feature)
            feature_importance['Pearson相关系数'].append(p_corr)
            feature_importance['Spearman等级相关'].append(s_corr)
            feature_importance['互信息分数'].append(mi)
            feature_importance['特征类型'].append('物理特征')
        except Exception as e:
            print(f"计算'{feature}'相关性时出错: {e}")
    
    # 创建DataFrame并按相关性排序
    importance_df = pd.DataFrame(feature_importance)
    importance_df = importance_df.sort_values(by='Pearson相关系数', key=lambda x: abs(x), ascending=False)
    
    # 保存相关性分析结果
    importance_df.to_csv('physics_analysis_results/特征重要性分析.csv', index=False, encoding='utf-8-sig')
    print("特征重要性分析结果已保存到 'physics_analysis_results/特征重要性分析.csv'")
    
    # 2. 可视化特征重要性
    plt.figure(figsize=(14, 10))
    
    # 绘制Pearson相关系数条形图
    importance_df['abs_pearson'] = importance_df['Pearson相关系数'].abs()
    sorted_df = importance_df.sort_values('abs_pearson')
    
    # 根据特征类型设置不同颜色
    colors = ['skyblue' if t == '原始特征' else 'lightgreen' for t in sorted_df['特征类型']]
    
    plt.barh(sorted_df['特征'], sorted_df['Pearson相关系数'], color=colors)
    
    # 添加零线
    plt.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
    
    plt.title('特征与亚表面裂纹深度的Pearson相关系数')
    plt.xlabel('相关系数')
    
    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='skyblue', label='原始特征'),
        Patch(facecolor='lightgreen', label='物理特征')
    ]
    plt.legend(handles=legend_elements)
    
    plt.tight_layout()
    plt.savefig('physics_analysis_results/特征重要性条形图.png', dpi=300)
    
    # 3. 绘制散点图矩阵
    # 选取最重要的3个原始特征和3个物理特征
    top_orig = importance_df[importance_df['特征类型'] == '原始特征'].head(3)['特征'].tolist()
    top_physics = importance_df[importance_df['特征类型'] == '物理特征'].head(3)['特征'].tolist()
    top_features = top_orig + top_physics + ['亚表面裂纹深度']
    
    # 绘制散点图矩阵
    plt.figure(figsize=(16, 14))
    scatter_df = features_df[top_features].copy()
    
    # 添加材料种类信息用于着色
    scatter_df['材料种类'] = features_df['材料种类']
    
    # 创建散点图矩阵
    g = sns.pairplot(
        scatter_df, 
        hue='材料种类', 
        corner=True, 
        diag_kind='kde',
        plot_kws={'alpha': 0.6, 's': 30}
    )
    
    plt.suptitle('重要特征散点图矩阵', y=1.02, fontsize=16)
    plt.tight_layout()
    plt.savefig('physics_analysis_results/重要特征散点图矩阵.png', dpi=300)
    
    return importance_df

def compare_feature_groups(features_df, physics_features, importance_df):
    """比较原始特征和物理特征的性能差异"""
    print("\n=== 原始特征与物理特征对比分析 ===")
    
    # 1. 计算每组特征的平均相关性强度
    orig_features = ['莫氏硬度', '弹性模量', '断裂韧性', '晶粒粒径', '压溃压力', '磨粒速度']
    
    orig_corr = importance_df[importance_df['特征类型'] == '原始特征']['Pearson相关系数'].abs().mean()
    physics_corr = importance_df[importance_df['特征类型'] == '物理特征']['Pearson相关系数'].abs().mean()
    
    # 创建比较表格
    comparison_data = {
        '特征组': ['原始特征', '物理特征'],
        '特征数量': [len(orig_features), len(physics_features)],
        '平均相关性强度': [orig_corr, physics_corr],
        '最大相关性强度': [
            importance_df[importance_df['特征类型'] == '原始特征']['Pearson相关系数'].abs().max(),
            importance_df[importance_df['特征类型'] == '物理特征']['Pearson相关系数'].abs().max()
        ],
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv('physics_analysis_results/特征组对比.csv', index=False, encoding='utf-8-sig')
    print("特征组对比结果已保存到 'physics_analysis_results/特征组对比.csv'")
    
    # 2. 绘制对比条形图
    plt.figure(figsize=(10, 6))
    
    x = np.arange(2)
    width = 0.35
    
    plt.bar(x - width/2, comparison_df['平均相关性强度'], width, label='平均相关性强度')
    plt.bar(x + width/2, comparison_df['最大相关性强度'], width, label='最大相关性强度')
    
    plt.xlabel('特征组')
    plt.ylabel('相关系数（绝对值）')
    plt.title('原始特征与物理特征相关性强度对比')
    plt.xticks(x, comparison_df['特征组'])
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('physics_analysis_results/特征组对比.png', dpi=300)
    
    # 3. 降维可视化对比
    # 准备原始特征和物理特征
    X_orig = features_df[orig_features].values
    X_physics = features_df[physics_features].values
    
    # 标准化
    scaler = RobustScaler()
    X_orig_scaled = scaler.fit_transform(X_orig)
    X_physics_scaled = scaler.fit_transform(X_physics)
    
    # 使用t-SNE进行降维
    tsne = TSNE(n_components=2, random_state=42)
    X_orig_tsne = tsne.fit_transform(X_orig_scaled)
    
    tsne = TSNE(n_components=2, random_state=42)
    X_physics_tsne = tsne.fit_transform(X_physics_scaled)
    
    # 绘制降维结果
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # 目标变量
    target = features_df['亚表面裂纹深度'].values
    
    # t-SNE可视化 - 原始特征
    scatter1 = axes[0].scatter(X_orig_tsne[:, 0], X_orig_tsne[:, 1], c=target, cmap='viridis', alpha=0.8, s=60)
    axes[0].set_title('原始特征 t-SNE投影')
    axes[0].set_xlabel('t-SNE维度1')
    axes[0].set_ylabel('t-SNE维度2')
    plt.colorbar(scatter1, ax=axes[0], label='亚表面裂纹深度')
    
    # t-SNE可视化 - 物理特征
    scatter2 = axes[1].scatter(X_physics_tsne[:, 0], X_physics_tsne[:, 1], c=target, cmap='viridis', alpha=0.8, s=60)
    axes[1].set_title('物理特征 t-SNE投影')
    axes[1].set_xlabel('t-SNE维度1')
    axes[1].set_ylabel('t-SNE维度2')
    plt.colorbar(scatter2, ax=axes[1], label='亚表面裂纹深度')
    
    plt.tight_layout()
    plt.savefig('physics_analysis_results/特征组降维对比.png', dpi=300)

def analyze_by_material_type(features_df, physics_features):
    """按材料种类分析物理特征差异"""
    print("\n=== 不同材料种类的物理特征差异分析 ===")
    
    # 1. 按材料种类分组计算统计量
    material_groups = features_df.groupby('材料种类')
    
    # 计算每个材料的平均物理特征值
    material_physics_means = pd.DataFrame({
        feature: material_groups[feature].mean() 
        for feature in physics_features
    })
    
    # 添加目标变量
    material_physics_means['亚表面裂纹深度'] = material_groups['亚表面裂纹深度'].mean()
    material_physics_means['样本数量'] = material_groups.size()
    
    # 保存结果
    material_physics_means.to_csv('physics_analysis_results/材料种类物理特征平均值.csv', encoding='utf-8-sig')
    print("材料种类物理特征平均值已保存到 'physics_analysis_results/材料种类物理特征平均值.csv'")
    
    # 2. 物理特征与裂纹深度的相关性热力图（按材料种类）
    # 选择样本量较多的前8种材料
    top_materials = material_physics_means.nlargest(8, '样本数量').index.tolist()
    
    # 计算每种材料的相关性
    corr_by_material = {}
    for material in top_materials:
        material_df = features_df[features_df['材料种类'] == material]
        if len(material_df) > 10:  # 确保有足够样本计算相关性
            try:
                corr_matrix = material_df[physics_features + ['亚表面裂纹深度']].corr()
                corr_with_target = corr_matrix.loc[physics_features, '亚表面裂纹深度']
                corr_by_material[material] = corr_with_target
            except Exception as e:
                print(f"计算'{material}'相关性时出错: {e}")
    
    # 创建相关性DataFrame
    if corr_by_material:
        corr_df = pd.DataFrame(corr_by_material)
        
        # 绘制热力图
        plt.figure(figsize=(14, 10))
        sns.heatmap(
            corr_df, 
            cmap='RdBu_r', 
            vmax=1, 
            vmin=-1, 
            center=0,
            annot=True, 
            fmt='.2f'
        )
        
        plt.title('不同材料种类的物理特征与裂纹深度相关性')
        plt.tight_layout()
        plt.savefig('physics_analysis_results/材料种类物理特征相关性.png', dpi=300)
        
        # 保存相关性表
        corr_df.to_csv('physics_analysis_results/材料种类物理特征相关性.csv', encoding='utf-8-sig')
    else:
        print("警告: 没有足够的材料种类样本来计算相关性")
    
    # 3. 材料种类与物理特征的箱线图
    # 为了避免图过于复杂，只选择最重要的3个物理特征
    # 根据与目标变量的相关性选择
    target_corrs = []
    for feature in physics_features:
        try:
            corr, _ = pearsonr(features_df[feature], features_df['亚表面裂纹深度'])
            target_corrs.append((feature, abs(corr)))
        except Exception as e:
            print(f"计算'{feature}'相关性时出错: {e}")
            target_corrs.append((feature, 0.0))
    
    # 排序并选择前3个
    target_corrs.sort(key=lambda x: x[1], reverse=True)
    top_features = [x[0] for x in target_corrs[:3]]
    
    # 绘制箱线图
    fig, axes = plt.subplots(len(top_features), 1, figsize=(14, 5*len(top_features)))
    
    for i, feature in enumerate(top_features):
        try:
            sns.boxplot(
                x='材料种类', 
                y=feature, 
                data=features_df,
                ax=axes[i] if len(top_features) > 1 else axes,
                palette='Set3'
            )
            
            axes[i].set_title(f'{feature}在不同材料种类中的分布') if len(top_features) > 1 else axes.set_title(f'{feature}在不同材料种类中的分布')
            axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45) if len(top_features) > 1 else axes.set_xticklabels(axes.get_xticklabels(), rotation=45)
        except Exception as e:
            print(f"绘制'{feature}'箱线图时出错: {e}")
    
    plt.tight_layout()
    plt.savefig('physics_analysis_results/材料种类物理特征分布.png', dpi=300)

def main():
    """主函数"""
    print("=== 开始改进材料物理特征分析 ===\n")
    
    # 1. 加载数据
    df = load_data()
    
    # 2. 提取物理特征
    features_df, data_dict = extract_physics_features(df)
    
    # 3. 分析特征统计数据
    physics_features = analyze_feature_statistics(features_df)
    
    # 4. 分析特征重要性
    importance_df = analyze_feature_importance(features_df, physics_features)
    
    # 5. 比较原始特征和物理特征
    compare_feature_groups(features_df, physics_features, importance_df)
    
    # 6. 按材料种类分析物理特征
    analyze_by_material_type(features_df, physics_features)
    
    print("\n=== 物理特征分析完成，结果已保存至 'physics_analysis_results' 目录 ===")

if __name__ == "__main__":
    main()
