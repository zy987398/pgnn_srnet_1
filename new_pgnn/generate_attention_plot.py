"""
生成物理特征注意力权重图
单独文件，用于生成论文所需的注意力权重柱状图
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

# 导入自定义模块
from models import MaterialCrackPGNN
from utils import prepare_material_data


def get_feature_symbols():
    """获取特征符号（用于图表显示）- 与论文表格对应"""
    return [
        # 原始特征 (6个) - 对应论文中的输入特征
        r"$H$",              # Mohs hardness
        r"$E$",              # Elastic modulus
        r"$K_c$",            # Fracture toughness
        r"$d$",              # Abrasive grain size
        r"$\sigma$",         # Grinding pressure
        r"$v$",              # Grinding velocity
        # 物理特征 (12个) - 按表格顺序
        r"$G$",              # Griffith Energy
        r"$K_r$",            # Stress Intensity Ratio
        r"$H_p$",            # Hall-Petch Term
        r"$\chi$",           # Crack Shielding Factor
        r"$\rho_1$",         # Load-Strength Ratio
        r"$\rho_2$",         # Power-Strength Ratio
        r"$d_c$",            # Critical BDT Depth
        r"$R_a$",            # Critical Crack Ratio
        r"$R_p$",            # Plastic Zone Ratio
        r"$B$",              # Brittleness Index
        r"$R_k$",            # Toughness-Modulus Ratio
        r"$U$"               # Strain Energy Density
    ]


def extract_attention_weights(model, data_loader, device):
    """从模型中提取注意力权重"""
    model.eval()
    all_var_weights = []
    
    with torch.no_grad():
        for batch in data_loader:
            material_ids, numeric_features, numeric_raw, _ = batch
            material_ids = material_ids.to(device)
            numeric_raw = numeric_raw.to(device)
            
            # 前向传播以触发符号层
            _ = model(material_ids, numeric_raw)
            
            # 提取变量权重
            interp = model.get_symbolic_interpretation()
            if 'variable_weights' in interp and interp['variable_weights'] is not None:
                var_weights = interp['variable_weights'].cpu()
                all_var_weights.append(var_weights)
    
    if len(all_var_weights) == 0:
        print("警告: 无法提取变量权重")
        return None
    
    # 合并所有批次的权重
    all_var_weights = torch.cat(all_var_weights, dim=0).numpy()
    return all_var_weights


def plot_attention_weights(weights, save_path, top_n=15):
    """
    绘制注意力权重柱状图
    
    参数:
        weights: 注意力权重数组 [N, num_features]
        save_path: 保存路径
        top_n: 显示前N个特征
    """
    # 确保保存目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 设置 Times New Roman 字体
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['mathtext.fontset'] = 'stix'  # 数学符号使用STIX字体
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # 计算平均注意力权重和标准差
    mean_weights = weights.mean(axis=0)
    std_weights = weights.std(axis=0)
    
    # 获取排序后的索引（降序）
    indices = np.argsort(mean_weights)[::-1][:top_n]
    
    # 获取排序后的数据
    sorted_means = mean_weights[indices]
    sorted_stds = std_weights[indices]
    
    # 获取特征符号
    feature_symbols = get_feature_symbols()
    sorted_labels = [feature_symbols[i] for i in indices]
    
    # 绘制垂直条形图
    x_positions = np.arange(len(indices))
    bars = ax.bar(x_positions, sorted_means,
                  color='steelblue', edgecolor='black', alpha=0.8)
    
    # 绘制不确定度范围（只用误差棒线段，不填充）
    ax.errorbar(x_positions, sorted_means, yerr=sorted_stds,
                fmt='none',  # 不绘制中心点
                ecolor='#2C3E50',  # 深灰色
                elinewidth=2,
                capsize=5,
                capthick=2,
                alpha=0.8,
                zorder=4)
    
    # 设置纵坐标为对数刻度
    ax.set_yscale('log')
    
    # 设置坐标轴（英文标签）- 横坐标标签水平显示
    ax.set_xticks(x_positions)
    ax.set_xticklabels(sorted_labels, rotation=0, ha='center', fontsize=20)
    ax.set_xlabel('Physical Features', fontsize=20)
    ax.set_ylabel('Mean Attention Weight (log scale)', fontsize=20)
    
    # 添加网格线
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    # 设置刻度标签字体大小
    for label in ax.get_yticklabels():
        label.set_fontsize(12)
    
    # 在条形上方添加数值标注（调整位置避免与误差棒重叠）
    for i, (bar, mean_val) in enumerate(zip(bars, sorted_means)):
        # 对数刻度下，位置需要用乘法调整
        text_y = sorted_means[i] + sorted_stds[i]
        text_y = text_y * 1.15  # 在误差棒上方留出空间
        ax.text(bar.get_x() + bar.get_width()/2, 
                text_y,
                f'{mean_val:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 恢复默认字体设置
    plt.rcParams['font.family'] = 'sans-serif'
    
    print(f"注意力权重图已保存: {save_path}")
    
    # 打印排名信息
    print("\n物理特征重要性排名:")
    print("-" * 50)
    for rank, idx in enumerate(indices, 1):
        feature_name = feature_symbols[idx]
        importance = mean_weights[idx]
        std = std_weights[idx]
        print(f"{rank:2d}. {feature_name:15s}: {importance:.4f} ± {std:.4f}")


def main():
    """主函数"""
    print("="*70)
    print("生成物理特征注意力权重图")
    print("="*70)
    
    # 配置
    DATA_PATH = "3011916164精确材料性能数据.csv"
    MODEL_PATH = "results/run_20251229_210602/best_model.pth"  # 修改为你的模型路径
    SAVE_DIR = "results/attention"
    SEED = 66
    
    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 检查模型文件是否存在
    if not os.path.exists(MODEL_PATH):
        print(f"错误: 模型文件不存在: {MODEL_PATH}")
        print("请修改 MODEL_PATH 为正确的模型路径")
        return
    
    # 加载数据
    print(f"\n加载数据: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"数据集大小: {len(df)} 样本")
    
    # 分割数据
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=SEED, stratify=df["材料种类"]
    )
    
    # 准备数据
    train_data = prepare_material_data(train_df)
    test_data = prepare_material_data(
        test_df,
        scaler=train_data["scaler"],
        le=train_data["label_encoder"],
    )
    
    num_materials = train_data["num_materials"]
    print(f"材料种类数: {num_materials}")
    
    # 创建模型
    print(f"\n加载模型: {MODEL_PATH}")
    model = MaterialCrackPGNN(
        num_materials=num_materials,
        material_embed_dim=16,
        latent_dim=32,
        dropout_rate=0.2,
        symbolic_type="advanced"
    )
    
    # 加载模型权重
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()
    print("模型加载成功")
    
    # 创建数据加载器
    test_dataset = TensorDataset(
        torch.LongTensor(test_data["material_ids"]),
        torch.FloatTensor(test_data["numeric_features"]),
        torch.FloatTensor(test_data["numeric_raw"]),
        torch.FloatTensor(test_data["target"]).unsqueeze(1),
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 提取注意力权重
    print("\n提取注意力权重...")
    weights = extract_attention_weights(model, test_loader, device)
    
    if weights is None:
        print("无法提取注意力权重，程序退出")
        return
    
    print(f"提取到 {len(weights)} 个样本的注意力权重")
    
    # 确保保存目录存在
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # 生成并保存图片
    save_path = os.path.join(SAVE_DIR, "attention_weights.png")
    plot_attention_weights(weights, save_path, top_n=15)
    
    # 同时保存高分辨率PDF版本（适合论文）
    save_path_pdf = os.path.join(SAVE_DIR, "attention_weights.pdf")
    
    # 重新绘制PDF版本
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['mathtext.fontset'] = 'stix'
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    mean_weights = weights.mean(axis=0)
    std_weights = weights.std(axis=0)
    indices = np.argsort(mean_weights)[::-1][:15]
    sorted_means = mean_weights[indices]
    sorted_stds = std_weights[indices]
    
    feature_symbols = get_feature_symbols()
    sorted_labels = [feature_symbols[i] for i in indices]
    
    x_positions = np.arange(len(indices))
    
    # 绘制柱状图
    bars = ax.bar(x_positions, sorted_means,
                  color='steelblue', edgecolor='black', alpha=0.8)
    
    # 绘制不确定度范围（只用误差棒线段）
    ax.errorbar(x_positions, sorted_means, yerr=sorted_stds,
                fmt='none',
                ecolor='#2C3E50',
                elinewidth=2,
                capsize=5,
                capthick=2,
                alpha=0.8,
                zorder=4)
    
    # 设置纵坐标为对数刻度
    ax.set_yscale('log')
    
    # 横坐标标签水平显示
    ax.set_xticks(x_positions)
    ax.set_xticklabels(sorted_labels, rotation=0, ha='center', fontsize=14)
    ax.set_xlabel('Physical Features', fontsize=16, fontweight='bold')
    ax.set_ylabel('Mean Attention Weight (log scale)', fontsize=16, fontweight='bold')
    
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    for label in ax.get_yticklabels():
        label.set_fontsize(12)
    
    for i, (bar, mean_val) in enumerate(zip(bars, sorted_means)):
        text_y = sorted_means[i] + sorted_stds[i]
        text_y = text_y * 1.15
        ax.text(bar.get_x() + bar.get_width()/2, 
                text_y,
                f'{mean_val:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path_pdf, dpi=300, bbox_inches='tight', format='pdf')
    plt.close()
    
    # 恢复默认字体设置
    plt.rcParams['font.family'] = 'sans-serif'
    
    print(f"PDF版本已保存: {save_path_pdf}")
    
    print("\n" + "="*70)
    print("完成！")
    print(f"图片保存位置: {SAVE_DIR}")
    print("="*70)


if __name__ == "__main__":
    main()
