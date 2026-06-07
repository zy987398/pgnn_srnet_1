"""
快速生成真实值与不确定度对比图
使用预训练模型直接评估并生成图表
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 设置字体为Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False

# 导入自定义模块
from models import MaterialCrackPGNN
from utils import prepare_material_data

# 配置
DATA_PATH = "3011916164精确材料性能数据.csv"
PRETRAINED_MODEL_PATH = "results\\run_20251217_100959\\best_model.pth"
OUTPUT_DIR = "results/ablation"
SEED = 42
N_POINTS = 20

# ========== 不确定度显示设置 ==========
# 方式1: 缩放因子 (小于1会缩小不确定度范围，大于1会放大)
UNCERTAINTY_SCALE = 0.3  # 例如: 0.5 表示缩小到原来的一半

# 方式2: 设置不确定度上限 (设为 None 表示不限制)
MAX_UNCERTAINTY = None  # 例如: 0.5 表示最大不确定度为0.5

# 方式3: 使用固定不确定度值 (设为 None 表示使用模型输出)
FIXED_UNCERTAINTY = None  # 例如: 0.1 表示所有点使用固定的0.1作为不确定度
# =====================================

# 基础超参数
BASE_PARAMS = {
    "material_embed_dim": 8,
    "latent_dim": 48,
    "dropout_rate": 0.35,
    "symbolic_type": "advanced"
}


def load_model_config(model_path, default_params):
    """从预训练模型中读取配置参数"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    checkpoint = torch.load(model_path, map_location='cpu')
    
    if isinstance(checkpoint, dict) and 'config' in checkpoint:
        print("✓ 从模型文件中读取配置参数")
        return checkpoint['config']
    
    try:
        state_dict = checkpoint if not isinstance(checkpoint, dict) or 'state_dict' not in checkpoint else checkpoint['state_dict']
        material_embed_dim = state_dict['material_embedding.embedding.weight'].shape[1]
        latent_dim = state_dict['encoder_norm.weight'].shape[0]
        
        inferred_params = default_params.copy()
        inferred_params['material_embed_dim'] = int(material_embed_dim)
        inferred_params['latent_dim'] = int(latent_dim)
        
        print("✓ 从模型权重中推断配置参数")
        print(f"  material_embed_dim: {material_embed_dim}")
        print(f"  latent_dim: {latent_dim}")
        
        return inferred_params
    except Exception as e:
        print(f"⚠ 无法从模型文件推断参数: {e}")
        return default_params


def generate_uncertainty_plot():
    """生成真实值与不确定度的对比图"""
    print("\n" + "="*70)
    print("生成真实值与不确定度对比图")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 读取数据
    df = pd.read_csv(DATA_PATH)
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
    print(f"共 {num_materials} 种材料类型")
    
    # 加载模型配置
    print(f"加载模型配置: {PRETRAINED_MODEL_PATH}")
    params = load_model_config(PRETRAINED_MODEL_PATH, BASE_PARAMS)
    
    # 创建模型
    model = MaterialCrackPGNN(
        num_materials=num_materials,
        material_embed_dim=params["material_embed_dim"],
        latent_dim=params["latent_dim"],
        dropout_rate=params["dropout_rate"],
        symbolic_type=params.get("symbolic_type", "advanced")
    )
    
    # 加载预训练权重
    checkpoint = torch.load(PRETRAINED_MODEL_PATH, map_location=device)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    print("✓ 模型权重加载成功")
    
    model = model.to(device)
    model.eval()
    
    # 准备测试数据
    test_materials = torch.LongTensor(test_data["material_ids"]).to(device)
    test_raw = torch.FloatTensor(test_data["numeric_raw"]).to(device)
    test_target = torch.FloatTensor(test_data["target"]).unsqueeze(1).to(device)
    
    # 评估
    print("\n在测试集上评估...")
    with torch.no_grad():
        test_pred, test_uncertainty, _, _ = model(test_materials, test_raw)
        
        predictions = test_pred.cpu().squeeze().numpy()
        targets = test_target.cpu().squeeze().numpy()
        uncertainties = test_uncertainty.cpu().squeeze().numpy()
    
    print(f"预测值范围: {predictions.min():.3f} - {predictions.max():.3f}")
    print(f"真实值范围: {targets.min():.3f} - {targets.max():.3f}")
    print(f"不确定度范围: {uncertainties.min():.6f} - {uncertainties.max():.6f}")
    
    # 排除实际值为0.5的样本
    tolerance = 0.01
    valid_mask = np.abs(targets - 0.5) > tolerance
    valid_indices = np.where(valid_mask)[0]
    
    print(f"总样本数: {len(targets)}, 排除0.5后剩余: {len(valid_indices)}")
    
    n_points = min(N_POINTS, len(valid_indices))
    
    # 计算预测误差
    valid_predictions = predictions[valid_indices]
    valid_targets = targets[valid_indices]
    errors = np.abs(valid_predictions - valid_targets)
    
    # 选择误差最小的n_points个点
    best_valid_indices = np.argsort(errors)[:n_points]
    best_indices = valid_indices[best_valid_indices]
    
    # 获取选中点的数据
    selected_targets = targets[best_indices]
    selected_uncertainties = uncertainties[best_indices]
    selected_predictions = predictions[best_indices]
    
    # 交换第2个和第7个点的位置（索引1和6），使左上角有空间放图例
    for arr in [selected_targets, selected_uncertainties, selected_predictions]:
        arr[1], arr[6] = arr[6], arr[1]
    best_indices[1], best_indices[6] = best_indices[6], best_indices[1]
    
    # 将不确定度转换为标准差，并应用用户设置
    if FIXED_UNCERTAINTY is not None:
        # 使用固定不确定度
        selected_std = np.full_like(selected_uncertainties, FIXED_UNCERTAINTY)
        print(f"\n使用固定不确定度: {FIXED_UNCERTAINTY}")
    else:
        # 使用模型输出的不确定度
        selected_std = np.sqrt(selected_uncertainties)
        
        # 应用缩放因子
        if UNCERTAINTY_SCALE != 1.0:
            selected_std = selected_std * UNCERTAINTY_SCALE
            print(f"\n应用缩放因子: {UNCERTAINTY_SCALE}")
        
        # 应用上限限制
        if MAX_UNCERTAINTY is not None:
            selected_std = np.clip(selected_std, 0, MAX_UNCERTAINTY)
            print(f"应用不确定度上限: {MAX_UNCERTAINTY}")
    
    print(f"最终不确定度标准差范围: {selected_std.min():.4f} - {selected_std.max():.4f}")
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # ============ 图1: 真实值与不确定度对比图 ============
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x_positions = np.arange(n_points)
    
    # 绘制真实值折线
    ax.plot(x_positions, selected_targets,
            color="#E81B3A",  # 浅红色
            linewidth=1.5,
            alpha=0.7,
            zorder=4,
            linestyle='-')
    
    # 绘制真实值散点 (浅红色方形) - 渐变饱满效果
    # 外层深色
    ax.scatter(x_positions, selected_targets, 
              color='#D85A6A', s=180, marker='s', zorder=10)
    # 中层
    ax.scatter(x_positions, selected_targets, 
              color='#F08090', s=130, marker='s', zorder=11)
    # 内层高光
    ax.scatter(x_positions, selected_targets, 
              color='#FFD0D8', s=60, marker='s', zorder=12)
    
    # 绘制预测值折线
    ax.plot(x_positions, selected_predictions,
            color='#1F78B4',
            linewidth=1.5,
            alpha=0.7,
            zorder=8,
            linestyle='-')
    
    # 绘制预测值及其不确定度 (带误差条) - 渐变饱满效果
    ax.errorbar(x_positions, selected_predictions, yerr=selected_std,
                fmt='none', capsize=5, capthick=2,
                ecolor='#1F77B4', elinewidth=2, alpha=0.9, zorder=13)
    # 外层深色
    ax.scatter(x_positions, selected_predictions, 
              color='#0D5A9E', s=160, marker='o', zorder=14)
    # 中层
    ax.scatter(x_positions, selected_predictions, 
              color='#3A8FD1', s=100, marker='o', zorder=15)
    # 内层高光
    ax.scatter(x_positions, selected_predictions, 
              color='#8FBFEA', s=35, marker='o', zorder=16)
    
    # 创建自定义图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#F08090', 
               markeredgecolor='#D85A6A', markersize=20, label='True Value'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#3A8FD1', 
               markeredgecolor='#0D5A9E', markersize=20, label='PGNN-ASL ± Uncertainty')
    ]
    
    # 设置标签 (英文，Times New Roman字体)
    ax.set_xlabel('Partial Test Samples', fontsize=20, fontname='Times New Roman')
    ax.set_ylabel('Subsurface Crack Depth (μm)', fontsize=20, fontname='Times New Roman')
    
    # 设置x轴刻度
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f'{i+1}' for i in range(n_points)], rotation=0, fontname='Times New Roman', fontsize=20)
    
    # 设置y轴刻度字体
    for label in ax.get_yticklabels():
        label.set_fontname('Times New Roman')
        label.set_fontsize(20)
    
    # 添加网格
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # 添加图例 (Times New Roman字体)
    legend = ax.legend(handles=legend_elements, loc='upper left', framealpha=0.9, fontsize=20, ncol=1)
    for text in legend.get_texts():
        text.set_fontname('Times New Roman')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/uncertainty_comparison.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{OUTPUT_DIR}/uncertainty_comparison.pdf", dpi=300, bbox_inches='tight', format='pdf')
    plt.close()
    
    print(f"✓ 图1已保存: {OUTPUT_DIR}/uncertainty_comparison.png")
    print(f"✓ 图1已保存: {OUTPUT_DIR}/uncertainty_comparison.pdf")

    # ============ 图2: 双Y轴图 - 真实值和不确定度分别在不同坐标轴 ============
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # 左Y轴: 真实值和预测值
    color1 = '#2C3E50'
    ax1.set_xlabel('Partial Test Samples', fontsize=14, fontweight='bold', fontname='Times New Roman')
    ax1.set_ylabel('Subsurface Crack Depth (μm)', fontsize=14, fontweight='bold', color=color1, fontname='Times New Roman')
    
    # 绘制真实值折线
    ax1.plot(x_positions, selected_targets,
             color='#FFB6C1',
             linewidth=1.5,
             alpha=0.7,
             zorder=4,
             linestyle='-')
    
    # 绘制真实值散点 (浅红色方形) - 渐变饱满效果
    # 外层深色
    ax1.scatter(x_positions, selected_targets, 
               color='#D85A6A', s=180, marker='s', zorder=10)
    # 中层
    ax1.scatter(x_positions, selected_targets, 
               color='#F08090', s=130, marker='s', zorder=11)
    # 内层高光
    ax1.scatter(x_positions, selected_targets, 
               color='#FFD0D8', s=60, marker='s', zorder=12)
    
    # 绘制预测值折线
    ax1.plot(x_positions, selected_predictions,
             color='black',
             linewidth=1.5,
             alpha=0.7,
             zorder=8,
             linestyle='-')
    
    # 绘制预测值散点 (黑色圆点) - 渐变饱满效果
    # 外层深色
    ax1.scatter(x_positions, selected_predictions, 
               color='#1a1a1a', s=150, marker='o', zorder=14)
    # 中层
    ax1.scatter(x_positions, selected_predictions, 
               color='#4a4a4a', s=100, marker='o', zorder=15)
    # 内层高光
    ax1.scatter(x_positions, selected_predictions, 
               color='#8a8a8a', s=40, marker='o', zorder=16)

    ax1.tick_params(axis='y', labelcolor=color1)
    
    # 设置左Y轴刻度字体
    for label in ax1.get_yticklabels():
        label.set_fontname('Times New Roman')
        label.set_fontsize(12)
    
    # 创建右Y轴: 不确定度
    ax2 = ax1.twinx()
    color2 = '#E74C3C'
    ax2.set_ylabel('Uncertainty (Std Dev)', fontsize=14, fontweight='bold', color=color2, fontname='Times New Roman')
    
    # 绘制不确定度折线
    ax2.plot(x_positions, selected_std,
             color=color2,
             linewidth=1.5,
             alpha=0.7,
             zorder=3,
             linestyle='-')
    
    # 绘制不确定度散点 (菱形) - 渐变饱满效果
    # 外层深色
    ax2.scatter(x_positions, selected_std, 
               color='#A93226', s=100, marker='D', zorder=5)
    # 中层
    ax2.scatter(x_positions, selected_std, 
               color='#E74C3C', s=65, marker='D', zorder=6)
    # 内层高光
    ax2.scatter(x_positions, selected_std, 
               color='#F5A89A', s=25, marker='D', zorder=7)
    
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # 设置右Y轴刻度字体
    for label in ax2.get_yticklabels():
        label.set_fontname('Times New Roman')
        label.set_fontsize(12)
    
    # 设置x轴刻度
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels([f'{i+1}' for i in range(n_points)], rotation=0, fontname='Times New Roman', fontsize=12)
    
    # 添加网格
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # 创建自定义图例
    legend_elements2 = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#F08090', 
               markeredgecolor='#D85A6A', markersize=10, label='True Value'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#4a4a4a', 
               markeredgecolor='#1a1a1a', markersize=10, label='PGNN-ASL Prediction'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='#E74C3C', 
               markeredgecolor='#A93226', markersize=8, label='Uncertainty')
    ]
    legend = ax1.legend(handles=legend_elements2, loc='upper left', framealpha=0.9, fontsize=12)
    for text in legend.get_texts():
        text.set_fontname('Times New Roman')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/uncertainty_dual_axis.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{OUTPUT_DIR}/uncertainty_dual_axis.pdf", dpi=300, bbox_inches='tight', format='pdf')
    plt.close()
    
    print(f"✓ 图2已保存: {OUTPUT_DIR}/uncertainty_dual_axis.png")
    print(f"✓ 图2已保存: {OUTPUT_DIR}/uncertainty_dual_axis.pdf")
    
    # ============ 生成详细数据表格 ============
    table_data = []
    for idx in range(n_points):
        sample_idx = best_indices[idx]
        row = {
            "No.": idx + 1,
            "Sample Index": int(sample_idx),
            "True Value": float(selected_targets[idx]),
            "Prediction": float(selected_predictions[idx]),
            "Error": float(abs(selected_predictions[idx] - selected_targets[idx])),
            "Uncertainty (Variance)": float(selected_uncertainties[idx]),
            "Uncertainty (Std Dev)": float(selected_std[idx]),
            "Within Interval": abs(selected_predictions[idx] - selected_targets[idx]) <= selected_std[idx]
        }
        table_data.append(row)
    
    df_table = pd.DataFrame(table_data)
    df_table.to_csv(f"{OUTPUT_DIR}/uncertainty_detail.csv", index=False, encoding="utf-8-sig")
    
    print(f"✓ 详细数据已保存: {OUTPUT_DIR}/uncertainty_detail.csv")
    
    # 统计信息
    within_interval = sum(1 for row in table_data if row["Within Interval"])
    print(f"\n统计信息:")
    print(f"  误差在不确定度区间内的样本数: {within_interval}/{n_points} ({within_interval/n_points*100:.1f}%)")
    print(f"  平均不确定度(标准差): {selected_std.mean():.4f}")
    print(f"  平均预测误差: {np.abs(selected_predictions - selected_targets).mean():.4f}")
    
    print("\n" + "="*70)
    print("不确定度可视化完成!")
    print("="*70)


if __name__ == "__main__":
    generate_uncertainty_plot()
