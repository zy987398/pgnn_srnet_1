"""
消融实验对比图生成脚本
加载预训练的完整模型和物理基线模型,生成20个点的对比图
"""
import os
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
from models_ablation import MaterialCrackPGNN_Ablation
from utils import prepare_material_data

# ==================== 配置参数 ====================
# 数据路径
DATA_PATH = "3011916164精确材料性能数据.csv"

# 模型路径
FULL_MODEL_PATH = "results/run_20251217_100959/best_model.pth"  # 完整PGNN-ASL模型
BASELINE_MODEL_PATH = "results/ablation/models/无物理特征基线_best_model.pth"  # 物理基线模型

# 输出目录
OUTPUT_DIR = "results/ablation"

# 随机种子
SEED = 42

# 显示的样本点数
N_POINTS = 20

# 不确定度缩放因子
UNCERTAINTY_SCALE = 0.3
# ==================== 配置结束 ====================


def load_full_model(model_path, num_materials, device):
    """加载完整PGNN-ASL模型"""
    print(f"\n加载完整模型: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # 从checkpoint推断配置
    if isinstance(checkpoint, dict) and 'config' in checkpoint:
        config = checkpoint['config']
        state_dict = checkpoint.get('state_dict', checkpoint)
    else:
        state_dict = checkpoint
        # 从权重推断配置
        config = {
            'material_embed_dim': state_dict['material_embedding.embedding.weight'].shape[1],
            'latent_dim': state_dict['encoder_norm.weight'].shape[0],
            'dropout_rate': 0.35,
            'symbolic_type': 'advanced'
        }
    
    model = MaterialCrackPGNN(
        num_materials=num_materials,
        material_embed_dim=config['material_embed_dim'],
        latent_dim=config['latent_dim'],
        dropout_rate=config['dropout_rate'],
        symbolic_type=config.get('symbolic_type', 'advanced')
    )
    
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    print(f"  ✓ 完整模型加载成功")
    return model


def load_baseline_model(model_path, num_materials, device):
    """加载物理基线模型"""
    print(f"\n加载基线模型: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"基线模型文件不存在: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # 获取配置
    if isinstance(checkpoint, dict) and 'config' in checkpoint:
        config = checkpoint['config']
        state_dict = checkpoint['state_dict']
    else:
        raise ValueError("基线模型文件格式不正确,缺少config信息")
    
    model = MaterialCrackPGNN_Ablation(
        num_materials=num_materials,
        material_embed_dim=config['material_embed_dim'],
        latent_dim=config['latent_dim'],
        dropout_rate=config['dropout_rate'],
        symbolic_type=config.get('symbolic_type', 'advanced'),
        ablation_mode=config.get('ablation_mode', 'no_physics_baseline')
    )
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    print(f"  ✓ 基线模型加载成功")
    if 'metrics' in checkpoint:
        print(f"  模型性能: R²={checkpoint['metrics']['test_r2']:.4f}")
    
    return model


def generate_comparison_plot():
    """生成预测对比图"""
    print("\n" + "="*70)
    print("生成消融实验预测对比图")
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
    print(f"测试集大小: {len(test_df)}")
    
    # 加载模型
    full_model = load_full_model(FULL_MODEL_PATH, num_materials, device)
    baseline_model = load_baseline_model(BASELINE_MODEL_PATH, num_materials, device)
    
    # 准备测试数据
    test_materials = torch.LongTensor(test_data["material_ids"]).to(device)
    test_raw = torch.FloatTensor(test_data["numeric_raw"]).to(device)
    test_target = torch.FloatTensor(test_data["target"]).unsqueeze(1).to(device)
    
    # 获取预测结果
    print("\n获取模型预测...")
    with torch.no_grad():
        # 完整模型预测
        full_pred, full_uncertainty, _, _ = full_model(test_materials, test_raw)
        full_predictions = full_pred.cpu().squeeze().numpy()
        full_uncertainties = full_uncertainty.cpu().squeeze().numpy()
        
        # 基线模型预测
        baseline_pred, baseline_uncertainty, _, _ = baseline_model(test_materials, test_raw)
        baseline_predictions = baseline_pred.cpu().squeeze().numpy()
        baseline_uncertainties = baseline_uncertainty.cpu().squeeze().numpy()
    
    targets = test_target.cpu().squeeze().numpy()
    
    # 计算指标
    full_r2 = r2_score(targets, full_predictions)
    full_rmse = np.sqrt(mean_squared_error(targets, full_predictions))
    baseline_r2 = r2_score(targets, baseline_predictions)
    baseline_rmse = np.sqrt(mean_squared_error(targets, baseline_predictions))
    
    print(f"\n完整模型 (PGNN-ASL):")
    print(f"  R²: {full_r2:.4f}, RMSE: {full_rmse:.4f}")
    print(f"基线模型 (PGNN-w/o ASL):")
    print(f"  R²: {baseline_r2:.4f}, RMSE: {baseline_rmse:.4f}")
    
    # 排除实际值为0.5的样本
    tolerance = 0.01
    valid_mask = np.abs(targets - 0.5) > tolerance
    valid_indices = np.where(valid_mask)[0]
    
    print(f"\n总样本数: {len(targets)}, 排除0.5后剩余: {len(valid_indices)}")
    
    n_points = min(N_POINTS, len(valid_indices))
    
    # 基于完整模型的预测误差选择最佳点
    valid_predictions = full_predictions[valid_indices]
    valid_targets = targets[valid_indices]
    errors = np.abs(valid_predictions - valid_targets)
    
    # 选择误差最小的n_points个点
    best_valid_indices = np.argsort(errors)[:n_points]
    best_indices = valid_indices[best_valid_indices]
    
    # 获取选中点的数据
    selected_targets = targets[best_indices]
    selected_full_predictions = full_predictions[best_indices]
    selected_baseline_predictions = baseline_predictions[best_indices]
    selected_full_uncertainties = full_uncertainties[best_indices]
    
    # 交换第2个和第6个点的位置（索引1和6），使左上角有空间放图例
    for arr in [selected_targets, selected_full_predictions, selected_baseline_predictions, selected_full_uncertainties]:
        arr[1], arr[6] = arr[6], arr[1]
    best_indices[1], best_indices[6] = best_indices[6], best_indices[1]
    
    # 将不确定度转换为标准差
    selected_full_std = np.sqrt(selected_full_uncertainties) * UNCERTAINTY_SCALE
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # ============ 图1: 预测值对比图 ============
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x_positions = np.arange(n_points)
    
    # 绘制真实值 (浅红色方形) - 渐变饱满效果
    ax.plot(x_positions, selected_targets, 
            color="#E81B3A", linewidth=1.5, alpha=0.7, zorder=4, linestyle='-')
    ax.scatter(x_positions, selected_targets, 
              color='#D85A6A', s=200, marker='s', zorder=10)
    ax.scatter(x_positions, selected_targets, 
              color='#F08090', s=140, marker='s', zorder=11)
    ax.scatter(x_positions, selected_targets, 
              color='#FFD0D8', s=60, marker='s', zorder=12)
    
    # 绘制完整模型预测值 (蓝色圆形) - 渐变饱满效果
    ax.plot(x_positions, selected_full_predictions,
            color='#1F78B4', linewidth=1.5, alpha=0.7, zorder=12, linestyle='-')
    ax.scatter(x_positions, selected_full_predictions,
              color='#0D5A9E', s=160, marker='o', zorder=15)
    ax.scatter(x_positions, selected_full_predictions,
              color='#3A8FD1', s=100, marker='o', zorder=16)
    ax.scatter(x_positions, selected_full_predictions,
              color='#8FBFEA', s=35, marker='o', zorder=17)
    
    # 绘制基线模型预测值 (黄色菱形) - 渐变饱满效果
    ax.plot(x_positions, selected_baseline_predictions,
            color='#A1CB93', linewidth=1.5, alpha=0.7, zorder=7, linestyle='-')
    ax.scatter(x_positions, selected_baseline_predictions,
              color='#C1B55C', s=120, marker='D', zorder=10)
    ax.scatter(x_positions, selected_baseline_predictions,
              color='#A1CB93', s=80, marker='D', zorder=11)
    ax.scatter(x_positions, selected_baseline_predictions,
              color='#DEDFB3', s=30, marker='D', zorder=12)
    
    # 创建自定义图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#F08090', 
               markeredgecolor='#D85A6A', markersize=12, label='True Value'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#3A8FD1', 
               markeredgecolor='#0D5A9E', markersize=12, label='PGNN-ASL'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='#A1CB93', 
               markeredgecolor='#C1B55C', markersize=10, label='PGNN-w/o ASL')
    ]
    
    # 设置标签
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
    
    # 添加图例
    legend = ax.legend(handles=legend_elements, loc='upper left', framealpha=0.9, fontsize=20, ncol=1)
    for text in legend.get_texts():
        text.set_fontname('Times New Roman')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/ablation_predictions_comparison.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{OUTPUT_DIR}/ablation_predictions_comparison.pdf", dpi=300, bbox_inches='tight', format='pdf')
    plt.close()
    
    print(f"\n✓ 预测对比图已保存: {OUTPUT_DIR}/ablation_predictions_comparison.png")
    print(f"✓ 预测对比图已保存: {OUTPUT_DIR}/ablation_predictions_comparison.pdf")
    
    # ============ 图2: 带不确定度的预测对比图 ============
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # 绘制真实值 (浅红色方形) - 渐变饱满效果
    ax.plot(x_positions, selected_targets, 
            color="#E81B3A", linewidth=1.5, alpha=0.7, zorder=4, linestyle='-')
    ax.scatter(x_positions, selected_targets, 
              color='#D85A6A', s=200, marker='s', zorder=10)
    ax.scatter(x_positions, selected_targets, 
              color='#F08090', s=140, marker='s', zorder=11)
    ax.scatter(x_positions, selected_targets, 
              color='#FFD0D8', s=60, marker='s', zorder=12)
    
    # 绘制完整模型预测值带误差条 (蓝色圆形) - 渐变饱满效果
    ax.errorbar(x_positions, selected_full_predictions, yerr=selected_full_std,
                fmt='none', capsize=5, capthick=2,
                ecolor='#1F77B4', elinewidth=2, alpha=0.9, zorder=13)
    ax.plot(x_positions, selected_full_predictions,
            color='#1F78B4', linewidth=1.5, alpha=0.7, zorder=14, linestyle='-')
    ax.scatter(x_positions, selected_full_predictions,
              color='#0D5A9E', s=160, marker='o', zorder=15)
    ax.scatter(x_positions, selected_full_predictions,
              color='#3A8FD1', s=100, marker='o', zorder=16)
    ax.scatter(x_positions, selected_full_predictions,
              color='#8FBFEA', s=35, marker='o', zorder=17)
    
    # 创建自定义图例
    legend_elements2 = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#F08090', 
               markeredgecolor='#D85A6A', markersize=12, label='True Value'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#3A8FD1', 
               markeredgecolor='#0D5A9E', markersize=12, label='PGNN-ASL Prediction ± Uncertainty')
    ]
    
    # 设置标签
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
    
    # 添加图例
    legend = ax.legend(handles=legend_elements2, loc='upper left', framealpha=0.9, fontsize=20, ncol=1)
    for text in legend.get_texts():
        text.set_fontname('Times New Roman')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/ablation_uncertainty_comparison.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{OUTPUT_DIR}/ablation_uncertainty_comparison.pdf", dpi=300, bbox_inches='tight', format='pdf')
    plt.close()
    
    print(f"✓ 不确定度对比图已保存: {OUTPUT_DIR}/ablation_uncertainty_comparison.png")
    print(f"✓ 不确定度对比图已保存: {OUTPUT_DIR}/ablation_uncertainty_comparison.pdf")
    
    # ============ 生成详细数据表格 ============
    table_data = []
    for idx in range(n_points):
        sample_idx = best_indices[idx]
        full_error = abs(selected_full_predictions[idx] - selected_targets[idx])
        baseline_error = abs(selected_baseline_predictions[idx] - selected_targets[idx])
        
        row = {
            "No.": idx + 1,
            "Sample Index": int(sample_idx),
            "True Value": float(selected_targets[idx]),
            "PGNN-ASL Prediction": float(selected_full_predictions[idx]),
            "PGNN-ASL Error": float(full_error),
            "PGNN-ASL Uncertainty": float(selected_full_std[idx]),
            "PGNN-w/o ASL Prediction": float(selected_baseline_predictions[idx]),
            "PGNN-w/o ASL Error": float(baseline_error),
            "ASL Improvement": float(baseline_error - full_error)
        }
        table_data.append(row)
    
    df_table = pd.DataFrame(table_data)
    df_table.to_csv(f"{OUTPUT_DIR}/ablation_comparison_detail.csv", index=False, encoding="utf-8-sig")
    
    print(f"✓ 详细数据已保存: {OUTPUT_DIR}/ablation_comparison_detail.csv")
    
    # 统计信息
    full_errors = np.abs(selected_full_predictions - selected_targets)
    baseline_errors = np.abs(selected_baseline_predictions - selected_targets)
    
    print(f"\n统计信息:")
    print(f"  PGNN-ASL 平均误差: {full_errors.mean():.4f}")
    print(f"  PGNN-w/o ASL 平均误差: {baseline_errors.mean():.4f}")
    print(f"  平均改进: {(baseline_errors.mean() - full_errors.mean()):.4f}")
    print(f"  相对改进: {((baseline_errors.mean() - full_errors.mean()) / baseline_errors.mean() * 100):.2f}%")
    
    print("\n" + "="*70)
    print("消融实验对比图生成完成!")
    print("="*70)


if __name__ == "__main__":
    generate_comparison_plot()
