import os
import time
import warnings
import traceback
import copy
import joblib
import json
import torch
from torch.optim.lr_scheduler import OneCycleLR
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from torch.utils.data import TensorDataset, DataLoader

# 配置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi', 'FangSong']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['font.size'] = 12  # 设置字体大小

# 设置matplotlib后端，确保中文显示
import matplotlib
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
    
# 设置绘图样式
plt.style.use('seaborn-v0_8-whitegrid')

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
    ax.figure.canvas.draw_idle() if hasattr(ax.figure, 'canvas') else None

def setup_plot_style(fig=None, ax=None):
    """设置图表风格以确保中文正确显示和样式统一"""
    if fig is None:
        fig = plt.gcf()
    if ax is None:
        ax = plt.gca()
        
    # 增强网格线和轴线的可见性
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 设置轴标签和标题的字体属性
    font_name = get_chinese_font()
    
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(font_manager.FontProperties(family=font_name))
        
    # 设置标题和轴标签字体
    if ax.get_title():
        ax.set_title(ax.get_title(), fontproperties=font_manager.FontProperties(family=font_name))
    if ax.get_xlabel():
        ax.set_xlabel(ax.get_xlabel(), fontproperties=font_manager.FontProperties(family=font_name))
    if ax.get_ylabel():
        ax.set_ylabel(ax.get_ylabel(), fontproperties=font_manager.FontProperties(family=font_name))
        
    # 边框设置
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    
    # 图例样式
    legend = ax.get_legend()
    if legend is not None:
        legend.get_frame().set_alpha(0.8)
        for text in legend.get_texts():
            text.set_fontproperties(font_manager.FontProperties(family=font_name))
    
    return fig, ax

# 导入自定义模块
from models import MaterialCrackPGNN
from losses import PhysicsInformedLoss
from utils import prepare_material_data, augment_dataset, reconstruct_df

# 忽略警告
warnings.filterwarnings("ignore")

# 数据路径
DATA_PATH = "3011916164精确材料性能数据.csv"

# 固定随机种子,确保结果可重复
SEED = 66
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# 配置参数
USE_CROSS_VALIDATION = True  # 是否使用交叉验证
CV_FOLDS = 5  # 交叉验证折数


def load_best_params_from_study(study_path):
    """从已保存的Optuna study文件中加载最佳参数"""
    print(f"\n加载Optuna study: {study_path}")
    
    try:
        # 加载study
        study = joblib.load(study_path)
        
        # 获取最佳trial
        best_trial = study.best_trial
        best_params = best_trial.params
        
        print(f"\n最佳trial信息:")
        print(f"  Trial编号: {best_trial.number}")
        print(f"  最佳验证R²: {best_trial.value:.4f}")
        print(f"\n最佳参数:")
        for key, value in best_params.items():
            print(f"  {key}: {value}")
        
        # 保存最佳参数到JSON文件
        os.makedirs("results", exist_ok=True)
        with open("results/loaded_best_params.json", "w", encoding="utf-8") as f:
            json.dump(best_params, f, indent=4, ensure_ascii=False)
        print(f"\n最佳参数已保存至 results/loaded_best_params.json")
        
        return best_params
        
    except Exception as e:
        print(f"加载study失败: {e}")
        traceback.print_exc()
        return None


def train_single_fold(train_data, val_data, params, device, fold_idx=None):
    """训练单个折叠或数据集划分的模型"""
    fold_str = f"折叠 {fold_idx+1}/{params.get('cv_folds', 5)}: " if fold_idx is not None else ""
    
    # 创建模型
    num_materials = train_data["num_materials"]
    model = MaterialCrackPGNN(
        num_materials=num_materials,
        material_embed_dim=params.get("material_embed_dim", 16),
        latent_dim=params.get("latent_dim", 32),
        dropout_rate=params.get("dropout_rate", 0.2),
        symbolic_type=params.get("symbolic_type", "advanced")
    )
    model = model.to(device)
    
    # 数据增强 (如果启用)
    if params.get("augmentation", False):
        train_df = reconstruct_df(train_data)
        original_size = len(train_df)
        train_df = augment_dataset(train_df)
        if fold_idx is not None:
            print(f"  {fold_str}数据增强: {original_size} -> {len(train_df)} 样本")
        
        # 使用增强后的数据更新训练集
        train_data = prepare_material_data(
            train_df,
            scaler=train_data["scaler"],
            le=train_data["label_encoder"],
            augment=False,
        )
    
    # 创建DataLoader
    batch_size = params.get("batch_size", 32)
    train_dataset = TensorDataset(
        torch.LongTensor(train_data["material_ids"]),
        torch.FloatTensor(train_data["numeric_features"]),
        torch.FloatTensor(train_data["numeric_raw"]),
        torch.FloatTensor(train_data["target"]).unsqueeze(1),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=True,
        num_workers=0,
    )
    
    # 准备验证数据
    val_materials = torch.LongTensor(val_data["material_ids"]).to(device)
    val_features = torch.FloatTensor(val_data["numeric_features"]).to(device)
    val_raw = torch.FloatTensor(val_data["numeric_raw"]).to(device)
    val_target = torch.FloatTensor(val_data["target"]).unsqueeze(1).to(device)
    
    # 优化器
    lr = params.get("lr", 0.001)
    optimizer_name = params.get("optimizer", "AdamW")
    weight_decay = params.get("weight_decay", 0.01)
    
    if optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "RMSprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "RAdam":
        optimizer = torch.optim.RAdam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, 
                                   weight_decay=weight_decay, nesterov=True)
    elif optimizer_name == "NAdam":
        optimizer = torch.optim.NAdam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        print(f"未知的优化器: {optimizer_name}, 使用默认的AdamW")
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # 学习率调度
    epochs = params.get("epochs", 300)
    total_steps = epochs * len(train_loader)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=lr,
        total_steps=total_steps,
        pct_start=0.3,
        div_factor=25.0,
        final_div_factor=1e4,
        anneal_strategy='cos'
    )
    
    # 损失函数
    loss_fn = PhysicsInformedLoss(
        lambda_complexity=params.get("lambda_complexity", 0.0001),
        lambda_physics=params.get("lambda_physics", 0.2),
        weight_decay=0.0,
    )
    loss_fn.total_epochs = epochs
    
    # 训练循环
    best_val_r2 = -float("inf")
    patience_counter = 0
    best_model_wts = None
    train_losses = []
    val_losses = []
    val_r2_scores = []
    patience = params.get("patience", 50)
    
    for epoch in range(epochs):
        # 更新当前epoch
        loss_fn.update_epoch(epoch, epochs)
        
        # 训练
        model.train()
        epoch_loss = 0
        
        for batch_materials, batch_features, batch_raw, batch_target in train_loader:
            batch_materials = batch_materials.to(device, non_blocking=True)
            batch_features = batch_features.to(device, non_blocking=True)
            batch_raw = batch_raw.to(device, non_blocking=True)
            batch_target = batch_target.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            pred, uncertainty, symbolic, residual = model(batch_materials, batch_raw)
            loss, metrics = loss_fn(pred, uncertainty, batch_target, model, batch_raw)
            
            # 检查NaN
            if torch.isnan(loss) or torch.isinf(loss):
                if fold_idx is not None:
                    print(f"  {fold_str}警告：检测到NaN或Inf损失，跳过此批次")
                continue
                
            loss.backward()
            # 使用与train_hyperopt.py相同的梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=params.get("max_norm", 1.0))
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item() * batch_target.size(0)
        
        epoch_loss /= len(train_dataset)
        train_losses.append(epoch_loss)
        
        # 验证
        model.eval()
        with torch.no_grad():
            val_pred, val_uncertainty, _, _ = model(val_materials, val_raw)
            val_loss, _ = loss_fn(val_pred, val_uncertainty, val_target, model, val_raw)
            val_losses.append(val_loss.item())
            
            val_pred_np = val_pred.cpu().squeeze().numpy()
            val_target_np = val_target.cpu().squeeze().numpy()
            
            val_r2 = r2_score(val_target_np, val_pred_np)
            val_rmse = np.sqrt(mean_squared_error(val_target_np, val_pred_np))
            val_r2_scores.append(val_r2)
            
            # 每10个epoch打印一次进度
            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch+1}/{epochs} - 训练损失: {epoch_loss:.4f}, "
                      f"验证损失: {val_loss.item():.4f}, 验证R²: {val_r2:.4f}, 验证RMSE: {val_rmse:.4f}")
                
                # 门控监控
                if hasattr(model, 'get_gate_statistics'):
                    gate_stats = model.get_gate_statistics()
                    print(f"         门控统计 - 平均值: {gate_stats['mean']:.3f}, "
                          f"标准差: {gate_stats['std']:.3f}, "
                          f"符号层权重: {gate_stats['symbolic_weight']:.3f}")
        
        # 早停检查
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            patience_counter = 0
            best_model_wts = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
        
        if patience_counter >= params.get("patience", 50):
            if fold_idx is not None:
                print(f"  {fold_str}早停：{params.get('patience', 50)} 轮未改善，在第 {epoch+1} 轮停止")
            else:
                print(f"\n早停：{params.get('patience', 50)} 轮未改善，在第 {epoch+1} 轮停止")
            break
    
    # 恢复最佳模型状态
    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)
    
    # 最终验证结果
    model.eval()
    with torch.no_grad():
        val_pred, val_uncertainty, val_symbolic, val_residual = model(val_materials, val_raw)
        val_pred_np = val_pred.cpu().squeeze().numpy()
        val_target_np = val_target.cpu().squeeze().numpy()
        
        final_val_r2 = r2_score(val_target_np, val_pred_np)
        final_val_rmse = np.sqrt(mean_squared_error(val_target_np, val_pred_np))
    
    if fold_idx is not None:
        print(f"  {fold_str}最终验证 R²: {final_val_r2:.4f}, RMSE: {final_val_rmse:.4f}")
    
    return {
        "model": model,
        "val_r2": final_val_r2,
        "val_rmse": final_val_rmse,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_r2_scores": val_r2_scores,
        "best_epoch": len(val_r2_scores) - patience_counter - 1
    }


def evaluate_model(model, test_data, device):
    """评估模型在测试数据上的性能"""
    model.eval()
    
    # 准备测试数据
    test_materials = torch.LongTensor(test_data["material_ids"]).to(device)
    test_features = torch.FloatTensor(test_data["numeric_features"]).to(device)
    test_raw = torch.FloatTensor(test_data["numeric_raw"]).to(device)
    test_target = torch.FloatTensor(test_data["target"]).unsqueeze(1).to(device)
    
    with torch.no_grad():
        test_pred, test_uncertainty, test_symbolic, test_residual = model(test_materials, test_raw)
        
        # 转换为NumPy数组
        test_pred_np = test_pred.cpu().squeeze().numpy()
        test_target_np = test_target.cpu().squeeze().numpy()
        test_uncertainty_np = test_uncertainty.cpu().squeeze().numpy()
        test_symbolic_np = test_symbolic.cpu().squeeze().numpy() if test_symbolic is not None else None
        test_residual_np = test_residual.cpu().squeeze().numpy() if test_residual is not None else None
        
        # 计算指标
        test_r2 = r2_score(test_target_np, test_pred_np)
        test_rmse = np.sqrt(mean_squared_error(test_target_np, test_pred_np))
        test_mae = mean_absolute_error(test_target_np, test_pred_np)
        
        # 评估符号层和残差网络的单独性能
        results = {
            "test_r2": test_r2,
            "test_rmse": test_rmse,
            "test_mae": test_mae,
            "test_pred": test_pred_np,
            "test_target": test_target_np,
            "test_uncertainty": test_uncertainty_np,
        }
        
        if test_symbolic_np is not None and test_residual_np is not None:
            symbolic_r2 = r2_score(test_target_np, test_symbolic_np)
            residual_r2 = r2_score(test_target_np, test_residual_np)
            results["symbolic_r2"] = symbolic_r2
            results["residual_r2"] = residual_r2
            results["symbolic_pred"] = test_symbolic_np
            results["residual_pred"] = test_residual_np
            
            # 获取权重信息
            if hasattr(model, 'get_symbolic_interpretation'):
                interpretation = model.get_symbolic_interpretation()
                if "alpha_weight" in interpretation:
                    results["alpha_weight"] = interpretation["alpha_weight"]
    
    return results


def train_with_fixed_params(params, data_path=DATA_PATH, seed=SEED):
    """使用固定超参数训练和评估模型 - 支持单次分割和交叉验证"""
    print("\n" + "="*50)
    print("使用固定超参数训练模型")
    print("="*50)
    
    # 打印使用的参数
    for key, value in params.items():
        print(f"{key}: {value}")
    
    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n使用设备: {device}")
    
    # 读取数据
    print("\n加载数据...")
    df = pd.read_csv(data_path)
    print(f"数据集大小: {len(df)} 样本")
    
    # 检查是否使用交叉验证
    use_cv = params.get("use_cross_validation", False)
    cv_folds = params.get("cv_folds", 5)
    
    if use_cv:
        # 执行交叉验证训练
        print(f"\n使用 {cv_folds} 折交叉验证...")
        
        # 创建分层K折交叉验证
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
        cv_results = []
        
        # 用于记录所有折叠的平均测试性能
        cv_test_r2_values = []
        cv_test_rmse_values = []
        cv_test_mae_values = []
        cv_symbolic_r2_values = []
        cv_residual_r2_values = []
        cv_alpha_values = []
        
        # 准备可视化测试结果
        all_predictions = []
        all_targets = []
        
        # 保存最佳模型
        best_model = None
        best_model_r2 = -float("inf")
        
        # 执行交叉验证
        fold_idx = 0
        for train_idx, test_idx in skf.split(df, df["材料种类"]):
            print(f"\n训练折叠 {fold_idx+1}/{cv_folds}...")
            train_fold = df.iloc[train_idx]
            test_fold = df.iloc[test_idx]
            
            print(f"  折叠 {fold_idx+1}: 训练集 {len(train_idx)} 样本, 测试集 {len(test_idx)} 样本")
            
            # 准备数据
            train_data = prepare_material_data(train_fold)
            test_data = prepare_material_data(
                test_fold,
                scaler=train_data["scaler"],
                le=train_data["label_encoder"],
            )
            
            # 训练该折叠
            fold_result = train_single_fold(train_data, test_data, params, device, fold_idx)
            
            # 评估测试集性能
            eval_results = evaluate_model(fold_result["model"], test_data, device)
            
            # 记录结果
            cv_test_r2_values.append(eval_results["test_r2"])
            cv_test_rmse_values.append(eval_results["test_rmse"])
            cv_test_mae_values.append(eval_results["test_mae"])
            
            if "symbolic_r2" in eval_results:
                cv_symbolic_r2_values.append(eval_results["symbolic_r2"])
                cv_residual_r2_values.append(eval_results["residual_r2"])
            
            if "alpha_weight" in eval_results:
                cv_alpha_values.append(eval_results["alpha_weight"])
                
            # 收集预测结果
            all_predictions.append(eval_results["test_pred"])
            all_targets.append(eval_results["test_target"])
            
            # 保存结果
            fold_result.update({
                "test_r2": eval_results["test_r2"],
                "test_rmse": eval_results["test_rmse"],
                "test_mae": eval_results["test_mae"]
            })
            cv_results.append(fold_result)
            
            # 更新最佳模型
            if eval_results["test_r2"] > best_model_r2:
                best_model_r2 = eval_results["test_r2"]
                best_model = fold_result["model"]
            
            fold_idx += 1
        
        # 计算平均性能
        mean_test_r2 = np.mean(cv_test_r2_values)
        std_test_r2 = np.std(cv_test_r2_values)
        mean_test_rmse = np.mean(cv_test_rmse_values)
        mean_test_mae = np.mean(cv_test_mae_values)
        
        # 打印交叉验证结果
        print("\n" + "="*30)
        print(f"交叉验证结果 ({cv_folds}折):")
        print("="*30)
        print(f"测试集 R²: {mean_test_r2:.4f} ± {std_test_r2:.4f}")
        print(f"测试集 RMSE: {mean_test_rmse:.4f}")
        print(f"测试集 MAE: {mean_test_mae:.4f}")
        
        if cv_symbolic_r2_values:
            mean_symbolic_r2 = np.mean(cv_symbolic_r2_values)
            mean_residual_r2 = np.mean(cv_residual_r2_values)
            print(f"\n组件性能分析:")
            print(f"  符号层 R²: {mean_symbolic_r2:.4f}")
            print(f"  残差网络 R²: {mean_residual_r2:.4f}")
            print(f"  组合模型 R²: {mean_test_r2:.4f}  (提升: +{mean_test_r2-max(mean_symbolic_r2, mean_residual_r2):.4f})")
            
        if cv_alpha_values:
            mean_alpha = np.mean(cv_alpha_values)
            print(f"  符号层平均权重: {mean_alpha:.4f}")
            print(f"  残差网络平均权重: {1-mean_alpha:.4f}")
        
        # 每折的R²值
        for i, r2 in enumerate(cv_test_r2_values):
            print(f"  折叠 {i+1} R²: {r2:.4f}")
        
        # 可视化交叉验证结果
        print("\n生成交叉验证结果可视化...")
        
        # 创建结果目录
        os.makedirs("results", exist_ok=True)
        
        # 折叠性能对比
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(range(cv_folds), cv_test_r2_values, alpha=0.7, color='#4472C4')
        ax.axhline(y=mean_test_r2, color='r', linestyle='-', label=f'平均 R² = {mean_test_r2:.4f}')
        ax.set_xlabel('交叉验证折叠', fontproperties=chinese_font_prop)
        ax.set_ylabel('测试集 R²', fontproperties=chinese_font_prop)
        ax.set_title('交叉验证性能', fontweight='bold', fontproperties=chinese_font_prop)
        ax.set_xticks(range(cv_folds))
        ax.set_xticklabels([f'折叠 {i+1}' for i in range(cv_folds)])
        ax.legend(prop=chinese_font_prop)
        
        # 为刻度标签设置字体
        for label in ax.get_xticklabels():
            label.set_fontproperties(chinese_font_prop)
        for label in ax.get_yticklabels():
            label.set_fontproperties(chinese_font_prop)
        
        # 应用样式
        fig, ax = setup_plot_style(fig, ax)
        
        plt.savefig("results/cv_performance.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 所有折叠的预测散点图
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = ['#4472C4', '#ED7D31', '#A5A5A5', '#FFC000', '#5B9BD5', 
                '#70AD47', '#264478', '#9E480E', '#636363', '#997300']
                
        for i in range(len(all_predictions)):
            ax.scatter(all_targets[i], all_predictions[i], alpha=0.5, 
                     label=f'折叠 {i+1}' if i < 5 else None,  # 只显示前5个折叠的图例
                     color=colors[i % len(colors)])
        
        # 添加理想线
        min_val = min(np.min(np.concatenate(all_targets)), np.min(np.concatenate(all_predictions)))
        max_val = max(np.max(np.concatenate(all_targets)), np.max(np.concatenate(all_predictions)))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        
        ax.set_xlabel('实际裂纹深度', fontproperties=chinese_font_prop)
        ax.set_ylabel('预测裂纹深度', fontproperties=chinese_font_prop)
        ax.set_title(f'交叉验证预测 (平均 R² = {mean_test_r2:.4f} ± {std_test_r2:.4f})', fontweight='bold', fontproperties=chinese_font_prop)
        if cv_folds <= 5:  # 只有当折叠数较小时才显示图例
            ax.legend(prop=chinese_font_prop)
            
        # 为刻度标签设置字体
        for label in ax.get_xticklabels():
            label.set_fontproperties(chinese_font_prop)
        for label in ax.get_yticklabels():
            label.set_fontproperties(chinese_font_prop)
            
        # 应用样式
        fig, ax = setup_plot_style(fig, ax)
        
        plt.savefig("results/cv_predictions.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 保存最佳模型
        if best_model is not None:
            torch.save(best_model.state_dict(), "results/best_model.pth")
            print("\n最佳交叉验证模型已保存至 results/best_model.pth")
        
        # 返回交叉验证结果
        return {
            "cv_results": cv_results,
            "mean_test_r2": mean_test_r2,
            "std_test_r2": std_test_r2,
            "mean_test_rmse": mean_test_rmse,
            "mean_test_mae": mean_test_mae,
            "best_model": best_model
        }
    
    else:
        # 单次训练-测试分割
        print("\n使用单次训练-测试分割 (80%-20%)...")
        
        # 分割数据
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=seed, stratify=df["材料种类"])
        print(f"训练集: {len(train_df)} 样本, 测试集: {len(test_df)} 样本")
        
        # 准备数据
        train_data = prepare_material_data(train_df)
        test_data = prepare_material_data(
            test_df,
            scaler=train_data["scaler"],
            le=train_data["label_encoder"],
        )
        
        num_materials = train_data["num_materials"]
        print(f"共 {num_materials} 种材料类型")
        
        # 单次训练
        print("\n开始训练...")
        start_time = time.time()
        
        # 训练模型
        result = train_single_fold(train_data, test_data, params, device)
        model = result["model"]
        
        # 训练时间
        train_time = time.time() - start_time
        print(f"\n训练完成，用时: {train_time:.2f} 秒 ({train_time/60:.2f} 分钟)")
        
        # 评估模型
        print("\n在测试集上评估模型...")
        eval_results = evaluate_model(model, test_data, device)
        
        # 显示结果
        test_r2 = eval_results["test_r2"]
        test_rmse = eval_results["test_rmse"]
        test_mae = eval_results["test_mae"]
        
        print(f"\n最终测试集结果:")
        print(f"  R²: {test_r2:.4f}")
        print(f"  RMSE: {test_rmse:.4f}")
        print(f"  MAE: {test_mae:.4f}")
        
        # 门控融合分析
        if hasattr(model, 'get_fusion_interpretation'):
            print(f"\n门控融合分析:")
            print(model.get_fusion_interpretation())
            
            gate_stats = model.get_gate_statistics()
            print(f"门控详细统计: 平均={gate_stats['mean']:.3f}, "
                  f"最小={gate_stats['min']:.3f}, 最大={gate_stats['max']:.3f}")
        
        # 符号层可解释性分析
        if hasattr(model, 'get_symbolic_interpretation'):
            print(f"\n符号层可解释性分析:")
            interp = model.get_symbolic_interpretation()
            if 'complexity' in interp:
                print(f"符号表达式复杂度: {interp['complexity']:.4f}")
    
        # 可视化训练过程
        print("\n生成训练过程可视化...")
        train_losses = result["train_losses"]
        val_losses = result["val_losses"] 
        val_r2_scores = result["val_r2_scores"]
        test_pred_np = eval_results["test_pred"]
        test_target_np = eval_results["test_target"]
        
        plt.figure(figsize=(15, 5))
        
        # 损失曲线
        plt.subplot(1, 3, 1)
        plt.plot(train_losses, label='训练损失', alpha=0.7)
        plt.plot(val_losses, label='验证损失', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('损失', fontproperties=chinese_font_prop)
        plt.legend(prop=chinese_font_prop)
        plt.title('训练和验证损失', fontproperties=chinese_font_prop)
        plt.grid(True, alpha=0.3)
        
        # R²曲线
        plt.subplot(1, 3, 2)
        plt.plot(val_r2_scores, label='验证R²', alpha=0.7)
        plt.axhline(y=test_r2, color='r', linestyle='--', label=f'测试R²: {test_r2:.4f}')
        plt.xlabel('Epoch')
        plt.ylabel('R²')
        plt.legend(prop=chinese_font_prop)
        plt.title('验证和测试R²', fontproperties=chinese_font_prop)
        plt.grid(True, alpha=0.3)
        
        # 预测散点图
        plt.subplot(1, 3, 3)
        plt.scatter(test_target_np, test_pred_np, alpha=0.5, s=30)
        min_val = min(test_target_np.min(), test_pred_np.min())
        max_val = max(test_target_np.max(), test_pred_np.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        plt.xlabel('实际裂纹深度', fontproperties=chinese_font_prop)
        plt.ylabel('预测裂纹深度', fontproperties=chinese_font_prop)
        plt.title(f'预测结果 (R²={test_r2:.4f})', fontproperties=chinese_font_prop)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("results/training_results.png", dpi=150)
        plt.close()
        print("训练结果图已保存至 results/training_results.png")
    
    # 门控权重可视化
    if hasattr(model, 'last_gate') and model.last_gate is not None:
        plt.figure(figsize=(12, 5))
        gate_values = model.last_gate.cpu().detach().numpy().flatten()
        
        plt.subplot(1, 2, 1)
        plt.hist(gate_values, bins=30, alpha=0.7, edgecolor='black', color='steelblue')
        plt.xlabel('门控权重值', fontproperties=chinese_font_prop)
        plt.ylabel('频次', fontproperties=chinese_font_prop)
        plt.title('门控权重分布', fontproperties=chinese_font_prop)
        plt.axvline(x=0.5, color='r', linestyle='--', linewidth=2, label='平衡点 (0.5)')
        plt.legend(prop=chinese_font_prop)
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        sample_size = min(100, len(gate_values))
        plt.plot(gate_values[:sample_size], 'o-', alpha=0.6, markersize=4)
        plt.xlabel('样本索引', fontproperties=chinese_font_prop)
        plt.ylabel('门控权重', fontproperties=chinese_font_prop)
        plt.title(f'门控权重示例 (前{sample_size}个样本)', fontproperties=chinese_font_prop)
        plt.axhline(y=0.5, color='r', linestyle='--', linewidth=2, label='平衡点')
        plt.legend(prop=chinese_font_prop)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("results/gate_analysis.png", dpi=150)
        plt.close()
        print("门控分析图已保存至 results/gate_analysis.png")
    
        # 不确定性可视化
        test_uncertainty_np = eval_results["test_uncertainty"]
        plt.figure(figsize=(12, 6))
        indices = np.arange(len(test_pred_np))
        plt.errorbar(indices, test_pred_np, yerr=np.sqrt(test_uncertainty_np), 
                     fmt='o', alpha=0.3, elinewidth=1, capsize=2, label='预测±不确定性')
        plt.plot(indices, test_target_np, 'ro', markersize=3, alpha=0.6, label='实际值')
        plt.xlabel('测试样本索引', fontproperties=chinese_font_prop)
        plt.ylabel('裂纹深度', fontproperties=chinese_font_prop)
        plt.title('预测值与不确定性', fontproperties=chinese_font_prop)
        plt.legend(prop=chinese_font_prop)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("results/uncertainty_visualization.png", dpi=150)
        plt.close()
        print("不确定性可视化已保存至 results/uncertainty_visualization.png")
    
        # 保存模型
        model_save_path = "results/best_model_from_study.pth"
        torch.save(model.state_dict(), model_save_path)
        print(f"模型已保存至 {model_save_path}")
        
        # 保存训练结果
        results_summary = {
            "test_r2": float(test_r2),
            "test_rmse": float(test_rmse),
            "test_mae": float(test_mae),
            "best_val_r2": float(result["val_r2"]),
            "training_time_seconds": float(train_time),
            "epochs_trained": len(train_losses),
            "parameters": params
        }
        
        with open("results/training_summary.json", "w", encoding="utf-8") as f:
            json.dump(results_summary, f, indent=4, ensure_ascii=False)
        print("训练摘要已保存至 results/training_summary.json")
    
    # 返回结果
    results = {
        "model": model,
        "test_r2": test_r2,
        "test_rmse": test_rmse,
        "test_mae": test_mae,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_r2_scores": val_r2_scores
    }
    
    return results


def predict_on_new_data(model, new_data_path, train_data_path=DATA_PATH):
    """使用训练好的模型对新数据进行预测"""
    print("\n" + "="*50)
    print("对新数据进行预测")
    print("="*50)
    
    # 加载原始训练数据以获取缩放器和标签编码器
    train_df = pd.read_csv(train_data_path)
    train_data = prepare_material_data(train_df)
    
    # 加载新数据
    new_df = pd.read_csv(new_data_path)
    print(f"新数据大小: {len(new_df)} 样本")
    
    # 使用训练数据的缩放器和标签编码器处理新数据
    new_data = prepare_material_data(
        new_df,
        scaler=train_data["scaler"],
        le=train_data["label_encoder"],
    )
    
    # 准备预测
    device = next(model.parameters()).device
    new_materials = torch.LongTensor(new_data["material_ids"]).to(device)
    new_raw = torch.FloatTensor(new_data["numeric_raw"]).to(device)
    
    # 进行预测
    model.eval()
    with torch.no_grad():
        preds, uncertainties, _, _ = model(new_materials, new_raw)
        
        preds_np = preds.cpu().squeeze().numpy()
        uncertainties_np = uncertainties.cpu().squeeze().numpy()
    
    # 创建结果DataFrame
    results_df = new_df.copy()
    results_df["预测裂纹深度"] = preds_np
    results_df["预测不确定性"] = uncertainties_np
    
    # 如果有真实值，计算指标
    if "亚表面裂纹深度" in results_df.columns:
        r2 = r2_score(results_df["亚表面裂纹深度"], results_df["预测裂纹深度"])
        rmse = np.sqrt(mean_squared_error(results_df["亚表面裂纹深度"], results_df["预测裂纹深度"]))
        mae = mean_absolute_error(results_df["亚表面裂纹深度"], results_df["预测裂纹深度"])
        
        print(f"\n预测指标:")
        print(f"  R²: {r2:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")
        
        # 可视化
        plt.figure(figsize=(10, 8))
        plt.scatter(results_df["亚表面裂纹深度"], results_df["预测裂纹深度"], alpha=0.5)
        min_val = min(results_df["亚表面裂纹深度"].min(), results_df["预测裂纹深度"].min())
        max_val = max(results_df["亚表面裂纹深度"].max(), results_df["预测裂纹深度"].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        plt.xlabel('实际裂纹深度')
        plt.ylabel('预测裂纹深度')
        plt.title(f'新数据预测结果 (R²={r2:.4f})')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("results/new_data_predictions.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("预测结果图已保存至 results/new_data_predictions.png")
    
    # 保存预测结果
    results_df.to_csv("results/prediction_results.csv", index=False, encoding="utf-8-sig")
    print("预测结果已保存至 results/prediction_results.csv")
    
    return results_df


def train_with_best_params(params, data_path=DATA_PATH, seed=SEED):
    """使用最佳超参数训练和评估模型"""
    print("\n" + "="*50)
    print("使用Optuna最佳超参数训练模型")
    print("="*50)
    
    # 打印使用的参数
    print("\n训练参数:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    
    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n使用设备: {device}")
    
    # 读取和分割数据
    print("\n加载数据...")
    df = pd.read_csv(data_path)
    print(f"数据集大小: {len(df)} 样本")
    
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=seed, stratify=df["材料种类"])
    print(f"训练集: {len(train_df)} 样本, 测试集: {len(test_df)} 样本")
    
    # 准备数据
    train_data = prepare_material_data(train_df)
    test_data = prepare_material_data(
        test_df,
        scaler=train_data["scaler"],
        le=train_data["label_encoder"],
    )
    
    num_materials = train_data["num_materials"]
    print(f"共 {num_materials} 种材料类型")
    
    # 数据增强 (如果启用)
    train_df_for_aug = train_df.copy()
    if params.get("augmentation", False):
        print("\n应用数据增强...")
        train_df_for_aug = augment_dataset(train_df_for_aug)
        print(f"数据集大小: {len(train_df)} -> {len(train_df_for_aug)} (增加 {len(train_df_for_aug)-len(train_df)} 个样本)")
    
    augmented_train_data = prepare_material_data(
        train_df_for_aug,
        scaler=train_data["scaler"],
        le=train_data["label_encoder"],
        augment=False,
    )
    
    # 使用单次训练
    print("\n开始训练...")
    start_time = time.time()
    
    # 训练模型
    result = train_single_fold(augmented_train_data, test_data, params, device)
    model = result["model"]
    
    # 训练时间
    train_time = time.time() - start_time
    print(f"\n训练完成，用时: {train_time:.2f} 秒 ({train_time/60:.2f} 分钟)")
    
    # 评估模型
    print("\n在测试集上评估模型...")
    eval_results = evaluate_model(model, test_data, device)
    
    # 显示结果
    test_r2 = eval_results["test_r2"]
    test_rmse = eval_results["test_rmse"]
    test_mae = eval_results["test_mae"]
    
    print(f"\n最终测试集结果:")
    print(f"  R²: {test_r2:.4f}")
    print(f"  RMSE: {test_rmse:.4f}")
    print(f"  MAE: {test_mae:.4f}")
        
    # 门控融合分析
    if hasattr(model, 'get_fusion_interpretation'):
        print(f"\n门控融合分析:")
        print(model.get_fusion_interpretation())
        
        gate_stats = model.get_gate_statistics()
        print(f"门控详细统计: 平均={gate_stats['mean']:.3f}, "
              f"最小={gate_stats['min']:.3f}, 最大={gate_stats['max']:.3f}")
    
    # 符号层可解释性分析
    if hasattr(model, 'get_symbolic_interpretation'):
        print(f"\n符号层可解释性分析:")
        interp = model.get_symbolic_interpretation()
        if 'complexity' in interp:
            print(f"符号表达式复杂度: {interp['complexity']:.4f}")

    # 可视化训练过程
    print("\n生成训练过程可视化...")
    train_losses = result["train_losses"]
    val_losses = result["val_losses"] 
    val_r2_scores = result["val_r2_scores"]
    test_pred_np = eval_results["test_pred"]
    test_target_np = eval_results["test_target"]
    test_uncertainty_np = eval_results["test_uncertainty"]
    
    plt.figure(figsize=(15, 5))
    
    # 损失曲线
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='训练损失', alpha=0.7)
    plt.plot(val_losses, label='验证损失', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('损失', fontproperties=chinese_font_prop)
    plt.legend(prop=chinese_font_prop)
    plt.title('训练和验证损失', fontproperties=chinese_font_prop)
    plt.grid(True, alpha=0.3)
    
    # R²曲线
    plt.subplot(1, 3, 2)
    plt.plot(val_r2_scores, label='验证R²', alpha=0.7)
    plt.axhline(y=test_r2, color='r', linestyle='--', label=f'测试R²: {test_r2:.4f}')
    plt.xlabel('Epoch')
    plt.ylabel('R²')
    plt.legend(prop=chinese_font_prop)
    plt.title('验证和测试R²', fontproperties=chinese_font_prop)
    plt.grid(True, alpha=0.3)
    
    # 预测散点图
    plt.subplot(1, 3, 3)
    plt.scatter(test_target_np, test_pred_np, alpha=0.5, s=30)
    min_val = min(test_target_np.min(), test_pred_np.min())
    max_val = max(test_target_np.max(), test_pred_np.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    plt.xlabel('实际裂纹深度', fontproperties=chinese_font_prop)
    plt.ylabel('预测裂纹深度', fontproperties=chinese_font_prop)
    plt.title(f'预测结果 (R²={test_r2:.4f})', fontproperties=chinese_font_prop)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("results/training_results.png", dpi=150)
    plt.close()
    print("训练结果图已保存至 results/training_results.png")
    
    # 保存模型
    model_save_path = "results/best_model_from_study.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"模型已保存至 {model_save_path}")
    
    # 保存训练结果
    results_summary = {
        "test_r2": float(test_r2),
        "test_rmse": float(test_rmse),
        "test_mae": float(test_mae),
        "training_time_seconds": float(train_time),
        "epochs_trained": len(train_losses),
        "parameters": params
    }
    
    with open("results/training_summary.json", "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=4, ensure_ascii=False)
    print("训练摘要已保存至 results/training_summary.json")
    
    # 返回结果
    results = {
        "model": model,
        "test_r2": test_r2,
        "test_rmse": test_rmse,
        "test_mae": test_mae,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_r2_scores": val_r2_scores
    }
    
    return results


if __name__ == "__main__":
    print("\n" + "="*70)
    print("材料裂纹深度预测 - 从Optuna Study加载最佳参数并训练")
    print("="*70)
    
    # 确保结果目录存在
    os.makedirs("results", exist_ok=True)
    
    # 指定study文件路径 (请根据实际文件名修改)
    STUDY_PATH = "results\\best_hyperopt_study.pkl"  # 修改为您的study文件路径
    
    # 检查study文件是否存在
    if not os.path.exists(STUDY_PATH):
        print(f"\n错误: 找不到study文件 '{STUDY_PATH}'")
        print("请确保study文件存在，或修改STUDY_PATH变量指向正确的文件路径")
        print("\n可用的.pkl文件:")
        pkl_files = [f for f in os.listdir(".") if f.endswith(".pkl")]
        if pkl_files:
            for f in pkl_files:
                print(f"  - {f}")
        else:
            print("  当前目录下没有找到.pkl文件")
        exit(1)
    
    # 从study中加载最佳参数
    best_params = load_best_params_from_study(STUDY_PATH)
    
    if best_params is None:
        print("\n加载参数失败，程序退出")
        exit(1)
    
    # 添加交叉验证参数（如果没有的话）
    if 'use_cross_validation' not in best_params:
        best_params['use_cross_validation'] = USE_CROSS_VALIDATION
        best_params['cv_folds'] = CV_FOLDS
        print(f"\n添加交叉验证参数: {'启用' if USE_CROSS_VALIDATION else '禁用'} {CV_FOLDS}折交叉验证")
    
    # 使用最佳参数训练模型
    print("\n" + "="*70)
    results = train_with_fixed_params(best_params, DATA_PATH, SEED)
    
    print("\n" + "="*70)
    # 区分交叉验证结果和普通测试结果
    if 'mean_test_r2' in results:
        print(f"模型训练与评估完成，交叉验证平均 R² = {results['mean_test_r2']:.4f} ± {results['std_test_r2']:.4f}")
    elif 'test_r2' in results:
        print(f"模型训练完成!")
        print(f"  最终测试集 R² = {results['test_r2']:.4f}")
        print(f"  最终测试集 RMSE = {results['test_rmse']:.4f}")
        print(f"  最终测试集 MAE = {results['test_mae']:.4f}")
    print("="*70)
    