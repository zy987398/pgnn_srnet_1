import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# 传统机器学习模型
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.linear_model import ElasticNet
from sklearn.neural_network import MLPRegressor

def evaluate_traditional_models(X_train, X_test, y_train, y_test):
    """评估传统机器学习模型"""
    
    models = {
        'Random Forest': RandomForestRegressor(
            n_estimators=200, 
            max_depth=15,
            min_samples_split=5,
            random_state=42
        ),
        'XGBoost': XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        ),
        'SVR': SVR(
            kernel='rbf',
            C=10,
            gamma='scale'
        ),
        'MLP': MLPRegressor(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            max_iter=1000,
            random_state=42
        ),
        'ElasticNet': ElasticNet(
            alpha=0.1,
            l1_ratio=0.5,
            random_state=42
        )
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n训练 {name}...")
        
        # 训练模型
        model.fit(X_train, y_train)
        
        # 预测
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # 评估指标
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        
        results[name] = {
            'model': model,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'test_mae': test_mae,
            'predictions': y_pred_test
        }
        
        print(f"  训练 R²: {train_r2:.4f}")
        print(f"  测试 R²: {test_r2:.4f}")
        print(f"  测试 MSE: {test_mse:.4f}")
        print(f"  测试 MAE: {test_mae:.4f}")
        
        # 特征重要性（如果支持）
        if hasattr(model, 'feature_importances_'):
            results[name]['feature_importance'] = model.feature_importances_
    
    return results

def create_ensemble_features(df):
    """创建集成学习的特征"""
    features = []
    feature_names = []
    
    # 原始数值特征
    numeric_cols = ['莫氏硬度', '弹性模量', '断裂韧性', '晶粒粒径', '压溃压力', '磨粒速度']
    features.append(df[numeric_cols].values)
    feature_names.extend(numeric_cols)
    
    # 物理特征工程
    # 1. 脆性指数
    brittleness = df['莫氏硬度'] / df['断裂韧性']
    features.append(brittleness.values.reshape(-1, 1))
    feature_names.append('脆性指数')
    
    # 2. 能量释放率代理
    energy_release = (df['压溃压力'] ** 2) * np.sqrt(df['晶粒粒径']) / df['弹性模量']
    features.append(energy_release.values.reshape(-1, 1))
    feature_names.append('能量释放率')
    
    # 3. 应力强度因子比
    k_ratio = df['压溃压力'] * np.sqrt(df['晶粒粒径']) / df['断裂韧性']
    features.append(k_ratio.values.reshape(-1, 1))
    feature_names.append('应力强度因子比')
    
    # 4. 接触应力
    contact_stress = np.power(df['压溃压力'] * df['弹性模量'] ** 2 / df['晶粒粒径'] ** 2, 1/3)
    features.append(contact_stress.values.reshape(-1, 1))
    feature_names.append('接触应力')
    
    # 5. 能量输入
    energy_input = df['压溃压力'] * df['磨粒速度']
    features.append(energy_input.values.reshape(-1, 1))
    feature_names.append('能量输入')
    
    # 6. 硬度弹性模量比
    h_e_ratio = df['莫氏硬度'] / df['弹性模量']
    features.append(h_e_ratio.values.reshape(-1, 1))
    feature_names.append('硬度弹性模量比')
    
    # 7. Hall-Petch特征
    hall_petch = 1 / np.sqrt(df['晶粒粒径'])
    features.append(hall_petch.values.reshape(-1, 1))
    feature_names.append('Hall-Petch特征')
    
    # 8. 材料类别编码（one-hot）
    le = LabelEncoder()
    material_encoded = le.fit_transform(df['材料种类'])
    
    # 材料类别特征（使用目标编码的思想）
    material_mean_crack = df.groupby('材料种类')['亚表面裂纹深度'].transform('mean')
    features.append(material_mean_crack.values.reshape(-1, 1))
    feature_names.append('材料平均裂纹深度')
    
    # 组合所有特征
    X = np.hstack(features)
    
    return X, feature_names, material_encoded, le

def plot_model_comparison(results, y_test):
    """绘制模型比较图"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (name, res) in enumerate(results.items()):
        ax = axes[idx]
        y_pred = res['predictions']
        
        # 散点图
        ax.scatter(y_test, y_pred, alpha=0.5, s=20)
        
        # 理想预测线
        min_val, max_val = y_test.min(), y_test.max()
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        
        # 添加误差带
        rmse = np.sqrt(res['test_mse'])
        ax.fill_between([min_val, max_val], 
                       [min_val - rmse, max_val - rmse],
                       [min_val + rmse, max_val + rmse],
                       alpha=0.2, color='red')
        
        ax.set_xlabel('实际裂纹深度 (μm)')
        ax.set_ylabel('预测裂纹深度 (μm)')
        ax.set_title(f'{name}\nR² = {res["test_r2"]:.3f}, RMSE = {rmse:.3f}')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 性能对比条形图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    model_names = list(results.keys())
    r2_scores = [res['test_r2'] for res in results.values()]
    rmse_scores = [np.sqrt(res['test_mse']) for res in results.values()]
    
    # R² 分数
    bars1 = ax1.bar(model_names, r2_scores, color='skyblue', edgecolor='navy')
    ax1.set_ylabel('R² Score')
    ax1.set_title('模型R²分数对比')
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bar, score in zip(bars1, r2_scores):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom')
    
    # RMSE
    bars2 = ax2.bar(model_names, rmse_scores, color='lightcoral', edgecolor='darkred')
    ax2.set_ylabel('RMSE (μm)')
    ax2.set_title('模型RMSE对比')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bar, score in zip(bars2, rmse_scores):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{score:.3f}', ha='center', va='bottom')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def analyze_feature_importance(results, feature_names):
    """分析特征重要性"""
    # 提取支持特征重要性的模型
    importance_models = {}
    for name, res in results.items():
        if 'feature_importance' in res:
            importance_models[name] = res['feature_importance']
    
    if not importance_models:
        print("没有模型支持特征重要性分析")
        return
    
    # 创建特征重要性热图
    importance_df = pd.DataFrame(importance_models, index=feature_names)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(importance_df, cmap='YlOrRd', annot=True, fmt='.3f', cbar_kws={'label': '重要性'})
    plt.title('不同模型的特征重要性对比')
    plt.xlabel('模型')
    plt.ylabel('特征')
    plt.tight_layout()
    plt.show()
    
    # 平均特征重要性
    mean_importance = importance_df.mean(axis=1).sort_values(ascending=False)
    
    plt.figure(figsize=(10, 6))
    mean_importance.plot(kind='barh')
    plt.xlabel('平均重要性')
    plt.title('特征平均重要性排名')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("\n特征重要性排名（平均）：")
    for feat, imp in mean_importance.items():
        print(f"{feat}: {imp:.4f}")

# 主函数
def main():
    """比较不同模型在材料裂纹深度预测上的性能"""
    
    # 读取数据
    print("读取数据...")
    df = pd.read_csv('3011916164精确材料性能数据.csv')
    
    # 创建特征
    print("创建特征...")
    X, feature_names, material_encoded, le = create_ensemble_features(df)
    y = df['亚表面裂纹深度'].values
    
    # 数据分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 评估传统模型
    print("\n评估传统机器学习模型...")
    results = evaluate_traditional_models(X_train_scaled, X_test_scaled, y_train, y_test)
    
    # 可视化比较
    print("\n生成对比图表...")
    plot_model_comparison(results, y_test)
    
    # 特征重要性分析
    print("\n分析特征重要性...")
    analyze_feature_importance(results, feature_names)
    
    # 误差分析
    print("\n误差分析...")
    best_model_name = max(results.keys(), key=lambda k: results[k]['test_r2'])
    best_model = results[best_model_name]
    
    print(f"\n最佳模型: {best_model_name}")
    print(f"测试集 R²: {best_model['test_r2']:.4f}")
    print(f"测试集 RMSE: {np.sqrt(best_model['test_mse']):.4f} μm")
    print(f"测试集 MAE: {best_model['test_mae']:.4f} μm")
    
    # 误差分布
    errors = y_test - best_model['predictions']
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.hist(errors, bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel('预测误差 (μm)')
    plt.ylabel('频数')
    plt.title(f'{best_model_name} 误差分布')
    plt.axvline(x=0, color='red', linestyle='--')
    
    plt.subplot(1, 3, 2)
    plt.scatter(best_model['predictions'], errors, alpha=0.5)
    plt.xlabel('预测值 (μm)')
    plt.ylabel('残差 (μm)')
    plt.title('残差图')
    plt.axhline(y=0, color='red', linestyle='--')
    
    plt.subplot(1, 3, 3)
    from scipy import stats
    stats.probplot(errors, dist="norm", plot=plt)
    plt.title('Q-Q图')
    
    plt.tight_layout()
    plt.show()
    
    return results, feature_names

# 模型推荐
def recommend_models():
    """基于数据特点推荐模型"""
    print("\n=== 模型推荐 ===")
    print("\n基于您的数据特点，我推荐以下模型方案：")
    
    print("\n1. **集成学习方法**（推荐优先级：高）")
    print("   - XGBoost/LightGBM：处理类别特征效果好，可解释性强")
    print("   - Random Forest：稳定性好，不易过拟合")
    print("   - 优势：无需复杂特征工程，自动处理特征交互")
    
    print("\n2. **物理信息神经网络（PGNN）**（推荐优先级：高）")
    print("   - 适合您的数据，因为有明确的物理机制")
    print("   - 可以融入断裂力学知识")
    print("   - 提供不确定性量化")
    
    print("\n3. **深度学习方法**")
    print("   - TabNet：专门为表格数据设计")
    print("   - SAINT：自注意力机制处理特征交互")
    print("   - 需要更多数据才能充分发挥优势")
    
    print("\n4. **混合方法**（推荐优先级：中）")
    print("   - 使用PGNN提取物理特征")
    print("   - 结合XGBoost进行最终预测")
    print("   - 平衡了可解释性和预测性能")

if __name__ == "__main__":
    # 运行对比实验
    results, feature_names = main()
    
    # 给出推荐
    recommend_models()
