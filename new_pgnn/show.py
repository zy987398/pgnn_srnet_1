import joblib
import os

print("=== 提取最佳超参数 ===")

# 首先尝试从study对象获取参数
study_path = "results/best_hyperopt_study.pkl"
if os.path.exists(study_path):
    print(f"从 {study_path} 加载study对象...")
    study = joblib.load(study_path)
    
    # 获取完整参数集
    if hasattr(study, "_complete_params") and study._complete_params:
        params = study._complete_params
        print("找到了完整参数集 (_complete_params):")
    else:
        # 尝试合并阶段1和阶段2的参数
        params = {}
        if hasattr(study, "_study1") and hasattr(study._study1, "best_params"):
            print("找到阶段1参数，合并中...")
            params.update(study._study1.best_params)
        if study.best_params:
            print("找到最终最佳参数，合并中...")
            params.update(study.best_params)
        if not params:
            params = study.best_params or {}
    
    print("\n最佳超参数:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    
    # 获取最佳值
    print(f"\n最佳验证R²: {study.best_value:.4f}")
else:
    print(f"找不到study对象文件: {study_path}")

# 然后尝试从训练历史获取参数
history_path = "results/training_history.pkl"
if os.path.exists(history_path):
    print(f"\n从 {history_path} 加载训练历史...")
    history = joblib.load(history_path)
    
    if "best_params" in history and history["best_params"]:
        print("\n从训练历史中提取的最佳超参数:")
        for key, value in history["best_params"].items():
            print(f"  {key}: {value}")
        
        print(f"\n训练历史中的最佳R²: {history['best_value']:.4f}")
    else:
        print("训练历史中没有保存超参数")
else:
    print(f"找不到训练历史文件: {history_path}")