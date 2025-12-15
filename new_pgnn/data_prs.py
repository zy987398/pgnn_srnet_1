import pandas as pd
import numpy as np

# ================= 配置区域 (请根据实际情况修改) =================

# 1. 砂轮直径 (单位: mm) —— 图片里没写，你需要自己测量填入！
#    警告：这个数值直接影响磨削速度的计算，必须准确。
WHEEL_DIAMETER_MM = 10 

# 2. 压力校准系数 (Calibration Factor)
#    用于将 (进给*深度/速度) 的计算结果调整到你训练数据中“压溃压力”的数量级。
#    建议：先填 500000，运行后看生成的压力值。如果你的训练数据压力大概是 500，而这里算出是 0.5，就把这个系数乘 1000。
CALIBRATION_FACTOR = 500000.0

# 3. 材料属性 (必须填入你训练数据中使用的数值)
#    图片备注："与之前微晶试件加工方式相同"，说明是同一种材料。
MATERIAL_PROPS = {
    "材料种类": 0,       # 对应 LabelEncoder 的 ID
    "莫氏硬度": 6.5,     # 示例值，请改为训练集中的真实值
    "弹性模量": 200.0,   # 示例值 (GPa?)
    "断裂韧性": 3.0,     # 示例值
    "晶粒粒径": 21    # 示例值 (注意：这是工件材料晶粒，不是磨粒)
}

# ===============================================================

def generate_model_data():
    # 1. 录入图片中的 9 行原始数据
    # 列顺序：序号, 粒度(mesh), 转速(r/min), 进给(mm/min), 深度(mm)
    raw_data = [
        # 行1: 10000 rpm, 400 mm/min, 0.005 mm
        {"id": 1, "rpm": 10000, "feed": 400,  "depth": 0.005},
        # 行2: 10000 rpm, 800 mm/min, 0.010 mm
        {"id": 2, "rpm": 10000, "feed": 800,  "depth": 0.010},
        # 行3: 10000 rpm, 1200 mm/min, 0.015 mm
        {"id": 3, "rpm": 10000, "feed": 1200, "depth": 0.015},
        
        # 行4: 15000 rpm, 400 mm/min, 0.01 mm
        {"id": 4, "rpm": 15000, "feed": 400,  "depth": 0.010},
        # 行5: 15000 rpm, 800 mm/min, 0.015 mm
        {"id": 5, "rpm": 15000, "feed": 800,  "depth": 0.015},
        # 行6: 15000 rpm, 1200 mm/min, 0.005 mm
        {"id": 6, "rpm": 15000, "feed": 1200, "depth": 0.005},
        
        # 行7: 20000 rpm, 400 mm/min, 0.015 mm
        {"id": 7, "rpm": 20000, "feed": 400,  "depth": 0.015},
        # 行8: 20000 rpm, 800 mm/min, 0.005 mm
        {"id": 8, "rpm": 20000, "feed": 800,  "depth": 0.005},
        # 行9: 20000 rpm, 1200 mm/min, 0.015 mm
        {"id": 9, "rpm": 20000, "feed": 1200, "depth": 0.015},
    ]

    df = pd.DataFrame(raw_data)

    print("="*50)
    print("开始数据转换...")
    print(f"设定砂轮直径: {WHEEL_DIAMETER_MM} mm")
    
    # 2. 计算【磨粒速度】 (Grinding Velocity, m/s)
    # 公式: vs = (π * D * n) / 60000
    df['磨粒速度'] = (np.pi * WHEEL_DIAMETER_MM * df['rpm']) / 60000

    # 3. 计算【压溃压力】 (Estimated Stress)
    # 物理代理公式: Pressure ∝ (vw * ap) / vs
    # vw (进给速度 m/s) = feed / 60000
    # ap (磨削深度 mm) = depth
    
    vw_ms = df['feed'] / 60000.0  # 换算为 m/s
    ap_mm = df['depth']
    vs_ms = df['磨粒速度']
    
    # 应用公式
    df['压溃压力'] = CALIBRATION_FACTOR * (vw_ms * ap_mm) / vs_ms

    # 4. 填充【材料固有属性】
    # 你的模型输入特征：材料种类, 莫氏硬度, 弹性模量, 断裂韧性, 晶粒粒径, 压溃压力, 磨粒速度
    for key, value in MATERIAL_PROPS.items():
        df[key] = value

    # 5. 整理最终列顺序 (必须与训练时完全一致)
    # 注意：这里的“晶粒粒径”指的是工件材料的晶粒尺寸，而不是磨料的800目
    final_cols = ['材料种类', '莫氏硬度', '弹性模量', '断裂韧性', '晶粒粒径', '压溃压力', '磨粒速度']
    result_df = df[final_cols]
    
    # 6. 保存文件
    output_filename = "test.csv"
    result_df.to_csv(output_filename, index=False)
    
    print("\n转换结果预览 (前3行):")
    print(result_df[['磨粒速度', '压溃压力']].head(3))
    print("\n" + "="*50)
    print(f"数据已生成并保存至: {output_filename}")
    print("请检查生成的'压溃压力'数值大小是否与你的训练数据接近。")
    print("如果不接近，请调整代码顶部的 CALIBRATION_FACTOR。")
    print("="*50)

if __name__ == "__main__":
    generate_model_data()