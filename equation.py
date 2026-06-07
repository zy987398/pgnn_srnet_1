import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

DATA_PATH = "3011916164精确材料性能数据.csv"

COL_MAT   = "材料种类"
COL_D     = "晶粒粒径"                       # d
COL_E     = "弹性模量"                         # E
COL_H     = "莫氏硬度"                           # H
COL_KC    = "断裂韧性"          # Kc
COL_SIGMA = "压溃压力"                         # sigma（按你数据中最接近压力的列）
COL_Y     = "亚表面裂纹深度"                    # y_true

RANDOM_SEED = 42

def H_poly(x):
    return 0.53*x - 1.42*(x**2) + 1.33*(x**3) - 0.61*np.sqrt(x)

def H_op(x):
    return 0.17*np.log(x) + 0.16*(x**2) + 0.15*(x**(-1))

def y_sym_from_x(x, gate=0.61, eps=1e-12):
    x = np.clip(np.asarray(x, dtype=float), eps, None)
    return gate * H_poly(x) + (1 - gate) * H_op(x)

def x_full(d, E, H, Kc, sigma, eps=1e-12):
    d = np.clip(np.asarray(d, dtype=float), eps, None)
    E = np.clip(np.asarray(E, dtype=float), eps, None)
    H = np.clip(np.asarray(H, dtype=float), eps, None)
    Kc = np.clip(np.asarray(Kc, dtype=float), eps, None)
    sigma = np.clip(np.asarray(sigma, dtype=float), eps, None)
    return (0.088*d
            + 0.089*(E*(Kc**2)/(H**3))
            + 0.12*((Kc**2)/(np.pi*d*(sigma**2))))

def x_donly(d):
    return 0.088*np.asarray(d, dtype=float)

def rmse_r2(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return rmse, r2

# --------- Load data ----------
df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
for c in [COL_D, COL_E, COL_H, COL_KC, COL_SIGMA, COL_Y]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=[COL_MAT, COL_D, COL_E, COL_H, COL_KC, COL_SIGMA, COL_Y]).copy()

d = df[COL_D].to_numpy()
E = df[COL_E].to_numpy()
H = df[COL_H].to_numpy()
Kc = df[COL_KC].to_numpy()
sigma = df[COL_SIGMA].to_numpy()
y = df[COL_Y].to_numpy()

idx = np.arange(len(df))
train_idx, test_idx = train_test_split(
    idx, test_size=0.2, random_state=RANDOM_SEED, stratify=df[COL_MAT]
)

# --------- Full equation ----------
y_sym_full = y_sym_from_x(x_full(d, E, H, Kc, sigma))
cal_full = LinearRegression().fit(y_sym_full[train_idx].reshape(-1,1), y[train_idx])
y_pred_full = cal_full.predict(y_sym_full[test_idx].reshape(-1,1))
rmse_full, r2_full = rmse_r2(y[test_idx], y_pred_full)

# --------- d-only ablation ----------
y_sym_d = y_sym_from_x(x_donly(d))
cal_d = LinearRegression().fit(y_sym_d[train_idx].reshape(-1,1), y[train_idx])
y_pred_d = cal_d.predict(y_sym_d[test_idx].reshape(-1,1))
rmse_d, r2_d = rmse_r2(y[test_idx], y_pred_d)

print(f"Full Eq.(28)-(31): RMSE={rmse_full:.4f} μm, R2={r2_full:.4f}")
print(f"Only 0.088d      : RMSE={rmse_d:.4f} μm, R2={r2_d:.4f}")
print(f"Drop             : ΔRMSE={rmse_d-rmse_full:+.4f} μm, ΔR2={r2_d-r2_full:+.4f}")