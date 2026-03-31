import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import catboost as cb         
import optuna
import matplotlib.pyplot as plt

# 设置绘图支持中文显示 (针对 Windows/Mac 可选)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ========================================================= 
# 阶段一：数据集成与严苛的物理清洗 
# ========================================================= 
print("开始执行阶段一：数据加载与清洗...")

# ⚠️ 注意：这里的 CSV 必须是包含了 几何特征 和 RACs 特征 的合并版数据集
df = pd.read_csv(r'E:\CODE\cif2des\train_2\train_Di\clean_data\A_final_cleaned_data_Di.csv')

# 1. 定义几何特征与新增的 153 个 RACs 特征
geo_features = ['Di_x', 'Df', 'Dif', 'rho', 'VSA', 'GSA', 'VPOV', 'GPOV', 'POAV_vol_frac', 'GPOAV', 'POAV']

rac_features = [
        "lc-chi-0-all", "lc-chi-1-all", "lc-chi-2-all", "lc-chi-3-all", 
        "lc-Z-0-all", "lc-Z-1-all", "lc-Z-2-all", "lc-Z-3-all",
        "lc-I-0-all", "lc-I-1-all", "lc-I-2-all", "lc-I-3-all",
        "lc-T-0-all", "lc-T-1-all", "lc-T-2-all", "lc-T-3-all",
        "lc-S-0-all", "lc-S-1-all", "lc-S-2-all", "lc-S-3-all",
        "lc-alpha-0-all", "lc-alpha-1-all", "lc-alpha-2-all", "lc-alpha-3-all",
        "D_lc-chi-0-all", "D_lc-chi-1-all", "D_lc-chi-2-all", "D_lc-chi-3-all",
        "D_lc-Z-0-all", "D_lc-Z-1-all", "D_lc-Z-2-all", "D_lc-Z-3-all",
        "D_lc-I-0-all", "D_lc-I-1-all", "D_lc-I-2-all", "D_lc-I-3-all",
        "D_lc-T-0-all", "D_lc-T-1-all", "D_lc-T-2-all", "D_lc-T-3-all",
        "D_lc-S-0-all", "D_lc-S-1-all", "D_lc-S-2-all", "D_lc-S-3-all",
        "D_lc-alpha-0-all", "D_lc-alpha-1-all", "D_lc-alpha-2-all", "D_lc-alpha-3-all",
        "func-chi-0-all", "func-chi-1-all", "func-chi-2-all", "func-chi-3-all",
        "func-Z-0-all", "func-Z-1-all", "func-Z-2-all", "func-Z-3-all",
        "func-I-0-all", "func-I-1-all", "func-I-2-all", "func-I-3-all",
        "func-T-0-all", "func-T-1-all", "func-T-2-all", "func-T-3-all",
        "func-S-0-all", "func-S-1-all", "func-S-2-all", "func-S-3-all",
        "func-alpha-0-all", "func-alpha-1-all", "func-alpha-2-all", "func-alpha-3-all",
        "D_func-chi-0-all", "D_func-chi-1-all", "D_func-chi-2-all", "D_func-chi-3-all",
        "D_func-Z-0-all", "D_func-Z-1-all", "D_func-Z-2-all", "D_func-Z-3-all",
        "D_func-I-0-all", "D_func-I-1-all", "D_func-I-2-all", "D_func-I-3-all",
        "D_func-T-0-all", "D_func-T-1-all", "D_func-T-2-all", "D_func-T-3-all",
        "D_func-S-0-all", "D_func-S-1-all", "D_func-S-2-all", "D_func-S-3-all",
        "D_func-alpha-0-all", "D_func-alpha-1-all", "D_func-alpha-2-all", "D_func-alpha-3-all",
        "f-chi-0-all", "f-chi-1-all", "f-chi-2-all", "f-chi-3-all",
        "f-Z-0-all", "f-Z-1-all", "f-Z-2-all", "f-Z-3-all",
        "f-I-0-all", "f-I-1-all", "f-I-2-all", "f-I-3-all",
        "f-T-0-all", "f-T-1-all", "f-T-2-all", "f-T-3-all",
        "f-S-0-all", "f-S-1-all", "f-S-2-all", "f-S-3-all",
        "mc-chi-0-all", "mc-chi-1-all", "mc-chi-2-all", "mc-chi-3-all",
        "mc-Z-0-all", "mc-Z-1-all", "mc-Z-2-all", "mc-Z-3-all",
        "mc-I-0-all", "mc-I-1-all", "mc-I-2-all", "mc-I-3-all",
        "mc-T-0-all", "mc-T-1-all", "mc-T-2-all", "mc-T-3-all",
        "mc-S-0-all", "mc-S-1-all", "mc-S-2-all", "mc-S-3-all",
        "D_mc-chi-0-all", "D_mc-chi-1-all", "D_mc-chi-2-all", "D_mc-chi-3-all",
        "D_mc-Z-0-all", "D_mc-Z-1-all", "D_mc-Z-2-all", "D_mc-Z-3-all",
        "D_mc-I-0-all", "D_mc-I-1-all", "D_mc-I-2-all", "D_mc-I-3-all",
        "D_mc-T-0-all", "D_mc-T-1-all", "D_mc-T-2-all", "D_mc-T-3-all",
        "D_mc-S-0-all", "D_mc-S-1-all", "D_mc-S-2-all", "D_mc-S-3-all",
        "f-lig-chi-0", "f-lig-chi-1", "f-lig-chi-2", "f-lig-chi-3",
        "f-lig-Z-0", "f-lig-Z-1", "f-lig-Z-2", "f-lig-Z-3",
        "f-lig-I-0", "f-lig-I-1", "f-lig-I-2", "f-lig-I-3",
        "f-lig-T-0", "f-lig-T-1", "f-lig-T-2", "f-lig-T-3",
        "f-lig-S-0", "f-lig-S-1", "f-lig-S-2", "f-lig-S-3"
    ]
target = 'Di_y'

# 将潜在的正负无穷大统一替换为空值
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# 物理边界限制
df = df[(df[target] > 0) & (df[target] < 100) & (df['Di_x'] > 0) & (df['Di_x'] < 100)]

# 【关键保护】：仅仅在“几何特征”和“目标Y”上执行严格的 dropna
# 如果把 RACs 加进去 dropna，会导致包含0或者空图的样本被大面积误删
df.dropna(subset=geo_features + [target], inplace=True)

# 对于 RACs 中的缺失值 (通常代表缺乏某类化学键)，安全填充为 0
for col in rac_features:
    if col in df.columns:
        df[col] = df[col].fillna(0)

# 去重
if 'filename' in df.columns:
    df.drop_duplicates(subset=['filename'], inplace=True)
elif 'cif_name' in df.columns:
    df.drop_duplicates(subset=['cif_name'], inplace=True)

print(f"基础清洗完毕，当前有效样本数：{len(df)}")

# ========================================================= 
# 阶段二：特征工程 (零方差死特征剔除)
# ========================================================= 
print("\n正在扫描并剔除零方差(死特征)...")
valid_features = []
all_candidates = geo_features + rac_features

for col in all_candidates:
    if col in df.columns:
        # 如果一列中独立的值大于 1 个，说明方差不为 0，保留
        if df[col].nunique() > 1:
            valid_features.append(col)
        else:
            print(f"  🗑️ 剔除死特征: {col} (所有行的值完全一样)")
    else:
        print(f"  ⚠️ 警告: 数据集中未找到特征列 -> {col}")

print(f"特征筛选完毕！保留了 {len(valid_features)} / {len(all_candidates)} 个有效特征。")

# 提取最终特征 X 和 目标 y
X = df[valid_features]
y = df[target]

# ========================================================= 
# 阶段三：严谨的数据集划分 (8:1:1) 
# ========================================================= 
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
print(f"数据集划分完毕：训练集 {len(X_train)} | 验证集 {len(X_val)} | 测试集 {len(X_test)}")

# ========================================================= 
# 阶段四：基于 Optuna 的超参数调优 
# ========================================================= 
print("\n启动 Optuna 超参数寻优引擎 (高维特征自适应模式)...")

def objective(trial):
    param = {
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1),
        'depth': trial.suggest_int('depth', 5, 9),  # 稍微缩小上限防止高维特征死记硬背
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 20.0), # 放宽 L2 正则上限，强力防止过拟合
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 15),
        'grow_policy': 'Depthwise', 
        'iterations': 2000,
        'loss_function': 'RMSE', 
        'eval_metric': 'RMSE',   
        'random_seed': 42
    }

    model = cb.CatBoostRegressor(**param)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=100,
        verbose=False 
    )
    return model.get_best_score()['validation']['RMSE']

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=30)

print("✅ 调优完成！最佳参数为：")
print(study.best_params)

# ========================================================= 
# 阶段五：模型终极拟合与全过程监控 
# ========================================================= 
print("\n开始注入最佳参数并进行终极拟合...")
final_params = study.best_params.copy()
final_params.update({
    'grow_policy': 'Depthwise',
    'iterations': 2000,
    'loss_function': 'RMSE',
    'eval_metric': 'RMSE',
    'random_seed': 42,
    'task_type': 'GPU', 
    'devices': '0' 
})

final_model = cb.CatBoostRegressor(**final_params, custom_metric=['R2', 'MAE', 'RMSE'])
final_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=100,
    use_best_model=True, 
    verbose=100,          
    save_snapshot=True,   
    snapshot_file='catboost_snapshot.cbs'
)

# ========================================================= 
# 阶段六：终极盲测、物理规律解析与持久化 
# ========================================================= 
print("\n=============== 启封测试集 ===============")
y_pred = final_model.predict(X_test)

final_r2 = r2_score(y_test, y_pred)
final_mae = mean_absolute_error(y_test, y_pred)
final_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n🎯 终极盲测 (Test Set) 成绩单 🎯")
print(f"R2:   {final_r2:.4f}")
print(f"MAE:  {final_mae:.4f}")
print(f"RMSE: {final_rmse:.4f}")

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6, edgecolors='w', linewidth=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel("真实值 (优化后 Di)")
plt.ylabel("预测值")
plt.title(f"真实值 vs 预测值 (Test Set)\nR2 = {final_r2:.4f}")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

final_model.save_model('catboost_mof_di_predictor_best_with_RACs.cbm')
print("✅ 带有 RACs 的最佳高维模型已安全落盘 (catboost_mof_di_predictor_best_with_RACs.cbm)！")