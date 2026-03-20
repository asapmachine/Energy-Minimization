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
# 阶段一：数据集成与严苛的物理清洗 (Data Integration & Cleaning) 
# ========================================================= 
print("开始执行阶段一：数据加载与清洗...")

# 1. 读取原始数据集
df = pd.read_csv(r'E:\CODE\cif2des\train_VSA\clean_data\train_data_VSA_final.csv')

# 2. 物理学与完整性清洗
features = ['Di', 'Df', 'Dif', 'rho', 'VSA_x', 'GSA', 'VPOV', 'GPOV', 'POAV_vol_frac', 'GPOAV', 'POAV']
target = 'VSA_y'

# 将数据集中潜在的正负无穷大 (Inf/-Inf) 统一替换为空值 (NaN)
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# 剔除骨架坍塌或异常巨无霸的结构。同时对目标 (Y) 和 核心输入特征 (X) 施加物理边界限制
df = df[(df[target] > 0) & (df[target] < 2500) & (df['VSA_x'] > 0) & (df['VSA_x'] < 2500)]

# 剔除在任何核心特征上存在缺失值 (NaN) 的残缺样本
check_columns = features + [target]
df.dropna(subset=check_columns, inplace=True)

# 最后的双重保险去重
if 'filename' in df.columns:
    df.drop_duplicates(subset=['filename'], inplace=True)
else:
    print("⚠️ 警告：数据集中未检测到 'filename' 列，已跳过基于文件名的去重。")

print(f"清洗完毕，当前有效样本数：{len(df)}")

# ========================================================= 
# 阶段二：特征工程与目标界定 (Feature Engineering) 
# ========================================================= 
# 1. 提取输入特征 (X)
X = df[features]

# 2. 确定预测目标 (Y)
# 【核心改动 1】：使用 np.log1p 对跨度极大的 VSA 进行对数压缩
# 真实物理空间 (0~2500) -> 对数空间 (0~7.82)
print("正在对目标变量 VSA_y 进行对数转换 (log1p)...")
y = np.log1p(df[target])

# ========================================================= 
# 阶段三：严谨的数据集划分 (Data Splitting - 8:1:1 黄金法则) 
# ========================================================= 
# 第一次切分：80% 训练集，20% 临时保留集
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 第二次切分：将 20% 的保留集对半切分，得到 10% 验证集 和 10% 测试集
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)
print(f"数据集划分完毕：训练集 {len(X_train)} | 验证集 {len(X_val)} | 测试集 {len(X_test)}")

# ========================================================= 
# 阶段四：基于 Optuna 的超参数调优 (Hyperparameter Tuning) 
# ========================================================= 
print("\n启动 Optuna 超参数寻优引擎...")

def objective(trial):
    param = {
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1),
        'depth': trial.suggest_int('depth', 6, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
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
    
    best_rmse = model.get_best_score()['validation']['RMSE']
    return best_rmse

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=30)

print("✅ 调优完成！最佳参数为：")
print(study.best_params)

# ========================================================= 
# 阶段五：模型终极拟合与全过程监控 (Final Training & Monitoring) 
# ========================================================= 
print("\n开始注入最佳参数并进行终极拟合...")

final_params = study.best_params.copy()
final_params.update({
    'grow_policy': 'Depthwise',
    'iterations': 2000,
    'loss_function': 'RMSE',
    'eval_metric': 'RMSE',
    'random_seed': 42,
    'task_type': 'GPU',  # 启用 GPU 加速
    'devices': '0' 
})

final_model = cb.CatBoostRegressor(
    **final_params,
    custom_metric=['R2', 'MAE', 'RMSE']
)

print("\n⚠️ 注意：训练过程中打印的 R2, MAE, RMSE 均处于【对数空间】下。")
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
# 阶段六：终极盲测、物理规律解析与持久化 (Blind Test & Interpretation) 
# ========================================================= 
print("\n=============== 启封测试集 ===============")

# 1. 启封测试集进行盲测 (此时模型吐出的是对数尺度 0~9 的预测值)
y_pred_log = final_model.predict(X_test)

# 【核心改动 2】：使用 np.expm1 解除封印，将数据全部还原回 0~8000 的真实物理尺度
y_test_real = np.expm1(y_test)
y_pred_real = np.expm1(y_pred_log)

# 2. 核心指标严苛计算 (使用还原后的真实物理数值，保证论文数据的物理意义)
final_r2 = r2_score(y_test_real, y_pred_real)
final_mae = mean_absolute_error(y_test_real, y_pred_real)
final_rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))

print("\n🎯 终极盲测 (Test Set - 物理真实尺度) 成绩单 🎯")
print(f"真实 R2:   {final_r2:.4f}")
print(f"真实 MAE:  {final_mae:.4f} m^2/g")
print(f"真实 RMSE: {final_rmse:.4f} m^2/g")

# 3. 结果可视化 (展示还原后的真实物理数值)
plt.figure(figsize=(8, 6))
plt.scatter(y_test_real, y_pred_real, alpha=0.6, edgecolors='w', linewidth=0.5)
plt.plot([y_test_real.min(), y_test_real.max()], [y_test_real.min(), y_test_real.max()], 'k--', lw=2)
plt.xlabel("真实值 (优化后 VSA, $m^2/g$)")
plt.ylabel("预测值 (优化后 VSA, $m^2/g$)")
plt.title("真实值 vs 预测值 (Test Set)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# 4. 模型持久化入库
final_model.save_model('catboost_mof_VSA_predictor_best.cbm')
print("✅ 最佳模型已安全落盘，随时可用于高通量预测！")