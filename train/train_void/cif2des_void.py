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
df = pd.read_csv(r'E:\CODE\cif2des\train_void\clean_data\train_data_void_final.csv')

# 2. 物理学与完整性清洗
features = ['Di', 'Df', 'Dif', 'rho', 'VSA', 'GSA', 'VPOV', 'GPOV', 'POAV_vol_frac_x', 'GPOAV', 'POAV']
target = 'POAV_vol_frac_y'

# 将数据集中潜在的正负无穷大 (Inf/-Inf) 统一替换为空值 (NaN)
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# 剔除骨架坍塌或异常巨无霸的结构。同时对目标 (Y) 和 核心输入特征 (X) 施加物理边界限制
df = df[(df[target] > 0) & (df[target] < 1) & (df['POAV_vol_frac_x'] > 0) & (df['POAV_vol_frac_x'] < 1)]

# 剔除在任何核心特征上存在缺失值 (NaN) 的残缺样本
check_columns = features + [target]
df.dropna(subset=check_columns, inplace=True)

# 最后的双重保险去重 (注：前提是你的 CSV 中确实包含 'filename' 列)
if 'cif_name' in df.columns:
    df.drop_duplicates(subset=['cif_name'], inplace=True)
else:
    print("⚠️ 警告：数据集中未检测到 'cif_name' 列，已跳过基于文件名的去重。")

print(f"清洗完毕，当前有效样本数：{len(df)}")

# ========================================================= 
# 阶段二：特征工程与目标界定 (Feature Engineering) 
# ========================================================= 
# 1. 提取输入特征 (X)
X = df[features]

# 2. 确定预测目标 (Y)
y = df[target]

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
    # 定义搜索空间
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

    # 初始化模型
    model = cb.CatBoostRegressor(**param)

    # 拟合与评估 (保持调参过程静默 verbose=False)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=100,
        verbose=False 
    )
    
    # 获取验证集上的最佳 RMSE 并返回给引擎
    best_rmse = model.get_best_score()['validation']['RMSE']
    return best_rmse

# 实例化寻优引擎并执行
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=30)

print("✅ 调优完成！最佳参数为：")
print(study.best_params)

# ========================================================= 
# 阶段五：模型终极拟合与全过程监控 (Final Training & Monitoring) 
# ========================================================= 
print("\n开始注入最佳参数并进行终极拟合...")

# 将静态参数与找到的最佳动态参数合并
final_params = study.best_params.copy()
final_params.update({
    'grow_policy': 'Depthwise',
    'iterations': 2000,
    'loss_function': 'RMSE',
    'eval_metric': 'RMSE',
    'random_seed': 42,
    'task_type': 'GPU', 
    'devices': '0' # 如果你只有一张显卡就是 '0'；
})

# 1. 注入最佳参数，配置多指标监控
final_model = cb.CatBoostRegressor(
    **final_params,
    custom_metric=['R2', 'MAE', 'RMSE']
)

# 2. 最终拟合：开启早停、快照续训、多指标打印
final_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=100,
    use_best_model=True, 
    verbose=100,          # 每 100 轮打印一次指标
    save_snapshot=True,   # 防断电机制
    snapshot_file='catboost_snapshot.cbs'
)

# ========================================================= 
# 阶段六：终极盲测、物理规律解析与持久化 (Blind Test & Interpretation) 
# ========================================================= 
print("\n=============== 启封测试集 ===============")

# 1. 启封测试集进行盲测
y_pred = final_model.predict(X_test)

# 2. 核心指标严苛计算
final_r2 = r2_score(y_test, y_pred)
final_mae = mean_absolute_error(y_test, y_pred)
final_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n🎯 终极盲测 (Test Set) 成绩单 🎯")
print(f"R2:   {final_r2:.4f}")
print(f"MAE:  {final_mae:.4f}")
print(f"RMSE: {final_rmse:.4f}")

# 3. 结果可视化
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6, edgecolors='w', linewidth=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2) # 绘制 y=x 对角线
plt.xlabel("真实值 (优化后 void)")
plt.ylabel("预测值")
plt.title("真实值 vs 预测值 (Test Set)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# 4. 模型持久化入库
final_model.save_model('catboost_mof_void_predictor_best.cbm')
print("✅ 最佳模型已安全落盘 (catboost_mof_void_predictor_best.cbm)，随时可用于高通量预测！")