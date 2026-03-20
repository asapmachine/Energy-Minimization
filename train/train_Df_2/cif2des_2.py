import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import catboost as cb         
import shap
import matplotlib.pyplot as plt

# ========================================================= 
# 全局设置：中文字体与输出目录准备
# ========================================================= 
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

output_dir = 'shap_plots'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# ========================================================= 
# 阶段一：数据集成与严苛的物理清洗 
# ========================================================= 
print("开始执行阶段一：数据加载与清洗...")
df = pd.read_csv(r'e:\CODE\cif2des\train_Df\clean_data_Df\train_data_Df_final.csv')

features = ['Di', 'Df_x', 'Dif', 'rho', 'VSA', 'GSA', 'VPOV', 'GPOV', 'POAV_vol_frac', 'GPOAV', 'POAV']
target = 'Df_y'

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df = df[(df[target] > 0) & (df[target] < 100) & (df['Df_x'] > 0) & (df['Df_x'] < 100)]

check_columns = features + [target]
df.dropna(subset=check_columns, inplace=True)

if 'filename' in df.columns:
    df.drop_duplicates(subset=['filename'], inplace=True)
else:
    print("⚠️ 警告：数据集中未检测到 'filename' 列，已跳过基于文件名的去重。")

print(f"清洗完毕，当前有效样本数：{len(df)}")

# ========================================================= 
# 阶段二 & 阶段三：特征提取与数据集划分 (8:1:1)
# ========================================================= 
X = df[features]
y = df[target]

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
print(f"数据集划分完毕：训练集 {len(X_train)} | 验证集 {len(X_val)} | 测试集 {len(X_test)}")

# ========================================================= 
# 阶段五：模型终极拟合 (注入最佳参数)
# ========================================================= 
print("\n开始注入最佳参数并进行终极拟合...")
final_params = {
    'learning_rate': 0.020403712643468016, 
    'depth': 10, 
    'l2_leaf_reg': 3.0399551236344924, 
    'min_data_in_leaf': 1,
    'grow_policy': 'Depthwise',  
    'iterations': 2000,
    'loss_function': 'RMSE',
    'eval_metric': 'RMSE',
    'random_seed': 42
}

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

# 绘制并保存散点图
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6, edgecolors='w', linewidth=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2) 
plt.xlabel("真实值 (优化后 Df)")
plt.ylabel("预测值")
plt.title("真实值 vs 预测值 (Test Set)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(f'{output_dir}/model_performance_scatter.png', dpi=300, bbox_inches='tight')
plt.show() 

final_model.save_model('catboost_mof_df_predictor_best.cbm')
print("✅ 最佳模型已安全落盘！")

# ========================================================= 
# 阶段七：基于 SHAP 的物理规律可解释性分析
# ========================================================= 
print("\n=============== 启动 SHAP 物理规律解析 ===============")
explainer = shap.TreeExplainer(final_model)
print("正在计算测试集的 SHAP 值，请稍候...")
shap_values = explainer(X_test)

# 1. 全局影响图 (Beeswarm)
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test, show=False)
plt.title("SHAP 物理特征全局影响图 (Beeswarm)", fontsize=14, pad=20)
plt.tight_layout()
plt.savefig(f'{output_dir}/shap_beeswarm.png', dpi=300, bbox_inches='tight') 
plt.savefig(f'{output_dir}/shap_beeswarm.pdf', bbox_inches='tight') 
plt.show()

# 2. 绝对重要性柱状图 (Bar Plot)
plt.figure(figsize=(10, 8))
shap.plots.bar(shap_values, show=False)
plt.title("SHAP 特征重要性绝对排名", fontsize=14, pad=20)
plt.tight_layout()
plt.savefig(f'{output_dir}/shap_bar_importance.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. 局部解释瀑布图 (Waterfall)
plt.figure(figsize=(10, 8))
shap.plots.waterfall(shap_values[0], show=False)
plt.title("单个样本预测值的微观推导过程 (Waterfall)", fontsize=14, pad=20)
plt.tight_layout()
plt.savefig(f'{output_dir}/shap_waterfall_sample_0.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\n✅ 所有高质量图表均已保存至当前目录下的 '{output_dir}' 文件夹！")