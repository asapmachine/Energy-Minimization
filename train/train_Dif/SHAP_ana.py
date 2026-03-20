import pandas as pd
import numpy as np
import catboost as cb         
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os

# 1. 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

print("=============== 准备数据与加载模型 ===============")
# 2. 完美复刻昨天的数据清洗与切分 (必须保证 random_state=42，以获得相同的 X_test)
df = pd.read_csv(r'E:\CODE\cif2des\train_Dif\clean_data\train_data_Dif_final.csv')
features = ['Di', 'Df', 'Dif_x', 'rho', 'VSA', 'GSA', 'VPOV', 'GPOV', 'POAV_vol_frac', 'GPOAV', 'POAV']
target = 'Dif_y'
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df = df[(df[target] > 0) & (df[target] < 100) & (df['Dif_x'] > 0) & (df['Dif_x'] < 100)]
df.dropna(subset=features + [target], inplace=True)
if 'filename' in df.columns:
    df.drop_duplicates(subset=['filename'], inplace=True)

X = df[features]
y = df[target]
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# 3. 唤醒落盘的模型！
print("正在从硬盘加载已训练好的模型...")
final_model = cb.CatBoostRegressor()
final_model.load_model('catboost_mof_dif_predictor_best.cbm')

# ========================================================= 
# 阶段七：基于 SHAP 的物理规律可解释性分析
# ========================================================= 
print("\n=============== 启动 SHAP 物理规律解析 ===============")
explainer = shap.TreeExplainer(final_model)
print("正在计算测试集的 SHAP 值，请稍候...")
shap_values = explainer(X_test)

# 画图 1: 全局影响
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test, show=False)
plt.title("SHAP 物理特征全局影响图 (Beeswarm)", fontsize=14, pad=20)
plt.tight_layout()
plt.show()

# 画图 2: 绝对重要性
plt.figure(figsize=(10, 8))
shap.plots.bar(shap_values, show=False)
plt.title("SHAP 特征重要性绝对排名", fontsize=14, pad=20)
plt.tight_layout()
plt.show()
# 创建一个专门存图的文件夹，保持工程目录整洁
output_dir = 'shap_plots'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# ==========================================
# 画图 1: 全局影响图 (Beeswarm)
# ==========================================
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test, show=False)
plt.title("SHAP 物理特征全局影响图 (Beeswarm)", fontsize=14, pad=20)
plt.tight_layout()
# 【新增】：以 300 DPI 的高分辨率保存，bbox_inches='tight' 确保边缘的坐标轴标签不会被截断
plt.savefig(f'{output_dir}/shap_beeswarm.png', dpi=300, bbox_inches='tight') 
plt.savefig(f'{output_dir}/shap_beeswarm.pdf', bbox_inches='tight') # 顺便存一份矢量图，方便插入论文
plt.show()

# ==========================================
# 画图 2: 绝对重要性柱状图 (Bar Plot)
# ==========================================
plt.figure(figsize=(10, 8))
shap.plots.bar(shap_values, show=False)
plt.title("SHAP 特征重要性绝对排名", fontsize=14, pad=20)
plt.tight_layout()
# 【新增】：保存柱状图
plt.savefig(f'{output_dir}/shap_bar_importance.png', dpi=300, bbox_inches='tight')
plt.show()

# ==========================================
# 画图 3: 局部解释瀑布图 (Waterfall)
# ==========================================
plt.figure(figsize=(10, 8))
# 解剖测试集里第 1 个样本
shap.plots.waterfall(shap_values[0], show=False)
plt.title("单个 MOF 样本预测值的微观推导过程 (Waterfall)", fontsize=14, pad=20)
plt.tight_layout()
# 【新增】：保存瀑布图
plt.savefig(f'{output_dir}/shap_waterfall_sample_0.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\n✅ 所有 SHAP 高清图表已成功保存至 '{output_dir}' 文件夹中！")