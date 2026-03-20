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

def main():
    print("=============== 准备数据与加载模型 ===============")
    # 2. 完美复刻训练时的数据清洗与切分 (必须保证 random_state=42)
    df = pd.read_csv(r'E:\CODE\cif2des\train_GSA\clean_data\train_data_GSA_final.csv')
    features = ['Di', 'Df', 'Dif', 'rho', 'VSA', 'GSA_x', 'VPOV', 'GPOV', 'POAV_vol_frac', 'GPOAV', 'POAV']
    target = 'GSA_y'
    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df[(df[target] > 0) & (df[target] < 8000) & (df['GSA_x'] > 0) & (df['GSA_x'] < 8000)]
    df.dropna(subset=features + [target], inplace=True)
    
    if 'filename' in df.columns:
        df.drop_duplicates(subset=['filename'], inplace=True)

    X = df[features]
    
    # 【核心修复】：为目标变量加上对数转换，确保切分出的 X_test 与训练时完全同频！
    y = np.log1p(df[target])
    
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # 3. 唤醒落盘的模型
    print("正在从硬盘加载已训练好的 GSA 模型...")
    final_model = cb.CatBoostRegressor()
    final_model.load_model('catboost_mof_GSA_predictor_best.cbm')

    # ========================================================= 
    # 阶段七：基于 SHAP 的物理规律可解释性分析
    # ========================================================= 
    print("\n=============== 启动 SHAP 物理规律解析 ===============")
    explainer = shap.TreeExplainer(final_model)
    print("正在计算测试集的 SHAP 值，请稍候... (数据量大时可能需要几十秒)")
    shap_values = explainer(X_test)

    # 创建专属的存图文件夹，防止文件杂乱
    output_dir = 'shap_plots'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("\n⚠️ 提示：因为 GSA 模型是基于对数空间训练的，以下图表中的 SHAP 值 (横坐标) 代表的是【对数 GSA 的变化量】。")

    # ==========================================
    # 画图 1: 全局影响图 (Beeswarm)
    # ==========================================
    print("-> 正在渲染全局影响图...")
    plt.figure(figsize=(10, 8))
    # 更新为最新版 SHAP 推荐的 beeswarm API
    shap.plots.beeswarm(shap_values, show=False)
    plt.title("SHAP 物理特征全局影响图 (Beeswarm)", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/shap_beeswarm.png', dpi=300, bbox_inches='tight') 
    plt.savefig(f'{output_dir}/shap_beeswarm.pdf', bbox_inches='tight') # 保留矢量图
    plt.show()

    # ==========================================
    # 画图 2: 绝对重要性柱状图 (Bar Plot)
    # ==========================================
    print("-> 正在渲染特征重要性柱状图...")
    plt.figure(figsize=(10, 8))
    shap.plots.bar(shap_values, show=False)
    plt.title("SHAP 特征重要性绝对排名", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/shap_bar_importance.png', dpi=300, bbox_inches='tight')
    plt.show()

    # ==========================================
    # 画图 3: 局部解释瀑布图 (Waterfall)
    # ==========================================
    print("-> 正在渲染单样本微观解析瀑布图...")
    plt.figure(figsize=(10, 8))
    shap.plots.waterfall(shap_values[0], show=False)
    plt.title("单个 MOF 样本预测值的微观推导过程 (Waterfall)", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/shap_waterfall_sample_0.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\n✅ 所有 SHAP 高清图表已成功保存至 '{output_dir}' 文件夹中！")

if __name__ == "__main__":
    main()