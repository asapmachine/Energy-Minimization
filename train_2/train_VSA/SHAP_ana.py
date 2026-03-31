import pandas as pd
import numpy as np
import catboost as cb         
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def main():
    print("=============== 准备高维数据与加载模型 ===============")
    df = pd.read_csv(r'E:\CODE\cif2des\train_2\train_VSA\clean_data\A_final_cleaned_data.csv')
    
    geo_features = ['Di', 'Df', 'Dif', 'rho', 'VSA_x', 'GSA', 'VPOV', 'GPOV', 'POAV_vol_frac', 'GPOAV', 'POAV']
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
    target = 'VSA_y'
    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df[(df[target] > 0) & (df[target] < 2500) & (df['VSA_x'] > 0) & (df['VSA_x'] < 2500)]
    df.dropna(subset=geo_features + [target], inplace=True)
    
    for col in rac_features:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    if 'filename' in df.columns:
        df.drop_duplicates(subset=['filename'], inplace=True)

    valid_features = []
    for col in geo_features + rac_features:
        if col in df.columns and df[col].nunique() > 1:
            valid_features.append(col)

    X = df[valid_features]
    y = np.log1p(df[target])
    
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    print("正在从硬盘加载已训练好的高维 VSA 模型...")
    final_model = cb.CatBoostRegressor()
    final_model.load_model('catboost_mof_VSA_predictor_best_with_RACs.cbm')

    print("\n=============== 启动 SHAP 物理规律解析 ===============")
    explainer = shap.TreeExplainer(final_model)
    print("正在计算高维测试集的 SHAP 值，请稍候...")
    shap_values = explainer(X_test)

    output_dir = 'shap_plots_with_RACs'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("\n⚠️ 提示：图表中的 SHAP 值 (横坐标) 代表的是【对数 VSA 的变化量】。")

    print("-> 正在渲染全局影响图...")
    plt.figure(figsize=(10, 8))
    shap.plots.beeswarm(shap_values, show=False, max_display=20)
    plt.title("SHAP 物理与化学特征全局影响图 (Top 20)", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/shap_beeswarm.png', dpi=300, bbox_inches='tight') 
    plt.show()

    print("-> 正在渲染特征重要性柱状图...")
    plt.figure(figsize=(10, 8))
    shap.plots.bar(shap_values, show=False, max_display=20)
    plt.title("SHAP 特征重要性绝对排名 (Top 20)", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/shap_bar_importance.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("-> 正在渲染单样本微观解析瀑布图...")
    plt.figure(figsize=(10, 8))
    shap.plots.waterfall(shap_values[0], show=False, max_display=15)
    plt.title("单个 MOF 样本预测值的微观推导过程", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/shap_waterfall_sample_0.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\n✅ 所有 SHAP 高清图表已成功保存至 '{output_dir}' 文件夹中！")

if __name__ == "__main__":
    main()