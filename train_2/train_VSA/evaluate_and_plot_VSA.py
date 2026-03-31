import pandas as pd
import numpy as np
import catboost as cb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# 设置绘图支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def main():
    print("步骤一：正在加载数据与高维特征工程...")
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

    # 1. 物理清洗 
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df[(df[target] > 0) & (df[target] < 2500) & (df['VSA_x'] > 0) & (df['VSA_x'] < 2500)]
    df.dropna(subset=geo_features + [target], inplace=True)

    # 填充 RACs
    for col in rac_features:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    if 'filename' in df.columns:
        df.drop_duplicates(subset=['filename'], inplace=True)

    # 剔除死特征
    valid_features = []
    for col in geo_features + rac_features:
        if col in df.columns and df[col].nunique() > 1:
            valid_features.append(col)

    # 2. 特征与目标对数化
    print("-> 正在应用对数转换 (log1p) 匹配模型输入格式...")
    y_log = np.log1p(df[target])        

    X = df[valid_features]
    y = y_log

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val_log, y_test_log = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    print("步骤二：正在加载高维模型...")
    model = cb.CatBoostRegressor()
    model.load_model('catboost_mof_VSA_predictor_best_with_RACs.cbm')

    print("步骤三：正在进行对数预测与物理尺度还原...")
    y_pred_log = model.predict(X_test)
    y_test_real = np.expm1(y_test_log)
    y_pred_real = np.expm1(y_pred_log)

    final_r2 = r2_score(y_test_real, y_pred_real)
    final_mae = mean_absolute_error(y_test_real, y_pred_real)
    final_rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))

    print(f"🎯 评估结果 (真实物理尺度) -> R²: {final_r2:.4f} | MAE: {final_mae:.4f} | RMSE: {final_rmse:.4f}")

    print("步骤四：👉 正在渲染终极双面图并执行高清保存！👈") 
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    metrics_text = (f"$R^2$ = {final_r2:.4f}\n"
                    f"MAE  = {final_mae:.4f}\n"
                    f"RMSE = {final_rmse:.4f}")

    ax1 = axes[0]
    ax1.scatter(y_test_real, y_pred_real, alpha=0.6, edgecolors='w', linewidth=0.5, color='#1f77b4')
    min_val = min(y_test_real.min(), y_pred_real.min())
    max_val = max(y_test_real.max(), y_pred_real.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='y=x (理想线)')
    ax1.text(0.05, 0.95, metrics_text, transform=ax1.transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.6', facecolor='#f8f9fa', edgecolor='#ced4da', alpha=0.9))
    ax1.set_xlabel("真实值 (Actual VSA, m²/cm³)", fontsize=12)
    ax1.set_ylabel("预测值 (Predicted VSA, m²/cm³)", fontsize=12)
    ax1.set_title("散点对比图 (融合 RACs 特征)", fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2 = axes[1]
    ax2.hist(y_test_real, bins=30, alpha=0.55, label='真实值', color='#1f77b4', edgecolor='black')
    ax2.hist(y_pred_real, bins=30, alpha=0.55, label='预测值', color='#ff7f0e', edgecolor='black')
    ax2.set_xlabel("VSA 值 (m²/cm³)", fontsize=12)
    ax2.set_ylabel("频数", fontsize=12)
    ax2.set_title("分布直方图", fontsize=14)
    ax2.legend()
    ax2.grid(axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()

    save_filename = r'E:\CODE\cif2des\train_2\train_VSA\final_result\FINAL_Plot_with_RACs_300dpi.png'
    plt.savefig(save_filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ 大功告成！图表已保存为：{save_filename}")

    plt.show()

if __name__ == "__main__":
    main()