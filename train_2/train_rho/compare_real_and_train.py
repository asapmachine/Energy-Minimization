import pandas as pd
import numpy as np
import catboost as cb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def main():
    print("=============== 1. 加载数据与严格对齐 ===============")
    csv_path = r'E:\CODE\cif2des\train_2\train_rho\clean_data\A_final_cleaned_data.csv'
    model_path = 'catboost_mof_rho_predictor_best_with_RACs.cbm' 
    
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"❌ 找不到文件 {csv_path}，请手动确认路径！")
        return

    geo_features = ['Di', 'Df', 'Dif', 'rho_x', 'VSA', 'GSA', 'VPOV', 'GPOV', 'POAV_vol_frac', 'GPOAV', 'POAV']
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
    target = 'rho_y'

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df[(df[target] > 0) & (df[target] < 50) & (df['rho_x'] > 0) & (df['rho_x'] < 50)]
    df.dropna(subset=geo_features + [target], inplace=True)
    
    for col in rac_features:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    if 'filename' in df.columns:
        df.drop_duplicates(subset=['filename'], inplace=True)
    elif 'cif_name' in df.columns:
        df.drop_duplicates(subset=['cif_name'], inplace=True)

    valid_features = []
    for col in geo_features + rac_features:
        if col in df.columns and df[col].nunique() > 1:
            valid_features.append(col)

    X = df[valid_features]
    y = df[target]

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    print("=============== 2. 算力对决：基准 vs 机器学习 ===============")
    y_baseline = X_test['rho_x']
    
    base_r2 = r2_score(y_test, y_baseline)
    base_mae = mean_absolute_error(y_test, y_baseline)
    base_rmse = np.sqrt(mean_squared_error(y_test, y_baseline))

    print("正在加载高维增强版模型...")
    model = cb.CatBoostRegressor()
    try:
        model.load_model(model_path)
    except Exception as e:
        print(f"❌ 模型加载失败，请确保 {model_path} 在当前目录下！错误信息: {e}")
        return
    
    y_pred = model.predict(X_test)
    
    model_r2 = r2_score(y_test, y_pred)
    model_mae = mean_absolute_error(y_test, y_pred)
    model_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print("\n" + "🔥" * 25)
    print("     物理形变基准 vs 高维机器学习预测 对比报告 (rho)")
    print("🔥" * 25)
    
    print("\n【未优化基准 (Baseline - rho_x)】:")
    print(f"  -> R²:   {base_r2:.4f}")
    print(f"  -> MAE:  {base_mae:.4f} g/cm³")
    print(f"  -> RMSE: {base_rmse:.4f} g/cm³")

    print("\n【CatBoost 模型预测 (ML + RACs)】:")
    print(f"  -> R²:   {model_r2:.4f}")
    print(f"  -> MAE:  {model_mae:.4f} g/cm³")
    print(f"  -> RMSE: {model_rmse:.4f} g/cm³")

    print("\n" + "=" * 50)
    print(f"🚀 【核心结论：RACs 高维模型带来的价值】 🚀")
    print(f"平均预测误差 (MAE) 缩小了: {base_mae - model_mae:.4f} g/cm³")
    print(f"解释方差 (R²) 提升了:      {(model_r2 - base_r2)*100:.2f} %")
    print("=" * 50)

    print("\n=============== 3. 正在渲染终极对比图 ===============")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    min_val = min(y_test.min(), y_baseline.min(), y_pred.min())
    max_val = max(y_test.max(), y_baseline.max(), y_pred.max())
    
    ax1 = axes[0]
    ax1.scatter(y_test, y_baseline, alpha=0.5, color='#7f8c8d', edgecolors='w', linewidth=0.5)
    ax1.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)
    ax1.set_title("未优化结构 (Baseline: $\\rho_{unopt}$ vs $\\rho_{opt}$)", fontsize=14, pad=15)
    ax1.set_xlabel("真实骨架密度 (Optimized $\\rho$, g/cm³)", fontsize=12)
    ax1.set_ylabel("未优化骨架密度 (Unoptimized $\\rho$, g/cm³)", fontsize=12)
    
    text_base = f"$R^2$ = {base_r2:.4f}\nMAE = {base_mae:.4f}"
    ax1.text(0.05, 0.95, text_base, transform=ax1.transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax2 = axes[1]
    ax2.scatter(y_test, y_pred, alpha=0.6, color='#2980b9', edgecolors='w', linewidth=0.5)
    ax2.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)
    ax2.set_title("高维机器学习预测 (ML+RACs: $\\rho_{pred}$ vs $\\rho_{opt}$)", fontsize=14, pad=15)
    ax2.set_xlabel("真实骨架密度 (Optimized $\\rho$, g/cm³)", fontsize=12)
    ax2.set_ylabel("预测骨架密度 (Predicted $\\rho$, g/cm³)", fontsize=12)
    
    text_model = f"$R^2$ = {model_r2:.4f}\nMAE = {model_mae:.4f}"
    ax2.text(0.05, 0.95, text_model, transform=ax2.transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    
    out_dir = 'evaluation_results'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    save_path = f"{out_dir}/Rho_Baseline_vs_ML_Comparison_with_RACs.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 高清对比图已保存至: {save_path}")
    
    plt.show()

if __name__ == "__main__":
    main()