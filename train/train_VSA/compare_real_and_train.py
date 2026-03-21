import pandas as pd
import numpy as np
import catboost as cb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import os

# ==========================================
# 0. 绘图与基础配置
# ==========================================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def main():
    print("=============== 1. 加载数据与严格对齐 ===============")
    # ⚠️ 匹配了你的 VSA 数据集路径
    csv_path = r'E:\CODE\cif2des\train\train_VSA\clean_data\train_data_VSA_final.csv'
    model_path = 'catboost_mof_VSA_predictor_best.cbm' # 你的 VSA 模型路径
    
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"❌ 找不到文件 {csv_path}，请手动确认路径！")
        return

    # ⚠️ 确保特征包含 VSA_x
    features = ['Di', 'Df', 'Dif', 'rho', 'VSA_x', 'GSA', 'VPOV', 'GPOV', 'POAV_vol_frac', 'GPOAV', 'POAV']
    target = 'VSA_y'

    # 1. 物理清洗 (去除了 < 2500 的范围限制，仅保留 > 0 的物理底线)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df[(df[target] > 0) & (df['VSA_x'] > 0)]
    df.dropna(subset=features + [target], inplace=True)
    
    if 'filename' in df.columns:
        df.drop_duplicates(subset=['filename'], inplace=True)
    elif 'cif_name' in df.columns:
        df.drop_duplicates(subset=['cif_name'], inplace=True)

    # 此时的 X 包含了所有的 VSA_x
    X = df[features]
    # 提取原始的物理 VSA_y
    y_real = df[target]

    # 使用相同的随机种子进行切分，确保拿到与训练时绝对一致的 X_test 和 y_test_real
    X_train, X_temp, y_train_real, y_temp_real = train_test_split(X, y_real, test_size=0.2, random_state=42)
    X_val, X_test, y_val_real, y_test_real = train_test_split(X_temp, y_temp_real, test_size=0.5, random_state=42)

    print("=============== 2. 算力对决：基准 vs 机器学习 ===============")
    
    # ---------------------------------------------------------
    # 对手 A：未优化的物理基准 (Baseline)
    # ---------------------------------------------------------
    y_baseline = X_test['VSA_x']
    
    base_r2 = r2_score(y_test_real, y_baseline)
    base_mae = mean_absolute_error(y_test_real, y_baseline)
    base_rmse = np.sqrt(mean_squared_error(y_test_real, y_baseline))

    # ---------------------------------------------------------
    # 对手 B：CatBoost 机器学习模型
    # ---------------------------------------------------------
    print("正在加载训练好的 VSA 机器学习模型...")
    model = cb.CatBoostRegressor()
    try:
        model.load_model(model_path)
    except Exception as e:
        print(f"❌ 模型加载失败，请确保 {model_path} 在当前目录下！错误信息: {e}")
        return
    
    # ⚠️ 模型预测 (0~7.82的对数结果)
    y_pred_log = model.predict(X_test)
    # 还原回真实体积比表面积尺度
    y_pred_real = np.expm1(y_pred_log)
    
    # 计算还原后的真实误差
    model_r2 = r2_score(y_test_real, y_pred_real)
    model_mae = mean_absolute_error(y_test_real, y_pred_real)
    model_rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))

    # ==========================================
    # 3. 打印对决结果 (震撼终端输出)
    # ==========================================
    print("\n" + "🔥" * 25)
    print("     物理形变基准 vs 机器学习预测 对比报告 (VSA - 无上限版本)")
    print("🔥" * 25)
    
    print("\n【未优化基准 (Baseline - VSA_x)】:")
    print(f"  -> R²:   {base_r2:.4f}")
    print(f"  -> MAE:  {base_mae:.4f} m²/cm³")
    print(f"  -> RMSE: {base_rmse:.4f} m²/cm³")

    print("\n【CatBoost 模型预测 (Machine Learning)】:")
    print(f"  -> R²:   {model_r2:.4f}")
    print(f"  -> MAE:  {model_mae:.4f} m²/cm³")
    print(f"  -> RMSE: {model_rmse:.4f} m²/cm³")

    print("\n" + "=" * 50)
    print(f"🚀 【核心结论：你的 ML 模型带来的价值】 🚀")
    print(f"平均预测误差 (MAE) 缩小了: {base_mae - model_mae:.4f} m²/cm³")
    print(f"解释方差 (R²) 提升了:      {(model_r2 - base_r2)*100:.2f} %")
    print("=" * 50)

    # ==========================================
    # 4. 论文级高质量可视化 (双面板对比)
    # ==========================================
    print("\n=============== 3. 正在渲染终极对比图 ===============")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    min_val = min(y_test_real.min(), y_baseline.min(), y_pred_real.min())
    max_val = max(y_test_real.max(), y_baseline.max(), y_pred_real.max())
    
    # --- 左图：未优化基准 ---
    ax1 = axes[0]
    ax1.scatter(y_test_real, y_baseline, alpha=0.5, color='#7f8c8d', edgecolors='w', linewidth=0.5)
    ax1.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)
    ax1.set_title("未优化结构 (Baseline: $VSA_{unopt}$ vs $VSA_{opt}$)", fontsize=14, pad=15)
    ax1.set_xlabel("真实体积比表面积 (Optimized VSA, m²/cm³)", fontsize=12)
    ax1.set_ylabel("未优化体积比表面积 (Unoptimized VSA, m²/cm³)", fontsize=12)
    
    text_base = f"$R^2$ = {base_r2:.4f}\nMAE = {base_mae:.4f}"
    ax1.text(0.05, 0.95, text_base, transform=ax1.transAxes, fontsize=12, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax1.grid(True, linestyle='--', alpha=0.5)

    # --- 右图：机器学习预测 ---
    ax2 = axes[1]
    ax2.scatter(y_test_real, y_pred_real, alpha=0.6, color='#2980b9', edgecolors='w', linewidth=0.5)
    ax2.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)
    ax2.set_title("机器学习预测 (CatBoost: $VSA_{pred}$ vs $VSA_{opt}$)", fontsize=14, pad=15)
    ax2.set_xlabel("真实体积比表面积 (Optimized VSA, m²/cm³)", fontsize=12)
    ax2.set_ylabel("预测体积比表面积 (Predicted VSA, m²/cm³)", fontsize=12)
    
    text_model = f"$R^2$ = {model_r2:.4f}\nMAE = {model_mae:.4f}"
    ax2.text(0.05, 0.95, text_model, transform=ax2.transAxes, fontsize=12, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    
    # 保存图片
    out_dir = 'evaluation_results'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    save_path = f"{out_dir}/VSA_Baseline_vs_ML_Comparison_NoLimit.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 高清对比图已保存至: {save_path}")
    
    plt.show()

if __name__ == "__main__":
    main()