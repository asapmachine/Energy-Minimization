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
    # ⚠️ 请确认你的 Di 训练集实际路径（这里我根据你之前的命名习惯做了推测）
    csv_path = r'E:\CODE\cif2des\train\train_Di\clean_data\train_data_Di_final.csv'
    model_path = 'catboost_mof_Di_predictor_best.cbm' # 你的 Di 模型路径
    
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"❌ 找不到文件 {csv_path}，请手动修改为你电脑上的真实路径！")
        return

    # ⚠️ 确保特征列表与训练模型时一模一样！注意这里是 Di_x
    features = ['Di_x', 'Df', 'Dif', 'rho', 'VSA', 'GSA', 'VPOV', 'GPOV', 'POAV_vol_frac', 'GPOAV', 'POAV']
    target = 'Di_y'

    # 基础物理清洗 (与训练保持绝对一致)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=features + [target], inplace=True)
    if 'filename' in df.columns:
        df.drop_duplicates(subset=['filename'], inplace=True)
    elif 'cif_name' in df.columns:
        df.drop_duplicates(subset=['cif_name'], inplace=True)

    X = df[features]
    y = df[target]

    # 使用固定随机种子切分，确保能拿到与训练时一模一样的 X_test 和 y_test
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    print("=============== 2. 算力对决：基准 vs 机器学习 ===============")
    
    # ---------------------------------------------------------
    # 对手 A：未优化的物理基准 (Baseline) 
    # ---------------------------------------------------------
    y_baseline = X_test['Di_x']
    
    base_r2 = r2_score(y_test, y_baseline)
    base_mae = mean_absolute_error(y_test, y_baseline)
    base_rmse = np.sqrt(mean_squared_error(y_test, y_baseline))

    # ---------------------------------------------------------
    # 对手 B：CatBoost 机器学习模型
    # ---------------------------------------------------------
    print("正在加载训练好的机器学习模型...")
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

    # ==========================================
    # 3. 打印对决结果 (震撼终端输出)
    # ==========================================
    print("\n" + "🔥" * 25)
    print("     物理形变基准 vs 机器学习预测 对比报告 (Di)")
    print("🔥" * 25)
    
    print("\n【未优化基准 (Baseline - Di_x)】:")
    print(f"  -> R²:   {base_r2:.4f}")
    print(f"  -> MAE:  {base_mae:.4f} Å")
    print(f"  -> RMSE: {base_rmse:.4f} Å")

    print("\n【CatBoost 模型预测 (Machine Learning)】:")
    print(f"  -> R²:   {model_r2:.4f}")
    print(f"  -> MAE:  {model_mae:.4f} Å")
    print(f"  -> RMSE: {model_rmse:.4f} Å")

    print("\n" + "=" * 50)
    print(f"🚀 【核心结论：你的 ML 模型带来的价值】 🚀")
    print(f"平均预测误差 (MAE) 缩小了: {base_mae - model_mae:.4f} Å")
    print(f"解释方差 (R²) 提升了:      {(model_r2 - base_r2)*100:.2f} %")
    print("=" * 50)

    # ==========================================
    # 4. 论文级高质量可视化 (双面板对比)
    # ==========================================
    print("\n=============== 3. 正在渲染终极对比图 ===============")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    min_val = min(y_test.min(), y_baseline.min(), y_pred.min())
    max_val = max(y_test.max(), y_baseline.max(), y_pred.max())
    
    # --- 左图：未优化基准 ---
    ax1 = axes[0]
    ax1.scatter(y_test, y_baseline, alpha=0.5, color='#7f8c8d', edgecolors='w', linewidth=0.5)
    ax1.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)
    # 修改标题与物理单位
    ax1.set_title("未优化结构 (Baseline: $Di_{unopt}$ vs $Di_{opt}$)", fontsize=14, pad=15)
    ax1.set_xlabel("真实最大孔径 (Optimized Di, Å)", fontsize=12)
    ax1.set_ylabel("未优化最大孔径 (Unoptimized Di, Å)", fontsize=12)
    
    text_base = f"$R^2$ = {base_r2:.4f}\nMAE = {base_mae:.4f}"
    ax1.text(0.05, 0.95, text_base, transform=ax1.transAxes, fontsize=12, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax1.grid(True, linestyle='--', alpha=0.5)

    # --- 右图：机器学习预测 ---
    ax2 = axes[1]
    ax2.scatter(y_test, y_pred, alpha=0.6, color='#2980b9', edgecolors='w', linewidth=0.5)
    ax2.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)
    # 修改标题与物理单位
    ax2.set_title("机器学习预测 (CatBoost: $Di_{pred}$ vs $Di_{opt}$)", fontsize=14, pad=15)
    ax2.set_xlabel("真实最大孔径 (Optimized Di, Å)", fontsize=12)
    ax2.set_ylabel("预测最大孔径 (Predicted Di, Å)", fontsize=12)
    
    text_model = f"$R^2$ = {model_r2:.4f}\nMAE = {model_mae:.4f}"
    ax2.text(0.05, 0.95, text_model, transform=ax2.transAxes, fontsize=12, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    
    # 保存图片 (文件名修改为 Di)
    out_dir = 'evaluation_results'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    save_path = f"{out_dir}/Di_Baseline_vs_ML_Comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 高清对比图已保存至: {save_path}")
    
    plt.show()

if __name__ == "__main__":
    main()