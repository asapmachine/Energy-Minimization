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
    print("步骤一：正在加载数据...")
    df = pd.read_csv(r'E:\CODE\cif2des\train_Df\clean_data_Df\train_data_Df_final.csv')

    features = ['Di', 'Df_x', 'Dif', 'rho', 'VSA', 'GSA', 'VPOV', 'GPOV', 'POAV_vol_frac', 'GPOAV', 'POAV']
    target = 'Df_y'

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df[(df[target] > 0) & (df[target] < 100) & (df['Df_x'] > 0) & (df['Df_x'] < 100)]
    df.dropna(subset=features + [target], inplace=True)

    if 'filename' in df.columns:
        df.drop_duplicates(subset=['filename'], inplace=True)

    X = df[features]
    y = df[target]

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    print("步骤二：正在加载模型...")
    model = cb.CatBoostRegressor()
    model.load_model('catboost_mof_df_predictor_best.cbm')

    print("步骤三：正在计算指标...")
    y_pred = model.predict(X_test)
    final_r2 = r2_score(y_test, y_pred)
    final_mae = mean_absolute_error(y_test, y_pred)
    final_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"🎯 评估结果 -> R²: {final_r2:.4f} | MAE: {final_mae:.4f} | RMSE: {final_rmse:.4f}")

    print("步骤四：👉 正在渲染终极双面图并执行高清保存！👈") # 如果终端打印出这句话，说明代码用对了！
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    metrics_text = (f"$R^2$ = {final_r2:.4f}\n"
                    f"MAE  = {final_mae:.4f}\n"
                    f"RMSE = {final_rmse:.4f}")

    # 左图
    ax1 = axes[0]
    ax1.scatter(y_test, y_pred, alpha=0.6, edgecolors='w', linewidth=0.5, color='#1f77b4')
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='y=x (理想线)')
    ax1.text(0.05, 0.95, metrics_text, transform=ax1.transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.6', facecolor='#f8f9fa', edgecolor='#ced4da', alpha=0.9))
    ax1.set_xlabel("真实值 (Actual Df)", fontsize=12)
    ax1.set_ylabel("预测值 (Predicted Df)", fontsize=12)
    ax1.set_title("散点对比图", fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # 右图
    ax2 = axes[1]
    ax2.hist(y_test, bins=30, alpha=0.55, label='真实值', color='#1f77b4', edgecolor='black')
    ax2.hist(y_pred, bins=30, alpha=0.55, label='预测值', color='#ff7f0e', edgecolor='black')
    ax2.set_xlabel("Df 值", fontsize=12)
    ax2.set_ylabel("频数", fontsize=12)
    ax2.set_title("分布直方图", fontsize=14)
    ax2.legend()
    ax2.grid(axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()

    # 保存逻辑
    save_filename = 'FINAL_Plot_300dpi.png'
    plt.savefig(save_filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ 大功告成！图表已保存为：{save_filename}")

    plt.show()

if __name__ == "__main__":
    main()