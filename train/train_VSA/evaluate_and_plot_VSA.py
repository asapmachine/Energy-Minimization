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
    print("步骤一：正在加载数据与特征工程...")
    df = pd.read_csv(r'E:\CODE\cif2des\train_VSA\clean_data\train_data_VSA_final.csv')

    features = ['Di', 'Df', 'Dif', 'rho', 'VSA_x', 'GSA', 'VPOV', 'GPOV', 'POAV_vol_frac', 'GPOAV', 'POAV']
    target = 'VSA_y'

    # 1. 物理清洗 (保持与训练集完全一致)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df[(df[target] > 0) & (df[target] < 2500) & (df['VSA_x'] > 0) & (df['VSA_x'] < 2500)]
    df.dropna(subset=features + [target], inplace=True)

    if 'filename' in df.columns:
        df.drop_duplicates(subset=['filename'], inplace=True)

    # 2. 特征与目标对数化 (极其重要！修复了之前的维度灾难)
    print("-> 正在应用对数转换 (log1p) 匹配模型输入格式...")
    
    # 【修复点】：移除了 df['GSA_x'] = np.log1p(df['GSA_x'])
    # 保持 GSA_x 为原始的 0~8000 尺度，因为你的模型是基于原始尺度训练的！
    
    # 仅对目标变量 y 进行对数转换，因为模型吐出的是对数结果
    y_log = np.log1p(df[target])        

    X = df[features]  # 此时的 X 完美还原了训练时的真实物理尺度
    y = y_log

    # 3. 按照完全相同的随机种子 (42) 进行切分，确保拿到的 X_test 完全一致
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val_log, y_test_log = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    print("步骤二：正在加载模型...")
    model = cb.CatBoostRegressor()
    model.load_model('catboost_mof_VSA_predictor_best.cbm')

    print("步骤三：正在进行对数预测与物理尺度还原...")
    # 此时模型吐出的预测值是 0~9 的对数尺度
    y_pred_log = model.predict(X_test)

    # 使用 expm1 解除封印，将测试集真实标签和预测值全部还原回 0~8000 的真实物理尺度
    y_test_real = np.expm1(y_test_log)
    y_pred_real = np.expm1(y_pred_log)

    # 计算指标 (使用还原后的真实数值)
    final_r2 = r2_score(y_test_real, y_pred_real)
    final_mae = mean_absolute_error(y_test_real, y_pred_real)
    final_rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))

    print(f"🎯 评估结果 (真实物理尺度) -> R²: {final_r2:.4f} | MAE: {final_mae:.4f} | RMSE: {final_rmse:.4f}")

    print("步骤四：👉 正在渲染终极双面图并执行高清保存！👈") 
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    metrics_text = (f"$R^2$ = {final_r2:.4f}\n"
                    f"MAE  = {final_mae:.4f}\n"
                    f"RMSE = {final_rmse:.4f}")

    # 左图 (散点图)
    ax1 = axes[0]
    ax1.scatter(y_test_real, y_pred_real, alpha=0.6, edgecolors='w', linewidth=0.5, color='#1f77b4')
    min_val = min(y_test_real.min(), y_pred_real.min())
    max_val = max(y_test_real.max(), y_pred_real.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='y=x (理想线)')
    ax1.text(0.05, 0.95, metrics_text, transform=ax1.transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.6', facecolor='#f8f9fa', edgecolor='#ced4da', alpha=0.9))
    ax1.set_xlabel("真实值 (Actual VSA, $m^2/g$)", fontsize=12)
    ax1.set_ylabel("预测值 (Predicted VSA, $m^2/g$)", fontsize=12)
    ax1.set_title("散点对比图", fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # 右图 (直方图)
    ax2 = axes[1]
    ax2.hist(y_test_real, bins=30, alpha=0.55, label='真实值', color='#1f77b4', edgecolor='black')
    ax2.hist(y_pred_real, bins=30, alpha=0.55, label='预测值', color='#ff7f0e', edgecolor='black')
    ax2.set_xlabel("VSA 值 ($m^2/g$)", fontsize=12)
    ax2.set_ylabel("频数", fontsize=12)
    ax2.set_title("分布直方图", fontsize=14)
    ax2.legend()
    ax2.grid(axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()

    # 保存逻辑
    save_filename = r'E:\CODE\cif2des\train_VSA\final_result\FINAL_Plot_300dpi.png'
    plt.savefig(save_filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ 大功告成！图表已保存为：{save_filename}")

    plt.show()

if __name__ == "__main__":
    main()