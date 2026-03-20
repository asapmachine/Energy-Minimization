import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. 配置参数 (请根据你的实际情况修改)
# ==========================================
csv_path = r'E:\CODE\cif2des\calc\clean_data\goal_data\RAC_and_geom_1inor_1org_1edge.csv'  # 替换为你实际的优化后特征 CSV 文件路径
target_col = 'VSA'        # 替换为你表格中代表“优化后VSA”的真实列名

print(f"正在读取数据: {csv_path} ...")
# ==========================================
# 2. 数据加载与基础处理
# ==========================================
try:
    df_opt = pd.read_csv(csv_path)
except FileNotFoundError:
    print(f"错误: 找不到文件 '{csv_path}'，请检查路径。")
    exit()

if target_col not in df_opt.columns:
    print(f"错误: 表格中找不到列名 '{target_col}'。当前包含的列有: {df_opt.columns.tolist()}")
    exit()

# 剔除空值 (NaN)，仅用于可视化和统计分析
data = df_opt[target_col].dropna()

# ==========================================
# 3. 核心统计指标输出
# ==========================================
print("\n" + "="*30)
print(f"优化后 {target_col} 的统计描述:")
print("="*30)
print(data.describe())
print("="*30)

# ==========================================
# 4. 绘制数据分布图 (直方图 + 箱线图)
# ==========================================
# 设置绘图风格
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 图1: 直方图与核密度估计 (KDE)
# 观察数据是正态分布、偏态分布，还是在 0 附近有异常堆积
sns.histplot(data, bins=100, kde=True, color="skyblue", ax=axes[0])
axes[0].set_title(f'Distribution of Optimized {target_col}', fontsize=14, fontweight='bold')
axes[0].set_xlabel(f'{target_col} (Å)', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)

# 图2: 箱线图 (Boxplot)
# 快速识别离群点 (Outliers)
sns.boxplot(x=data, color="lightgreen", ax=axes[1])
axes[1].set_title(f'Boxplot of Optimized {target_col}', fontsize=14, fontweight='bold')
axes[1].set_xlabel(f'{target_col} (Å)', fontsize=12)

# 调整布局并保存/显示
plt.tight_layout()
plt.savefig(r'E:\CODE\cif2des\train_VSA\property_png\VSA_distribution_analysis_2.png', dpi=300)
print("\n分布图已保存为 'VSA_distribution_analysis_2.png'。")
plt.show()