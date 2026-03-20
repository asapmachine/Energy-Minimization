import pandas as pd

# ==================== 配置区 ====================
INPUT_CSV = "E:\CODE\cif2des\calc\clean_data\origin_ABC_features_optimal.csv"      # 你合并出来的原始文件路径
OUTPUT_CSV = r"E:\CODE\cif2des\calc\clean_data\goal_data\final_ABC_features_optimal.csv"   # 清洗后保存的新文件路径

# 需要检查的列名 (如果你的表格里叫 PV_cm3_g，请改为 'PV_cm3_g')
TARGET_COLUMNS = ['rho', 'POAV'] 
# ================================================

def clean_data():
    print(f"正在读取原始数据: {INPUT_CSV}")
    try:
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        print(f"错误: 找不到文件 '{INPUT_CSV}'，请检查路径。")
        return

    original_count = len(df)
    print(f"原始数据行数: {original_count}")

    # 第一步：检查目标列是否存在
    missing_cols = [col for col in TARGET_COLUMNS if col not in df.columns]
    if missing_cols:
        print(f"错误: 在 CSV 中找不到以下列: {missing_cols}")
        print(f"当前 CSV 拥有的列名为: {list(df.columns)}")
        return

    # 第二步：删除包含空白 (NaN/Null) 的行
    # subset 参数确保只检查指定的列
    df_cleaned = df.dropna(subset=TARGET_COLUMNS)

    # 第三步：删除数值等于 0 (或 0.0) 的行
    for col in TARGET_COLUMNS:
        # 仅保留该列数值不等于 0 的行
        df_cleaned = df_cleaned[df_cleaned[col] != 0]

    # 第四步：保存清洗后的数据
    cleaned_count = len(df_cleaned)
    removed_count = original_count - cleaned_count

    df_cleaned.to_csv(OUTPUT_CSV, index=False)

    print("\n================ 清洗完成 ================")
    print(f"剔除了 {removed_count} 行含有空白或 0 的异常数据。")
    print(f"剩余有效数据行数: {cleaned_count}")
    print(f"清洗后的纯净数据已保存至: {OUTPUT_CSV}")

if __name__ == "__main__":
    clean_data()