import pandas as pd


def clean_csv_empty_rows(input_csv, output_csv):
    """
    读取 CSV 文件，删除 Di, Df, GSA, VSA 四列全部为空值的行，
    并将清洗后的数据保存为新的 CSV 文件。
    """
    # 1. 加载原始数据
    try:
        df = pd.read_csv(input_csv)
        original_count = len(df)
        print(f"原始数据共 {original_count} 行。")
    except FileNotFoundError:
        print(f"错误: 找不到 CSV 文件 '{input_csv}'。")
        return

    # 2. 核心清洗逻辑：使用 dropna 剔除空值行
    # subset: 指定我们要检查哪几列
    # how='all': 表示只有当 subset 指定的这几列【全部】为空值时，才删除该行
    cleaned_df = df.dropna(subset=['Di', 'Df', 'GSA', 'VSA'], how='all')

    # 3. 统计清洗成果
    cleaned_count = len(cleaned_df)
    deleted_count = original_count - cleaned_count

    if deleted_count > 0:
        print(f"成功删除了 {deleted_count} 行全为空值的异常数据！")

        # 4. 保存清洗后的数据（建议存为新文件，保留原始数据作为备份）
        cleaned_df.to_csv(output_csv, index=False)
        print(f"清洗后的数据已保存至: {output_csv}，当前剩余 {cleaned_count} 行有效数据。")
    else:
        print("没有发现需要删除的空值行，数据已经很干净了。")


# ==========================================
# 运行区域
# ==========================================
if __name__ == "__main__":
    INPUT_CSV = r'E:\CODE\cif2des\zeo_features_only.csv'  # 包含空值的原始 CSV
    OUTPUT_CSV = r'E:\CODE\cif2des\clean_data\zeo_features_cleaned_1.csv'  # 清洗后生成的新 CSV

    clean_csv_empty_rows(INPUT_CSV, OUTPUT_CSV)