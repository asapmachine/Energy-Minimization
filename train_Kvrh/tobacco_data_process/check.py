import pandas as pd

def compare_column_names(file_a_path, file_b_path):
    # 1. 读取两个表格的表头
    # 只读取前 0 行，这样速度极快，不需要加载整个大文件
    df_a = pd.read_csv(file_a_path, nrows=0)
    df_b = pd.read_csv(file_b_path, nrows=0)

    # 2. 获取列名集合
    cols_a = set(df_a.columns)
    cols_b = set(df_b.columns)

    # 3. 计算差异
    only_in_a = cols_a - cols_b  # 在 A 中有，B 中没有
    only_in_b = cols_b - cols_a  # 在 B 中有，A 中没有
    common_cols = cols_a & cols_b # 两个表共有的列

    # 4. 打印结果
    print("--- 表格列名对比报告 ---")
    print(f"表格 A 总列数: {len(cols_a)}")
    print(f"表格 B 总列数: {len(cols_b)}")
    print(f"共有列数: {len(common_cols)}")
    print("-" * 30)

    if only_in_a:
        print(f"🚩 仅在表格 A 中存在的列 ({len(only_in_a)}个):")
        print(f"   {sorted(list(only_in_a))}")
    else:
        print("✅ 表格 A 的所有列在表格 B 中都能找到。")

    print("-" * 30)

    if only_in_b:
        print(f"🚩 仅在表格 B 中存在的列 ({len(only_in_b)}个):")
        print(f"   {sorted(list(only_in_b))}")
    else:
        print("✅ 表格 B 的所有列在表格 A 中都能找到。")

# --- 运行配置 ---
path_a = r'E:\CODE\cif2des\train_Kvrh\tobacco_data_process\final_result.csv'
path_b = r'E:\CODE\cif2des\train_Kvrh\train_feature_files\net_short_symb_combined_data_frame_all.csv'

compare_column_names(path_a, path_b)