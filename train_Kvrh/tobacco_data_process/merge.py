import pandas as pd

def transfer_with_specific_order(file_a_path, file_b_path, output_path):
    # 1. 读取表格 A，保留 'cif_file' 和 'KVRH'
    df_a_raw = pd.read_csv(file_a_path)
    if not {'cif_file', 'KVRH'}.issubset(df_a_raw.columns):
        print("错误：表格 A 中缺少 'cif_file' 或 'KVRH' 列")
        return
    df_a = df_a_raw[['cif_file', 'KVRH']]

    # 2. 读取表格 B
    df_b = pd.read_csv(file_b_path)
    if 'cif_file' not in df_b.columns:
        print("错误：表格 B 中找不到 'cif_file' 列")
        return
    if 'name' not in df_b.columns:
        print("错误：表格 B 中找不到 'name' 列，无法执行后续操作")
        return

    # 3. 合并数据 (Left Join)
    # 合并后，KVRH 默认会排在比较靠前的位置
    result = pd.merge(df_a, df_b, on='cif_file', how='left')

    # 4. 清理：删除 'name' 为空的行
    result.dropna(subset=['name'], inplace=True)
    # 进一步清理可能的空字符串
    result = result[result['name'].astype(str).str.strip() != ""]

    # 5. 核心：调整列顺序，将 'KVRH' 放到 'name' 后面
    cols = list(result.columns)
    # 先把 KVRH 从列表中移除
    cols.remove('KVRH')
    # 找到 name 所在的位置，并在其后插入 KVRH
    name_index = cols.index('name')
    cols.insert(name_index + 1, 'KVRH')
    
    # 重新按这个顺序排列 DataFrame
    result = result[cols]

    # 6. 保存与反馈
    result.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"--- 处理完成 ---")
    print(f"最终表格列顺序: {' -> '.join(result.columns[:5])} ... (共{len(result.columns)}列)")
    print(f"最终保留有效行数: {len(result)}")
    print(f"文件已保存至: {output_path}")

# --- 配置区 ---
path_a = r'E:\CODE\cif2des\train_Kvrh\tobacco_data_process\total_tobacco_new.csv'
path_b = r'E:\CODE\cif2des\train_Kvrh\tobacco_data_process\ToBaCCo.csv'
path_output = r'E:\CODE\cif2des\train_Kvrh\tobacco_data_process\final_result.csv'

transfer_with_specific_order(path_a, path_b, path_output)