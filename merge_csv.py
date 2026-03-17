import pandas as pd

def merge_and_deduplicate_csv(file_list, output_file):
    """
    读取多个CSV文件，合并并根据 'cif_file' 列进行去重
    """
    df_list = []
    
    # 1. 遍历读取所有的 CSV 文件
    for file in file_list:
        try:
            # 读取 CSV 文件
            df = pd.read_csv(file)
            df_list.append(df)
            print(f"成功读取: {file}, 包含 {len(df)} 行数据。")
        except Exception as e:
            print(f"读取文件 {file} 时出错: {e}")
            
    if not df_list:
        print("没有成功读取任何数据，程序结束。")
        return

    # 2. 将所有读取的 DataFrame 进行纵向合并
    merged_df = pd.concat(df_list, ignore_index=True)
    print(f"\n合并后总数据量: {len(merged_df)} 行。")

    # 3. 针对 'cif_file' 列进行查重与去重
    # keep='first' 表示如果出现重复，保留第一次出现的数据行
    deduplicated_df = merged_df.drop_duplicates(subset=['cif_file'], keep='first')
    
    # 计算被清理的重复行数
    duplicates_removed = len(merged_df) - len(deduplicated_df)
    print(f"根据 'cif_file' 列查重，共发现并移除了 {duplicates_removed} 行重复数据。")
    print(f"去重后最终数据量: {len(deduplicated_df)} 行。")

    # 4. 将最终结果输出为新的 CSV 文件
    # index=False 表示不保存 DataFrame 的行索引
    deduplicated_df.to_csv(output_file, index=False)
    print(f"\n处理完成！合并且去重后的文件已保存至: {output_file}")

# ================= 运行示例 =================
if __name__ == "__main__":
    # 请将这里的列表替换为你实际的三个 CSV 文件名或路径
    input_files = [r'E:\CODE\cif2des\clean_data\zeo_features_cleaned_1.csv', r'E:\CODE\cif2des\clean_data\zeo_features_cleaned_2.csv', r'E:\CODE\cif2des\clean_data\zeo_features_cleaned_3.csv']
    # 指定输出的合并文件名
    output_filename = r'E:\CODE\cif2des\clean_data\merged_features.csv'

    merge_and_deduplicate_csv(input_files, output_filename)