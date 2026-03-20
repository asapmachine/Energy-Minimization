import os
import glob
import pandas as pd

def merge_csvs_in_folder(folder_c, output_file):
    """
    遍历文件夹 C 及其所有子文件夹，精准抓取 *_descriptors.csv 文件，
    合并为一个大的 CSV，并进行数据去重清洗。
    """
    # 1. 安全检查
    if not os.path.exists(folder_c):
        print(f"错误: 找不到指定的文件夹 '{folder_c}'。请检查路径。")
        return

    print(f"开始在 '{folder_c}' 及其子文件夹中精准搜索特征 CSV 文件...")
    
    # 2. 核心修改 1：精准打击
    # 将 '*.csv' 改为 '*_descriptors.csv'，完美避开 geometric_parameters.csv 的干扰
    search_pattern = os.path.join(folder_c, '**', '*_descriptors.csv')
    csv_files = glob.glob(search_pattern, recursive=True)

    if not csv_files:
        print(f"在 '{folder_c}' 中没有找到任何 '_descriptors.csv' 文件！")
        return

    print(f"共找到 {len(csv_files)} 个目标文件，开始读取并合并...")

    # 3. 逐个读取并存入列表
    all_dfs = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            if not df.empty:
                all_dfs.append(df)
        except Exception as e:
            print(f"  [警告] 读取文件失败 {file}: {e}")

    # 4. 执行合并与数据清洗
    if all_dfs:
        # 一次性合并所有表格
        final_df = pd.concat(all_dfs, ignore_index=True)

        # 调整表头：把 'name' 和 'cif_file' 移到最左侧
        cols = final_df.columns.tolist()
        if 'cif_file' in cols:
            cols.insert(0, cols.pop(cols.index('cif_file')))
        if 'name' in cols:
            cols.insert(0, cols.pop(cols.index('name')))
        
        final_df = final_df[cols]
        
        # 5. 核心修改 2：双保险去重
        # 以防万一某些文件夹里混入了重复计算的结果，基于 'name' 列强行去重
        if 'name' in final_df.columns:
            original_len = len(final_df)
            final_df = final_df.drop_duplicates(subset=['name'], keep='first')
            dedup_len = len(final_df)
            
            if original_len > dedup_len:
                print(f"  [提示] 触发双保险：成功清理了 {original_len - dedup_len} 行由于历史残留导致的重复数据！")

            # 按照 name 列进行字母表排序，让最终表格井井有条
            final_df = final_df.sort_values(by=['name'])

        # 6. 保存最终结果
        final_df.to_csv(output_file, index=False)
        print(f"\n完美！合并与去重全部完成。最终的表格包含 {len(final_df)} 行独立且有效的数据。")
        print(f"已保存为: {output_file}")
    else:
        print("所有找到的 CSV 文件都是空的，未能生成合并文件。")

# ==========================================
# 运行区域（请根据您的实际情况修改这里的路径）
# ==========================================
if __name__ == "__main__":
    DIR_C = r'E:\CODE\cif2des\error\remain'                                # 存放提取出来的文件夹的 C 文件夹路径
    OUTPUT_CSV = r'E:\CODE\cif2des\error\merged_features.csv' 
    
    merge_csvs_in_folder(DIR_C, OUTPUT_CSV)