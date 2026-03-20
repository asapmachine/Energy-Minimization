import pandas as pd

def transfer_and_split_data(csv_a_path, source_csvs, matched_output, unmatched_output):
    """
    将 BCD 表的 Di 列转移到 A 表。
    匹配规则：A表的 'name' 列  <==>  BCD表的 'filename' 列。
    最终将匹配成功与失败的数据分流保存。
    """
    print(">>> 开始执行跨表头数据转移与分流任务...\n")
    
    # 1. 读取并清洗主表 A
    try:
        df_a = pd.read_csv(csv_a_path)
        print(f"[1] 成功读取主表 A，共 {len(df_a)} 行数据。")
    except Exception as e:
        print(f"读取主表 A 失败: {e}")
        return

    if 'name' not in df_a.columns:
        print("错误：主表 A 中未找到 'name' 表头，请检查文件。")
        return
    # 清理主表 A 的匹配键
    df_a['name'] = df_a['name'].astype(str).str.strip()
    
    # 2. 读取 B, C, D 数据源
    source_dfs = []
    for file in source_csvs:
        try:
            df_temp = pd.read_csv(file, usecols=['filename', 'POAV_vol_frac'])
            source_dfs.append(df_temp)
        except Exception as e:
            print(f"    - 读取源文件 {file} 出错: {e}")
            
    if not source_dfs:
        print("未成功读取任何 BCD 数据源，程序终止。")
        return
        
    # 3. 数据源的合并、清洗与重命名
    combined_source = pd.concat(source_dfs, ignore_index=True)
    combined_source = combined_source.dropna(subset=['POAV_vol_frac'])
    combined_source = combined_source.drop_duplicates(subset=['filename'], keep='first')
    
    combined_source['filename'] = combined_source['filename'].astype(str).str.strip()
    
    # 在合并前，将源表的 'filename' 统一重命名为主表的 'name'
    combined_source = combined_source.rename(columns={
        'filename': 'name', 
        'POAV_vol_frac': 'POAV_vol_frac_y'
    })
    print(f"[2] 整合 BCD 数据源完毕，共提取了 {len(combined_source)} 个唯一的匹配键。")
    
    # 4. 执行同名键合并 (Left Join)
    result_df = pd.merge(df_a, combined_source, on='name', how='left')
    
    # 5. 【新增核心逻辑】：数据分流 (分离匹配与未匹配的数据)
    # 创建一个布尔掩码：如果 'Di_已匹配' 是空值，则为 True
    unmatched_mask = result_df['POAV_vol_frac_y'].isna()
    
    # 利用掩码将数据分为两部分
    unmatched_df = result_df[unmatched_mask]       # 提取未匹配的 30 行
    matched_df = result_df[~unmatched_mask]      # 提取成功匹配的 46944 行 (~ 表示取反)
    
    print("\n>>> 匹配结果核对与分流：")
    print(f"    - 主表 A 总行数: {len(result_df)}")
    print(f"    - 成功匹配并保留的行数: {len(matched_df)}")
    print(f"    - 未找到对应数据的行数: {len(unmatched_df)}")
    
    # 6. 分别保存结果
    # 保存匹配成功的数据
    matched_df.to_csv(matched_output, index=False)
    print(f"\n[3] 任务完成！匹配成功的干净数据已保存至: {matched_output}")
    
    # 保存未匹配的数据（如果存在）
    if len(unmatched_df) > 0:
        unmatched_df.to_csv(unmatched_output, index=False)
        print(f"[4] 注意：未匹配的 {len(unmatched_df)} 条异常数据已单独提取至: {unmatched_output}")

# ================= 运行示例 =================
if __name__ == "__main__":
    # 请根据你实际的文件路径进行替换
    file_a = r'E:\CODE\cif2des\train_void\clean_data\train_data.csv'
    # 数据源表 B, C, D 的路径列表
    files_bcd = [r'E:\CODE\cif2des\calc\clean_data\goal_data\RAC_and_geom_1inor_1edge.csv', r'E:\CODE\cif2des\calc\clean_data\goal_data\RAC_and_geom_1inor_1org_1edge.csv', r'E:\CODE\cif2des\calc\clean_data\goal_data\RAC_and_geom_2inor_1edge.csv']
    # 成功匹配的最终文件（即你需要的 train_data.csv）
    matched_target_file = r'E:\CODE\cif2des\train_void\clean_data\train_data_void.csv'
    
    # 没匹配上的 30 个文件单独存放在这里，供人工排查
    unmatched_error_file = r'E:\CODE\cif2des\train_void\clean_data\unmatched_30_error.csv'
    
    transfer_and_split_data(file_a, files_bcd, matched_target_file, unmatched_error_file)