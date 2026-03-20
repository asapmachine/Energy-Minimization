import pandas as pd

def update_poav(file_a, file_b, matched_output, unmatched_output):
    print(">>> 开始执行特征更新: A表(cif_name) <== B表(name) 的 POAV 替换...\n")
    
    # ================= 1. 读取并检查 A 表 =================
    try:
        df_a = pd.read_csv(file_a)
        print(f"[1] 读取主表 A 成功，共 {len(df_a)} 行。")
    except Exception as e:
        print(f"读取主表 A 失败: {e}")
        return

    if 'cif_name' not in df_a.columns or 'POAV' not in df_a.columns:
        print("错误：A表中未找到 'cif_name' 或 'POAV' 列！请检查表头。")
        return
        
    # 创建一个专门用于匹配的临时列，剥离空格和可能的 .cif 后缀（不影响原数据）
    df_a['match_key'] = df_a['cif_name'].astype(str).str.strip().str.replace(r'\.cif$', '', regex=True)

    # ================= 2. 读取并检查 B 表 =================
    try:
        df_b = pd.read_csv(file_b, usecols=['name', 'POAV'])
        print(f"[2] 读取源表 B 成功，共提取 {len(df_b)} 行数据。")
    except Exception as e:
        print(f"读取源表 B 失败: {e}")
        return
        
    # 剔除 B 表中 POAV 本身为缺失值的数据，避免用 NaN 覆盖 A 表的有效值
    df_b = df_b.dropna(subset=['POAV'])
    
    # 同样创建干净的临时匹配列，并将 B 表的 POAV 改名为 POAV_b，防止合并时名称冲突
    df_b['match_key'] = df_b['name'].astype(str).str.strip().str.replace(r'\.cif$', '', regex=True)
    df_b = df_b.rename(columns={'POAV': 'POAV_b'})
    
    # 去重：如果 B 表有同名的 MOF，保留第一个
    df_b = df_b.drop_duplicates(subset=['match_key'], keep='first')
    # 丢弃不需要的原始 name 列
    df_b = df_b.drop(columns=['name'])

    # ================= 3. 执行合并与替换 =================
    print("[3] 正在比对名字并替换 POAV...")
    result_df = pd.merge(df_a, df_b, on='match_key', how='left')
    
    # ================= 4. 数据分流 =================
    # 如果 POAV_b 是空的，说明在 B 表里没找到对应的名字
    unmatched_mask = result_df['POAV_b'].isna()
    
    # 【处理未匹配数据】：保留原始 A 表状态，扔掉临时列
    unmatched_df = result_df[unmatched_mask].drop(columns=['match_key', 'POAV_b'])
    
    # 【处理匹配数据】：用 B 表的 POAV_b 强行替换掉 A 表的 POAV
    matched_df = result_df[~unmatched_mask].copy()
    matched_df['POAV'] = matched_df['POAV_b']
    # 替换完成后，扔掉临时列
    matched_df = matched_df.drop(columns=['match_key', 'POAV_b'])
    
    print("\n>>> 更新结果核对：")
    print(f"    - A 表初始总行数: {len(df_a)}")
    print(f"    - 成功匹配并替换 POAV 的行数: {len(matched_df)}")
    print(f"    - 未匹配到 B 表数据的行数: {len(unmatched_df)}")
    
    # ================= 5. 保存结果 =================
    matched_df.to_csv(matched_output, index=False)
    print(f"\n[4] 匹配并更新成功的数据已保存至:\n    {matched_output}")
    
    if len(unmatched_df) > 0:
        unmatched_df.to_csv(unmatched_output, index=False)
        print(f"[5] 未匹配的数据 (保留原A表状态) 已单独提取至:\n    {unmatched_output}")

# ================= 运行区 =================
if __name__ == "__main__":
    # 替换为你实际的路径 (这里填入你的文件路径)
    file_a = r'E:\CODE\cif2des\train_POAV\clean_data\train_data.csv'  # 你的 A 表
    file_b = r'E:\CODE\cif2des\calc\clean_data\goal_data\final_ABC_features_not_optimal.csv' # 你的 B 表
    
    matched_target = r'E:\CODE\cif2des\train_POAV\clean_data\train_data_poav_updated.csv' # 更新好 POAV 的表
    unmatched_error = r'E:\CODE\cif2des\train_POAV\clean_data\unmatched_poav_error.csv' # 没匹配上的表
    
    update_poav(file_a, file_b, matched_target, unmatched_error)