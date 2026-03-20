import pandas as pd

def merge_and_compare_rho(file_a, file_b, matched_output, unmatched_output):
    print(">>> 开始执行特征合并: A表(cif_name) <==> B表(name)...\n")
    
    # ================= 1. 处理 A 表 =================
    try:
        df_a = pd.read_csv(file_a)
    except Exception as e:
        print(f"读取主表 A 失败: {e}")
        return

    df_a['cif_name'] = df_a['cif_name'].astype(str).str.strip()
    df_a['cif_name'] = df_a['cif_name'].str.replace(r'\.cif$', '', regex=True)
    
    if 'rho' in df_a.columns:
        df_a = df_a.rename(columns={'rho': 'rho_x'})

    # ================= 2. 处理 B 表 =================
    try:
        df_b = pd.read_csv(file_b, usecols=['name', 'rho'])
    except Exception as e:
        print(f"读取源表 B 失败: {e}")
        return
        
    df_b['name'] = df_b['name'].astype(str).str.strip()
    df_b['name'] = df_b['name'].str.replace(r'\.cif$', '', regex=True)
    
    df_b = df_b.dropna(subset=['rho'])
    df_b = df_b.drop_duplicates(subset=['name'], keep='first')
    df_b = df_b.rename(columns={'name': 'cif_name', 'rho': 'rho_y'})

    # ================= 关键 Debug 打印 =================
    print("\n🚨 【DEBUG 模式】请检查以下输出的名字是否完全一致！🚨")
    print("A表 (主表) 的前 5 个名字:")
    print(df_a['cif_name'].head(5).tolist())
    
    print("\nB表 (Zeo++特征表) 的前 5 个名字:")
    print(df_b['cif_name'].head(5).tolist())
    print("====================================================\n")

    # ================= 3. 执行合并 =================
    print("[3] 正在以 'cif_name' 为键进行数据拼接...")
    result_df = pd.merge(df_a, df_b, on='cif_name', how='left')
    
    # ================= 4. 数据分流 =================
    unmatched_mask = result_df['rho_y'].isna()
    unmatched_df = result_df[unmatched_mask]
    matched_df = result_df[~unmatched_mask]
    
    print("\n>>> 合并结果核对：")
    print(f"    - 匹配成功并拿到 rho_y 的行数: {len(matched_df)}")
    print(f"    - 未匹配到数据的行数: {len(unmatched_df)}")
    
    matched_df.to_csv(matched_output, index=False)
    if len(unmatched_df) > 0:
        unmatched_df.to_csv(unmatched_output, index=False)

if __name__ == "__main__":
    file_a = r'E:\CODE\cif2des\train_rho\clean_data\train_data.csv'
    file_b = r'E:\CODE\cif2des\calc\clean_data\goal_data\final_ABC_features_optimal.csv'
    matched_target = r'E:\CODE\cif2des\train_rho\clean_data\train_data_rho.csv'
    unmatched_error = r'E:\CODE\cif2des\train_rho\clean_data\unmatched_error.csv'
    
    merge_and_compare_rho(file_a, file_b, matched_target, unmatched_error)