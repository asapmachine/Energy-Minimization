import pandas as pd

def merge_csv_files(file_a, file_b, file_c, file_d, output_file_e):
    # 1. 读取表格A
    print("正在读取表格A...")
    df_a = pd.read_csv(file_a)
    
    # 保存A表格的所有原始表头
    cols_a = df_a.columns.tolist()
    
    # 给A表格创建一个用于匹配的临时列
    df_a['match_name'] = 'optimized_' + df_a['cif_name'].astype(str)
    
    # 2. 读取表格B, C, D
    print("正在读取表格B, C, D...")
    df_b = pd.read_csv(file_b)
    df_c = pd.read_csv(file_c)
    df_d = pd.read_csv(file_d)
    
    # 提取B,C,D表中我们需要的 'name' 和 'KVRH' 列，并将它们纵向合并
    df_bcd = pd.concat([
        df_b[['name', 'KVRH']],
        df_c[['name', 'KVRH']],
        df_d[['name', 'KVRH']]
    ], ignore_index=True)
    
    # 去重处理，防止数据膨胀
    df_bcd = df_bcd.drop_duplicates(subset=['name'])
    
    # 【修复重点】：把 BCD 表的 'name' 列名改成 'match_name'，避免和 A 表中原有的 'name' 列冲突
    df_bcd = df_bcd.rename(columns={'name': 'match_name'})
    
    # 3. 寻找匹配（重复）的数据：直接用 match_name 合并
    print("正在比对匹配项...")
    df_e = pd.merge(df_a, df_bcd, on='match_name', how='inner')
    
    # 4. 整理最终需要的列：A的所有表头 + BCD表中的'KVRH'
    final_cols = cols_a + ['KVRH']
    df_e = df_e[final_cols]
    
    # 5. 导出到表格E
    df_e.to_csv(output_file_e, index=False)
    print(f"处理完成！成功匹配到 {len(df_e)} 条数据，结果已保存至 {output_file_e}")

if __name__ == "__main__":
    # 文件路径配置
    FILE_A = r'E:\CODE\cif2des\prediction\merged_A_B.csv'
    FILE_B = r'E:\CODE\cif2des\prediction\csv\moduli_1inor_1edge.csv'
    FILE_C = r'E:\CODE\cif2des\prediction\csv\moduli_1inor_1org_1edge.csv'
    FILE_D = r'E:\CODE\cif2des\prediction\csv\moduli_2inor_1edge.csv'
    FILE_E = r'E:\CODE\cif2des\prediction\final.csv'  # 生成的最终表格名称
    
    # 执行函数
    merge_csv_files(FILE_A, FILE_B, FILE_C, FILE_D, FILE_E)