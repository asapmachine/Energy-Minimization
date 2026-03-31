import pandas as pd

def merge_common_columns(file_a, file_b, file_c, output_path):
    # 1. 读取三个表格
    df_a = pd.read_csv(file_a)
    df_b = pd.read_csv(file_b)
    df_c = pd.read_csv(file_c)

    # 2. 找出三个表格共同拥有的列名 (取交集)
    common_cols = list(set(df_a.columns) & set(df_b.columns) & set(df_c.columns))
    
    # 按照 A 表的原始顺序排一下列名，防止乱序
    common_cols = [col for col in df_a.columns if col in common_cols]

    print(f"--- 共有列识别报告 ---")
    print(f"找到三个表共有的列 ({len(common_cols)}个):")
    print(f"   {common_cols}")
    print("-" * 30)

    if not common_cols:
        print("警告：三个表没有完全相同的列名，无法合并！")
        return

    # 3. 提取共有列的数据并进行堆叠
    # 我们只取 common_cols 里面的列，防止 pandas 自动填充 NaN
    result = pd.concat([
        df_a[common_cols], 
        df_b[common_cols], 
        df_c[common_cols]
    ], axis=0, ignore_index=True)

    # 4. 打印合并结果统计
    print(f"表格 A 行数: {len(df_a)}")
    print(f"表格 B 行数: {len(df_b)}")
    print(f"表格 C 行数: {len(df_c)}")
    print(f"合并后总行数: {len(result)}")

    # 5. 保存结果
    result.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n✅ 合并完成！结果已保存至: {output_path}")

# --- 运行配置 ---
path_a = r'E:\CODE\cif2des\train_Kvrh\tobacco_data_process\final_result.csv'
path_b = r'E:\CODE\cif2des\train_Kvrh\train_feature_files\net_short_symb_combined_data_frame_all.csv'
path_c = r'E:\CODE\cif2des\train_Kvrh\test_feature_files\net_short_symb_combined_data_frame_all.csv'
path_out = 'triple_merged_data.csv'

merge_common_columns(path_a, path_b, path_c, path_out)