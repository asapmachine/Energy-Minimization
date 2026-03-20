import pandas as pd

def clean_zero_values_with_summary(input_csv, output_csv):
    """
    清理数据：只要指定列中任意一列的值为0，就删除该行。
    并宏观统计出每一个特征列中包含 0 值的总行数。
    """
    print(">>> 开始执行异常零值清洗与特征分布统计...\n")
    
    # 1. 读取数据
    try:
        df = pd.read_csv(input_csv)
        print(f"[1] 成功读取待清洗数据，共 {len(df)} 行。")
    except Exception as e:
        print(f"读取文件失败: {e}")
        return

    # 2. 定义目标列
    columns_to_check = ['Df', 'Di', 'GSA', 'VSA', 'rho']
    missing_cols = [col for col in columns_to_check if col not in df.columns]
    if missing_cols:
        print(f"错误：在表格中未找到以下列：{missing_cols}")
        return

    # 3. 【核心提取】：生成布尔矩阵
    bool_df = df[columns_to_check].isin([0, 0.0, '0'])
    
    # 4. 【新增功能】：按列统计 0 值的数量
    # sum() 默认按列 (axis=0) 将 True 的个数加起来
    zero_counts_per_col = bool_df.sum()
    
    # 找到哪些行至少包含一个 True (用于执行删除操作)
    zero_mask = bool_df.any(axis=1)
    deleted_count = zero_mask.sum()
    
    # 5. 打印美观的汇总清单
    if deleted_count > 0:
        print("\n>>> 零值分布统计清单：")
        # 遍历每列的统计结果并打印
        for col, count in zero_counts_per_col.items():
            print(f"    - {col}: {count} 行")
            
        print(f"\n    (注：若某一行有多个列同时为0，上述统计会分别计数，但该行只会被剔除一次)")
    else:
        print("\n>>> 完美！没有发现任何包含 0 值的异常行。")

    # 6. 执行删除并保存
    cleaned_df = df[~zero_mask]
    
    print("\n>>> 最终清洗核对：")
    print(f"    - 实际剔除的总行数: {deleted_count} 行")
    print(f"    - 清洗后保留的有效数据: {len(cleaned_df)} 行")

    cleaned_df.to_csv(output_csv, index=False)
    print(f"\n[2] 任务完成！极致干净的数据已保存至: {output_csv}")

# ================= 运行示例 =================
if __name__ == "__main__":
    input_file = r'E:\CODE\cif2des\train_Dif\clean_data\train_data_Dif.csv'
    output_file = r'E:\CODE\cif2des\train_Dif\clean_data\train_data_Dif_final.csv'
    
    clean_zero_values_with_summary(input_file, output_file)