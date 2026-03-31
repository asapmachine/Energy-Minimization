import pandas as pd

def extract_topology_mapping(input_files, output_file):
    """
    从原始数据表中提取拓扑与循环特征的唯一映射关系
    input_files: 输入的表格列表，例如 ['Table_A.csv', 'Table_B.csv']
    output_file: 提取后的精简映射表保存路径
    """
    all_extracted_data = []
    
    # 1. 定义我们要提取的 11 个表头
    target_columns = [
        'net', 
        '3_cycle_norm', '4_cycle_norm', '5_cycle_norm', '6_cycle_norm', 
        '7_cycle_norm', '8_cycle_norm', '9_cycle_norm', '10_cycle_norm', 
        '11_cycle_norm', '12_cycle_norm'
    ]

    for file in input_files:
        try:
            print(f"正在读取: {file} ...")
            # 读取原始大表
            df = pd.read_csv(file)
            
            # 检查表头是否完整
            missing_cols = [col for col in target_columns if col not in df.columns]
            if missing_cols:
                print(f"⚠️ 警告: 文件 {file} 缺失以下列: {missing_cols}")
                # 只取存在的列
                available_cols = [col for col in target_columns if col in df.columns]
                temp_df = df[available_cols]
            else:
                temp_df = df[target_columns]
                
            all_extracted_data.append(temp_df)
            
        except Exception as e:
            print(f"❌ 读取 {file} 失败: {e}")

    # 2. 合并所有提取到的数据
    combined_df = pd.concat(all_extracted_data, ignore_index=True)

    # 3. 核心步骤：去重
    # 只要 net 相同，对应的 10 个特征就应该是一样的，所以我们按 net 去重
    unique_mapping = combined_df.drop_duplicates(subset=['net']).copy()

    # 4. 清理数据：去掉 net 为空或者特征全为 0 的异常行
    unique_mapping.dropna(subset=['net'], inplace=True)
    
    # 5. 按拓扑名排序，方便查找
    unique_mapping.sort_values(by='net', inplace=True)

    # 6. 保存提取出来的“拓扑密码本”
    unique_mapping.to_csv(output_file, index=False)
    
    print("\n" + "="*30)
    print(f"✅ 提取成功！")
    print(f"共识别出 {len(unique_mapping)} 种唯一的拓扑结构。")
    print(f"映射表已保存至: {output_file}")
    print("="*30)
    
    # 打印前 5 行展示一下
    print("\n提取结果预览 (前 5 条):")
    print(unique_mapping.head())

# ==========================================================
# 执行：把你的文件名填在这里
# ==========================================================
if __name__ == "__main__":
    # 假设你的原始表格叫 Table_A.csv 和 Table_B.csv
    raw_tables = [
    r"E:\CODE\cif2des\train_Kvrh\test_feature_files\net_short_symb_combined_data_frame_all.csv",
    r"E:\CODE\cif2des\train_Kvrh\train_feature_files\net_short_symb_combined_data_frame_all.csv"
] 
    extract_topology_mapping(raw_tables, "topology_lookup_table.csv")