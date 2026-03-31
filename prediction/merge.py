import pandas as pd
import os

def merge_predictions(file_a, file_b, output_file):
    print(f"正在读取表格 A: {file_a}")
    print(f"正在读取表格 B: {file_b}")
    
    # 1. 读取两个表格
    df_a = pd.read_csv(file_a)
    df_b = pd.read_csv(file_b)
    
    # 检查必要的列是否存在
    for df, file_name in zip([df_a, df_b], [file_a, file_b]):
        if 'name' not in df.columns or 'predicted_bulk_modulus' not in df.columns:
            print(f"[错误] 表格 {file_name} 中找不到 'name' 或 'predicted_bulk_modulus' 列，请检查！")
            return
            
    # 2. 为了保持结果干净，只提取我们需要的两列
    df_a_subset = df_a[['name', 'predicted_bulk_modulus']]
    df_b_subset = df_b[['name', 'predicted_bulk_modulus']]
    
    # 3. 按照 'name' 列进行合并
    # how='inner' 表示只保留在 A 和 B 中都存在的 name（交集）
    # suffixes=('_A', '_B') 表示遇到同名列时，分别加上 _A 和 _B 的后缀
    print("正在比对并合并数据...")
    merged_df = pd.merge(
        df_a_subset, 
        df_b_subset, 
        on='name', 
        how='inner', 
        suffixes=('_A', '_B')
    )
    
    # 4. 导出结果
    merged_df.to_csv(output_file, index=False)
    print(f"[成功] 处理完成！共找到 {len(merged_df)} 个匹配的 name。")
    print(f"合并后的数据已保存至: {output_file}")
    print("合并后的表头将包含: 'name', 'predicted_bulk_modulus_A', 'predicted_bulk_modulus_B'")

# ================= 使用示例 =================
if __name__ == "__main__":
    # 请根据实际情况修改以下文件名/路径
    FILE_A = r'E:\CODE\cif2des\prediction\predicted_kvrh_output_1.csv'    # 你的表格 A
    FILE_B = r'E:\CODE\cif2des\prediction\predicted_kvrh_output_2.csv'    # 你的表格 B
    OUTPUT_FILE = r'E:\CODE\cif2des\prediction\merged_A_B.csv'  # 合并后生成的新表格
    
    if not os.path.exists(FILE_A) or not os.path.exists(FILE_B):
        print("请确认 FILE_A 和 FILE_B 的路径正确且文件存在！")
    else:
        merge_predictions(FILE_A, FILE_B, OUTPUT_FILE)