import pandas as pd
import numpy as np

print("=============== 启动缺失值(NaN)探测雷达 ===============")

# 1. 配置文件路径（请替换为你刚才那个报出1个缺失值的 CSV 路径）
input_file = r'E:\CODE\cif2des\train_Df\clean_data_Df\train_data_Df_second.csv'

print(f"正在读取文件: {input_file} ...")
df = pd.read_csv(input_file)

# 将潜在的正负无穷大替换为 NaN，统一按缺失值处理
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# 2. 核心探测逻辑：揪出所有至少包含一个 NaN 的行
# isnull().any(axis=1) 的意思是：横向扫描每一行，只要碰到一个空值，就把它标记为 True
nan_rows = df[df.isnull().any(axis=1)]

if len(nan_rows) == 0:
    print("✅ 恭喜！完美的数据集，未发现任何缺失值 (NaN)。")
else:
    print(f"\n🚨 警报！共发现 {len(nan_rows)} 行包含缺失值。详细信息如下：\n")
    
    # 3. 逐行解剖，指出具体是哪些列缺失
    for index, row in nan_rows.iterrows():
        # 提取这一行中值为 NaN 的列名
        missing_cols = row.index[row.isnull()].tolist()
        
        # 如果你的表里有 filename 这一列，我们就打印它的名字，否则打印它在表格里的行号
        sample_id = row['filename'] if 'filename' in row.index else f"CSV文件中的第 {index + 2} 行 (包含表头)"
        
        print(f"  -> 样本鉴定: [{sample_id}]")
        print(f"  -> 缺失的列: {missing_cols}\n")
    
    # 4. 把这些“刺客”单独关押到一个新表里，方便你用 Excel 打开审查
    output_file = r'E:\CODE\cif2des\train_Df\clean_data_Df\nan_samples_report.csv'
    nan_rows.to_csv(output_file, index=False)
    print(f"✅ 已将所有异常样本单独提取并保存至: {output_file}")