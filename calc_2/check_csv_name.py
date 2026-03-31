import pandas as pd
import os

def print_raw_feature_names(csv_path):
    if not os.path.exists(csv_path):
        print(f"错误：找不到文件 {csv_path}")
        return

    # 只读取表头
    df = pd.read_csv(csv_path, nrows=0)
    
    # 获取除 filename 以外的所有列名
    features = [col for col in df.columns if col != 'filename']
    
    # 直接循环输出每一个特征名
    for i, name in enumerate(features, 1):
        print(f"{i}: {name}")

if __name__ == "__main__":
    # 如果你的文件名不同，请修改此处
    FILE_NAME = r"E:\CODE\cif2des\calc_2\RAC_features_only_1.csv"
    print_raw_feature_names(FILE_NAME)