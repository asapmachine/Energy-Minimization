import pandas as pd
import os

def merge_and_drop_nas():
    # 1. 你的 176 个特征名单
    rac_features = [
        "lc-chi-0-all", "lc-chi-1-all", "lc-chi-2-all", "lc-chi-3-all", 
        "lc-Z-0-all", "lc-Z-1-all", "lc-Z-2-all", "lc-Z-3-all",
        "lc-I-0-all", "lc-I-1-all", "lc-I-2-all", "lc-I-3-all",
        "lc-T-0-all", "lc-T-1-all", "lc-T-2-all", "lc-T-3-all",
        "lc-S-0-all", "lc-S-1-all", "lc-S-2-all", "lc-S-3-all",
        "lc-alpha-0-all", "lc-alpha-1-all", "lc-alpha-2-all", "lc-alpha-3-all",
        "D_lc-chi-0-all", "D_lc-chi-1-all", "D_lc-chi-2-all", "D_lc-chi-3-all",
        "D_lc-Z-0-all", "D_lc-Z-1-all", "D_lc-Z-2-all", "D_lc-Z-3-all",
        "D_lc-I-0-all", "D_lc-I-1-all", "D_lc-I-2-all", "D_lc-I-3-all",
        "D_lc-T-0-all", "D_lc-T-1-all", "D_lc-T-2-all", "D_lc-T-3-all",
        "D_lc-S-0-all", "D_lc-S-1-all", "D_lc-S-2-all", "D_lc-S-3-all",
        "D_lc-alpha-0-all", "D_lc-alpha-1-all", "D_lc-alpha-2-all", "D_lc-alpha-3-all",
        "func-chi-0-all", "func-chi-1-all", "func-chi-2-all", "func-chi-3-all",
        "func-Z-0-all", "func-Z-1-all", "func-Z-2-all", "func-Z-3-all",
        "func-I-0-all", "func-I-1-all", "func-I-2-all", "func-I-3-all",
        "func-T-0-all", "func-T-1-all", "func-T-2-all", "func-T-3-all",
        "func-S-0-all", "func-S-1-all", "func-S-2-all", "func-S-3-all",
        "func-alpha-0-all", "func-alpha-1-all", "func-alpha-2-all", "func-alpha-3-all",
        "D_func-chi-0-all", "D_func-chi-1-all", "D_func-chi-2-all", "D_func-chi-3-all",
        "D_func-Z-0-all", "D_func-Z-1-all", "D_func-Z-2-all", "D_func-Z-3-all",
        "D_func-I-0-all", "D_func-I-1-all", "D_func-I-2-all", "D_func-I-3-all",
        "D_func-T-0-all", "D_func-T-1-all", "D_func-T-2-all", "D_func-T-3-all",
        "D_func-S-0-all", "D_func-S-1-all", "D_func-S-2-all", "D_func-S-3-all",
        "D_func-alpha-0-all", "D_func-alpha-1-all", "D_func-alpha-2-all", "D_func-alpha-3-all",
        "f-chi-0-all", "f-chi-1-all", "f-chi-2-all", "f-chi-3-all",
        "f-Z-0-all", "f-Z-1-all", "f-Z-2-all", "f-Z-3-all",
        "f-I-0-all", "f-I-1-all", "f-I-2-all", "f-I-3-all",
        "f-T-0-all", "f-T-1-all", "f-T-2-all", "f-T-3-all",
        "f-S-0-all", "f-S-1-all", "f-S-2-all", "f-S-3-all",
        "mc-chi-0-all", "mc-chi-1-all", "mc-chi-2-all", "mc-chi-3-all",
        "mc-Z-0-all", "mc-Z-1-all", "mc-Z-2-all", "mc-Z-3-all",
        "mc-I-0-all", "mc-I-1-all", "mc-I-2-all", "mc-I-3-all",
        "mc-T-0-all", "mc-T-1-all", "mc-T-2-all", "mc-T-3-all",
        "mc-S-0-all", "mc-S-1-all", "mc-S-2-all", "mc-S-3-all",
        "D_mc-chi-0-all", "D_mc-chi-1-all", "D_mc-chi-2-all", "D_mc-chi-3-all",
        "D_mc-Z-0-all", "D_mc-Z-1-all", "D_mc-Z-2-all", "D_mc-Z-3-all",
        "D_mc-I-0-all", "D_mc-I-1-all", "D_mc-I-2-all", "D_mc-I-3-all",
        "D_mc-T-0-all", "D_mc-T-1-all", "D_mc-T-2-all", "D_mc-T-3-all",
        "D_mc-S-0-all", "D_mc-S-1-all", "D_mc-S-2-all", "D_mc-S-3-all",
        "f-lig-chi-0", "f-lig-chi-1", "f-lig-chi-2", "f-lig-chi-3",
        "f-lig-Z-0", "f-lig-Z-1", "f-lig-Z-2", "f-lig-Z-3",
        "f-lig-I-0", "f-lig-I-1", "f-lig-I-2", "f-lig-I-3",
        "f-lig-T-0", "f-lig-T-1", "f-lig-T-2", "f-lig-T-3",
        "f-lig-S-0", "f-lig-S-1", "f-lig-S-2", "f-lig-S-3"
    ]

    # 路径配置
    file_a = r"E:\CODE\cif2des\train_2\train_POAV\clean_data\train_data_POAV_final.csv"
    data_sources = [
        r"E:\CODE\cif2des\calc_2\RAC_features_only_1.csv", 
        r"E:\CODE\cif2des\calc_2\RAC_features_only_2.csv", 
        r"E:\CODE\cif2des\calc_2\RAC_features_only_3.csv",
        r"E:\CODE\cif2des\calc_2\RACs_features_summary.csv"
    ]

    def clean_id(s):
        return str(s).replace('.cif', '').strip()

    # 2. 构建特征库
    all_feature_dfs = []
    for path in data_sources:
        if os.path.exists(path):
            df_temp = pd.read_csv(path)
            id_col = 'cif_name' if 'cif_name' in df_temp.columns else 'filename'
            df_temp['merge_id'] = df_temp[id_col].apply(clean_id)
            # 只取 ID 和名单中存在的特征
            available = ['merge_id'] + [f for f in rac_features if f in df_temp.columns]
            all_feature_dfs.append(df_temp[available])

    full_pool = pd.concat(all_feature_dfs, axis=0, ignore_index=True).drop_duplicates('merge_id')

    # 3. 读取主表并合并
    df_main = pd.read_csv(file_a)
    df_main['merge_id'] = df_main['cif_name'].apply(clean_id)
    df_merged = pd.merge(df_main, full_pool, on='merge_id', how='left')

    # 4. 【新任务】：清理 176 个特征中的空白行
    # 找出在合并后真正出现在表里的特征列名
    existing_rac_cols = [f for f in rac_features if f in df_merged.columns]
    
    initial_count = len(df_merged)
    # 删除在这些特征列中含有 NaN 的行
    df_cleaned = df_merged.dropna(subset=existing_rac_cols)
    final_count = len(df_cleaned)

    # 5. 保存结果
    df_cleaned = df_cleaned.drop(columns=['merge_id'])
    df_cleaned.to_csv("A_final_cleaned_data.csv", index=False)

    # 6. 报告
    print("\n--- 数据清理报告 ---")
    print(f"合并后的初始行数: {initial_count}")
    print(f"检测到含空值并删除的行数: {initial_count - final_count}")
    print(f"最终保留的行数: {final_count}")
    print(f"特征完整度检查: 176 个特征中已匹配并清理了 {len(existing_rac_cols)} 个列。")
    print(f"结果已保存为: A_final_cleaned_data.csv")

if __name__ == "__main__":
    merge_and_drop_nas()