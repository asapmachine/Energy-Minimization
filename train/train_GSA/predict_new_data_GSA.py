import pandas as pd
import numpy as np
import catboost as cb

print("=============== 启动工业级高通量预测引擎 ===============")

# 1. 核心参数与特征定义
features = ['Di_x', 'Df', 'Dif', 'rho', 'VSA', 'GSA', 'VPOV', 'GPOV', 'POAV_vol_frac', 'GPOAV', 'POAV']
model_path = 'catboost_mof_df_predictor_best.cbm' 
input_file = r'E:\CODE\cif2des\train_Di\clean_data\train_data_Di.csv' # 你的待预测文件
output_file = r'E:\CODE\cif2des\train_Di\clean_data\predicted_results.csv'

# 2. 加载模型
print(f"正在加载模型: {model_path} ...")
model = cb.CatBoostRegressor()
model.load_model(model_path)
print("✅ 模型加载成功！")

# 3. 读取待预测数据
print(f"正在读取待预测数据: {input_file} ...")
df_new = pd.read_csv(input_file)
original_len = len(df_new)

# 💡 核心升级：使用掩码技术 (Mask)，坚决不删除任何一行！
# 先把潜在的无穷大替换为 NaN
df_new.replace([np.inf, -np.inf], np.nan, inplace=True)

# 找出特征列中没有任何缺失值的“健康样本”的索引
valid_mask = df_new[features].notna().all(axis=1)
invalid_count = original_len - valid_mask.sum()

print(f"数据扫描完毕：共 {original_len} 个样本。其中 {valid_mask.sum()} 个健康，{invalid_count} 个包含特征缺失。")

# 4. 执行精准推理预测 (只对健康样本做预测)
print("正在执行推理预测...")
# 提取健康的 X 矩阵
X_valid = df_new.loc[valid_mask, features]

# 预先在原表中创建预测结果列，全部默认填入空值 (NaN)
df_new['Predicted_Di_y'] = np.nan

# 只有健康的样本才填入模型预测出的真实数值
if valid_mask.sum() > 0:
    predictions = model.predict(X_valid)
    df_new.loc[valid_mask, 'Predicted_Di_y'] = predictions

# 5. 保存落盘
df_new.to_csv(output_file, index=False)
print(f"✅ 预测完成！结果已成功保存至: {output_file}")
print("（注：输出文件与输入文件的行数已完美对齐，包含缺失特征的残缺样本其预测值已标记为空）")