import pandas as pd
import numpy as np
import catboost as cb
import os

def main():
    print("=====================================================")
    print("🚀 MOF 高通量全属性预测引擎 (8合1终极版) 启动！")
    print("=====================================================")

    # 1. 配置文件路径 (请确保模型都放在了对应的目录下)
    models_dir = r'E:\CODE\cif2des\prediction\models'  # 建议把8个模型集中放在这里
    input_csv_path = r'E:\CODE\cif2des\prediction\kvrh4prediction.csv' # 你要预测的 MOF 特征文件
    output_csv_path = r'E:\CODE\cif2des\prediction\predicted_results.csv'     # 预测结果的输出路径

    # 2. 定义目标属性与对应模型的映射字典
    # 格式: '目标名': ('模型文件名', '输入数据中对应的未优化特征列名')
    # 例如预测 Di_y 时，模型需要的核心几何特征叫 Di_x，而你的输入表里可能叫 Di
    model_configs = {
        'Di':   ('catboost_mof_di_predictor_best_with_RACs.cbm',   'Di_x'),
        'Df':   ('catboost_mof_df_predictor_best_with_RACs.cbm',   'Df_x'),
        'Dif':  ('catboost_mof_dif_predictor_best_with_RACs.cbm',  'Dif_x'),
        'VSA':  ('catboost_mof_VSA_predictor_best_with_RACs.cbm',  'VSA_x'),
        'GSA':  ('catboost_mof_GSA_predictor_best_with_RACs.cbm',  'GSA_x'),
        'POAV': ('catboost_mof_POAV_predictor_best_with_RACs.cbm', 'POAV_x'),
        'rho':  ('catboost_mof_rho_predictor_best_with_RACs.cbm',  'rho_x'),
        'void': ('catboost_mof_void_predictor_best_with_RACs.cbm', 'POAV_vol_frac_x')
    }

    # 这些目标在训练时做了 log1p 对数转换，预测后必须用 expm1 还原！
    log_transformed_targets = ['VSA', 'GSA']

    # 3. 加载待预测的输入数据
    try:
        df_input = pd.read_csv(input_csv_path)
        print(f"✅ 成功加载输入数据，共发现 {len(df_input)} 个待预测 MOF。")
    except FileNotFoundError:
        print(f"❌ 找不到输入文件 {input_csv_path}，请检查路径！")
        return

    # 准备一个空字典，用于存储预测结果
    predictions_dict = {}
    if 'name' in df_input.columns:
        predictions_dict['name'] = df_input['name'].values
    elif 'cif_name' in df_input.columns:
        predictions_dict['cif_name'] = df_input['cif_name'].values
    elif 'filename' in df_input.columns:
        predictions_dict['filename'] = df_input['filename'].values

    # 4. 循环调用 8 个专家模型进行推演
    for target_name, (model_filename, unopt_feature_name) in model_configs.items():
        model_path = os.path.join(models_dir, model_filename)
        
        if not os.path.exists(model_path):
            print(f"⚠️ 警告: 找不到模型 {model_filename}，已跳过 {target_name} 的预测。")
            continue
        
        print(f"\n🤖 正在唤醒 {target_name} 预测专家...")
        model = cb.CatBoostRegressor()
        model.load_model(model_path)

        # 获取当前模型在训练时【真正使用】的特征列表（自动绕过零方差问题）
        expected_features = model.feature_names_
        
        # 为了适配模型，我们需要在预测时对特征列进行重命名
        # 例如：原始表里叫 'Di'，但在预测 Di_y 时，模型认的名字是 'Di_x'
        # 我们用一个临时 DataFrame 来组装特征，绝不污染原始输入数据
        df_temp = df_input.copy()
        
        # 映射未优化的基础几何名称
        base_geo_name = target_name
        if target_name == 'void':
            base_geo_name = 'POAV_vol_frac'
            
        if base_geo_name in df_temp.columns and unopt_feature_name not in df_temp.columns:
            df_temp.rename(columns={base_geo_name: unopt_feature_name}, inplace=True)

        # 使用 reindex 强制对齐特征：
        # 如果模型需要的特征 (如某个 RAC) 在输入数据里找不到，自动填充为 0
        X_predict = df_temp.reindex(columns=expected_features, fill_value=0)

        # 执行预测
        print(f"   -> 正在推演 {target_name} 的优化后属性...")
        raw_preds = model.predict(X_predict)

        # 【核心】：如果目标是 VSA 或 GSA，必须解除对数封印！
        if target_name in log_transformed_targets:
            real_preds = np.expm1(raw_preds)
            print(f"   -> 已触发物理尺度还原 (expm1)。")
        else:
            real_preds = raw_preds

        # 将预测结果存入字典，后缀加上 _predicted
        predictions_dict[f'{target_name}_predicted'] = real_preds
        print(f"   ✅ {target_name} 预测完成！")

    # 5. 组合并保存最终结果
    print("\n=====================================================")
    print("📦 所有模型推演完毕，正在打包结果...")
    df_results = pd.DataFrame(predictions_dict)
    
    # 你可以选择把预测结果和原始特征拼在一起，这里为了清晰，只保存名字和预测值
    df_results.to_csv(output_csv_path, index=False)
    print(f"🎉 大功告成！全属性预测结果已保存至: {output_csv_path}")
    print("=====================================================")

    # 打印前 5 个结果给你看看
    print("\n预览前 5 个 MOF 的预测结果：")
    print(df_results.head())

if __name__ == "__main__":
    main()