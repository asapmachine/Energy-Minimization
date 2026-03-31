import pandas as pd
import numpy as np
import joblib
import os
import sys
import tensorflow as tf
from tensorflow.keras.models import load_model

# ================= 配置区域 =================
# 请根据实际情况修改以下路径
MODEL_DIR = r'E:\CODE\Core_bulk\NEW\ASSEMBLE\fitness_func\Bulk_prediction'

CONFIG = {
    # 输入与输出文件
    "input_file": r"E:\CODE\cif2des\prediction\output_2.csv",    # 你手上的待预测表格
    "output_file": r"E:\CODE\cif2des\prediction\predicted_kvrh_output_2.csv",   # 预测完成后生成的新表格

    # 模型与预处理文件路径
    "model_path": os.path.join(MODEL_DIR, "pinn_bulk_modulus_best.h5"),
    "scaler_path": os.path.join(MODEL_DIR, "scaler_bulk_modulus.pkl"),
    "encoder_path": os.path.join(MODEL_DIR, "encoder_topology.pkl"),

    # 列名映射 (Mapping):
    # 如果你当前表格的表头与模型训练时用的表头不完全一致，请在这里配置。
    # 格式: '当前表格的表头' : '模型需要的表头'
    "col_mapping": {
        'Gravimetric Surface Area (m2/g)': 'GSA',
        'Void_Fraction': 'VF'
    },

    # 模型预测严格需要的连续数值特征顺序
    "feature_order": ['Density', 'VSA', 'GSA', 'VF', 'PLD', 'PV', 'LCD']
}
# ==========================================

def main():
    print("--- 开始批量预测体积模量 (Bulk Modulus) ---")

    # 1. 加载模型与预处理器
    print("正在加载模型与预处理文件...")
    try:
        model = load_model(CONFIG["model_path"], compile=False)
        scaler = joblib.load(CONFIG["scaler_path"])
        label_encoder = joblib.load(CONFIG["encoder_path"])
    except Exception as e:
        print(f"[Error] 加载模型失败: {e}", file=sys.stderr)
        sys.exit(1)

    # 2. 读取原始数据
    if not os.path.exists(CONFIG["input_file"]):
        print(f"[Error] 找不到输入文件: {CONFIG['input_file']}", file=sys.stderr)
        sys.exit(1)
        
    print(f"正在读取表格: {CONFIG['input_file']}")
    df = pd.read_csv(CONFIG["input_file"])
    
    if df.empty:
        print("[Warning] 输入表格为空，结束任务。")
        sys.exit(0)

    # 3. 数据预处理
    # 拷贝一份用于模型输入，通过 rename 统一表头，这样不会改变原 df 的表头
    df_model_input = df.rename(columns=CONFIG["col_mapping"])

    # 检查数值特征列是否齐全
    missing_cols = [c for c in CONFIG["feature_order"] if c not in df_model_input.columns]
    if missing_cols:
        print(f"[Error] 缺少模型所需的特征列: {missing_cols}", file=sys.stderr)
        print("请检查原始表格表头，或者在 CONFIG['col_mapping'] 中进行映射配置。")
        sys.exit(1)
        
    # 检查拓扑(Topology)类别列是否存在
    if 'Topology' not in df_model_input.columns:
        print("[Error] 缺少模型所需的 'Topology' 列", file=sys.stderr)
        sys.exit(1)

    print("正在进行数据标准化和编码...")
    # 提取数值特征并标准化
    X_num_raw = df_model_input[CONFIG["feature_order"]].values
    X_num_scaled = scaler.transform(X_num_raw)

    # 提取拓扑特征并编码 (处理可能出现的未知拓扑结构)
    known_topologies = set(label_encoder.classes_)
    current_topos = df_model_input['Topology'].astype(str).values
    # 安全机制：遇到模型没见过的拓扑，默认用编码器第一种拓扑代替，防止报错中断
    safe_topos = [t if t in known_topologies else label_encoder.classes_[0] for t in current_topos]
    X_cat = label_encoder.transform(safe_topos).reshape(-1, 1)

    # 4. 执行预测
    print(f"正在为 {len(df)} 条数据执行预测...")
    preds = model.predict([X_num_scaled, X_cat], verbose=0).flatten()

    # 5. 将结果追加到原表格的最后一列
    # 直接在原始 DataFrame 上新增一列
    df['predicted_bulk_modulus'] = preds

    # 6. 保存最终表格
    df.to_csv(CONFIG["output_file"], index=False)
    print(f"[Success] 预测完成！结果已新增至最后一列，并保存至: {CONFIG['output_file']}")

if __name__ == "__main__":
    main()