import pandas as pd
import os

def calculate_absolute_errors(input_file, output_file):
    print(f"正在读取表格: {input_file}")
    
    # 读取文件，使用 gbk 编码防止 Windows 下的中文/特殊字符报错
    try:
        df = pd.read_csv(input_file, encoding='gbk')
    except Exception as e:
        print(f"[读取错误] {e}。尝试使用 utf-8 读取...")
        df = pd.read_csv(input_file)
        
    # 检查必要的列是否存在
    required_cols = ['predicted_bulk_modulus_A', 'predicted_bulk_modulus_B', 'KVRH']
    for col in required_cols:
        if col not in df.columns:
            print(f"[错误] 表格中找不到列 '{col}'，请检查表头！")
            return
            
    print("正在计算每一行的绝对误差...")
    
    # 1. 计算每一行的绝对误差 (预测值 - 真实值 的绝对值)
    df['Absolute_Error_A'] = abs(df['predicted_bulk_modulus_A'] - df['KVRH'])
    df['Absolute_Error_B'] = abs(df['predicted_bulk_modulus_B'] - df['KVRH'])
    
    # 2. 导出包含每一行误差的新表格
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"[成功] 每一行的误差已计算完毕！结果保存至: {output_file}")
    
    # 3. 计算并打印整体的平均绝对误差 (MAE)
    mae_a = df['Absolute_Error_A'].mean()
    mae_b = df['Absolute_Error_B'].mean()
    
    print("\n" + "="*40)
    print("          整体平均绝对误差 (MAE)")
    print("="*40)
    print(f"【模型 A】 平均绝对误差: {mae_a:.4f}")
    print(f"【模型 B】 平均绝对误差: {mae_b:.4f}")
    print("="*40)
    
    if mae_a < mae_b:
        print("结论: 整体来看，【模型 A】 的误差更小，预测更准确！")
    elif b < mae_a:
        print("结论: 整体来看，【模型 B】 的误差更小，预测更准确！")
    else:
        print("结论: 两个模型的平均绝对误差完全一样！")

# ================= 使用示例 =================
if __name__ == "__main__":
    # 请将这里的 'huibao.csv' 替换为你实际的文件名
    INPUT_FILE = r'E:\CODE\cif2des\prediction\zuhui.csv'               
    OUTPUT_FILE = 'absolute_errors_output.csv' # 生成的新表格名称
    
    if not os.path.exists(INPUT_FILE):
        print(f"找不到文件 {INPUT_FILE}，请确认文件就在当前文件夹下。")
    else:
        calculate_absolute_errors(INPUT_FILE, OUTPUT_FILE)