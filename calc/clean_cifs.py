import pandas as pd
import os
import shutil


def extract_zero_feature_cifs(csv_path, source_folder, dest_folder):
    """
    读取 CSV 文件，识别 Di, Df, GSA, VSA 皆为 0 的行，
    并根据 cif_file 表头将对应的文件从 source_folder 提取到 dest_folder。
    """
    # 1. 安全检查：确保目标文件夹 B 存在，如果不存在则自动创建
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
        print(f"已创建目标文件夹: {dest_folder}")

    # 2. 加载数据
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"错误: 找不到 CSV 文件 '{csv_path}'，请检查路径。")
        return

    # 3. 稳健性检查：确保所有需要的表头都在 CSV 中，避免代码中途崩溃
    required_columns = ['Di', 'Df', 'GSA', 'VSA', 'cif_file']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"错误: CSV 文件缺少必要的表头: {missing_columns}")
        return

    # 4. 核心逻辑：多条件空值筛选
    # 使用 isna() 函数来判断特征是否为 NaN (空值)
    condition = df['Di'].isna() & df['Df'].isna() & df['GSA'].isna() & df['VSA'].isna()
    target_rows = df[condition]

    if target_rows.empty:
        print("完美！没有找到 Di, Df, GSA, VSA 全为 0 的异常数据。")
        return

    print(f"检测到 {len(target_rows)} 个全为 0 的异常 CIF 文件，开始提取...")

    # 5. 遍历异常名单，执行文件操作
    success_count = 0
    for cif_filename in target_rows['cif_file']:
        # 拼接出完整的文件路径
        source_path = os.path.join(source_folder, cif_filename)
        dest_path = os.path.join(dest_folder, cif_filename)

        # 检查源文件是否存在（可能已经被移走或删除了）
        if os.path.exists(source_path):
            try:
                # 默认使用 shutil.copy2 进行复制（保留文件的元数据如修改时间）
                # 【如果要剪切/移动文件，请把下面这行换成：shutil.move(source_path, dest_path)】
                shutil.copy2(source_path, dest_path)
                print(f"  [成功] 提取文件: {cif_filename}")
                success_count += 1
            except Exception as e:
                print(f"  [失败] 无法复制 {cif_filename}: {e}")
        else:
            print(f"  [跳过] 源文件夹中找不到该文件: {cif_filename}")

    print(f"\n执行完毕！共成功提取 {success_count} 个文件到 '{dest_folder}'。")


# ==========================================
# 运行区域（请在这里填入您的实际路径）
# ==========================================
if __name__ == "__main__":
    CSV_FILE_PATH = r'E:\CODE\cif2des\error\merged_features.csv'  # 你的 CSV 文件路径
    FOLDER_A = r'E:\CODE\cif2des\front'  # 源文件夹 A（包含所有 cif 的文件夹）
    FOLDER_B = r'E:\CODE\cif2des\error\Suspected_error_3'  # 目标文件夹 B（用来存放全0的 cif）

    extract_zero_feature_cifs(CSV_FILE_PATH, FOLDER_A, FOLDER_B)