import os
import shutil
import pandas as pd

def collect_cif_files(csv_path, source_dirs, dest_dir):
    """
    根据 CSV 文件中的 cif_file 列，从多个源文件夹中搜索对应文件，并集中提取到目标文件夹。
    """
    # 1. 安全检查：创建目标文件夹 D
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
        print(f"已创建目标文件夹: {dest_dir}")

    # 2. 读取 CSV 数据并提取目标名单
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"错误: 找不到 CSV 文件 '{csv_path}'，请检查路径。")
        return

    # 严谨性检查：确认表头是否存在
    if 'cif_file' not in df.columns:
        print("错误: CSV 中没有找到 'cif_file' 表头！请确认文件格式。")
        return

    # 提取所有文件名，并使用 .unique() 去重，防止 CSV 中有重复记录导致重复劳动
    target_cifs = df['cif_file'].dropna().unique()
    print(f"在 CSV 中共读取到 {len(target_cifs)} 个需要提取的独立 CIF 文件名。")
    print("开始在多个文件夹中搜寻并提取...\n")

    # 3. 核心逻辑：多路搜寻与复制
    success_count = 0
    missing_files = []

    for filename in target_cifs:
        file_found = False
        
        # 依次在 A, B, C 中寻找该文件
        for src_dir in source_dirs:
            source_path = os.path.join(src_dir, filename)
            
            # 如果在当前文件夹找到了文件
            if os.path.exists(source_path):
                target_path = os.path.join(dest_dir, filename)
                try:
                    # 执行复制 (如果想剪切，可替换为 shutil.move)
                    shutil.copy2(source_path, target_path)
                    success_count += 1
                    file_found = True
                    # 关键点：找到并成功复制后，使用 break 跳出当前的 src_dir 循环，不再去后面的文件夹找了
                    break 
                except Exception as e:
                    print(f"  [失败] 找到了 {filename}，但无法复制，原因: {e}")
                    file_found = True # 标记为已找到（避免列入 missing），跳出循环
                    break
        
        # 如果把 A, B, C 都翻遍了还没找到 (file_found 依然是 False)
        if not file_found:
            missing_files.append(filename)

    # 4. 打印最终的数据对账报告
    print("-" * 40)
    print(f"执行完毕！共成功提取 {success_count} 个文件到 '{dest_dir}'。")
    
    if missing_files:
        print(f"⚠️ 注意: 有 {len(missing_files)} 个文件在所有的源文件夹中均未找到。")
        # 如果缺失的不多，打印出来方便排查；如果太多，就只打印前 5 个
        preview_limit = 5
        print(f"缺失名单示例: {missing_files[:preview_limit]}" + ("..." if len(missing_files) > preview_limit else ""))
    else:
        print("完美！清单上的所有文件都已成功找到并提取。")

# ==========================================
# 运行区域（请根据您的实际情况修改这里的路径）
# ==========================================
if __name__ == "__main__":
    CSV_FILE = r'E:\CODE\cif2des\error\merged_features.csv'    # 包含 cif_file 表头的 CSV 文件
    
    SOURCE_FOLDERS = [r'E:\MOF database\zenodo_upload_ultrastable\zenodo_upload_ultrastable\initial_structures\1inorganic_1edge', r'E:\MOF database\zenodo_upload_ultrastable\zenodo_upload_ultrastable\initial_structures\1inorganic_1organic_1edge', r'E:\MOF database\zenodo_upload_ultrastable\zenodo_upload_ultrastable\initial_structures\2inorganic_1edge']     
    
    DEST_FOLDER = r'E:\CODE\cif2des\front'    
    
    collect_cif_files(CSV_FILE, SOURCE_FOLDERS, DEST_FOLDER)