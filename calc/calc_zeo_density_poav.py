import os
import sys
import glob
import shutil
import subprocess
import numpy as np
import pandas as pd
import platform
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

# ==================== 配置区 (可随时修改) ====================

# 1. 你要计算的三个文件夹的 WSL 绝对路径
TARGET_FOLDERS = [
    "/mnt/e/MOF database/zenodo_upload_ultrastable/zenodo_upload_ultrastable/optimized_structures/1inorganic_1edge",
    "/mnt/e/MOF database/zenodo_upload_ultrastable/zenodo_upload_ultrastable/optimized_structures/1inorganic_1organic_1edge",
    "/mnt/e/MOF database/zenodo_upload_ultrastable/zenodo_upload_ultrastable/optimized_structures/2inorganic_1edge"
]

# 2. 最终合并特征 CSV 文件的保存路径 (WSL 格式)
FINAL_CSV_OUTPUT_PATH = "/mnt/e/CODE/cif2des/calc/final_ABC_features.csv"

# =============================================================

# =============================================================

def is_wsl():
    if platform.system().lower() != "linux": return False
    if os.environ.get("WSL_INTEROP") or os.environ.get("WSL_DISTRO_NAME"): return True
    try:
        with open("/proc/version", "r", encoding="utf-8", errors="ignore") as f:
            return "microsoft" in f.read().lower()
    except OSError: return False

def convert_win_path_to_wsl(win_path):
    if is_wsl(): return os.path.abspath(win_path)
    abs_win_path = os.path.abspath(win_path).replace('\\', '/')
    return f'/mnt/{abs_win_path[0].lower()}/{abs_win_path[3:]}'

def delete_and_remake_folders(folder_names):
    for folder_name in folder_names:
        if os.path.isdir(folder_name): shutil.rmtree(folder_name)
        os.mkdir(folder_name)

def run_zeopp(cmd_args, timeout_sec=0):
    try:
        proc = subprocess.run(cmd_args, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, timeout=timeout_sec or None)
        return proc.returncode, proc.stderr.strip()
    except subprocess.TimeoutExpired:
        return 124, "Timeout"

def extract_value(pattern, text):
    """使用正则表达式安全提取数值"""
    match = re.search(pattern, text)
    return float(match.group(1)) if match else np.nan

def descriptor_generator(name, structure_path, source_folder, zeopp_executable_path):
    current_MOF_folder = f'feature_folders/{name}'
    cif_folder = f'{current_MOF_folder}/cifs'
    zeo_folder = f'{current_MOF_folder}/zeo++'
    merged_descriptors_folder = f'{current_MOF_folder}/merged_descriptors'
    
    delete_and_remake_folders([current_MOF_folder, cif_folder, zeo_folder, merged_descriptors_folder])

    target_cif_path = f'{cif_folder}/{name}.cif'
    shutil.copy(structure_path, target_cif_path)

    wsl_structure_path = convert_win_path_to_wsl(target_cif_path)
    wsl_pov_txt = convert_win_path_to_wsl(f'{zeo_folder}/{name}_pov.txt')

    zeo_prefix = zeopp_executable_path if isinstance(zeopp_executable_path, list) else [zeopp_executable_path]

    # 只运行 -volpo (孔体积和密度计算)
    run_zeopp(zeo_prefix + ["-volpo", "1.86", "1.86", "10000", wsl_pov_txt, wsl_structure_path])

    crystal_density, POAV_vol_frac, GPOAV = np.nan, np.nan, np.nan

    # 正则提取特征
    if os.path.exists(f'{zeo_folder}/{name}_pov.txt'):
        with open(f'{zeo_folder}/{name}_pov.txt') as f:
            content = f.read()
            crystal_density = extract_value(r'Density:\s*([0-9\.]+)', content)
            GPOAV = extract_value(r'POAV_cm\^3/g:\s*([0-9\.]+)', content)
            POAV_vol_frac = extract_value(r'POAV_Volume_fraction:\s*([0-9\.]+)', content)

    # 提取当前文件夹的简写名（取最后一部分），方便记录在表格中
    folder_short_name = os.path.basename(source_folder)

    geo_dict = {
        'name': name,
        'source_folder': folder_short_name, 
        'rho': crystal_density,
        'PV_cm3_g': GPOAV,
        'Porosity': POAV_vol_frac
    }
    
    geo_df = pd.DataFrame([geo_dict])
    out_csv = f'{merged_descriptors_folder}/{name}_descriptors.csv'
    geo_df.to_csv(out_csv, index=False)

def _process_one_mof(task):
    i, total, cp, source_folder, zeopp_path = task
    MOF_name = os.path.basename(cp).replace('.cif', '')
    out_csv = f'feature_folders/{MOF_name}/merged_descriptors/{MOF_name}_descriptors.csv'

    # ======= 断点续传核心逻辑 =======
    if os.path.exists(out_csv):
        try:
            # 尝试读取，确认文件没有因为断电而损坏
            df_test = pd.read_csv(out_csv)
            if not df_test.empty and 'rho' in df_test.columns:
                print(f'[{i + 1}/{total}] 已存在，跳过: {MOF_name}')
                return MOF_name, 'skipped'
        except Exception:
            # 文件损坏，继续执行后面的重新计算逻辑
            pass
    # ================================

    print(f'[{i + 1}/{total}] 正在计算: {MOF_name} ...')
    descriptor_generator(MOF_name, cp, source_folder, zeopp_path)
    return MOF_name, 'done'

if __name__ == "__main__":
    ZEOpp_EXECUTABLE_PATH = ["wsl", "/home/lilyaf/zeo++-0.3/network"] if not is_wsl() else "/home/lilyaf/zeo++-0.3/network"

    if not os.path.exists('feature_folders'):
        os.mkdir('feature_folders')

    all_cif_tasks = []
    
    # 扫描所有文件夹
    print("正在扫描目标文件夹...")
    for folder in TARGET_FOLDERS:
        if os.path.isdir(folder):
            cifs_in_folder = glob.glob(os.path.join(folder, '*.cif'))
            for cp in cifs_in_folder:
                all_cif_tasks.append({'path': cp, 'folder': folder})
            print(f" -> 找到 {len(cifs_in_folder)} 个文件于: {os.path.basename(folder)}")
        else:
            print(f" -> [警告] 文件夹不存在，请检查路径: {folder}")

    if not all_cif_tasks:
        print("未找到任何 .cif 文件，程序退出。")
        sys.exit(0)

    total_files = len(all_cif_tasks)
    print(f"\n总计找到 {total_files} 个 CIF 文件。准备开始计算（支持断点续传）...")

    # 准备任务队列
    tasks = [(i, total_files, item['path'], item['folder'], ZEOpp_EXECUTABLE_PATH) for i, item in enumerate(all_cif_tasks)]
    
    n_workers = max(1, int(multiprocessing.cpu_count() * 1.5))
    
    # 执行多进程计算
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = [ex.submit(_process_one_mof, t) for t in tasks]
        for fut in as_completed(futures):
            name, status = fut.result()

    # =========== 最终数据合并 ===========
    print("\n所有计算完成，正在合并数据...")
    all_merged_dfs = []
    for item in all_cif_tasks:
        MOF_name = os.path.basename(item['path']).replace('.cif', '')
        csv_path = f'feature_folders/{MOF_name}/merged_descriptors/{MOF_name}_descriptors.csv'
        if os.path.exists(csv_path):
            try:
                all_merged_dfs.append(pd.read_csv(csv_path))
            except Exception:
                pass # 忽略极少数可能依然损坏的文件

    if all_merged_dfs:
        final_df = pd.concat(all_merged_dfs, ignore_index=True)
        # 确保输出目录存在
        os.makedirs(os.path.dirname(FINAL_CSV_OUTPUT_PATH), exist_ok=True)
        final_df.to_csv(FINAL_CSV_OUTPUT_PATH, index=False)
        print(f"\n================ 任务完成 ================")
        print(f"共成功收集并合并了 {len(final_df)} 个结构的数据！")
        print(f"最终汇总表已保存至: {FINAL_CSV_OUTPUT_PATH}")
    else:
        print("\n未能收集到任何特征数据。")