import os
import sys

# =============================================================
# 🔧 核心修复：自动寻找并注入 OpenBabel 的数据路径 (解决 Windows conda 报错)
# =============================================================
conda_prefix = os.environ.get('CONDA_PREFIX', sys.prefix)
babel_share_dir = os.path.join(conda_prefix, 'share', 'openbabel')
if os.path.exists(babel_share_dir):
    for item in os.listdir(babel_share_dir):
        full_path = os.path.join(babel_share_dir, item)
        if os.path.isdir(full_path) and os.path.exists(os.path.join(full_path, 'space-groups.txt')):
            os.environ['BABEL_DATADIR'] = full_path
            print(f"🔧 已自动修复 OpenBabel 数据路径: {full_path}")
            break
# =============================================================

import glob
import shutil
import warnings
import traceback
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
# 直接导入所需的所有 molSimplify 函数
from molSimplify.Informatics.MOF.MOF_descriptors import get_primitive, get_MOF_descriptors

# 屏蔽所有的 UserWarning 和 FutureWarning，保持终端极致清爽
warnings.filterwarnings("ignore")

def delete_and_remake_folders(folder_names):
    """静默删除并重建文件夹"""
    for folder_name in folder_names:
        if os.path.isdir(folder_name):
            try:
                shutil.rmtree(folder_name)
            except Exception:
                pass # 忽略多进程可能带来的文件占用冲突
        os.makedirs(folder_name, exist_ok=True)

def process_single_mof(cif_path):
    """
    独立的工作进程函数：涵盖了特征提取的所有逻辑。
    """
    MOF_name = os.path.basename(cif_path).replace('.cif', '')
    wiggle_room = 1.0
    
    current_MOF_folder = f'feature_folders/{MOF_name}'
    cif_folder = f'{current_MOF_folder}/cifs'
    RACs_folder = f'{current_MOF_folder}/RACs'
    merged_descriptors_folder = f'{current_MOF_folder}/merged_descriptors'
    final_csv_path = f'{merged_descriptors_folder}/{MOF_name}_descriptors.csv'
    
    # 再次兜底检查：如果在进程分配期间发现算过了，直接跳过
    if os.path.exists(final_csv_path):
        return MOF_name, "SKIPPED"

    # 初始化文件夹 (会清理掉之前的脏数据)
    delete_and_remake_folders([current_MOF_folder, cif_folder, RACs_folder, merged_descriptors_folder])

    # 尝试提取原胞 (Primitive Cell)
    structure_path = cif_path
    try:
        primitive_path = f'{cif_folder}/{MOF_name}_primitive.cif'
        get_primitive(structure_path, primitive_path)
        if os.path.exists(primitive_path):
            structure_path = primitive_path
    except Exception:
        pass 

    # 核心计算：极速纯 RACs 提取
    try:
        # 重定向 stdout 和 stderr，防止底层 C++ 报错打断进度条
        devnull = open(os.devnull, 'w')
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = devnull, devnull

        try:
            full_names, full_descriptors = get_MOF_descriptors(
                f'{structure_path}',
                3,
                path=RACs_folder,
                xyzpath=f'{RACs_folder}/{MOF_name}.xyz', 
                wiggle_room=wiggle_room,
                max_num_atoms=6000,
                # 🚀 极速纯 RACs 提取参数
                get_sbu_linker_bond_info=False,       
                surrounded_sbu_file_generation=False, 
                detect_1D_rod_sbu=False               
            )
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr
            devnull.close()

        if (len(full_names) <= 1) and (len(full_descriptors) <= 1):
            return MOF_name, "FAILED"

    except Exception:
        return MOF_name, "FAILED"

    # 特征提取与合并
    try:
        lc_csv = f"{RACs_folder}/lc_descriptors.csv"
        sbu_csv = f"{RACs_folder}/sbu_descriptors.csv"
        linker_csv = f"{RACs_folder}/linker_descriptors.csv"

        if not (os.path.exists(lc_csv) and os.path.exists(sbu_csv) and os.path.exists(linker_csv)):
            return MOF_name, "FAILED"

        lc_df = pd.read_csv(lc_csv)
        sbu_df = pd.read_csv(sbu_csv)
        linker_df = pd.read_csv(linker_csv)

        if 'name' in lc_df.columns: lc_df = lc_df.drop(columns=['name'])
        if 'name' in sbu_df.columns: sbu_df = sbu_df.drop(columns=['name'])
        if 'name' in linker_df.columns: linker_df = linker_df.drop(columns=['name'])

        lc_df = lc_df.apply(pd.to_numeric, errors='coerce').mean().to_frame().transpose()
        sbu_df = sbu_df.apply(pd.to_numeric, errors='coerce').mean().to_frame().transpose()
        linker_df = linker_df.apply(pd.to_numeric, errors='coerce').mean().to_frame().transpose()

        merged_df = pd.concat([lc_df, sbu_df, linker_df], axis=1)
        merged_df.insert(0, 'cif_file', f"{MOF_name}.cif")
        merged_df.insert(0, 'name', MOF_name)

        merged_df.to_csv(final_csv_path, index=False)
        return MOF_name, "SUCCESS"
    
    except Exception:
        return MOF_name, "FAILED"

# ==========================================
# 主程序入口
# ==========================================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_RAC_only.py <path_to_cif_directory>")
        sys.exit(1)

    structure_directory = sys.argv[1]

    if not os.path.exists('feature_folders'):
        os.mkdir('feature_folders')

    all_cif_paths = glob.glob(f'{structure_directory}/*.cif')
    all_cif_paths.sort()

    if not all_cif_paths:
        print(f"❌ 未在 {structure_directory} 中找到任何 .cif 文件！")
        sys.exit(0)

    # =========================================================
    # 🚀 真正的预过滤断点续算机制：启动前剔除已完成任务
    # =========================================================
    cif_paths_to_run = []
    already_done_count = 0

    print("🔍 正在扫描已完成的任务记录，请稍候...")
    for cp in all_cif_paths:
        MOF_name = os.path.basename(cp).replace('.cif', '')
        # 如果你之前算好的文件夹不叫 feature_folders，需要在这里修改路径！
        expected_csv = f'feature_folders/{MOF_name}/merged_descriptors/{MOF_name}_descriptors.csv'
        
        if os.path.exists(expected_csv):
            already_done_count += 1
        else:
            cif_paths_to_run.append(cp)

    print(f"📂 总共发现 CIF 文件: {len(all_cif_paths)}")
    print(f"✅ 检测到已完成并跳过: {already_done_count}")
    print(f"⏳ 本次实际需要计算: {len(cif_paths_to_run)}\n")

    if cif_paths_to_run:
        MAX_WORKERS = min(20, os.cpu_count() or 4)
        print(f"🚀 启动并发加速引擎！开启 {MAX_WORKERS} 个独立进程同时计算...")
        print(f"⚡ 已精简计算参数，进入极速纯 RACs 提取模式！\n")
        
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # 仅将未完成的任务提交给进程池
            futures = {executor.submit(process_single_mof, cp): cp for cp in cif_paths_to_run}
            
            # 进度条总数对应剩余未计算的任务
            for future in tqdm(as_completed(futures), total=len(cif_paths_to_run), desc="高通量提取 RACs", unit="MOF", dynamic_ncols=True, colour='green'):
                pass 
    else:
        print("🎉 所有特征提取任务均已完成，直接进入特征拼接阶段！\n")

    # =========================================================
    # 收集特征阶段 (无论是否全部跳过，都会执行拼接)
    # =========================================================
    print("📦 正在收集并拼接所有特征矩阵...")
    all_merged_dfs = []
    unsuccessful_featurizations = []

    MOF_names = [os.path.basename(i).replace('.cif', '') for i in all_cif_paths]

    for MOF_name in MOF_names:
        csv_path = f'feature_folders/{MOF_name}/merged_descriptors/{MOF_name}_descriptors.csv'
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            all_merged_dfs.append(df)
        else:
            unsuccessful_featurizations.append(MOF_name)

    if all_merged_dfs:
        final_df = pd.concat(all_merged_dfs, ignore_index=True)

        cols = final_df.columns.tolist()
        if 'cif_file' in cols: cols.insert(0, cols.pop(cols.index('cif_file')))
        if 'name' in cols: cols.insert(0, cols.pop(cols.index('name')))

        final_df = final_df[cols]
        final_df = final_df.sort_values(by=['name'])
        
        final_df.to_csv('RAC_features_only.csv', index=False)
        print(f"✅ 成功提取并收集了 {len(all_merged_dfs)} 个 MOF 的特征！")
    else:
        print("❌ 未能成功提取任何特征。")

    if unsuccessful_featurizations:
        print(f"⚠️ 以下 {len(unsuccessful_featurizations)} 个 MOF 提取失败，可能是结构极其复杂或原胞破缺：")
        print(unsuccessful_featurizations[:20], "..." if len(unsuccessful_featurizations) > 20 else "")
    
    print("\n🎉 脚本运行结束！")