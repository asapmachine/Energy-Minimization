import os
import sys
import glob
import shutil
import subprocess
import numpy as np
import pandas as pd
import platform
import time
import re
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed

# This script generates ONLY Zeo++ features directly from the provided CIFs.

### Start of functions ###

def is_wsl():
    # Detect WSL so the script can run both on Windows and inside WSL.
    if platform.system().lower() != "linux":
        return False
    if os.environ.get("WSL_INTEROP") or os.environ.get("WSL_DISTRO_NAME"):
        return True
    try:
        with open("/proc/version", "r", encoding="utf-8", errors="ignore") as f:
            return "microsoft" in f.read().lower()
    except OSError:
        return False

def convert_win_path_to_wsl(win_path):
    """将一个 Windows 路径（绝对或相对）转换为 WSL 兼容的路径。"""
    # If we are already running inside WSL, keep Linux paths unchanged.
    if is_wsl():
        return os.path.abspath(win_path)
    abs_win_path = os.path.abspath(win_path)
    abs_win_path = abs_win_path.replace('\\', '/')
    drive = abs_win_path[0].lower()
    return f'/mnt/{drive}/{abs_win_path[3:]}'

def normalize_input_cif_dir(path_str):
    """标准化输入目录：在 WSL 下支持 E:\\... 这类 Windows 路径。"""
    p = (path_str or "").strip().strip('"').strip("'")
    if not p:
        return p

    # WSL: convert Windows drive path to /mnt/<drive>/...
    if is_wsl() and re.match(r'^[A-Za-z]:[\\/]', p):
        drive = p[0].lower()
        rest = p[2:].replace('\\', '/').lstrip('/').lstrip('\\')
        return f"/mnt/{drive}/{rest}"

    return os.path.abspath(os.path.expanduser(p))

def delete_and_remake_folders(folder_names):
    """
    Deletes the folder specified by each item in folder_names if it exists, then remakes it.
    """
    for folder_name in folder_names:
        if os.path.isdir(folder_name):
            shutil.rmtree(folder_name)
        os.mkdir(folder_name)

def _resolve_zeopp_network_path(candidate):
    if not candidate:
        return None
    # If it's an absolute/relative path, validate it; otherwise, try PATH lookup.
    if os.path.sep in candidate or candidate.startswith("."):
        p = os.path.expanduser(candidate)
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
        return None
    p = shutil.which(candidate)
    return p

def choose_zeopp_network():
    # Prefer explicit env override.
    env_path = os.environ.get("ZEOPP_NETWORK") or os.environ.get("ZEOpp_NETWORK") or os.environ.get("ZEO_NETWORK")
    resolved = _resolve_zeopp_network_path(env_path)
    if resolved:
        return resolved

    # Common install locations inside WSL/Linux.
    candidates = [
        "/home/lilyaf/zeo++-0.3/network",
        os.path.join(os.path.expanduser("~"), "zeo++-0.3", "network"),
        "network",
    ]
    for c in candidates:
        resolved = _resolve_zeopp_network_path(c)
        if resolved:
            return resolved
    return None

def run_zeopp(cmd_args, debug=False, timeout_sec=0):
    # Run Zeo++ and return (returncode, stderr_preview).
    if debug:
        print("Running:", " ".join(cmd_args))

    try:
        proc = subprocess.run(
            cmd_args,
            stdout=subprocess.PIPE if debug else subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=(timeout_sec if timeout_sec and timeout_sec > 0 else None),
        )
    except subprocess.TimeoutExpired:
        return 124, f"Timed out after {timeout_sec}s"

    stderr_text = (proc.stderr or "").strip()
    if debug and proc.returncode != 0:
        print("Zeo++ failed:", " ".join(cmd_args))
        if stderr_text:
            print(stderr_text[:4000])

    if proc.returncode == 0:
        return 0, ""

    # Keep preview short to avoid flooding logs.
    preview_lines = stderr_text.splitlines()[:3] if stderr_text else []
    return proc.returncode, "\n".join(preview_lines)

def descriptor_generator(name, structure_path, zeopp_executable_path):
    """
    descriptor_generator generates Zeo++ descriptors directly from the original CIF.
    """
    # Make the required folders for the current MOF.
    current_MOF_folder = f'feature_folders/{name}'
    cif_folder = f'{current_MOF_folder}/cifs'
    zeo_folder = f'{current_MOF_folder}/zeo++'
    merged_descriptors_folder = f'{current_MOF_folder}/merged_descriptors'
    
    # 仅创建需要的文件夹
    delete_and_remake_folders([current_MOF_folder, cif_folder, zeo_folder, merged_descriptors_folder])

    # 将原始 CIF 复制到 cif_folder 以保持文件结构完整
    target_cif_path = f'{cif_folder}/{name}.cif'
    shutil.copy(structure_path, target_cif_path)

    # --- 开始转换路径以适应 WSL ---
    wsl_structure_path = convert_win_path_to_wsl(target_cif_path)
    wsl_pd_txt = convert_win_path_to_wsl(f'{zeo_folder}/{name}_pd.txt')
    wsl_sa_txt = convert_win_path_to_wsl(f'{zeo_folder}/{name}_sa.txt')
    wsl_pov_txt = convert_win_path_to_wsl(f'{zeo_folder}/{name}_pov.txt')
    # -----------------------------

    # Zeo++ commands. Use 1.86 A probe radius.
    debug = os.environ.get("DEBUG", "").strip() in {"1", "true", "True", "YES", "yes"}
    timeout_sec = int(os.environ.get("ZEO_TIMEOUT", "0") or "0")
    # zeopp_executable_path can be a string (WSL/Linux) or a list (Windows calling into WSL).
    zeo_prefix = zeopp_executable_path if isinstance(zeopp_executable_path, list) else [zeopp_executable_path]

    t0 = time.time()
    rc1, err1 = run_zeopp(
        zeo_prefix + ["-ha", "-res", wsl_pd_txt, wsl_structure_path],
        debug=debug,
        timeout_sec=timeout_sec,
    )
    rc2, err2 = run_zeopp(
        zeo_prefix + ["-sa", "1.86", "1.86", "10000", wsl_sa_txt, wsl_structure_path],
        debug=debug,
        timeout_sec=timeout_sec,
    )
    rc3, err3 = run_zeopp(
        zeo_prefix + ["-volpo", "1.86", "1.86", "10000", wsl_pov_txt, wsl_structure_path],
        debug=debug,
        timeout_sec=timeout_sec,
    )
    if debug:
        print(f"Zeo++ elapsed for {name}: {time.time() - t0:.1f}s")

    if (rc1 != 0) or (rc2 != 0) or (rc3 != 0):
        print(f"Zeo++ failed for {name}: ha={rc1}, sa={rc2}, pov={rc3}. Set DEBUG=1 for details.")
        preview = err1 or err2 or err3
        if preview:
            print(preview)

    # 读取 Zeo++ 输出文件
    dict_list = []
    cif_file = name + '.cif'
    largest_included_sphere, largest_free_sphere, largest_included_sphere_along_free_sphere_path = np.nan, np.nan, np.nan
    unit_cell_volume, crystal_density, VSA, GSA = np.nan, np.nan, np.nan, np.nan
    VPOV, GPOV = np.nan, np.nan
    POAV, PONAV, GPOAV, GPONAV, POAV_volume_fraction, PONAV_volume_fraction = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    if (os.path.exists(f'{zeo_folder}/{name}_pd.txt') & os.path.exists(f'{zeo_folder}/{name}_sa.txt') &
            os.path.exists(f'{zeo_folder}/{name}_pov.txt')):
        with open(f'{zeo_folder}/{name}_pd.txt') as f:
            pore_diameter_data = f.readlines()
            for row in pore_diameter_data:
                largest_included_sphere = float(row.split()[1])
                largest_free_sphere = float(row.split()[2])
                largest_included_sphere_along_free_sphere_path = float(row.split()[3])
        with open(f'{zeo_folder}/{name}_sa.txt') as f:
            surface_area_data = f.readlines()
            for i, row in enumerate(surface_area_data):
                if i == 0:
                    unit_cell_volume = float(row.split('Unitcell_volume:')[1].split()[0])
                    crystal_density = float(row.split('Density:')[1].split()[0])
                    VSA = float(row.split('ASA_m^2/cm^3:')[1].split()[0])
                    GSA = float(row.split('ASA_m^2/g:')[1].split()[0])
        with open(f'{zeo_folder}/{name}_pov.txt') as f:
            pore_volume_data = f.readlines()
            for i, row in enumerate(pore_volume_data):
                if i == 0:
                    density = float(row.split('Density:')[1].split()[0])
                    POAV = float(row.split('POAV_A^3:')[1].split()[0])
                    PONAV = float(row.split('PONAV_A^3:')[1].split()[0])
                    GPOAV = float(row.split('POAV_cm^3/g:')[1].split()[0])
                    GPONAV = float(row.split('PONAV_cm^3/g:')[1].split()[0])
                    POAV_volume_fraction = float(row.split('POAV_Volume_fraction:')[1].split()[0])
                    PONAV_volume_fraction = float(row.split('PONAV_Volume_fraction:')[1].split()[0])
                    VPOV = POAV_volume_fraction + PONAV_volume_fraction
                    GPOV = VPOV / density
    else:
        print(f'Not all 3 files exist for {name}, so at least one Zeo++ call failed!', 'sa: ',
              os.path.exists(f'{zeo_folder}/{name}_sa.txt'),
              '; pd: ', os.path.exists(f'{zeo_folder}/{name}_pd.txt'), '; pov: ',
              os.path.exists(f'{zeo_folder}/{name}_pov.txt'))

    geo_dict = {'name': name, 'cif_file': cif_file, 'Di': largest_included_sphere, 'Df': largest_free_sphere,
                'Dif': largest_included_sphere_along_free_sphere_path,
                'rho': crystal_density, 'VSA': VSA, 'GSA': GSA, 'VPOV': VPOV, 'GPOV': GPOV,
                'POAV_vol_frac': POAV_volume_fraction,
                'PONAV_vol_frac': PONAV_volume_fraction, 'GPOAV': GPOAV, 'GPONAV': GPONAV, 'POAV': POAV, 'PONAV': PONAV}
    
    dict_list.append(geo_dict)
    geo_df = pd.DataFrame(dict_list)
    
    # 存入 zeo++ 文件夹
    geo_df.to_csv(f'{zeo_folder}/geometric_parameters.csv', index=False)
    
    # 直接将结果作为最终特征存入 merged_descriptors 文件夹
    geo_df.to_csv(f'{merged_descriptors_folder}/{name}_descriptors.csv', index=False)

def _process_one_mof(task):

    i, total, cp, zeopp_path = task
    MOF_name = os.path.basename(cp).replace('.cif', '')
    out_csv = f'feature_folders/{MOF_name}/merged_descriptors/{MOF_name}_descriptors.csv'

    if os.path.exists(out_csv):
        return MOF_name, 'skipped'

    print(f'The current MOF is {MOF_name}. Number {i + 1} of {total}.')
    descriptor_generator(MOF_name, cp, zeopp_path)
    return MOF_name, 'done'

### End of functions ###

if __name__ == "__main__":
    # Input CIF directory:
    # - If an argument is provided, use it.
    # - Otherwise, default to a sibling folder named "cifs" next to this script.
    if len(sys.argv) >= 2:
        structure_directory = normalize_input_cif_dir(sys.argv[1])
        print(f'Using input CIF directory: {structure_directory}')
    else:
        default_cifs_dir = os.path.join(os.path.dirname(__file__), "cifs")
        if os.path.isdir(default_cifs_dir):
            structure_directory = default_cifs_dir
            print(f'No input directory provided; using default: {structure_directory}')
        else:
            print("Usage: python feature_generation_only_zeo.py <path_to_cif_directory>")
            sys.exit(1)

    if is_wsl():
        ZEOpp_EXECUTABLE_PATH = choose_zeopp_network()
        if not ZEOpp_EXECUTABLE_PATH:
            print("Could not find Zeo++ 'network' executable.")
            print("Set env var ZEOPP_NETWORK=/full/path/to/network, or put 'network' on PATH.")
            sys.exit(2)
    else:
        # On Windows, prefix with "wsl" to run the Linux binary.
        zeopp_wsl_bin = "/home/lilyaf/zeo++-0.3/network"
        ZEOpp_EXECUTABLE_PATH = ["wsl", zeopp_wsl_bin]

    if os.environ.get("DEBUG", "").strip() in {"1", "true", "True", "YES", "yes"}:
        print("Using Zeo++ network:", ZEOpp_EXECUTABLE_PATH)
    if not os.path.exists('feature_folders'):
        os.mkdir('feature_folders')

    cif_paths = glob.glob(f'{structure_directory}/*.cif')
    cif_paths.sort()

    if not cif_paths:
        print(f"No .cif files found in {structure_directory}")
        sys.exit(0)

    '''
    Generating features for all MOFs.
    '''
    start_index = int(os.environ.get("START_INDEX", "0") or "0")
    max_mofs = int(os.environ.get("MAX_MOFS", "0") or "0")
    if start_index or max_mofs:
        end_index = (start_index + max_mofs) if max_mofs else None
        cif_paths = cif_paths[start_index:end_index]
        print(f"Processing CIFs slice: start={start_index}, max={max_mofs or 'all'}, total={len(cif_paths)}")


    physical_cores = multiprocessing.cpu_count()

    auto_workers = max(1, int(physical_cores * 1.5))
    
    n_workers = int(os.environ.get("N_WORKERS", str(auto_workers)) or str(auto_workers))
    n_workers = max(1, n_workers)
    
    if n_workers > 1:
        print(f"Parallel mode enabled: N_WORKERS={n_workers} (CPU Cores detected: {physical_cores})")
        tasks = [(i, len(cif_paths), cp, ZEOpp_EXECUTABLE_PATH) for i, cp in enumerate(cif_paths)]
        # 增加 thread_name_prefix 方便系统级监控资源
        with ThreadPoolExecutor(max_workers=n_workers, thread_name_prefix="Zeo_Worker") as ex:
            futures = [ex.submit(_process_one_mof, t) for t in tasks]
            for fut in as_completed(futures):
                name, status = fut.result()
                if status == 'skipped':
                    print(f'Skipping {name}, features already exist.')
    else:
        for i, cp in enumerate(cif_paths):
            MOF_name = os.path.basename(cp).replace('.cif', '')

            if os.path.exists(f'feature_folders/{MOF_name}/merged_descriptors/{MOF_name}_descriptors.csv'):
                print(f'Skipping {MOF_name}, features already exist.')
                continue

            print(f'The current MOF is {MOF_name}. Number {i + 1} of {len(cif_paths)}.')
            
            descriptor_generator(MOF_name, cp, ZEOpp_EXECUTABLE_PATH)

    '''
    Collecting MOF features together across all MOFs.
    '''
    all_merged_dfs = []
    unsuccessful_featurizations = []

    MOF_names = [os.path.basename(i).replace('.cif', '') for i in cif_paths]

    for MOF_name in MOF_names:
        csv_path = f'feature_folders/{MOF_name}/merged_descriptors/{MOF_name}_descriptors.csv'
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            all_merged_dfs.append(df)
        else:
            unsuccessful_featurizations.append(MOF_name)

    # 动态拼接所有列
    if all_merged_dfs:
        final_df = pd.concat(all_merged_dfs, ignore_index=True)

        cols = final_df.columns.tolist()
        if 'cif_file' in cols:
            cols.insert(0, cols.pop(cols.index('cif_file')))
        if 'name' in cols:
            cols.insert(0, cols.pop(cols.index('name')))

        final_df = final_df[cols]
        final_df = final_df.sort_values(by=['name'])
        

        final_df.to_csv('zeo_features_only.csv', index=False)
        print(f"\nSuccessfully collected features for {len(all_merged_dfs)} MOFs.")
    else:
        print("\nNo features were successfully collected.")

    print(f'Finished! Unsuccessful featurizations: {unsuccessful_featurizations}')