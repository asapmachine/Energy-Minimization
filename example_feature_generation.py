import os
import sys
import glob
import shutil
import subprocess
import numpy as np
import pandas as pd
from molSimplify.Informatics.MOF.MOF_descriptors import get_primitive


# Use molSimplify version 1.7.3 and pymatgen=2020.10.20
# Use Zeo++-0.3
# This script generates RAC and Zeo++ features for the specified CIFs.

### Start of functions ###

def convert_win_path_to_wsl(win_path):
    """将一个 Windows 路径（绝对或相对）转换为 WSL 兼容的路径。"""
    abs_win_path = os.path.abspath(win_path)
    abs_win_path = abs_win_path.replace('\\', '/')
    drive = abs_win_path[0].lower()
    return f'/mnt/{drive}/{abs_win_path[3:]}'


def delete_and_remake_folders(folder_names):
    """
    Deletes the folder specified by each item in folder_names if it exists, then remakes it.
    """
    for folder_name in folder_names:
        if os.path.isdir(folder_name):
            shutil.rmtree(folder_name)
        os.mkdir(folder_name)


def descriptor_generator(name, structure_path, wiggle_room, zeopp_executable_path):
    """
    descriptor_generator generates RAC and Zeo++ descriptors.
    """
    # Make the required folders for the current MOF.
    current_MOF_folder = f'feature_folders/{name}'
    cif_folder = f'{current_MOF_folder}/cifs'
    RACs_folder = f'{current_MOF_folder}/RACs'
    zeo_folder = f'{current_MOF_folder}/zeo++'
    merged_descriptors_folder = f'{current_MOF_folder}/merged_descriptors'
    delete_and_remake_folders([current_MOF_folder, cif_folder, RACs_folder, zeo_folder, merged_descriptors_folder])

    # Next, running MOF featurization.
    get_primitive_success = True

    try:
        # get_primitive 运行在当前的 Python 环境中
        get_primitive(structure_path, f'{cif_folder}/{name}_primitive.cif')
    except Exception as e:
        # 捕获所有异常，防止“占有率>1”等脏 CIF 文件导致主程序崩溃
        print(f'The primitive cell of {name} could not be found. Reason: {e}')
        get_primitive_success = False

    if get_primitive_success:
        structure_path = f'{cif_folder}/{name}_primitive.cif'

    # --- 开始转换路径以适应 WSL ---
    wsl_structure_path = convert_win_path_to_wsl(structure_path)
    wsl_pd_txt = convert_win_path_to_wsl(f'{zeo_folder}/{name}_pd.txt')
    wsl_sa_txt = convert_win_path_to_wsl(f'{zeo_folder}/{name}_sa.txt')
    wsl_pov_txt = convert_win_path_to_wsl(f'{zeo_folder}/{name}_pov.txt')
    # -----------------------------

    # 在 Windows CMD 中静默输出
    dev_null = "> NUL 2>&1"

    cmd1 = f'{zeopp_executable_path} -ha -res {wsl_pd_txt} "{wsl_structure_path}" {dev_null}'
    cmd2 = f'{zeopp_executable_path} -sa 1.86 1.86 10000 {wsl_sa_txt} "{wsl_structure_path}" {dev_null}'
    cmd3 = f'{zeopp_executable_path} -volpo 1.86 1.86 10000 {wsl_pov_txt} "{wsl_structure_path}" {dev_null}'

    # RAC_getter.py 直接使用当前的 Windows Python 环境执行
    cmd4 = f'"{sys.executable}" RAC_getter.py "{structure_path}" "{name}" "{RACs_folder}" {wiggle_room}'

    # four parallelized Zeo++ and RAC commands
    process1 = subprocess.Popen(cmd1, stdout=subprocess.PIPE, stderr=None, shell=True)
    process2 = subprocess.Popen(cmd2, stdout=subprocess.PIPE, stderr=None, shell=True)
    process3 = subprocess.Popen(cmd3, stdout=subprocess.PIPE, stderr=None, shell=True)
    process4 = subprocess.Popen(cmd4, stdout=subprocess.PIPE, stderr=None, shell=True)

    process1.communicate()
    process2.communicate()
    process3.communicate()
    process4.communicate()

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
    geo_df.to_csv(f'{zeo_folder}/geometric_parameters.csv', index=False)

    # error handling for cmd4
    log_file_path = f'{RACs_folder}/RAC_getter_log.txt'
    if os.path.exists(log_file_path):
        with open(log_file_path, 'r') as f:
            if 'FAILED' in f.readline():
                print(f'RAC generation failed for {name}. Continuing to next MOF.')
                return
    else:
        print(f'RAC generation log missing for {name}. It might have crashed. Continuing.')
        return

    # Merging geometric information with RAC information
    try:
        lc_df = pd.read_csv(f"{RACs_folder}/lc_descriptors.csv")
        sbu_df = pd.read_csv(f"{RACs_folder}/sbu_descriptors.csv")
        linker_df = pd.read_csv(f"{RACs_folder}/linker_descriptors.csv")

        # 1. 剔除包含文本的 'name' 列，防止 pandas 报错
        if 'name' in lc_df.columns: lc_df = lc_df.drop(columns=['name'])
        if 'name' in sbu_df.columns: sbu_df = sbu_df.drop(columns=['name'])
        if 'name' in linker_df.columns: linker_df = linker_df.drop(columns=['name'])

        # 2. 恢复原作者的做法：暴力转成数字并求平均值 (这会抹除文本特征并还原数值基准)
        lc_df = lc_df.apply(pd.to_numeric, errors='coerce').mean().to_frame().transpose()
        sbu_df = sbu_df.apply(pd.to_numeric, errors='coerce').mean().to_frame().transpose()
        linker_df = linker_df.apply(pd.to_numeric, errors='coerce').mean().to_frame().transpose()

        merged_df = pd.concat([geo_df, lc_df, sbu_df, linker_df], axis=1)
        merged_df.to_csv(f'{merged_descriptors_folder}/{name}_descriptors.csv', index=False)
    except FileNotFoundError:
        print(f"RAC descriptor files missing for {name}. Featurization incomplete.")
    except Exception as e:
        print(f"Error merging DataFrames for {name}: {e}")

### End of functions ###

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python example_feature_generation.py <path_to_cif_directory>")
        sys.exit(1)

    structure_directory = sys.argv[1]
    ZEOpp_EXECUTABLE_PATH = 'wsl /home/lilyaf/zeo++-0.3/network'

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
    for i, cp in enumerate(cif_paths):
        MOF_name = os.path.basename(cp).replace('.cif', '')

        if os.path.exists(f'feature_folders/{MOF_name}/merged_descriptors/{MOF_name}_descriptors.csv'):
            print(f'Skipping {MOF_name}, features already exist.')
            continue

        print(f'The current MOF is {MOF_name}. Number {i + 1} of {len(cif_paths)}.')
        wiggle_room = 1
        descriptor_generator(MOF_name, cp, wiggle_room, ZEOpp_EXECUTABLE_PATH)

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

    # 动态拼接所有列，无需硬编码列名
    if all_merged_dfs:
        final_df = pd.concat(all_merged_dfs, ignore_index=True)

        # 确保 name 和 cif_file 列排在最前面，方便查看
        cols = final_df.columns.tolist()
        if 'cif_file' in cols:
            cols.insert(0, cols.pop(cols.index('cif_file')))
        if 'name' in cols:
            cols.insert(0, cols.pop(cols.index('name')))

        final_df = final_df[cols]

        final_df = final_df.sort_values(by=['name'])
        final_df.to_csv('modified_RAC_and_zeo_features.csv', index=False)
        print(f"\nSuccessfully collected features for {len(all_merged_dfs)} MOFs.")
    else:
        print("\nNo features were successfully collected.")

    print(f'Finished! Unsuccessful featurizations: {unsuccessful_featurizations}')