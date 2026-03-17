import os
import shutil
import random
from pathlib import Path


def process_cif_files(folder_a, folder_b, folder_c):
    path_a = Path(folder_a)
    path_b = Path(folder_b)
    path_c = Path(folder_c)

    # 1. 确保 C 文件夹存在
    path_c.mkdir(parents=True, exist_ok=True)

    # 2. 获取 A 文件夹中所有“子文件夹”的名字
    # 使用 set 存储，去掉路径只保留文件夹名
    subfolder_names = {f.name for f in path_a.iterdir() if f.is_dir()}
    print(f"在 A 中识别到 {len(subfolder_names)} 个子文件夹。")

    # 3. 遍历 B 文件夹中的 .cif 文件并清理
    removed_count = 0
    # 只针对 .cif 文件进行操作
    for cif_file in path_b.glob("*.cif"):
        # cif_file.stem 获取的是不带后缀的文件名，例如 'MOF_1'
        if cif_file.stem in subfolder_names:
            cif_file.unlink()  # 执行物理删除
            removed_count += 1

    print(f"从 B 中删除了 {removed_count} 个与 A 同名的 .cif 文件。")

    # 4. 获取 B 中剩余的 .cif 文件列表
    remaining_cifs = list(path_b.glob("*.cif"))
    total_remaining = len(remaining_cifs)

    # 5. 随机抽取 1/2
    sample_size = total_remaining // 2
    print(f"B 中剩余 {total_remaining} 个 .cif 文件，准备提取 {sample_size} 个。")

    # 随机打乱列表
    random.seed(42)  # 设置随机种子，保证结果可复现（可选）
    random.shuffle(remaining_cifs)

    # 6. 移动到 C 文件夹
    moved_count = 0
    for i in range(sample_size):
        target_file = remaining_cifs[i]
        shutil.move(str(target_file), str(path_c / target_file.name))
        moved_count += 1

    print(f"任务完成！已移动 {moved_count} 个 .cif 文件到 {folder_c}。")

# --- 使用时替换你的路径 ---
process_cif_files(r'E:\CODE\cif2des\feature_folders', r'E:\CODE\cif2des\descriptors', r'E:\CODE\cif2des\help_cifs')