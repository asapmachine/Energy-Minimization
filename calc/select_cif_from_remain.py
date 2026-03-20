import os
import shutil

def extract_unique_folders(folder_a, folder_b, folder_c):
    """
    对比文件夹 A 和 B，将 A 中存在但 B 中不存在的子文件夹，提取到文件夹 C 中。
    """
    # 1. 安全检查：确保目标文件夹 C 存在，不存在则创建
    if not os.path.exists(folder_c):
        os.makedirs(folder_c)
        print(f"已创建目标文件夹: {folder_c}")

    # 2. 获取 A 和 B 中【仅限文件夹】的名称列表，并转化为集合 (Set)
    try:
        # 遍历目录，并使用 os.path.isdir 严格筛选出文件夹，排除普通文件
        folders_a = {f for f in os.listdir(folder_a) if os.path.isdir(os.path.join(folder_a, f))}
        folders_b = {f for f in os.listdir(folder_b) if os.path.isdir(os.path.join(folder_b, f))}
    except FileNotFoundError as e:
        print(f"错误: 找不到指定的源文件夹。详细信息: {e}")
        return

    # 3. 核心逻辑：利用集合求差集 (A - B)
    unique_to_a = folders_a - folders_b

    # 4. 检查是否有需要提取的文件夹
    if not unique_to_a:
        print("完美！文件夹 A 中的所有子文件夹都已经存在于文件夹 B 中，没有需要提取的内容。")
        return

    print(f"在 A 中但不在 B 中的文件夹共有 {len(unique_to_a)} 个，开始提取...\n")

    # 5. 遍历差集，执行文件夹拷贝操作
    success_count = 0
    for folder_name in unique_to_a:
        source_path = os.path.join(folder_a, folder_name)
        target_path = os.path.join(folder_c, folder_name)

        try:
            # 复制整个文件夹（包含其内部的所有文件和子文件夹）
            # 【注意：如果您想“剪切/移动”而不是复制，请把下面这行换成：shutil.move(source_path, target_path)】
            shutil.copytree(source_path, target_path)
            print(f"  [成功] 提取文件夹: {folder_name}")
            success_count += 1
        except FileExistsError:
            print(f"  [跳过] 文件夹 C 中已经存在同名文件夹，无法覆盖: {folder_name}")
        except Exception as e:
            print(f"  [失败] 无法提取 {folder_name}，原因: {e}")

    print(f"\n执行完毕！共成功提取 {success_count} 个文件夹到 '{folder_c}'。")

# ==========================================
# 运行区域（请根据您的实际情况修改这里的路径）
# ==========================================
if __name__ == "__main__":
    DIR_A = r'E:\CODE\cif2des\feature_folders_2'  # 文件夹 A 的路径
    DIR_B = r'E:\CODE\cif2des\feature_folders'  # 文件夹 B 的路径
    DIR_C = r'E:\CODE\cif2des\error\remain'  # 文件夹 C 的路径
    
    extract_unique_folders(DIR_A, DIR_B, DIR_C)