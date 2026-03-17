import os
import shutil

def get_unique_filename(target_folder, filename):
    """
    检查目标文件夹中是否存在同名文件。如果存在，则添加 (1), (2) 等后缀以防止覆盖。
    """
    target_path = os.path.join(target_folder, filename)
    if not os.path.exists(target_path):
        return target_path
    
    base_name, extension = os.path.splitext(filename)
    counter = 1
    
    while True:
        new_filename = f"{base_name}({counter}){extension}"
        new_target_path = os.path.join(target_folder, new_filename)
        if not os.path.exists(new_target_path):
            return new_target_path
        counter += 1

def extract_from_multiple_folders(ref_folders, src_folders, target_folder, prefix_to_remove="", action='copy'):
    # 1. 创建目标文件夹
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
        print(f"已创建目标文件夹: {target_folder}\n")

    # 2. 收集文件名并建立映射字典
    # 字典格式：{ '去掉前缀的搜索名' : '提取文件夹中的原始名' }
    reference_mapping = {}
    print("正在收集提取（参考）文件夹中的文件名...")
    for folder in ref_folders:
        if os.path.exists(folder):
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                if os.path.isfile(file_path):
                    # 核心逻辑：如果文件名带有指定前缀，则去掉前缀作为搜索依据
                    search_name = filename
                    if prefix_to_remove and filename.startswith(prefix_to_remove):
                        # 截取掉前缀部分，保留后面的真实名字
                        search_name = filename[len(prefix_to_remove):]
                    
                    reference_mapping[search_name] = filename
            print(f" - 已读取: {folder}")
        else:
            print(f" - [警告] 找不到提取文件夹: {folder}，已跳过。")
            
    total_ref_files = len(reference_mapping)
    print(f"共收集到 {total_ref_files} 个唯一的搜索文件名。\n")
    print("-" * 30)

    # 3. 遍历所有的寻找（源）文件夹并处理文件
    found_search_names = set() # 记录哪些“搜索暗号”被找到了
    match_count = 0         
    
    for folder in src_folders:
        if not os.path.exists(folder):
            print(f"[警告] 找不到寻找文件夹: {folder}，已跳过。")
            continue
            
        print(f"正在搜索文件夹: {folder} ...")
        for filename in os.listdir(folder):
            # 用源文件夹里的文件名去匹配我们的“搜索暗号”
            if filename in reference_mapping:
                source_path = os.path.join(folder, filename)

                if os.path.isfile(source_path):
                    # 这里默认复制过去的文件名保持为源文件名 (不带 optimized_ 的版本)
                    target_path = get_unique_filename(target_folder, filename)
                    
                    if action == 'copy':
                        shutil.copy2(source_path, target_path)
                    elif action == 'move':
                        shutil.move(source_path, target_path)
                    
                    found_search_names.add(filename)
                    match_count += 1
                    
    print("-" * 30)
    
    # 4. 统计与生成未找到的文件报告
    # 用所有的搜索暗号 减去 已经找到的搜索暗号
    unfound_search_names = set(reference_mapping.keys()) - found_search_names
    unfound_count = len(unfound_search_names)
    
    print("【任务统计】")
    print(f"目标要找的文件种类: {total_ref_files} 个")
    print(f"实际复制/移动的文件: {match_count} 个") 
    print(f"完全没有找到的文件: {unfound_count} 个")
    
    # 如果有没找到的文件，输出到文本文件中方便核对
    if unfound_count > 0:
        log_file = "未找到的文件清单.txt"
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(f"以下 {unfound_count} 个文件在所有的寻找文件夹中均未找到：\n")
            f.write(f"(注：已为你还原显示提取文件夹中的原始带有 '{prefix_to_remove}' 的文件名)\n")
            f.write("=" * 60 + "\n")
            
            # 为了方便你核对，报告里输出的是它原本带前缀的名字
            for search_name in sorted(unfound_search_names):
                original_name = reference_mapping[search_name]
                f.write(original_name + "\n")
                
        print(f"\n[提示] 未找到的文件列表已保存至当前目录下的 '{log_file}'，请查阅。")

# ==========================================
# 在这里修改为你的实际配置
# ==========================================
if __name__ == "__main__":
    # 需要被忽略的前缀（注意要包含下划线）
    PREFIX_TO_REMOVE = "optimized_"
    
    # 3 个提取文件夹（参考文件夹）路径
    REF_DIRS = [
        r"E:\MOF database\zenodo_upload_ultrastable\zenodo_upload_ultrastable\optimized_structures\1inorganic_1edge", 
        r"E:\MOF database\zenodo_upload_ultrastable\zenodo_upload_ultrastable\optimized_structures\1inorganic_1organic_1edge", 
        r"E:\MOF database\zenodo_upload_ultrastable\zenodo_upload_ultrastable\optimized_structures\2inorganic_1edge"
    ]
    
    # 3 个寻找文件夹（源文件夹）路径
    SRC_DIRS = [
        r"E:\MOF database\zenodo_upload_ultrastable\zenodo_upload_ultrastable\initial_structures\1inorganic_1edge", 
        r"E:\MOF database\zenodo_upload_ultrastable\zenodo_upload_ultrastable\initial_structures\1inorganic_1organic_1edge", 
        r"E:\MOF database\zenodo_upload_ultrastable\zenodo_upload_ultrastable\initial_structures\2inorganic_1edge"
    ]
    
    # 存放最终结果的目标文件夹
    TARGET_DIR = r"E:\CODE\cif2des\descriptors"

    # 执行脚本
    extract_from_multiple_folders(REF_DIRS, SRC_DIRS, TARGET_DIR, prefix_to_remove=PREFIX_TO_REMOVE, action='copy')