import sys
import traceback
from molSimplify.Informatics.MOF.MOF_descriptors import get_MOF_descriptors


# This is called by example_feature_generation.py.

def main():
    # user command line inputs
    structure_path = sys.argv[1]
    name = sys.argv[2]  # name of the MOF
    RACs_folder = sys.argv[3]
    wiggle_room = float(sys.argv[4])

    # result log
    f = open(f'{RACs_folder}/RAC_getter_log.txt', 'w')

    try:
        # 恢复作者独有的图切割高级参数，并保留 1.7.3 版本的正确参数名 xyzpath 和 6000 原子解封
        full_names, full_descriptors = get_MOF_descriptors(
            f'{structure_path}',
            3,
            path=RACs_folder,
            xyzpath=f'{RACs_folder}/{name}.xyz',  # 1.7.3 版必须且只能用无下划线的 xyzpath
            wiggle_room=wiggle_room,
            max_num_atoms=6000,
            get_sbu_linker_bond_info=True,  # 恢复: 提取深度成键信息
            surrounded_sbu_file_generation=True,  # 恢复: 生成特定文件
            detect_1D_rod_sbu=True  # 恢复: 1D棒状节点识别
        )

        if (len(full_names) <= 1) and (len(full_descriptors) <= 1):  # This is a featurization check
            f.write('FAILED - Featurization error')
            f.close()

    except Exception as e:
        f.write(f'FAILED - Error: {e}')
        f.close()
        print(f"\n[RAC_getter.py ERROR] for {name}: {e}\n")
        traceback.print_exc()


if __name__ == "__main__":
    main()