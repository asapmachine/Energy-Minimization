import pandas as pd

def extract_topology(input_file, output_file):
    print(f"正在读取表格: {input_file} ...")
    # 读取CSV文件
    df = pd.read_csv(input_file)
    
    # 检查是否存在 'name' 列
    if 'name' not in df.columns:
        print("错误：表格中找不到 'name' 列，请检查文件！")
        return
        
    print("正在提取 Topology 信息...")
    # 使用正则表达式提取 'net-' 和 '_' 之间的内容
    # r'net-([a-zA-Z]+)_' 意思是：匹配 'net-' 开头，'_' 结尾中间的所有英文字母
    # 如果你的拓扑结构含有数字或横杠（比如 bcu-a），可以把正则换成 r'net-([^_]+)_'
    df['Topology'] = df['name'].str.extract(r'net-([a-zA-Z]+)_')
    
    # 【可选优化】为了方便查看，把 'Topology' 列移动到 'name' 列的紧挨着后面
    cols = df.columns.tolist()
    # 找到 name 列的索引
    name_idx = cols.index('name')
    # 把 Topology 从列表最后弹出来，插入到 name 后面
    cols.insert(name_idx + 1, cols.pop(cols.index('Topology')))
    # 重新排列 DataFrame 的列
    df = df[cols]
    
    # 导出结果
    df.to_csv(output_file, index=False)
    print(f"处理完成！成功提取并新增了 'Topology' 列，结果已保存至 {output_file}")

# ================= 使用示例 =================
if __name__ == "__main__":
    # 请将下面单引号内的文件名替换为你的实际文件路径
    INPUT_FILE = r'E:\CODE\cif2des\prediction\output_2.csv'   # 需要处理的源文件
    OUTPUT_FILE = r'E:\CODE\cif2des\prediction\output_3.csv' # 处理后生成的新文件
    
    # 执行函数
    extract_topology(INPUT_FILE, OUTPUT_FILE)