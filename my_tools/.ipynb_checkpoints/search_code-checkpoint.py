import os
import argparse
import re
import sys

def search_in_files(root_dir, search_pattern, output_file, search_type="text", case_sensitive=False):
    """
    递归搜索指定目录下的文件，查找包含特定模式的内容
    
    Args:
        root_dir (str): 要搜索的根目录路径
        search_pattern (str): 要搜索的模式
        output_file (str): 输出结果的文件路径
        search_type (str): 搜索类型，可选值: "text", "class", "function", "import"
        case_sensitive (bool): 是否区分大小写
    """
    # 根据搜索类型设置不同的正则表达式模式
    patterns = {
        "text": search_pattern,  # 普通文本搜索
        "class": rf'^\s*class\s+{search_pattern}\s*(?:\(|:)',  # 类定义
        "function": rf'^\s*def\s+{search_pattern}\s*\(',  # 函数定义
        "import": rf'^\s*(?:from|import).*{search_pattern}'  # 导入语句
    }
    
    # 搜索类型描述
    search_type_desc = {
        "text": "文本",
        "class": "类定义",
        "function": "函数定义",
        "import": "导入语句"
    }.get(search_type, "内容")
    
    # 确定使用的模式
    if search_type in patterns:
        if search_type == "text":
            pattern = re.compile(re.escape(search_pattern), 
                                re.IGNORECASE if not case_sensitive else 0)
        else:
            pattern = re.compile(patterns[search_type], 
                                re.IGNORECASE if not case_sensitive else 0)
    else:
        print(f"错误: 不支持的搜索类型 '{search_type}'。可选值: text, class, function, import")
        return
    
    results = []
    searched_files = 0
    
    # 遍历所有目录和子目录
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                searched_files += 1
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        
                    # 检查每一行是否匹配模式
                    for line_num, line in enumerate(lines, 1):
                        if pattern.search(line):
                            results.append({
                                'file_path': file_path,
                                'line_number': line_num,
                                'line_content': line.strip()
                            })
                            
                except UnicodeDecodeError:
                    # 尝试使用其他编码
                    try:
                        with open(file_path, 'r', encoding='latin-1') as f:
                            lines = f.readlines()
                            
                        for line_num, line in enumerate(lines, 1):
                            if pattern.search(line):
                                results.append({
                                    'file_path': file_path,
                                    'line_number': line_num,
                                    'line_content': line.strip()
                                })
                    except Exception as e:
                        print(f"读取文件 {file_path} 时出错: {e}")
                except Exception as e:
                    print(f"读取文件 {file_path} 时出错: {e}")
    
    # 将结果写入输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        if results:
            f.write(f"在目录 '{root_dir}' 中搜索 {search_type_desc} '{search_pattern}' 的结果:\n")
            f.write(f"搜索了 {searched_files} 个文件，找到 {len(results)} 个匹配项\n\n")
            
            for result in results:
                f.write(f"文件: {result['file_path']}\n")
                f.write(f"行号: {result['line_number']}\n")
                f.write(f"内容: {result['line_content']}\n")
                f.write("-" * 80 + "\n\n")
        else:
            f.write(f"在目录 '{root_dir}' 中未找到 '{search_pattern}' 的{search_type_desc}。\n")
            f.write(f"搜索了 {searched_files} 个文件。\n")
    
    # 同时在控制台输出结果
    if results:
        print(f"搜索完成! 找到 {len(results)} 个匹配项，结果已保存到 {output_file}")
        print("前10个匹配项:")
        for i, result in enumerate(results[:10]):
            print(f"{i+1}. {result['file_path']} (第 {result['line_number']} 行)")
        if len(results) > 10:
            print(f"... 还有 {len(results) - 10} 个匹配项未显示")
    else:
        print(f"未找到 '{search_pattern}' 的{search_type_desc}。")
        print(f"搜索了 {searched_files} 个文件。")

def main():
    parser = argparse.ArgumentParser(description='递归搜索Python文件中的特定内容')
    parser.add_argument('directory', help='要搜索的根目录路径')
    parser.add_argument('pattern', help='要搜索的模式')
    parser.add_argument('-t', '--type', default='text', 
                       choices=['text', 'class', 'function', 'import'],
                       help='搜索类型: text(文本), class(类), function(函数), import(导入) (默认: text)')
    parser.add_argument('-o', '--output', default='search_results.txt', 
                       help='输出文件路径 (默认为 search_results.txt)')
    parser.add_argument('-c', '--case-sensitive', action='store_true',
                       help='区分大小写 (默认不区分)')
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.directory):
        print(f"错误: 目录 '{args.directory}' 不存在或不是有效目录。")
        sys.exit(1)
    
    search_in_files(args.directory, args.pattern, args.output, 
                   args.type, args.case_sensitive)

if __name__ == '__main__':
    main()