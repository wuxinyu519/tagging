import json
import os
from collections import Counter

def analyze_jsonl_files_summary(data_directory, save_results=True, output_file=None):
    """
    统计数据目录中所有jsonl文件的标签分布，只输出摘要
    
    Args:
        data_directory: 包含jsonl文件的目录路径
        save_results: 是否保存结果到文件
        output_file: 输出文件路径，如果为None则自动生成
    """
    # 获取所有jsonl文件
    jsonl_files = [f for f in os.listdir(data_directory) if f.endswith('.jsonl')]
    
    if not jsonl_files:
        print(f"在目录 {data_directory} 中没有找到jsonl文件")
        return
    
    # 准备输出内容
    output_lines = []
    output_lines.append(f"JSONL文件标签统计分析报告")
    output_lines.append(f"生成时间: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    output_lines.append(f"数据目录: {data_directory}")
    output_lines.append(f"找到 {len(jsonl_files)} 个jsonl文件\n")
    
    print(f"找到 {len(jsonl_files)} 个jsonl文件\n")
    
    # 标签类型名称
    tag_types = ['Domain', 'Task Type', 'Difficulty Level', 'Language']
    
    # 为每个jsonl文件统计
    for i, jsonl_file in enumerate(jsonl_files, 1):
        file_path = os.path.join(data_directory, jsonl_file)
        
        try:
            # 加载jsonl文件
            data = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                        data.append(item)
                    except json.JSONDecodeError as e:
                        print(f"⚠️  {jsonl_file} 第{line_num}行JSON解析错误: {e}")
                        continue
            
            print(f"{'='*60}")
            print(f"文件 {i}/{len(jsonl_files)}: {jsonl_file}")
            print(f"{'='*60}")
            
            # 添加到输出
            output_lines.append(f"{'='*60}")
            output_lines.append(f"文件 {i}/{len(jsonl_files)}: {jsonl_file}")
            output_lines.append(f"{'='*60}")
            
            # 提取所有parsed_tags
            all_records = []
            for item in data:
                if isinstance(item, dict) and 'parsed_tags' in item:
                    parsed_tags = item['parsed_tags']
                    
                    # 处理不同格式的parsed_tags
                    if isinstance(parsed_tags, str):
                        try:
                            # 尝试解析JSON字符串
                            if parsed_tags.strip().startswith('['):
                                parsed_tags = json.loads(parsed_tags)
                            else:
                                # 尝试eval解析
                                import ast
                                parsed_tags = ast.literal_eval(parsed_tags)
                        except:
                            continue
                    
                    if isinstance(parsed_tags, list):
                        all_records.append(parsed_tags)
            
            if not all_records:
                msg = f"警告: 文件 {jsonl_file} 中没有找到有效的parsed_tags"
                print(msg)
                output_lines.append(msg)
                continue
            
            record_count_msg = f"总记录数: {len(all_records)}\n"
            print(record_count_msg)
            output_lines.append(record_count_msg)
            
            # 为每种标签类型统计（前4个位置）
            for tag_index in range(4):
                tag_type = tag_types[tag_index]
                
                # 提取对应位置的标签
                position_tags = []
                for record_tags in all_records:
                    if isinstance(record_tags, list) and len(record_tags) > tag_index:
                        tag_item = record_tags[tag_index]
                        if isinstance(tag_item, dict) and 'tag' in tag_item:
                            tag_value = tag_item['tag']
                        else:
                            tag_value = str(tag_item)
                        
                        # 归一化Difficulty Level标签
                        if tag_index == 2:  # Difficulty Level是第3个位置(index=2)
                            if tag_value in ['Moderate', 'Medium']:
                                tag_value = 'Intermediate'
                        
                        position_tags.append(tag_value)
                        
                
                if not position_tags:
                    msg = f"{tag_type}: 没有找到数据"
                    print(msg)
                    output_lines.append(msg)
                    continue
                
                # 统计频次
                tag_counts = Counter(position_tags)
                unique_count = len(tag_counts)
                total_count = len(position_tags)
                
                # 输出到控制台和文件
                header = f"{tag_type}:"
                unique_msg = f"  唯一值数量: {unique_count}"
                total_msg = f"  总标签数量: {total_count}"
                dist_msg = f"  分布情况:"
                
                print(header)
                print(unique_msg)
                print(total_msg)
                print(dist_msg)
                
                output_lines.append(header)
                output_lines.append(unique_msg)
                output_lines.append(total_msg)
                output_lines.append(dist_msg)
                
                # 按频次排序显示
                for tag, count in tag_counts.most_common():
                    percentage = (count / total_count) * 100
                    item_msg = f"    {tag}: {count} ({percentage:.1f}%)"
                    print(item_msg)
                    output_lines.append(item_msg)
                
                print()
                output_lines.append("")
                
            # 如果有第5个及以后的位置，统计Topic(s)
            topic_tags = []
            for record_tags in all_records:
                if isinstance(record_tags, list) and len(record_tags) > 4:
                    for topic_index in range(4, len(record_tags)):
                        tag_item = record_tags[topic_index]
                        if isinstance(tag_item, dict) and 'tag' in tag_item:
                            topic_tags.append(tag_item['tag'])
                        else:
                            topic_tags.append(str(tag_item))
            
            if topic_tags:
                # 统计Topic(s)频次
                topic_counts = Counter(topic_tags)
                unique_count = len(topic_counts)
                total_count = len(topic_tags)
                
                header = f"Topic(s):"
                unique_msg = f"  唯一值数量: {unique_count}"
                total_msg = f"  总标签数量: {total_count}"
                dist_msg = f"  分布情况 (Top 20):"
                
                print(header)
                print(unique_msg)
                print(total_msg)
                print(dist_msg)
                
                output_lines.append(header)
                output_lines.append(unique_msg)
                output_lines.append(total_msg)
                output_lines.append(dist_msg)
                
                # 显示前20个最常见的Topic
                for tag, count in topic_counts.most_common(20):
                    percentage = (count / total_count) * 100
                    item_msg = f"    {tag}: {count} ({percentage:.1f}%)"
                    print(item_msg)
                    output_lines.append(item_msg)
                
                print()
                output_lines.append("")
            
        except Exception as e:
            error_msg = f"处理文件 {jsonl_file} 时出错: {str(e)}"
            print(error_msg)
            output_lines.append(error_msg)
        
        print()  # 文件之间的分隔
        output_lines.append("")
    
    # 保存结果到文件
    if save_results:
        if output_file is None:
            # 自动生成文件名
            timestamp = __import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"jsonl_tags_analysis_{timestamp}.txt"
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(output_lines))
            print(f"结果已保存到: {os.path.abspath(output_file)}")
        except Exception as e:
            print(f"保存文件时出错: {str(e)}")
    
    return output_lines

# 使用示例
if __name__ == "__main__":
    # 设置你的数据目录路径
    data_directory = "./"  # 请替换为你的实际路径
    
    print("JSONL文件标签统计分析工具 (带Difficulty归一化)")
    print("=" * 60)
    
    # 分析每个文件的详细统计
    analyze_jsonl_files_summary(data_directory)
    
    print("\n使用方法:")
    print("1. 将 data_directory 变量设置为你的jsonl文件目录路径")
    print("2. 运行函数:")
    print("   analyze_jsonl_files_summary(data_directory)  # 自动保存到文件")
    print("   analyze_jsonl_files_summary(data_directory, save_results=False)  # 不保存文件")
    print("   analyze_jsonl_files_summary(data_directory, output_file='my_report.txt')  # 自定义文件名")
    print("3. 结果保存文件名: jsonl_tags_analysis_YYYYMMDD_HHMMSS.txt")
    print("4. 标签提取规则:")
    print("   - Domain: 每条记录的第1个tag")
    print("   - Task Type: 每条记录的第2个tag") 
    print("   - Difficulty Level: 每条记录的第3个tag (Moderate和Medium自动归一化为Intermediate)")
    print("   - Language: 每条记录的第4个tag")
    print("   - Topic(s): 第5个及以后的所有tag")