import json
import os

def normalize_difficulty_in_jsonl(data_directory):
    """
    直接修改jsonl文件中的Difficulty Level标签，将Moderate和Medium改为Intermediate
    
    Args:
        data_directory: 包含jsonl文件的目录路径
    """
    # 获取所有jsonl文件
    jsonl_files = [f for f in os.listdir(data_directory) if f.endswith('.jsonl')]
    
    if not jsonl_files:
        print(f"在目录 {data_directory} 中没有找到jsonl文件")
        return
    
    print(f"找到 {len(jsonl_files)} 个jsonl文件\n")
    
    total_modified = 0
    
    for jsonl_file in jsonl_files:
        file_path = os.path.join(data_directory, jsonl_file)
        
        # 读取所有数据
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        
        modified_count = 0
        
        # 修改数据
        for item in data:
            if 'parsed_tags' in item:
                parsed_tags = item['parsed_tags']
                
                # 处理字符串格式的parsed_tags
                if isinstance(parsed_tags, str):
                    try:
                        if parsed_tags.strip().startswith('['):
                            parsed_tags = json.loads(parsed_tags)
                        else:
                            import ast
                            parsed_tags = ast.literal_eval(parsed_tags)
                    except:
                        continue
                
                # 修改第3个位置（index=2）的Difficulty Level
                if isinstance(parsed_tags, list) and len(parsed_tags) > 2:
                    tag_item = parsed_tags[2]
                    if isinstance(tag_item, dict) and 'tag' in tag_item:
                        if tag_item['tag'] in ['High']:
                            tag_item['tag'] = 'Hard'
                            modified_count += 1
                    
                    # 更新item
                    item['parsed_tags'] = parsed_tags
        
        # 写回文件
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"📝 {jsonl_file}: 修改了 {modified_count} 条记录")
        total_modified += modified_count
    
    print(f"\n✅ 完成！总共修改了 {total_modified} 条记录")

if __name__ == "__main__":
    # 设置你的数据目录路径
    data_directory = "./"  # 请替换为你的实际路径
    
    print("Difficulty Level 标签归一化工具")
    print("=" * 60)
    print("将 Moderate 和 Medium 统一改为 Intermediate")
    print("=" * 60)
    print()
    
    normalize_difficulty_in_jsonl(data_directory)