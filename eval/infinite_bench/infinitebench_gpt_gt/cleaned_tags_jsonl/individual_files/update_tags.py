#!/usr/bin/env python3
"""
JSONL文件第一个标签修改器(简化版)
将指定文件所有数据的parsed_tags字段的第一个tag改为"Programming"
"""
import json
import shutil

def modify_first_tag(file_path, new_tag="Programming"):
    """
    修改JSONL文件中所有数据的第一个标签
    
    Args:
        file_path: JSONL文件路径
        new_tag: 新的第一个标签名
    """
    print(f"🔧 处理文件: {file_path}")
    
    # 创建备份
    backup_path = file_path + '.backup'
    shutil.copy2(file_path, backup_path)
    print(f"💾 备份已创建: {backup_path}")
    
    modified_count = 0
    total_count = 0
    
    # 读取所有行
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 处理每一行
    new_lines = []
    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            new_lines.append('\n')
            continue
        
        try:
            data = json.loads(line)
            total_count += 1
            
            if 'parsed_tags' in data:
                parsed_tags = data['parsed_tags']
                
                # 处理字符串格式的parsed_tags
                if isinstance(parsed_tags, str):
                    try:
                        parsed_tags = json.loads(parsed_tags)
                    except:
                        import ast
                        parsed_tags = ast.literal_eval(parsed_tags)
                
                # 修改第一个标签
                if isinstance(parsed_tags, list) and len(parsed_tags) > 0:
                    if isinstance(parsed_tags[0], dict) and 'tag' in parsed_tags[0]:
                        old_tag = parsed_tags[0]['tag']
                        parsed_tags[0]['tag'] = new_tag
                        
                        data['parsed_tags'] = parsed_tags
                        modified_count += 1
                        
                        print(f"第{line_num}行: {old_tag} → {new_tag}")
            
            # 写入修改后的数据
            new_line = json.dumps(data, ensure_ascii=False) + '\n'
            new_lines.append(new_line)
            
        except Exception as e:
            print(f"⚠️  第{line_num}行处理失败: {e}")
            new_lines.append(line + '\n')
    
    # 写回文件
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    
    print(f"\n✅ 修改完成!")
    print(f"   总记录数: {total_count}")
    print(f"   成功修改: {modified_count}")

def main():
    file_path = input("请输入JSONL文件路径: ").strip()
    modify_first_tag(file_path, "General Knowledge")

if __name__ == "__main__":
    main()