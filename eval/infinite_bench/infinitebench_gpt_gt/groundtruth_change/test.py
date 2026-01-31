#!/usr/bin/env python3
import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import shutil


# 标签映射规则
TAG_MAPPING = {
    "Debugging":"Code Debugging",
    "Programming":"Code Programming",
    # "Retrieve.KV": "Key Value Retrieval",
    # "En.MutipleChoice": "English Multiple Choice",
    # "Zh.QA": "Chinese Question Answering",
    # "En.QA": "English Question Answering",
    # "En.Sum": "Summarization",
    # "En.Dia": "Character Identification",
    # "Math.Calc": "Math Calculation",
    # "Math.Find": "Math Finding",
    # "Retrieve.Number": "Number Retrieval",
    # "Retrieve.PassKey": "PassKey Retrieval",
}
def decide_file_tag(items):
    """
    根据一个文件里的样本，决定统一使用的 tag
    最简单策略：取第一个出现且在 TAG_MAPPING 里的 tag
    """
    for item in items:
        # 优先看 ground_truth
        gt = item.get("ground_truth", [])
        for obj in gt:
            if isinstance(obj, dict) and "tag" in obj:
                tag = obj["tag"]
                if tag in TAG_MAPPING:
                    return TAG_MAPPING[tag]
        # 其次看 output
        out = item.get("output", [])
        if isinstance(out, list):
            for obj in out:
                if isinstance(obj, dict) and "tag" in obj:
                    tag = obj["tag"]
                    if tag in TAG_MAPPING:
                        return TAG_MAPPING[tag]
    return None

def process_item(item):
    """处理单个样本，修改ground_truth中的tag"""
    modified = False
    
    # 处理output字段（可能是字符串或列表）
    if "output" in item:
        output = item["output"]
        
        # 如果是字符串，尝试解析为JSON
        if isinstance(output, str):
            try:
                output = json.loads(output)
            except json.JSONDecodeError:
                return item, modified
        
        # 如果是列表，遍历修改tag
        if isinstance(output, list):
            for obj in output:
                if isinstance(obj, dict) and "tag" in obj:
                    old_tag = obj["tag"]
                    if old_tag in TAG_MAPPING:
                        obj["tag"] = TAG_MAPPING[old_tag]
                        modified = True
            
            # 转回字符串（如果原来是字符串）
            if isinstance(item["output"], str):
                item["output"] = json.dumps(output, ensure_ascii=False)
            else:
                item["output"] = output
    
    # 处理ground_truth字段（如果存在）
    if "ground_truth" in item:
        ground_truth = item["ground_truth"]
        
        if isinstance(ground_truth, list):
            for obj in ground_truth:
                if isinstance(obj, dict) and "tag" in obj:
                    old_tag = obj["tag"]
                    if old_tag in TAG_MAPPING:
                        obj["tag"] = TAG_MAPPING[old_tag]
                        modified = True
    
    return item, modified


def process_jsonl_file(file_path, forced_tag, backup=True):
    """处理单个JSONL文件（原地修改，强制统一 tag）"""
    if backup:
        backup_path = str(file_path) + ".backup"
        shutil.copy2(file_path, backup_path)

    data = []
    modified_count = 0

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            processed_item, modified = process_item(item)
            data.append(processed_item)
            if modified:
                modified_count += 1

    file_tag = forced_tag

    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            if file_tag and "ground_truth" in item:
                for obj in item["ground_truth"]:
                    if isinstance(obj, dict) and "tag" in obj:
                        obj["tag"] = file_tag
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    return len(data), modified_count



def process_json_file(file_path, forced_tag, backup=True):
    """处理单个JSON文件（原地修改）"""
    # 备份原文件
    if backup:
        backup_path = str(file_path) + ".backup"
        shutil.copy2(file_path, backup_path)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    modified_count = 0
    
    # 如果是列表，遍历处理
    if isinstance(data, list):
        file_tag = forced_tag
        processed_data = []
        for item in data:
            processed_item, modified = process_item(item)
            if file_tag and "ground_truth" in processed_item:
                for obj in processed_item["ground_truth"]:
                    if isinstance(obj, dict) and "tag" in obj:
                        obj["tag"] = file_tag
                        modified = True

            processed_data.append(processed_item)
            if modified:
                modified_count += 1
        data = processed_data
    # 如果是单个对象
    elif isinstance(data, dict):
        data, modified = process_item(data)
        if modified:
            modified_count = 1
    
    # 写回原文件
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    total = len(data) if isinstance(data, list) else 1
    return total, modified_count


def rename_tags_inplace(directory, backup=True):
    """原地修改目录下所有JSON/JSONL文件"""
    print("=" * 60)
    print("标签重命名工具（原地修改）")
    print("=" * 60)
    print(f"目录: {directory}")
    print(f"备份: {'是' if backup else '否'}")
    print("=" * 60)
    print("\n标签映射规则:")
    for old, new in TAG_MAPPING.items():
        print(f"  {old} → {new}")
    print("=" * 60)
    
    dir_path = Path(directory)
    
    # 查找所有JSON和JSONL文件
    json_files = list(dir_path.glob("*.json"))
    jsonl_files = list(dir_path.glob("*.jsonl"))
    
    # 排除备份文件
    json_files = [f for f in json_files if not f.name.endswith('.backup')]
    jsonl_files = [f for f in jsonl_files if not f.name.endswith('.backup')]
    
    all_files = json_files + jsonl_files
    
    if not all_files:
        print(f"\n❌ 未找到JSON或JSONL文件: {directory}")
        return
    
    print(f"\n📂 找到 {len(all_files)} 个文件")
    print(f"   - JSON: {len(json_files)}")
    print(f"   - JSONL: {len(jsonl_files)}\n")
    
    if backup:
        print("💾 将创建 .backup 备份文件\n")
    
    total_samples = 0
    total_modified = 0
    
    # 处理所有文件
    for file in tqdm(all_files, desc="处理文件"):
        try:
            if file.suffix == ".jsonl":
                samples, modified = process_jsonl_file(file, backup)
            else:
                samples, modified = process_json_file(file, backup)
            
            total_samples += samples
            total_modified += modified
            
            if modified > 0:
                print(f"  ✅ {file.name}: {samples}样本, {modified}个被修改")
        
        except Exception as e:
            print(f"  ❌ {file.name}: 处理失败 - {e}")
    
    # 统计信息
    print("\n" + "=" * 60)
    print("处理完成")
    print("=" * 60)
    print(f"总文件数: {len(all_files)}")
    print(f"总样本数: {total_samples}")
    print(f"修改样本数: {total_modified} ({total_modified/total_samples*100:.1f}%)")
    print("=" * 60)
    
    if backup:
        print(f"\n💾 备份文件: *.backup")
        print("   如果确认无误，可以删除备份文件:")
        print(f"   rm {directory}/*.backup")
    
    # 保存统计
    stats_file = dir_path / "rename_stats.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump({
            "total_files": len(all_files),
            "total_samples": total_samples,
            "modified_samples": total_modified,
            "backup_created": backup,
            "tag_mapping": TAG_MAPPING
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 统计信息已保存到: {stats_file}")


def main():
    parser = argparse.ArgumentParser(description="强制统一单个 JSON/JSONL 文件的 tag")
    parser.add_argument("--file", type=str, required=True,
                        help="JSON / JSONL 文件路径")
    parser.add_argument("--tag", type=str, required=True,
                        help="要强制设置的 tag，例如: Code Debugging")
    parser.add_argument("--no-backup", action="store_true",
                        help="不创建备份文件（危险）")

    
    args = parser.parse_args()
    
    # 确认操作
    if args.no_backup:
        print("⚠️  警告: 将直接修改原文件，不创建备份！")
        confirm = input("确认继续? (yes/no): ")
        if confirm.lower() != "yes":
            print("已取消")
            return
    
    file_path = Path(args.file)
    forced_tag = args.tag

    if not file_path.exists():
        print(f"❌ 文件不存在: {file_path}")
        return

    if file_path.suffix == ".jsonl":
        process_jsonl_file(file_path, forced_tag, backup=not args.no_backup)
    else:
        process_json_file(file_path, forced_tag, backup=not args.no_backup)



if __name__ == "__main__":
    main()