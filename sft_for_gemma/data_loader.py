#!/usr/bin/env python3
import os
import json
import random


def _write_jsonl(path, data):
    """保存数据到 JSONL 文件"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for x in data:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")


def load_data(data_dir, limit_data=None, split_ratio=1, seed=42, save_splits_to=None, load_existing=False):

    if load_existing and save_splits_to:
        train_path = os.path.join(save_splits_to, "train.jsonl")
        test_path = os.path.join(save_splits_to, "test.jsonl")
        if os.path.exists(train_path) and os.path.exists(test_path):
            print(f"Loading existing split from {save_splits_to}")
            with open(train_path, "r", encoding="utf-8") as f:
                train_data = [json.loads(line) for line in f]
            with open(test_path, "r", encoding="utf-8") as f:
                test_data = [json.loads(line) for line in f]
            print(f"Loaded saved split: {len(train_data)} train, {len(test_data)} test")
            return train_data, test_data
        else:
            print(f"Saved split not found in {save_splits_to}, regenerating from raw data...")

    random.seed(seed)

    all_files = [
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.endswith(".jsonl")
    ]
    if not all_files:
        raise FileNotFoundError(f"No .jsonl files found in {data_dir}")

    all_data = []
    for file in all_files:
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    sample = json.loads(line)
                    if "input" in sample and "output" in sample:
                        all_data.append(sample)
                except json.JSONDecodeError:
                    continue

    if not all_data:
        raise ValueError("No valid samples found in any jsonl file")

    print(f"Loaded {len(all_data)} total samples from {len(all_files)} files")

    if limit_data is not None and limit_data < len(all_data):
        all_data = random.sample(all_data, limit_data)
        print(f"Sampled {limit_data} examples")

    random.shuffle(all_data)
    split = int(split_ratio * len(all_data))
    train_data = all_data[:split]
    test_data = all_data[split:]

    print(f"Split into {len(train_data)} train and {len(test_data)} test samples")


    if save_splits_to:
        os.makedirs(save_splits_to, exist_ok=True)
        train_path = os.path.join(save_splits_to, "train.jsonl")
        test_path = os.path.join(save_splits_to, "test.jsonl")
        _write_jsonl(train_path, train_data)
        _write_jsonl(test_path, test_data)

        meta = {
            "seed": seed,
            "limit_data": limit_data,
            "split_ratio": split_ratio,
            "train_size": len(train_data),
            "test_size": len(test_data),
            "source_files": [os.path.basename(f) for f in all_files],
        }
        with open(os.path.join(save_splits_to, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print(f"Saved split files to {save_splits_to}")

    return train_data, test_data
