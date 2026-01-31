#!/usr/bin/env python3
import os
import json
import torch
import argparse
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from vllm import LLM, SamplingParams


# ===============================
# Extract tags
# ===============================
def extract_tags_from_output(text):
    tags = []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            for t in parsed:
                if isinstance(t, dict) and "tag" in t:
                    tags.append(t["tag"].strip())
        elif isinstance(parsed, dict) and "tag" in parsed:
            tags.append(parsed["tag"].strip())
    except json.JSONDecodeError:
        for line in text.split("\n"):
            if "tag" in line.lower():
                parts = line.split(":")
                if len(parts) > 1:
                    tags.append(parts[1].strip())
    return tags



def evaluate_metrics(pred_tags, gold_tags, phrase_model, threshold=0.8):
    pred_set, gold_set = set(pred_tags), set(gold_tags)
    tp = len(pred_set & gold_set)
    precision = tp / len(pred_set) if pred_set else 0
    recall = tp / len(gold_set) if gold_set else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    acc = 1.0 if pred_set == gold_set else 0.0

    if not pred_tags or not gold_tags:
        return {"exact": (precision, recall, f1, acc), "semantic": (0, 0, 0, 0)}

    pred_vecs = phrase_model.encode(pred_tags, convert_to_tensor=True, normalize_embeddings=True)
    gold_vecs = phrase_model.encode(gold_tags, convert_to_tensor=True, normalize_embeddings=True)
    sim = util.cos_sim(gold_vecs, pred_vecs)
    matched = (sim > threshold).any(dim=1)
    sem_acc = matched.float().mean().item()
    sem_prec = (sim > threshold).any(dim=0).float().mean().item()
    sem_rec = sem_acc
    sem_f1 = 2 * sem_prec * sem_rec / (sem_prec + sem_rec) if (sem_prec + sem_rec) else 0

    return {"exact": (precision, recall, f1, acc),
            "semantic": (sem_prec, sem_rec, sem_f1, sem_acc)}



def load_test_data(test_file):
    test_data = []
    path = Path(test_file)

    if path.is_file():
        print(f"Loading test file: {test_file}")
        with open(test_file, 'r', encoding='utf-8') as f:
            test_data = [json.loads(line) for line in f]

    elif path.is_dir():
        print(f"Loading test data from directory: {test_file}")
        jsonl_files = list(path.glob("*.jsonl"))
        if not jsonl_files:
            raise ValueError(f"No .jsonl files found in {test_file}")
        print(f"Found {len(jsonl_files)} jsonl files")
        for file in sorted(jsonl_files):
            with open(file, 'r', encoding='utf-8') as f:
                file_data = [json.loads(line) for line in f]
                test_data.extend(file_data)
                print(f"   - {file.name}: {len(file_data)} samples")
    else:
        raise ValueError(f"Path does not exist: {test_file}")

    return test_data


# ===============================
# Main function (vLLM version)
# ===============================
def main():
    parser = argparse.ArgumentParser(description="Evaluate model with vLLM (supports file or directory)")
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--test_file", type=str, required=True,
                        help="Path to test file or directory containing .jsonl files")
    parser.add_argument("--output_file", type=str, default="eval_outputs/predictions.jsonl")
    parser.add_argument("--limit_eval", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    # ===== Load model via vLLM =====
    print(f"Loading model (vLLM) from {args.model_dir}")
    llm = LLM(
        model=args.model_dir,
        trust_remote_code=True,
        dtype="auto",
    
    )

    
    sampling_params = SamplingParams(
        max_tokens=256,
        temperature=0.6,
        top_p=0.9,
        repetition_penalty=1.2,
    )

    phrase_model = SentenceTransformer("whaleloops/phrase-bert")

    # ===== Load data =====
    test_data = load_test_data(args.test_file)
    if args.limit_eval:
        import random
        random.seed(42)
        test_data = random.sample(test_data, min(args.limit_eval, len(test_data)))

    print(f"Evaluating {len(test_data)} samples, batch size={args.batch_size}")

    # ===== Prompt =====
    template = (
        "You are a helpful assistant. "
        "{query}\nAssistant:"
    )

    all_exact, all_sem, predictions = [], [], []

    # ===== Batch evaluation (vLLM) =====
    for i in tqdm(range(0, len(test_data), args.batch_size), desc="Evaluating"):
        batch = test_data[i: i + args.batch_size]
        batch_prompts = [template.format(query=s["input"]) for s in batch]

     
        outputs = llm.generate(batch_prompts, sampling_params)

     
        for s, req_out in zip(batch, outputs):
         
            pred_text = (req_out.outputs[0].text if req_out.outputs else "").strip()

            try:
                gold = json.loads(s["output"]) if isinstance(s["output"], str) else s["output"]
                gold_tags = [t["tag"] for t in gold if isinstance(t, dict) and "tag" in t]
            except Exception:
                gold_tags = []

            pred_tags = extract_tags_from_output(pred_text)
            metrics = evaluate_metrics(pred_tags, gold_tags, phrase_model)
            all_exact.append(metrics["exact"])
            all_sem.append(metrics["semantic"])

            predictions.append({
                "input": s["input"],
                "predicted_results": pred_tags,
                "ground_truth": gold_tags,
                "raw_model_output": pred_text,
            })

   
    avg = lambda arr, i: sum(x[i] for x in arr) / len(arr) if arr else 0
    results = {
        "exact": {
            "precision": avg(all_exact, 0),
            "recall": avg(all_exact, 1),
            "f1": avg(all_exact, 2),
            "accuracy": avg(all_exact, 3),
        },
        "semantic": {
            "precision": avg(all_sem, 0),
            "recall": avg(all_sem, 1),
            "f1": avg(all_sem, 2),
            "accuracy": avg(all_sem, 3),
        },
    }

    # ===== Save =====
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    summary_path = os.path.join(os.path.dirname(args.output_file), "evaluation_results.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\nFinal Evaluation Results:")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nSaved predictions → {args.output_file}")
    print(f"Saved summary → {summary_path}")


if __name__ == "__main__":
    main()
