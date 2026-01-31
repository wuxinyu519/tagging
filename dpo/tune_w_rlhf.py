#!/usr/bin/env python3
import os
import json
import argparse
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig

def freeze_layers(model, num_freeze: int):
    """Freeze the first num_freeze transformer blocks."""
    if not hasattr(model, "model"):
        print("Warning: model has no `model` attribute, skipping freeze.")
        return

    try:
        layers = model.model.layers
    except AttributeError:
        print("Could not access model layers, skipping freeze.")
        return

    total = len(layers)
    num_freeze = min(num_freeze, total)
    print(f"Freezing first {num_freeze} layers (total {total})")

    for layer in layers[:num_freeze]:
        for param in layer.parameters():
            param.requires_grad = False

    print(f"Remaining {total - num_freeze} layers will be trained")


# ===============================
# Load DPO data dir
# ===============================

def load_dpo_data_dir(data_dir):
    """Load all DPO data from a directory."""
    all_data = []
    skipped=0
    for file in os.listdir(data_dir):
        if file.endswith(".jsonl"):
            file_path = os.path.join(data_dir, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        
                        # skip identical samples
                        if item.get("chosen") == item.get("rejected"):
                            skipped += 1
                            continue
                        
                        user_input = item.get("input", "")
                        chosen_answer = item.get("chosen", "")
                        rejected_answer = item.get("rejected", "")
                        
                        # build prompt
                        prompt_text = f"You are a helpful assistant. {user_input}\nAssistant: "
                        
                        all_data.append({
                            "chosen": prompt_text + chosen_answer,
                            "rejected": prompt_text + rejected_answer
                        })
                        
                    except json.JSONDecodeError:
                        continue
    

    return all_data

# ===============================
# main
# ===============================
def main():
    parser = argparse.ArgumentParser(description="DPO training (following train.py style)")
    parser.add_argument("--dpo_data_dir", type=str, default="./dpo_data")
    parser.add_argument("--sft_model_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./dpo_output")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--accum_steps", type=int, default=4)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_prompt_length", type=int, default=512)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--freeze_layers", type=int, default=20)
    parser.add_argument("--save_strategy", type=str, default="epoch", choices=["epoch", "steps"])
    args = parser.parse_args()

    # ===============================
    # Set device (cuda if available)
    # ===============================
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ===============================
    # Load DPO data
    # ===============================
    print(f"\nLoading DPO data from {args.dpo_data_dir}")
    train_data = load_dpo_data_dir(args.dpo_data_dir)
    
    if not train_data:
        raise ValueError(f"No DPO data found in {args.dpo_data_dir}")
    
    print(f"Loaded {len(train_data)} training samples from all files")

    # ===============================
    # Load tokenizer
    # ===============================
    print(f"\nLoading tokenizer from {args.sft_model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(args.sft_model_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ===============================
    # Load SFT model (trainable)
    # ===============================
    print(f"Loading SFT model from {args.sft_model_dir}")
    model = AutoModelForCausalLM.from_pretrained(
        args.sft_model_dir,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )

    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    freeze_layers(model, args.freeze_layers)


    model_ref = AutoModelForCausalLM.from_pretrained(
        args.sft_model_dir,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    model_ref.eval()
    for param in model_ref.parameters():
        param.requires_grad = False

    # ===============================
    # Prepare Dataset
    # ===============================
    train_dataset = Dataset.from_list(train_data)

    # ===============================
    # DPO training config
    # ===============================
    training_args = DPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.accum_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        beta=args.beta,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        save_strategy=args.save_strategy,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=10,
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        remove_unused_columns=False,
        report_to="none",
        optim="adamw_torch",
        max_grad_norm=1.0,
        save_total_limit=2,
    )

    # ===============================
    # Create DPO trainer
    # ===============================
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=model_ref,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer, 
    )

    # ===============================
    # Train
    # ===============================
    print("\nStart DPO training...")
    dpo_trainer.train()
    print("DPO training completed!")

    # ===============================
    # Save model
    # ===============================
    final_path = os.path.join(args.output_dir, "final_model")
    os.makedirs(final_path, exist_ok=True)
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"Model saved to {final_path}")

    print(f"output: {args.output_dir}")
    print(f" - final_model/ (DPO)")
    print(f" - logs/ (training logs)")


if __name__ == "__main__":
    main()