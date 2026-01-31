#!/usr/bin/env python3
import os
import argparse
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    Trainer, TrainingArguments,
    DataCollatorForLanguageModeling
)
from data_loader import load_data


class AugmentedTagDataset(Dataset):
    def __init__(self, data_list, tokenizer, max_length=1024):
        self.data = data_list
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.template = (
            "You are a helpful assistant. "
            "{query}\nAssistant: {answer}"
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        s = self.data[idx]
        query = s.get("input", s.get("prompt", None))
        answer = s.get("output", s.get("chosen", None))



        if self.tokenizer.eos_token:
            answer = answer.strip() + self.tokenizer.eos_token

        full_text = self.template.format(query=query, answer=answer)
        prompt_text = self.template.format(query=query, answer="").rstrip()

        full_encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        prompt_encoding = self.tokenizer(prompt_text, add_special_tokens=False)

        labels = full_encoding["input_ids"].clone()
        prompt_len = len(prompt_encoding["input_ids"])

        if (
            self.tokenizer.bos_token_id is not None
            and full_encoding["input_ids"][0, 0].item() == self.tokenizer.bos_token_id
        ):
            prompt_len += 1

        labels[0, :prompt_len] = -100
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": full_encoding["input_ids"].squeeze(0),
            "attention_mask": full_encoding["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0),
        }



def freeze_layers(model, num_freeze: int):
    """
    Freeze the first num_freeze transformer blocks, so they won't train.
    """
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
# main function — entry point
# ===============================
def main():
    parser = argparse.ArgumentParser(description="Refined Gemma fine-tuning with stable training tricks")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--limit_data", type=int, default=None)
    parser.add_argument("--model_name", type=str, default="google/gemma-3-1b-it")
    parser.add_argument("--output_dir", type=str, default="./finetuned_output_refined")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--accum_steps", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--freeze_layers", type=int, default=20)
    parser.add_argument("--save_strategy", type=str, default="epoch", choices=["epoch", "steps"])
    args = parser.parse_args()

    # ===============================
    # Device setup (use GPU if possible)
    # ===============================
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ===============================
    # Load and split data
    # ===============================
    split_dir = os.path.join(args.output_dir, "splits")
    train_data, test_data = load_data(
        args.data_dir,
        limit_data=args.limit_data,
        save_splits_to=split_dir,
        seed=42
    )

    # ===============================
    # Sample weights by template type
    # ===============================
    def get_sample_weight(sample):
        t = sample.get("template_type", "").lower()
        if "select_single_from_taglist" in t:
            return 4.0  
        else:
            return 1.0

    sample_weights = [get_sample_weight(s) for s in train_data]

    print(
        f"✓ Weighted samples: "
        f"{sum(w > 1.0 for w in sample_weights)} / {len(sample_weights)}"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ===============================
    # Load the model
    # ===============================
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )

    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    freeze_layers(model, args.freeze_layers)

    # ===============================
    # Prepare dataset
    # ===============================
    train_ds = AugmentedTagDataset(train_data, tokenizer, max_length=args.max_length)
    print(f"Training set: {len(train_ds)} samples")


    train_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.accum_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        save_strategy=args.save_strategy,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=10,
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        remove_unused_columns=False,
        report_to="none",
        fp16_full_eval=False,
        optim="adamw_torch",
        max_grad_norm=1.0,
        save_total_limit=2,
        label_smoothing_factor=0.1,
    )

    # ===============================
    # Create Trainer and start training
    # ===============================
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )


    print("Start fine-tuning...")
    trainer.train()
    print("Training completed!")

    # ===============================
    # Save final model and tokenizer
    # ===============================
    final_path = os.path.join(args.output_dir, "final_model")
    os.makedirs(final_path, exist_ok=True)
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"Model saved to {final_path}")
    print(f"Data splits saved to {split_dir}")


if __name__ == "__main__":
    main()
