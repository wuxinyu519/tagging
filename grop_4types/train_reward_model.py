#!/usr/bin/env python3
import os
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from dataclasses import dataclass
from typing import Dict, List


class RewardModel(nn.Module):
    """Reward Model based on causal LM"""
    def __init__(self, base_model):
        super().__init__()
        self.model = base_model
        self.value_head = nn.Linear(base_model.config.hidden_size, 1)
        
    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1]
        
        # Get last non-padding token's hidden state
        if attention_mask is not None:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = input_ids.shape[0]
            last_hidden = last_hidden_state[torch.arange(batch_size, device=input_ids.device), sequence_lengths]
        else:
            last_hidden = last_hidden_state[:, -1, :]
        
        reward = self.value_head(last_hidden)
        return reward


def load_dpo_data(data_dir):
    """Load DPO data and convert to reward model training format"""
    all_data = []
    for file in os.listdir(data_dir):
        if file.endswith(".jsonl"):
            file_path = os.path.join(data_dir, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        
                        if item.get("chosen") == item.get("rejected"):
                            continue
                        
                        user_input = item.get("input", "")
                        chosen_answer = item.get("chosen", "")
                        rejected_answer = item.get("rejected", "")
                        
                        prompt_text = f"You are a helpful assistant. {user_input}\nAssistant: "
                        
                        all_data.append({
                            "query": prompt_text,
                            "chosen": chosen_answer,
                            "rejected": rejected_answer
                        })
                        
                    except json.JSONDecodeError:
                        continue
    
    return all_data


@dataclass
class RewardDataCollator:
    """Collator for reward model training"""
    tokenizer: AutoTokenizer
    max_length: int = 1024
    
    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        queries = [f["query"] for f in features]
        chosen = [f["chosen"] for f in features]
        rejected = [f["rejected"] for f in features]
        
        # Tokenize chosen
        chosen_full = [q + c for q, c in zip(queries, chosen)]
        chosen_encodings = self.tokenizer(
            chosen_full,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        # Tokenize rejected
        rejected_full = [q + r for q, r in zip(queries, rejected)]
        rejected_encodings = self.tokenizer(
            rejected_full,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "chosen_input_ids": chosen_encodings.input_ids,
            "chosen_attention_mask": chosen_encodings.attention_mask,
            "rejected_input_ids": rejected_encodings.input_ids,
            "rejected_attention_mask": rejected_encodings.attention_mask,
        }


class RewardModelTrainer(Trainer):
    """Custom Trainer for Reward Model with Bradley-Terry loss"""
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Forward pass for chosen
        reward_chosen = model(
            input_ids=inputs["chosen_input_ids"],
            attention_mask=inputs["chosen_attention_mask"]
        ).squeeze(-1)
        
        # Forward pass for rejected
        reward_rejected = model(
            input_ids=inputs["rejected_input_ids"],
            attention_mask=inputs["rejected_attention_mask"]
        ).squeeze(-1)
        
        # Bradley-Terry loss: -log(sigmoid(r_chosen - r_rejected))
        loss = -F.logsigmoid(reward_chosen - reward_rejected).mean()
        
        if return_outputs:
            return loss, {"reward_chosen": reward_chosen, "reward_rejected": reward_rejected}
        return loss


def main():
    parser = argparse.ArgumentParser(description="Train Reward Model from DPO data")
    parser.add_argument("--dpo_data_dir", type=str, default="./dpo_data")
    parser.add_argument("--base_model_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./reward_model_output")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load data
    print(f"\nLoading DPO data from {args.dpo_data_dir}")
    train_data = load_dpo_data(args.dpo_data_dir)
    
    if not train_data:
        raise ValueError(f"No DPO data found in {args.dpo_data_dir}")
    
    print(f"Loaded {len(train_data)} training pairs")

    # Load tokenizer
    print(f"\nLoading tokenizer from {args.base_model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    print(f"Loading base model from {args.base_model_dir}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_dir,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    
    if hasattr(base_model.config, "use_cache"):
        base_model.config.use_cache = False
    
    # Create reward model
    print("Creating reward model with value head")
    reward_model = RewardModel(base_model)

    # Prepare dataset
    train_dataset = Dataset.from_list(train_data)
    data_collator = RewardDataCollator(tokenizer=tokenizer, max_length=args.max_length)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        save_strategy="no",  # Don't save during training, save manually at end
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=10,
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=False,
        remove_unused_columns=False,
        report_to="none",
    )

    # Trainer
    trainer = RewardModelTrainer(
        model=reward_model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    # Train
    print("\nStart Reward Model training...")
    trainer.train()
    print("Reward Model training completed!")

    # Save model manually to avoid shared tensor issues
    final_path = os.path.join(args.output_dir, "final_model")
    os.makedirs(final_path, exist_ok=True)
    
    # Save only the reward model state dict (base model + value head)
    torch.save({
        'model_state_dict': reward_model.state_dict(),
        'config': base_model.config,
    }, os.path.join(final_path, "reward_model.pt"))
    
    # Save base model separately for easier loading
    base_model.save_pretrained(final_path, safe_serialization=False)
    
    tokenizer.save_pretrained(final_path)
    print(f"Reward Model saved to {final_path}")


if __name__ == "__main__":
    main()