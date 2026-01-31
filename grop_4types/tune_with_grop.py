#!/usr/bin/env python3
import os
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm


def freeze_layers(model, num_freeze: int):
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


class RewardModel(nn.Module):
    """Reward Model for inference"""
    def __init__(self, base_model):
        super().__init__()
        self.model = base_model
        self.value_head = nn.Linear(base_model.config.hidden_size, 1)
        
        # Move value_head to the same device as base_model
        if hasattr(base_model, 'device'):
            self.value_head = self.value_head.to(base_model.device)
        elif next(base_model.parameters(), None) is not None:
            self.value_head = self.value_head.to(next(base_model.parameters()).device)
        
    def forward(self, input_ids, attention_mask=None):
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            last_hidden_state = outputs.hidden_states[-1]
            
            if attention_mask is not None:
                sequence_lengths = attention_mask.sum(dim=1) - 1
                batch_size = input_ids.shape[0]
                last_hidden = last_hidden_state[torch.arange(batch_size, device=input_ids.device), sequence_lengths]
            else:
                last_hidden = last_hidden_state[:, -1, :]
            
            reward = self.value_head(last_hidden)
            return reward.squeeze(-1)


def load_queries_from_dpo(data_dir):
    """Load queries from DPO data"""
    all_data = []
    for file in os.listdir(data_dir):
        if file.endswith(".jsonl"):
            file_path = os.path.join(data_dir, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        user_input = item.get("input", "")
                        prompt_text = f"You are a helpful assistant. {user_input}\nAssistant: "
                        
                        all_data.append({"query": prompt_text})
                        
                    except json.JSONDecodeError:
                        continue
    
    return all_data


class GRPOTrainer:
    def __init__(
        self, 
        policy_model, 
        ref_model,
        reward_model,
        tokenizer,
        args
    ):
        self.policy_model = policy_model
        self.ref_model = ref_model
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.args = args
        
        self.optimizer = torch.optim.AdamW(
            policy_model.parameters(), 
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        
        self.device = next(policy_model.parameters()).device
        
    def sample_outputs(self, query, num_samples):
        """Sample G outputs from current policy"""
        inputs = self.tokenizer(
            query, 
            return_tensors="pt",
            max_length=self.args.max_prompt_length,
            truncation=True
        ).to(self.device)
        
        outputs_list = []
        output_ids_list = []
        
        with torch.no_grad():
            for _ in range(num_samples):
                output_ids = self.policy_model.generate(
                    **inputs,
                    max_new_tokens=self.args.max_length - inputs.input_ids.shape[1],
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
                
                output_text = self.tokenizer.decode(
                    output_ids[0][inputs.input_ids.shape[1]:], 
                    skip_special_tokens=True
                )
                
                outputs_list.append(output_text)
                output_ids_list.append(output_ids[0])
                
        return outputs_list, output_ids_list, inputs.input_ids.shape[1]
    
    def compute_rewards(self, query, outputs):
        """Compute rewards using reward model"""
        rewards = []
        
        for output in outputs:
            full_text = query + output
            inputs = self.tokenizer(
                full_text,
                return_tensors="pt",
                max_length=self.args.max_length,
                truncation=True,
                padding=True
            ).to(self.device)
            
            reward = self.reward_model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask
            )
            
            rewards.append(reward.item())
        
        return np.array(rewards)
    
    def compute_weights(self, rewards):
        """Compute softmax weights for GRPO with group-relative baseline"""
        tau = getattr(self.args, 'tau', 1.0)
        G = len(rewards)
        
        # Compute group-relative advantages: A_i = r_i - mean(r_j for j != i)
        advantages = np.zeros(G)
        for i in range(G):
            others_mean = (np.sum(rewards) - rewards[i]) / (G - 1)
            advantages[i] = rewards[i] - others_mean
        
        # Softmax over advantages (not rewards)
        exp_advantages = np.exp(advantages / tau)
        weights = exp_advantages / np.sum(exp_advantages)
        return weights
    
    def compute_grpo_loss(self, query, outputs, output_ids_list, query_len, weights):
        """Compute GRPO loss with softmax weights and KL penalty"""
        policy_losses = []
        kl_losses = []
        
        for output_ids, weight in zip(output_ids_list, weights):
            # Prepare inputs
            inputs = self.tokenizer(
                query + self.tokenizer.decode(output_ids[query_len:], skip_special_tokens=True),
                return_tensors="pt",
                max_length=self.args.max_length,
                truncation=True
            ).to(self.device)
            
            # Policy model forward
            policy_outputs = self.policy_model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask
            )
            policy_logits = policy_outputs.logits
            
            # Reference model forward
            with torch.no_grad():
                ref_outputs = self.ref_model(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask
                )
                ref_logits = ref_outputs.logits
            
            # Compute log probs for answer tokens only
            answer_start = query_len - 1
            answer_end = inputs.input_ids.shape[1] - 1
            
            if answer_end <= answer_start:
                continue
            
            # Policy log probs
            policy_log_probs = F.log_softmax(policy_logits[0, answer_start:answer_end], dim=-1)
            answer_tokens = inputs.input_ids[0, answer_start+1:answer_end+1]
            policy_token_log_probs = policy_log_probs.gather(1, answer_tokens.unsqueeze(1)).squeeze(1)
            
            # Reference log probs
            ref_log_probs = F.log_softmax(ref_logits[0, answer_start:answer_end], dim=-1)
            ref_token_log_probs = ref_log_probs.gather(1, answer_tokens.unsqueeze(1)).squeeze(1)
            
            # GRPO policy gradient loss with softmax weight
            pg_loss = -weight * policy_token_log_probs.mean()
            
            # KL divergence: E[π_ref/π - log(π_ref/π) - 1]
            log_ratio = ref_token_log_probs - policy_token_log_probs
            kl = (torch.exp(log_ratio) - log_ratio - 1).mean()
            
            policy_losses.append(pg_loss)
            kl_losses.append(kl)
        
        if not policy_losses:
            return None
        
        # Total loss
        avg_pg_loss = torch.stack(policy_losses).mean()
        avg_kl_loss = torch.stack(kl_losses).mean()
        total_loss = avg_pg_loss + self.args.beta * avg_kl_loss
        
        return total_loss, avg_pg_loss.item(), avg_kl_loss.item()
    
    def train_step(self, batch):
        """Single training step - sample then update"""
        # Step 1: Sample all data
        all_data = []
        self.policy_model.eval()
        
        for query in batch['query']:
            outputs, output_ids_list, query_len = self.sample_outputs(
                query, 
                self.args.num_samples
            )
            
            rewards = self.compute_rewards(query, outputs)
            
            # Homogeneity filtering: skip if reward std < 0.1
            if np.std(rewards) < 0.1:
                continue
            
            weights = self.compute_weights(rewards)
            
            all_data.append({
                'query': query,
                'outputs': outputs,
                'output_ids_list': output_ids_list,
                'query_len': query_len,
                'weights': weights
            })
        
        # Step 2: Update policy
        self.policy_model.train()
        
        total_loss = 0
        total_pg_loss = 0
        total_kl_loss = 0
        valid_samples = 0
        
        for data in all_data:
            loss_result = self.compute_grpo_loss(
                data['query'], 
                data['outputs'], 
                data['output_ids_list'],
                data['query_len'], 
                data['weights']
            )
            
            if loss_result is None:
                continue
            
            loss, pg_loss, kl_loss = loss_result
            total_loss += loss
            total_pg_loss += pg_loss
            total_kl_loss += kl_loss
            valid_samples += 1
        
        if valid_samples == 0:
            return 0, 0, 0
        
        # Backward
        avg_loss = total_loss / valid_samples
        self.optimizer.zero_grad()
        avg_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), self.args.max_grad_norm)
        self.optimizer.step()
        
        return avg_loss.item(), total_pg_loss / valid_samples, total_kl_loss / valid_samples
    
    def train(self, train_dataset):
        """Training loop"""
        dataloader = DataLoader(
            train_dataset, 
            batch_size=self.args.batch_size,
            shuffle=True
        )
        
        print(f"\nStarting GRPO training...")
        print(f"Total epochs: {self.args.epochs}")
        print(f"Batch size: {self.args.batch_size}")
        print(f"Number of samples per query: {self.args.num_samples}")
        print(f"Learning rate: {self.args.lr}")
        print(f"Beta (KL coefficient): {self.args.beta}")
        
        global_step = 0
        for epoch in range(self.args.epochs):
            epoch_loss = 0
            epoch_pg_loss = 0
            epoch_kl_loss = 0
            num_batches = 0
            
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.args.epochs}")
            for batch in pbar:
                loss, pg_loss, kl_loss = self.train_step(batch)
                
                if loss > 0:
                    epoch_loss += loss
                    epoch_pg_loss += pg_loss
                    epoch_kl_loss += kl_loss
                    num_batches += 1
                    global_step += 1
                    
                    pbar.set_postfix({
                        'loss': f'{loss:.4f}',
                        'pg_loss': f'{pg_loss:.4f}',
                        'kl_loss': f'{kl_loss:.4f}'
                    })
                    
                    if global_step % 10 == 0:
                        print(f"\nStep {global_step} - Loss: {loss:.4f}, PG Loss: {pg_loss:.4f}, KL Loss: {kl_loss:.4f}")
            
            if num_batches > 0:
                avg_loss = epoch_loss / num_batches
                avg_pg_loss = epoch_pg_loss / num_batches
                avg_kl_loss = epoch_kl_loss / num_batches
                print(f"\nEpoch {epoch+1} completed:")
                print(f"  Avg Loss: {avg_loss:.4f}")
                print(f"  Avg PG Loss: {avg_pg_loss:.4f}")
                print(f"  Avg KL Loss: {avg_kl_loss:.4f}")
                
                # Save checkpoint after each epoch
                checkpoint_path = os.path.join(self.args.output_dir, f"checkpoint-epoch-{epoch+1}")
                os.makedirs(checkpoint_path, exist_ok=True)
                self.policy_model.save_pretrained(checkpoint_path, safe_serialization=False)
                print(f"  Checkpoint saved to {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser(description="GRPO training with reward model")
    parser.add_argument("--dpo_data_dir", type=str, default="./dpo_data")
    parser.add_argument("--sft_model_dir", type=str, required=True)
    parser.add_argument("--reward_model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./grpo_output")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--beta", type=float, default=0.04)
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_prompt_length", type=int, default=512)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--freeze_layers", type=int, default=20)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=1.0, help="Temperature for softmax weighting in GRPO")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load data
    print(f"\nLoading queries from {args.dpo_data_dir}")
    train_data = load_queries_from_dpo(args.dpo_data_dir)
    print(f"Loaded {len(train_data)} queries")

    # Load tokenizer
    print(f"\nLoading tokenizer from {args.sft_model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(args.sft_model_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load policy model
    print(f"Loading policy model from {args.sft_model_dir}")
    policy_model = AutoModelForCausalLM.from_pretrained(
        args.sft_model_dir,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    if hasattr(policy_model.config, "use_cache"):
        policy_model.config.use_cache = False
    freeze_layers(policy_model, args.freeze_layers)

    # Load reference model
    print(f"Loading reference model from {args.sft_model_dir}")
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.sft_model_dir,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    # Load reward model
    print(f"Loading reward model from {args.reward_model_path}")
    reward_base_model = AutoModelForCausalLM.from_pretrained(
        args.reward_model_path,  # Load from saved reward model path
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    
    reward_model = RewardModel(reward_base_model)
    checkpoint = torch.load(os.path.join(args.reward_model_path, "reward_model.pt"), weights_only=False)
    reward_model.load_state_dict(checkpoint['model_state_dict'])
    
    # Ensure value_head is on the correct device after loading
    if next(reward_base_model.parameters(), None) is not None:
        device = next(reward_base_model.parameters()).device
        reward_model.value_head = reward_model.value_head.to(device)
    
    reward_model.eval()
    for param in reward_model.parameters():
        param.requires_grad = False
    
    print("Reward model loaded successfully")

    # Prepare dataset
    train_dataset = Dataset.from_list(train_data)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # GRPO Trainer
    grpo_trainer = GRPOTrainer(
        policy_model=policy_model,
        ref_model=ref_model,
        reward_model=reward_model,
        tokenizer=tokenizer,
        args=args
    )

    # Train
    try:
        grpo_trainer.train(train_dataset)
        print("\nGRPO training completed!")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Always save model even if training fails
        print("\nSaving model...")
        final_path = os.path.join(args.output_dir, "final_model")
        os.makedirs(final_path, exist_ok=True)
        
        # Save with safe_serialization=False to avoid Gemma shared tensor issues
        policy_model.save_pretrained(final_path, safe_serialization=False)
        tokenizer.save_pretrained(final_path)
        
        # Also save config explicitly
        policy_model.config.save_pretrained(final_path)
        
        print(f"Model saved to {final_path}")
        print(f"Saved files: {os.listdir(final_path)}")


if __name__ == "__main__":
    main()