#!/usr/bin/env python3
"""
Helpers for final_infer.py — evaluator, retriever, and small utils.
"""

import os
import json
import re
import pickle
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import util


# ============================================================================
# Text Processing
# ============================================================================

def truncate_context(context: str, tokenizer, max_tokens: int = 600) -> str:
    """Shorten context to max tokens; keep start and end."""
    max_each = max_tokens // 2
    tokens = tokenizer.encode(context, add_special_tokens=False)
    if len(tokens) <= max_tokens:
        return context
    
    front_tokens, back_tokens = tokens[:max_each], tokens[-max_each:]
    front_text = tokenizer.decode(front_tokens, skip_special_tokens=True)
    back_text = tokenizer.decode(back_tokens, skip_special_tokens=True)
    return front_text + "\n\n[Content truncated]\n\n" + back_text


def extract_tags_with_explanations(tags_text):
    """Parse tags+explanations from model output (returns list of dicts)."""
    try:
        # Find individual JSON objects
        single_json_pattern = r'\{[^{}]*"tag"[^{}]*"explanation"[^{}]*\}'
        matches = re.findall(single_json_pattern, tags_text)
        
        if matches:
            valid_tags = []
            for match in matches:
                try:
                    item = json.loads(match)
                    if "tag" in item and "explanation" in item:
                        valid_tags.append({
                            "tag": str(item["tag"]).strip(),
                            "explanation": str(item["explanation"]).strip()
                        })
                except:
                    continue
            return valid_tags if valid_tags else [{"tag": "General", "explanation": "Unable to parse"}]
        
        return [{"tag": "General", "explanation": "Unable to parse"}]
    except:
        return [{"tag": "Error", "explanation": "Failed to parse"}]


def extract_tags_from_explanations(tags_explanations):
    """Get plain tag strings from parsed tag-explanation items."""
    if not tags_explanations:
        return []
    return [item['tag'] if isinstance(item, dict) and 'tag' in item else item 
            for item in tags_explanations]


# ============================================================================
# Few-shot RAG Retriever
# ============================================================================

class FewShotExampleRetriever:
    """Find similar few-shot examples using embeddings."""
    
    def __init__(self, examples_path, embedding_model="whaleloops/phrase-bert", top_k=1):
        """Load examples and encoder, cache example embeddings."""
        self.top_k = top_k
        
        # Load examples
        print(f"Loading few-shot examples from: {examples_path}")
        with open(examples_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.examples = data['examples']
        
        print(f"Loaded {len(self.examples)} examples")
        
        # Load embedding model
        self.encoder = SentenceTransformer(embedding_model)
        
        # Pre-compute example embeddings
        print(f"Pre-computing embeddings for {len(self.examples)} examples...")
        self.example_queries = [ex['query'] for ex in self.examples]
        self.example_embeddings = self.encoder.encode(
            self.example_queries,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=True
        )
        print(f"✓ Example embeddings cached")
    
    def retrieve_relevant_examples(self, query, top_k=None):
        """Return top-k examples most similar to the query."""
        if top_k is None:
            top_k = self.top_k
        
        # Compute query embedding
        query_embedding = self.encoder.encode(
            query,
            convert_to_tensor=True,
            normalize_embeddings=True
        )
        
        # Compute similarity
        similarities = util.cos_sim(query_embedding, self.example_embeddings)[0]
        
        # Get top-k
        top_indices = similarities.argsort(descending=True)[:top_k]
        
        # Return examples with similarity scores
        retrieved = []
        for idx in top_indices:
            example = self.examples[idx.item()].copy()
            example['similarity_score'] = similarities[idx].item()
            retrieved.append(example)
        
        return retrieved
    
    def get_fixed_example(self, example_id):
        """Return the example with given id (or empty list)."""
        for ex in self.examples:
            if ex['id'] == example_id:
                example = ex.copy()
                example['similarity_score'] = 1.0
                return [example]
        return []


# ============================================================================
# Tag Evaluator
# ============================================================================

class TagEvaluator:
    """Evaluate tag sets: exact match and semantic metrics."""
    
    def __init__(self, device=None):
        """Init evaluator: set device, load sentence model, init cache."""
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        try:
            self.sentence_model = SentenceTransformer('whaleloops/phrase-bert', device=self.device)
            print(f"Loaded PHRASEBERT on {self.device}")
        except Exception as e:
            print(f"Warning: Could not load PHRASEBERT: {e}")
            self.sentence_model = None
        
        self.embedding_cache = {}
    
    def get_embeddings(self, tags):
        """Get embeddings for tags; use cache to avoid re-encoding."""
        if not tags:
            return np.array([])
        
        # Find new tags
        new_tags = [tag for tag in tags if tag not in self.embedding_cache]
        
        if new_tags:
            try:
                new_embeddings = self.sentence_model.encode(
                    new_tags,
                    batch_size=64,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    device=self.device
                )
                for tag, emb in zip(new_tags, new_embeddings):
                    self.embedding_cache[tag] = emb
            except Exception as e:
                print(f"Error encoding tags: {e}")
                return np.array([])
        
        return np.array([self.embedding_cache[tag] for tag in tags])
    
    def calculate_exact_match_f1(self, pred_tags, gold_tags):
        """Compute exact-match F1 (case-insensitive)."""
        if not pred_tags and not gold_tags:
            return 1.0
        if not pred_tags or not gold_tags:
            return 0.0
        
        pred_set = set([tag.lower().strip() for tag in pred_tags])
        gold_set = set([tag.lower().strip() for tag in gold_tags])
        
        intersection = pred_set & gold_set
        precision = len(intersection) / len(pred_set) if pred_set else 0
        recall = len(intersection) / len(gold_set) if gold_set else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return f1
    
    def calculate_semantic_accuracy(self, pred_tags, gold_tags, threshold=0.8):
        """Compute semantic accuracy: fraction of gold tags matched by any pred tag."""
        if self.sentence_model is None:
            return 0.0, []
        if not pred_tags and not gold_tags:
            return 1.0, []
        if not pred_tags or not gold_tags:
            return 0.0, []
        
        try:
            pred_embeddings = self.get_embeddings(pred_tags)
            gold_embeddings = self.get_embeddings(gold_tags)
            
            if pred_embeddings.size == 0 or gold_embeddings.size == 0:
                return 0.0, []
            
            # Compute similarity matrix
            if self.device == 'cuda' and torch.cuda.is_available():
                pred_tensor = torch.tensor(pred_embeddings, device='cuda')
                gold_tensor = torch.tensor(gold_embeddings, device='cuda')
                sim_matrix = torch.cosine_similarity(
                    gold_tensor.unsqueeze(1), pred_tensor.unsqueeze(0), dim=2
                )
                matched = (sim_matrix > threshold).any(dim=1)
                accuracy = matched.sum().float() / len(gold_tags)
                sim_matrix_np = sim_matrix.cpu().numpy()
                accuracy = accuracy.cpu().item()
            else:
                sim_matrix_np = cosine_similarity(gold_embeddings, pred_embeddings)
                matched = (sim_matrix_np > threshold).any(axis=1)
                accuracy = matched.sum() / len(gold_tags)
            
            # Record max similarity pairs
            max_sim_pairs = []
            for gt_idx, gt_tag in enumerate(gold_tags):
                max_pred_idx = np.argmax(sim_matrix_np[gt_idx])
                max_similarity = sim_matrix_np[gt_idx, max_pred_idx]
                max_sim_pairs.append({
                    'gt_tag': gt_tag,
                    'pred_tag': pred_tags[max_pred_idx],
                    'similarity': float(max_similarity)
                })
            
            return accuracy, max_sim_pairs
        
        except Exception as e:
            print(f"Error calculating semantic accuracy: {e}")
            return 0.0, []
    
    def calculate_semantic_f1(self, pred_tags, gold_tags, threshold=0.8):
        """Compute semantic F1 using similarity threshold."""
        if self.sentence_model is None or not pred_tags or not gold_tags:
            return 0.0
        
        try:
            pred_embeddings = self.get_embeddings(pred_tags)
            gold_embeddings = self.get_embeddings(gold_tags)
            
            if pred_embeddings.size == 0 or gold_embeddings.size == 0:
                return 0.0
            
            # Compute similarity matrix
            if self.device == 'cuda' and torch.cuda.is_available():
                pred_tensor = torch.tensor(pred_embeddings, device='cuda')
                gold_tensor = torch.tensor(gold_embeddings, device='cuda')
                sim_matrix = torch.cosine_similarity(
                    gold_tensor.unsqueeze(1), pred_tensor.unsqueeze(0), dim=2
                )
                sim_matrix_np = sim_matrix.cpu().numpy()
            else:
                sim_matrix_np = cosine_similarity(gold_embeddings, pred_embeddings)
            
            # Calculate precision and recall
            gt_matched = (sim_matrix_np > threshold).any(axis=1).sum()
            recall = gt_matched / len(gold_tags)
            
            pred_matched = (sim_matrix_np > threshold).any(axis=0).sum()
            precision = pred_matched / len(pred_tags)
            
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            return f1
        
        except Exception as e:
            print(f"Error calculating semantic F1: {e}")
            return 0.0
    
    def calculate_precision_recall(self, pred_tags, gold_tags, threshold=0.8):
        """Return exact-match and semantic precision & recall."""
        # Exact match
        if not pred_tags and not gold_tags:
            em_precision = em_recall = 1.0
        elif not pred_tags or not gold_tags:
            em_precision = em_recall = 0.0
        else:
            pred_set = set([tag.lower().strip() for tag in pred_tags])
            gold_set = set([tag.lower().strip() for tag in gold_tags])
            intersection = pred_set & gold_set
            em_precision = len(intersection) / len(pred_set) if pred_set else 0
            em_recall = len(intersection) / len(gold_set) if gold_set else 0
        
        # Semantic match
        if self.sentence_model is None or not pred_tags or not gold_tags:
            sem_precision = sem_recall = 0.0
        else:
            try:
                pred_embeddings = self.get_embeddings(pred_tags)
                gold_embeddings = self.get_embeddings(gold_tags)
                
                if pred_embeddings.size == 0 or gold_embeddings.size == 0:
                    sem_precision = sem_recall = 0.0
                else:
                    if self.device == 'cuda' and torch.cuda.is_available():
                        pred_tensor = torch.tensor(pred_embeddings, device='cuda')
                        gold_tensor = torch.tensor(gold_embeddings, device='cuda')
                        sim_matrix = torch.cosine_similarity(
                            gold_tensor.unsqueeze(1), pred_tensor.unsqueeze(0), dim=2
                        )
                        sim_matrix_np = sim_matrix.cpu().numpy()
                    else:
                        sim_matrix_np = cosine_similarity(gold_embeddings, pred_embeddings)
                    
                    pred_matched = (sim_matrix_np > threshold).any(axis=0).sum()
                    gold_matched = (sim_matrix_np > threshold).any(axis=1).sum()
                    sem_precision = pred_matched / len(pred_tags) if pred_tags else 0.0
                    sem_recall = gold_matched / len(gold_tags) if gold_tags else 0.0
            except Exception as e:
                print(f"Error calculating precision/recall: {e}")
                sem_precision = sem_recall = 0.0
        
        return em_precision, em_recall, sem_precision, sem_recall


# ============================================================================
# Evaluation Functions
# ============================================================================

def evaluate_results(results, evaluator):
    """
    Compute metrics from inference results.
    Returns (metrics, failed_cases) or None if no valid samples.
    """
    total_samples = 0
    valid_samples = 0
    valid_pred_tags_list, valid_gold_tags_list = [], []
    failed_cases = []
    perfect_match_count = 0
    used_ground_truth_count = 0
    
    for idx, result in enumerate(results):
        if 'error' in result:
            continue
        
        total_samples += 1
        
        # Extract predicted tags
        pred_tags = extract_tags_from_explanations(result.get('generated_tags', []))
        
        # Extract ground truth tags (prefer parsed_tags)
        has_parsed_tags = 'parsed_tags' in result and result['parsed_tags']
        if has_parsed_tags:
            gold_tags = extract_tags_from_explanations(result.get('parsed_tags', []))
        else:
            gold_tags = extract_tags_from_explanations(result.get('ground_truth', []))
            used_ground_truth_count += 1
        
        if gold_tags:
            valid_samples += 1
            valid_pred_tags_list.append(pred_tags)
            valid_gold_tags_list.append(gold_tags)
            
            # Calculate metrics
            em_f1 = evaluator.calculate_exact_match_f1(pred_tags, gold_tags)
            sem_acc, max_sim_pairs = evaluator.calculate_semantic_accuracy(pred_tags, gold_tags)
            
            # Perfect match stats (only for ground_truth cases)
            if not has_parsed_tags:
                if len(pred_tags) == len(gold_tags) and set(pred_tags) == set(gold_tags):
                    perfect_match_count += 1
            
            # Record failed cases
            if em_f1 < 0.5 or sem_acc < 0.5:
                failed_cases.append({
                    'sample_idx': idx,
                    'predicted_tags': pred_tags,
                    'ground_truth_tags': gold_tags,
                    'em_f1': em_f1,
                    'semantic_accuracy': sem_acc,
                    'max_similarity_pairs': max_sim_pairs,
                    'truncated_input': result.get('truncated_input', 'N/A')[:300] + '...',
                    'used_ground_truth': not has_parsed_tags
                })
    
    if valid_samples == 0:
        return None
    
    # Calculate overall metrics
    all_em_f1, all_sem_acc, all_sem_f1 = [], [], []
    all_em_prec, all_em_rec, all_sem_prec, all_sem_rec = [], [], [], []
    
    for pred_tags, gold_tags in zip(valid_pred_tags_list, valid_gold_tags_list):
        em_f1 = evaluator.calculate_exact_match_f1(pred_tags, gold_tags)
        sem_acc, _ = evaluator.calculate_semantic_accuracy(pred_tags, gold_tags)
        sem_f1 = evaluator.calculate_semantic_f1(pred_tags, gold_tags)
        em_prec, em_rec, sem_prec, sem_rec = evaluator.calculate_precision_recall(pred_tags, gold_tags)
        
        all_em_f1.append(em_f1)
        all_sem_acc.append(sem_acc)
        all_sem_f1.append(sem_f1)
        all_em_prec.append(em_prec)
        all_em_rec.append(em_rec)
        all_sem_prec.append(sem_prec)
        all_sem_rec.append(sem_rec)
    
    metrics = {
        'total_samples': total_samples,
        'valid_samples': valid_samples,
        'exact_match_f1': np.mean(all_em_f1),
        'exact_match_precision': np.mean(all_em_prec),
        'exact_match_recall': np.mean(all_em_rec),
        'semantic_accuracy': np.mean(all_sem_acc),
        'semantic_f1': np.mean(all_sem_f1),
        'semantic_precision': np.mean(all_sem_prec),
        'semantic_recall': np.mean(all_sem_rec),
        'used_ground_truth_count': used_ground_truth_count,
        'perfect_match_count': perfect_match_count,
        'perfect_match_rate': perfect_match_count / used_ground_truth_count if used_ground_truth_count > 0 else 0.0
    }
    
    # Sort failed cases by F1
    failed_cases.sort(key=lambda x: x['em_f1'])
    
    return metrics, failed_cases[:10]


# ============================================================================
# File Loading
# ============================================================================

def load_single_file(file_path: str, num_samples: int = None):
    """Read one file (pkl/jsonl/json). Add inference_context to items."""
    file_data = []
    
    if file_path.endswith('.pkl'):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        if isinstance(data, list):
            file_data = data
    elif file_path.endswith('.jsonl'):
        with open(file_path, 'r', encoding='utf-8') as f:
            file_data = [json.loads(line) for line in f if line.strip()]
    else:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            file_data = data if isinstance(data, list) else [data]
    
    # Build inference_context
    for item in file_data:
        input_content = item.get('input') or item.get('prompt')
        context_content = item.get('context', '')
        item['inference_context'] = f"{input_content}\n\n{context_content}" if context_content else input_content
    
    if num_samples:
        file_data = file_data[:num_samples]
    
    print(f"Loaded {len(file_data)} samples from {os.path.basename(file_path)}")
    return file_data


def get_model_name_from_path(model_path: str) -> str:
    """Return a simple name from a path (fallback: sanitized path)."""
    if not os.path.exists(model_path):
        return model_path.replace('/', '_').replace('\\', '_')
    else:
        return os.path.basename(os.path.normpath(model_path))


# ============================================================================
# Few-shot Statistics
# ============================================================================

def analyze_few_shot_usage(results):
    """
    Count few-shot example usage and similarity stats per run.
    Returns basic usage dict.
    """
    from collections import Counter
    
    # Collect all retrieved examples
    all_retrieved_examples = []
    example_similarity_scores = {}
    
    for result in results:
        if 'retrieved_examples' in result:
            retrieved = result['retrieved_examples']
            scores = result.get('similarity_scores', [])
            
            all_retrieved_examples.extend(retrieved)
            
            # Record similarity scores per example
            for ex_id, score in zip(retrieved, scores):
                if ex_id not in example_similarity_scores:
                    example_similarity_scores[ex_id] = []
                example_similarity_scores[ex_id].append(score)
    
    # If no examples retrieved
    if not all_retrieved_examples:
        return {
            'total_retrievals': 0,
            'unique_examples_used': 0,
            'example_usage_frequency': {},
            'example_avg_similarity': {},
            'most_used_examples': []
        }
    
    # Count usage frequency
    example_usage_count = Counter(all_retrieved_examples)
    
    # Calculate average similarity per example
    example_avg_similarity = {
        ex_id: sum(scores) / len(scores)
        for ex_id, scores in example_similarity_scores.items()
    }
    
    # Build statistics
    few_shot_stats = {
        'total_retrievals': len(all_retrieved_examples),
        'unique_examples_used': len(example_usage_count),
        'example_usage_frequency': dict(example_usage_count.most_common()),
        'example_avg_similarity': example_avg_similarity,
        'most_used_examples': [
            {
                'example_id': ex_id,
                'usage_count': count,
                'avg_similarity': example_avg_similarity.get(ex_id, 0.0)
            }
            for ex_id, count in example_usage_count.most_common(10)
        ]
    }
    
    return few_shot_stats


def aggregate_few_shot_stats(all_file_results):
    """
    Combine few-shot stats from multiple files into overall stats.
    """
    from collections import Counter
    
    overall_example_usage = Counter()
    overall_example_similarities = {}
    
    for r in all_file_results:
        if 'few_shot_stats' in r:
            stats = r['few_shot_stats']
            
            # Accumulate usage frequency
            for ex_id, count in stats.get('example_usage_frequency', {}).items():
                overall_example_usage[ex_id] += count
            
            # Accumulate similarities
            for ex_id, avg_sim in stats.get('example_avg_similarity', {}).items():
                if ex_id not in overall_example_similarities:
                    overall_example_similarities[ex_id] = []
                overall_example_similarities[ex_id].append(avg_sim)
    
    # If no data
    if not overall_example_usage:
        return {
            'total_retrievals': 0,
            'unique_examples_used': 0,
            'example_usage_frequency': {},
            'example_avg_similarity': {},
            'most_used_examples': []
        }
    
    # Calculate overall average similarity
    overall_avg_similarity = {
        ex_id: sum(sims) / len(sims)
        for ex_id, sims in overall_example_similarities.items()
    }
    
    overall_few_shot_stats = {
        'total_retrievals': sum(overall_example_usage.values()),
        'unique_examples_used': len(overall_example_usage),
        'example_usage_frequency': dict(overall_example_usage.most_common()),
        'example_avg_similarity': overall_avg_similarity,
        'most_used_examples': [
            {
                'example_id': ex_id,
                'usage_count': count,
                'avg_similarity': overall_avg_similarity.get(ex_id, 0.0)
            }
            for ex_id, count in overall_example_usage.most_common()
        ]
    }
    
    return overall_few_shot_stats


def print_few_shot_stats(stats, title="FEW-SHOT STATISTICS"):
    """Print a short summary of few-shot stats."""
    print(f"\n{title}:")
    print(f"Total retrievals: {stats['total_retrievals']}")
    print(f"Unique examples used: {stats['unique_examples_used']}")
    
    if stats['most_used_examples']:
        print(f"\nMost used examples:")
        for item in stats['most_used_examples'][:10]:
            print(f"  - {item['example_id']}: {item['usage_count']} times "
                  f"(avg sim: {item['avg_similarity']:.3f})")
            

def extract_per_sample_few_shot_info(results):
    """
    Return per-sample few-shot details for inspection.
    Each item contains input, preds, golds, retrieved examples, scores.
    """
    per_sample_info = []
    
    for idx, result in enumerate(results):
        if 'error' in result:
            per_sample_info.append({
                'sample_idx': idx,
                'error': result['error']
            })
            continue
        
        # Extract few-shot info
        sample_info = {
            'sample_idx': idx,
            'input': result.get('truncated_input', result.get('input', 'N/A'))[:200] + '...',
            'predicted_tags': extract_tags_from_explanations(result.get('generated_tags', [])),
            'ground_truth_tags': extract_tags_from_explanations(
                result.get('parsed_tags', result.get('ground_truth', []))
            ),
            'retrieved_examples': result.get('retrieved_examples', []),
            'similarity_scores': result.get('similarity_scores', []),
            'used_constraints': result.get('used_constraints', False)
        }
        
        per_sample_info.append(sample_info)
    
    return per_sample_info
