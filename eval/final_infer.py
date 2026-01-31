#!/usr/bin/env python3
"""
RAG-based inference with dynamic few-shot example retrieval
"""

import os
import json
import glob
import pickle
import argparse
import time
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

from final_infer_utils import (
    FewShotExampleRetriever,
    TagEvaluator,
    truncate_context,
    extract_tags_with_explanations,
    load_single_file,
    evaluate_results,
    get_model_name_from_path,
    analyze_few_shot_usage,
    aggregate_few_shot_stats,
    print_few_shot_stats,
    extract_per_sample_few_shot_info,  
    extract_tags_from_explanations
)


def load_model(model_path):
    """Load model and tokenizer"""
    print(f"Loading model with vLLM: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = LLM(
        model=model_path,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=2048,
        enforce_eager=True,
        gpu_memory_utilization=0.9,
        tensor_parallel_size=1
    )
    
    print("vLLM model loaded")
    return model, tokenizer


def run_inference_with_rag(
    model, 
    tokenizer, 
    data, 
    output_file, 
    retriever, 
    batch_size=8, 
    save_interval=50,
    fixed_example_id=None,
    query_template=None
):
    """
    Run inference with RAG-based few-shot retrieval
    
    Args:
        model: vLLM model
        tokenizer: tokenizer
        data: list of samples
        output_file: path to save results
        retriever: FewShotExampleRetriever instance
        batch_size: batch size for inference
        save_interval: save checkpoint every N samples
        fixed_example_id: fixed example ID for upper bound analysis
        query_template: template string with {truncated_context} placeholder
    
    Returns:
        list of inference results
    """
    print(f"\nRunning vLLM inference with RAG on {len(data)} samples...")
    
    # Auto-detect template type
    if query_template is None:
        print("ERROR: query_template is required. Please provide --query_template parameter.")
        raise ValueError("query_template parameter is required")
    
    # Detect if this is TAG_LIST mode by checking for "tag list" keyword
    is_tag_list_template = "tag list" in query_template.lower()
    
    # Auto-configure features based on template
    use_few_shot = is_tag_list_template
    use_constraints = is_tag_list_template
    
    print(f"Template type: {'TAG_LIST' if is_tag_list_template else 'OPEN_GENERATE'}")
    print(f"Few-shot examples: {'ENABLED' if use_few_shot else 'DISABLED'}")
    print(f"Constrained decoding: {'ENABLED' if use_constraints else 'DISABLED'}")
    print(f"Top-k examples: {retriever.top_k if use_few_shot else 'N/A'}")
    
    start_time = time.time()
    all_results = []
    
    # Tag list for constrained decoding
    TAG_LIST = [
        "Code Debugging",
        "Code Programming",
        "English Multiple Choice",
        "English Question Answering",
        "Chinese Question Answering",
        "Summarization",
        "Character Identification",
        "Math Calculation",
        "Math Finding",
        "Number Retrieval",
        "PassKey Retrieval",
        "Key Value Retrieval"
    ]
    
    for start in tqdm(range(0, len(data), batch_size), desc="Processing batches"):
        batch = data[start:start + batch_size]
        batch_inputs = []
        batch_metadata = []
        
        for item in batch:
            inference_text = item.get('inference_context', item.get('input', ''))
            truncated_context = truncate_context(inference_text, tokenizer, max_tokens=300)
        
            # Retrieve few-shot examples (only for TAG_LIST template)
            if use_few_shot:
                if fixed_example_id:
                    relevant_examples = retriever.get_fixed_example(fixed_example_id)
                else:
                    relevant_examples = retriever.retrieve_relevant_examples(truncated_context)
                    # if relevant_examples and relevant_examples[0]['similarity_score'] < 0.85:
                    #     relevant_examples = retriever.retrieve_relevant_examples(truncated_context, top_k=2)
                relevant_examples = list(reversed(relevant_examples))
            else:
                relevant_examples = []
            
            # Build message list
            messages = []
            
            # Add system message (TAG_LIST only)
            if is_tag_list_template:
                messages.append({
                    "role": "system",
                    "content": [{
                        "type": "text",
                        "text": (
                            "Before choosing the tag, internally reason about the query step by step "
                            "(e.g., identify language, task type, and key cues), you must choose the most specific one. "
                            "But DO NOT include the reasoning in your output. "
                            "Only return the final answer in JSON format."
                        )
                    }]
                })

            # Add few-shot examples
            for ex in relevant_examples:
                messages.append({
                    "role": "user",
                    "content": [{"type": "text", "text": ex['query']}]
                })
                messages.append({
                    "role": "assistant",
                    "content": [{
                        "type": "text",
                        "text": json.dumps({
                            "tag": ex['tag'],
                            "explanation": ex['explanation']
                        }, ensure_ascii=False)
                    }]
                })
            
            # Build query prompt - replace placeholder with actual context
            query_prompt = query_template.replace("{truncated_context}", truncated_context)

            # Add query to messages
            messages.append({
                "role": "user",
                "content": [{"type": "text", "text": query_prompt}]
            })
            
            # Format prompt
            try:
                formatted_prompt = tokenizer.apply_chat_template(
                    messages, 
                    add_generation_prompt=True, 
                    tokenize=False
                )
            except:
                formatted_prompt = f"<start_of_turn>user\n{query_prompt}<end_of_turn>\n<start_of_turn>model\n"
            
            batch_inputs.append(formatted_prompt)
            batch_metadata.append({
                'item': item,
                'truncated_context': truncated_context,
                'retrieved_examples': [ex['id'] for ex in relevant_examples],
                'similarity_scores': [ex['similarity_score'] for ex in relevant_examples]
            })
        
        # Batch generation with optional constraints
        if use_constraints:
            schema = {
                "type": "object",
                "properties": {
                    "tag": {"type": "string", "enum": TAG_LIST},
                    "explanation": {"type": "string"}
                },
                "required": ["tag", "explanation"]
            }
            guided = GuidedDecodingParams(json=schema)
            sampling_params = SamplingParams(
                temperature=0.6,
                top_p=0.9,
                top_k=1,
                max_tokens=512,
                stop=["</s>", "\n\n\n"],
                guided_decoding=guided
            )
        else:
            sampling_params = SamplingParams(
                temperature=0.6,
                top_p=0.9,
                top_k=1,
                max_tokens=512,
                stop=["</s>", "\n\n\n"]
            )
        
        try:
            outputs = model.generate(batch_inputs, sampling_params)
            generated_texts = [output.outputs[0].text.strip() for output in outputs]
        except Exception as e:
            print(f"Error during generation: {e}")
            generated_texts = ["Generation failed"] * len(batch)
        
        # Parse and save results
        for i, (raw_text, metadata) in enumerate(zip(generated_texts, batch_metadata)):
            parsed_tags = extract_tags_with_explanations(raw_text)
            
            result = {
                **metadata['item'],
                'truncated_input': metadata['truncated_context'],
                'generated_tags': parsed_tags,
                'raw_response': raw_text,
                'retrieved_examples': metadata['retrieved_examples'],
                'similarity_scores': metadata['similarity_scores'],
                'used_constraints': use_constraints
            }
            all_results.append(result)
        
        # Checkpoint saving
        if len(all_results) % save_interval == 0 or (start + batch_size) >= len(data):
            with open(output_file, 'wb') as f:
                pickle.dump(all_results, f)
    
    end_time = time.time()
    print(f"\n✓ Inference completed in {end_time - start_time:.2f} seconds")
    print(f"  Total samples: {len(all_results)}")
    
    return all_results


def process_single_file(model, tokenizer, evaluator, retriever, json_file, output_dir, args, 
                       fixed_example_id=None, query_template=None):
    """Process a single data file"""
    rel_path = os.path.relpath(json_file, args.data_dir)
    rel_dir = os.path.dirname(rel_path)
    file_basename = os.path.splitext(os.path.basename(rel_path))[0]
    
    output_subdir = os.path.join(output_dir, rel_dir) if rel_dir else output_dir
    os.makedirs(output_subdir, exist_ok=True)
    
    output_file = os.path.join(output_subdir, f"{file_basename}_results.pkl")
    metrics_file = os.path.join(output_subdir, f"{file_basename}_metrics.json")
    
    print(f"\n{'='*80}\nProcessing: {rel_path}\n{'='*80}")
    retriever.current_file_id = file_basename
    
    # Load data
    data = load_single_file(json_file, args.num_samples)
    if not data:
        return None
    
    # Run inference
    try:
        results = run_inference_with_rag(
            model, tokenizer, data, output_file, retriever,
            batch_size=args.batch_size,
            save_interval=args.save_interval,
            fixed_example_id=fixed_example_id,
            query_template=query_template,
        )
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Evaluate
    eval_result = evaluate_results(results, evaluator)
    if eval_result:
        metrics, failed_cases = eval_result
        
        few_shot_stats = analyze_few_shot_usage(results)
        per_sample_few_shot = extract_per_sample_few_shot_info(results)
        
        print(f"\n{'='*60}")
        print(f"RESULTS FOR: {file_basename}")
        print(f"{'='*60}")
        print(f"Total samples: {metrics['total_samples']}")
        print(f"Valid samples: {metrics['valid_samples']}")
        print(f"Exact Match F1: {metrics['exact_match_f1']:.3f}")
        print(f"Semantic Accuracy: {metrics['semantic_accuracy']:.3f}")
        print(f"Semantic F1: {metrics['semantic_f1']:.3f}")
        
        if metrics['used_ground_truth_count'] > 0:
            print(f"\nPerfect match rate: {metrics['perfect_match_rate']:.1%}")

        print_few_shot_stats(few_shot_stats)
  
        detailed_metrics = {
            'file_name': file_basename,
            'metrics': metrics,
            'few_shot_statistics': few_shot_stats,
            'per_sample_few_shot_usage': per_sample_few_shot,  
            'top_failed_cases': failed_cases
        }
        
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_metrics, f, indent=2, ensure_ascii=False)
        print(f"✓ Metrics saved to: {metrics_file}")
        
        return {
            'file_name': file_basename,
            'metrics': metrics,
            'few_shot_stats': few_shot_stats,
            'output_file': output_file,
            'metrics_file': metrics_file
        }
    
    return None


def main():
    parser = argparse.ArgumentParser(description="Unified inference with RAG")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing data files")
    parser.add_argument("--examples_json", type=str, required=True,
                        help="Path to few_shot_examples.json")
    parser.add_argument("--top_k_examples", type=int, default=1,
                        help="Number of examples to retrieve per query")
    parser.add_argument("--output_prefix", type=str, default="results",
                        help="Output directory prefix")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Limit number of samples per file")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for inference")
    parser.add_argument("--save_interval", type=int, default=50,
                        help="Save results every N samples")
    parser.add_argument("--device", type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda'],
                        help="Device for evaluation")
    parser.add_argument("--query_template", type=str, required=True,
                        help="Query template string with {truncated_context} placeholder. Template containing 'tag list' triggers TAG_LIST mode (few-shot + constraints), others trigger OPEN_GENERATE mode")
    parser.add_argument("--upper_bound_analysis", action="store_true",
                        help="Run upper bound analysis: test each example separately on all data")
    
    args = parser.parse_args()
    
    # Setup output directory
    model_name = get_model_name_from_path(args.checkpoint_path)
    output_dir = f"{args.output_prefix}_{model_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Print configuration
    print(f"\n{'='*80}\nCONFIGURATION\n{'='*80}")
    print(f"Model: {args.checkpoint_path}")
    print(f"Data: {args.data_dir}")
    print(f"Examples: {args.examples_json}")
    print(f"Output: {output_dir}")
    print(f"Top-k examples: {args.top_k_examples}")
    print(f"Query template: {args.query_template[:100]}..." if len(args.query_template) > 100 else f"Query template: {args.query_template}")
    print(f"Note: Few-shot and constraints are auto-configured based on template type")
    
    # Find data files
    json_files = glob.glob(os.path.join(args.data_dir, "**", "*.json*"), recursive=True)
    pkl_files = glob.glob(os.path.join(args.data_dir, "**", "*.pkl"), recursive=True)
    all_files = json_files + pkl_files
    
    if not all_files:
        print("No data files found!")
        return
    
    print(f"\nFound {len(all_files)} data files")
    
    # Load model
    print(f"\n{'='*80}\nLOADING MODEL\n{'='*80}")
    model, tokenizer = load_model(args.checkpoint_path)
    
    # Load retriever
    print(f"\n{'='*80}\nLOADING FEW-SHOT RETRIEVER\n{'='*80}")
    retriever = FewShotExampleRetriever(
        examples_path=args.examples_json,
        top_k=args.top_k_examples
    )
    
    # Load evaluator
    print(f"\n{'='*80}\nLOADING EVALUATOR\n{'='*80}")
    device = args.device if args.device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
    evaluator = TagEvaluator(device=device)
    
    # Upper bound analysis mode
    if args.upper_bound_analysis:
        print(f"\n{'='*80}\nUPPER BOUND ANALYSIS MODE\n{'='*80}")
        print(f"Testing each example separately on all datasets")
        
        example_ids = [ex['id'] for ex in retriever.examples]
        print(f"Total examples to test: {len(example_ids)}")
        
        upper_bound_file = os.path.join(output_dir, "upper_bound_acc_matrix.json")
        
        # Load existing results if available
        if os.path.exists(upper_bound_file):
            with open(upper_bound_file, 'r', encoding='utf-8') as f:
                upper_bound_matrix = json.load(f)
            print(f"Loaded existing results from {upper_bound_file}")
        else:
            upper_bound_matrix = {}
        
        # Test each example
        for ex_idx, ex_id in enumerate(example_ids, 1):
            print(f"\n{'='*80}\nTESTING EXAMPLE {ex_idx}/{len(example_ids)}: {ex_id}\n{'='*80}")
            
            for file_idx, json_file in enumerate(all_files, 1):
                file_basename = os.path.splitext(os.path.basename(json_file))[0]
                
                # Skip if already processed
                if file_basename in upper_bound_matrix and ex_id in upper_bound_matrix[file_basename]:
                    print(f"[{file_idx}/{len(all_files)}] {file_basename} + {ex_id}: ALREADY PROCESSED, skipping")
                    continue
                
                print(f"\n[{file_idx}/{len(all_files)}] Processing: {file_basename} with example: {ex_id}")
                
                data = load_single_file(json_file, args.num_samples)
                if not data:
                    continue
                
                temp_output_file = os.path.join(output_dir, f"temp_{file_basename}_{ex_id}.pkl")
                
                try:
                    results = run_inference_with_rag(
                        model, tokenizer, data, temp_output_file, retriever,
                        batch_size=args.batch_size,
                        save_interval=args.save_interval,
                        fixed_example_id=ex_id,
                        query_template=args.query_template,
                    )
                    
                    eval_result = evaluate_results(results, evaluator)
                    if eval_result:
                        metrics, _ = eval_result
                        accuracy = metrics['semantic_accuracy']
                        
                        if file_basename not in upper_bound_matrix:
                            upper_bound_matrix[file_basename] = {}
                        upper_bound_matrix[file_basename][ex_id] = accuracy
                        
                        print(f"  -> Accuracy: {accuracy:.3f}")
                        
                        # Incremental save
                        with open(upper_bound_file, 'w', encoding='utf-8') as f:
                            json.dump(upper_bound_matrix, f, indent=2, ensure_ascii=False)
                        print(f"  -> Saved to {upper_bound_file}")
                    
                    if os.path.exists(temp_output_file):
                        os.remove(temp_output_file)
                        
                except Exception as e:
                    print(f"  -> Error: {e}")
                    import traceback
                    traceback.print_exc()
        
        print(f"\n{'='*80}\nUPPER BOUND ANALYSIS COMPLETED\n{'='*80}")
        print(f"Results saved to: {upper_bound_file}")
        return

    # Normal inference mode
    all_file_results = []
    for i, json_file in enumerate(all_files, 1):
        print(f"\n{'='*80}\nFILE {i}/{len(all_files)}\n{'='*80}")
        result = process_single_file(
            model, tokenizer, evaluator, retriever,
            json_file, output_dir, args,
            query_template=args.query_template
        )
        if result:
            all_file_results.append(result)
    
    # Overall summary
    if all_file_results:
        print(f"\n{'='*80}\nOVERALL SUMMARY\n{'='*80}")
        print(f"Successfully processed: {len(all_file_results)}/{len(all_files)} files")
        
        import numpy as np
        avg_metrics = {
            'exact_match_f1': np.mean([r['metrics']['exact_match_f1'] for r in all_file_results]),
            'semantic_accuracy': np.mean([r['metrics']['semantic_accuracy'] for r in all_file_results]),
            'semantic_f1': np.mean([r['metrics']['semantic_f1'] for r in all_file_results]),
            'total_samples': sum([r['metrics']['total_samples'] for r in all_file_results]),
        }
        
        overall_few_shot_stats = aggregate_few_shot_stats(all_file_results)
        
        print(f"\nAVERAGE METRICS:")
        print(f"Total samples: {avg_metrics['total_samples']}")
        print(f"Exact Match F1: {avg_metrics['exact_match_f1']:.3f}")
        print(f"Semantic Accuracy: {avg_metrics['semantic_accuracy']:.3f}")
        print(f"Semantic F1: {avg_metrics['semantic_f1']:.3f}")
        
        print_few_shot_stats(overall_few_shot_stats, title="OVERALL FEW-SHOT STATISTICS")
        
        # Save summary
        summary_file = os.path.join(output_dir, "summary_all_files.json")
        summary_data = {
            'overall_metrics': avg_metrics,
            'overall_few_shot_statistics': overall_few_shot_stats,
            'per_file_results': [
                {
                    'file_name': r['file_name'],
                    'metrics': r['metrics'],
                    'few_shot_stats': r.get('few_shot_stats', {})
                }
                for r in all_file_results
            ]
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        print(f"\nSummary saved to: {summary_file}")
    
    print(f"\n{'='*80}\nCOMPLETED!\n{'='*80}")
    print(f"All results saved in: {output_dir}")


if __name__ == "__main__":
    main()