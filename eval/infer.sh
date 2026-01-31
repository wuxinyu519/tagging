#!/bin/bash
#PBS -N tag
#PBS -o rag_tag.out
#PBS -e rag_tag.err
#PBS -l walltime=48:00:00
#PBS -q poderoso
#PBS -l select=1:ncpus=8:ngpus=1:host=gpu008

cd $PBS_O_WORKDIR

module load cuda12.6/toolkit
eval "$(conda shell.bash hook)"
conda activate gemma

export CUDA_VISIBLE_DEVICES=6
export TOKENIZERS_PARALLELISM=false
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

echo "Starting inference at $(date)"
nvidia-smi

# ============================================================================
# Config
# ============================================================================
# open generation GT: ./infinite_bench/infinitebench_gpt_gt/cleaned_tags_jsonl/individual_files 
# taglist GT: ./infinite_bench/infinitebench_gpt_gt/groundtruth_change
MODEL_PATH="../dpo/dpo_gemma1b_filtered_final/final_model" 
DATA_DIR="./infinite_bench/infinitebench_gpt_gt/groundtruth_change"
EXAMPLES_JSON="./few_shot_examples.json"
OUTPUT_PREFIX="./infinite_bench/final_dpo_gemma1b_final"

# ============================================================================
# Template Selection - Change this to 1 or 2
# ============================================================================
TEMPLATE_TYPE=1  # 1: TAG_LIST, 2: OPEN_GENERATE

# ============================================================================
# Template Definitions
# ============================================================================
case $TEMPLATE_TYPE in
    1)
        # TAG_LIST template 
        QUERY_TEMPLATE="You are a helpful assistant. Please identify the best tag from the list below that matches the user query. Provide an explanation for your choice. Respond in JSON: {\"tag\": str, \"explanation\": str}.

## Tag list
 - Tag name: Code Debugging, Description: Asking for which function in code causes an error.
 - Tag name: Code Programming, Description: Asking for the return value in code.
 - Tag name: English Multiple Choice, Description: Multiple choice questions using English.
 - Tag name: English Question Answering, Description: Question answering using English.
 - Tag name: Chinese Question Answering, Description: Question answering using Chinese.
 - Tag name: Summarization, Description: Summarization of a text.
 - Tag name: Character Identification, Description: Identify speakers in dialogue.
 - Tag name: Math Calculation, Description: Asking for results from mathematical calculations.
 - Tag name: Math Finding, Description: Find the integer in a list.
 - Tag name: Number Retrieval, Description: Finding a number from repeated text.
 - Tag name: PassKey Retrieval, Description:  Asking for what is a **Pass Key** in a text.
 - Tag name: Key Value Retrieval, Description: Finding the corresponding value from a JSON dictionary.

Query: {truncated_context}"
        OUTPUT_SUFFIX="taglist"
        ;;
    2)
        # OPEN_GENERATE template
        QUERY_TEMPLATE="You are a helpful assistant. For the user query below, generate tags in this order: 1) Domain, 2) Task Type, 3) Difficulty, 4) Language, 5) Topics (can be multiple). Explain each tag briefly. Output must be JSON: {\"tag\": str, \"explanation\": str}.

Query: {truncated_context}"
        OUTPUT_SUFFIX="open"
        ;;
    *)
        echo "Error: Invalid TEMPLATE_TYPE=$TEMPLATE_TYPE. Use 1 or 2."
        exit 1
        ;;
esac

echo "Using template type: $TEMPLATE_TYPE"

# ============================================================================
# Run Inference
# ============================================================================
python final_infer.py \
    --checkpoint_path "$MODEL_PATH" \
    --data_dir "$DATA_DIR" \
    --examples_json "$EXAMPLES_JSON" \
    --output_prefix "${OUTPUT_PREFIX}_${OUTPUT_SUFFIX}" \
    --batch_size 10 \
    --device auto \
    --query_template "$QUERY_TEMPLATE"

echo "Inference completed at $(date)"