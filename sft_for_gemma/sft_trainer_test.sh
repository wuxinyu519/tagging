#!/bin/bash
#PBS -N sft_gemma1b
#PBS -o sft_gemma1b.out
#PBS -e sft_gemma1b.err
#PBS -l walltime=24:00:00
#PBS -q poderoso
#PBS -l select=1:ncpus=8:ngpus=1:host=gpu008

cd $PBS_O_WORKDIR

module load cuda12.6/toolkit
eval "$(conda shell.bash hook)"
conda activate gemma

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

# MODEL_NAME="google/gemma-3-1b-it"
# MODEL_NAME="google/gemma-3-4b-it"
MODEL_NAME="Qwen/Qwen3-0.6B"
DATA_DIR="../data_creator/sft_data_4types"
OUTPUT_DIR="./March21_Qwen0.6b_6"
FREEZE_LAYERS=0
LIMIT_DATA=100000
EPOCHS=1
BATCH_SIZE=2
LR=1e-4
ACCUM_STEPS=8
MAX_LEN=1024
WARMUP_RATIO=0.05
WEIGHT_DECAY=0.01
SAVE_STRATEGY="epoch"

TEST_FILE="../data_creator/test_data_4types"
LIMIT_EVAL=100000
EVAL_BATCH=10

echo "Fine-tuning started at $(date)"
nvidia-smi

python train_test.py \
    --data_dir "$DATA_DIR" \
    --limit_data "$LIMIT_DATA" \
    --freeze_layers "$FREEZE_LAYERS" \
    --model_name "$MODEL_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --accum_steps "$ACCUM_STEPS" \
    --max_length "$MAX_LEN" \
    --warmup_ratio "$WARMUP_RATIO" \
    --weight_decay "$WEIGHT_DECAY" \
    --save_strategy "$SAVE_STRATEGY"

# echo "Fine-tuning completed at $(date)"
# echo "Starting evaluation..."

# python test_constrained.py \
#     --model_dir "$OUTPUT_DIR/final_model" \
#     --output_file "$OUTPUT_DIR/eval_outputs/predictions.jsonl" \
#     --test_file "$TEST_FILE" \
#     --use_constraints \
#     --limit_eval "$LIMIT_EVAL"

python test_test.py \
    --model_dir "$OUTPUT_DIR/final_model" \
    --output_file "$OUTPUT_DIR/eval_outputs/predictions.jsonl" \
    --batch_size "$EVAL_BATCH" \
    --test_file "$TEST_FILE" \
    --limit_eval "$LIMIT_EVAL"

echo "All processes completed at $(date)"