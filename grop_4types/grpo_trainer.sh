#!/bin/bash
#PBS -N grpo_gemma1b
#PBS -o grpo_gemma1b.out
#PBS -e grpo_gemma1b.err
#PBS -l walltime=24:00:00
#PBS -q instructional
#PBS -l select=1:ncpus=8:ngpus=1:host=gpu011

cd $PBS_O_WORKDIR

# ============ ENV SETUP ============
module load cuda12.6/toolkit
eval "$(conda shell.bash hook)"
conda activate gemma

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

echo "GRPO Training started at $(date)"
nvidia-smi

## Path setup - where I put my data and model paths
DPO_DATA_DIR="../data_creator/dpo_data_final_filtered"
SFT_MODEL_DIR="../sft/finetuned_gemma1b_tagging_4types_best/final_model"
REWARD_MODEL_PATH="./reward_model_output/final_model"
OUTPUT_DIR="./grpo_gemma1b_output"

## GRPO hyperparameters - my training settings
EPOCHS=1
BATCH_SIZE=1
LR=1e-6
BETA=0.04
NUM_SAMPLES=64
MAX_LENGTH=1024
MAX_PROMPT_LENGTH=512
WEIGHT_DECAY=0.01
FREEZE_LAYERS=18
MAX_GRAD_NORM=1.0

## Test settings - evaluation config
TEST_FILE="../data_creator/test_data_4types"
EVAL_OUTPUT="$OUTPUT_DIR/eval_outputs/predictions.jsonl"
EVAL_BATCH=10
LIMIT_EVAL=20000

# ==========================================
# STEP 1: GRPO training - run the training
# ==========================================
echo ""
echo "========== Step 1: GRPO training =========="

python tune_with_grop.py \
    --dpo_data_dir "$DPO_DATA_DIR" \
    --sft_model_dir "$SFT_MODEL_DIR" \
    --reward_model_path "$REWARD_MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --beta "$BETA" \
    --num_samples "$NUM_SAMPLES" \
    --max_length "$MAX_LENGTH" \
    --max_prompt_length "$MAX_PROMPT_LENGTH" \
    --weight_decay "$WEIGHT_DECAY" \
    --freeze_layers "$FREEZE_LAYERS" \
    --max_grad_norm "$MAX_GRAD_NORM"

echo ""
echo "GRPO training completed at $(date)"

# ==========================================
# STEP 2: Evaluate GRPO model (on the test set) - generate predictions
# ==========================================
echo ""
echo "========== Step 2: Evaluating GRPO model =========="

# Create eval output directory
mkdir -p "$OUTPUT_DIR/eval_outputs"

python ../sft/test.py \
    --model_dir "$OUTPUT_DIR/final_model" \
    --test_file "$TEST_FILE" \
    --output_file "$EVAL_OUTPUT" \
    --batch_size "$EVAL_BATCH" \
    --limit_eval "$LIMIT_EVAL"

# ==========================================
# done
# ==========================================
echo ""
echo "=========================================="
echo "All processes completed at $(date)"
echo "=========================================="
echo "Results saved in:"
echo "  - $OUTPUT_DIR/final_model/"
echo "  - $EVAL_OUTPUT"