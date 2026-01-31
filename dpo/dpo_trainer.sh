#!/bin/bash
#PBS -N dpo_gemma1b
#PBS -o dpo_gemma1b.out
#PBS -e dpo_gemma1b.err
#PBS -l walltime=24:00:00
#PBS -q poderoso
#PBS -l select=1:ncpus=8:ngpus=1:host=gpu008

cd $PBS_O_WORKDIR

# ===== ENV SETUP =====
module load cuda12.6/toolkit
eval "$(conda shell.bash hook)"
conda activate gemma

export CUDA_VISIBLE_DEVICES=3
export TOKENIZERS_PARALLELISM=false

echo "DPO Pipeline started at $(date)"
nvidia-smi


# paths
DPO_DATA_DIR="../data_creator/dpo_data_final_filtered"
SFT_MODEL_DIR="../sft/finetuned_gemma1b_tagging_4types_best/final_model"
OUTPUT_DIR="./dpo_gemma1b_filtered_final"

# DPO hyperparams
EPOCHS=1
BATCH_SIZE=4
LR=1e-5
ACCUM_STEPS=4
BETA=0.1
MAX_LENGTH=1024
MAX_PROMPT_LENGTH=512
WARMUP_RATIO=0.05
WEIGHT_DECAY=0.01
FREEZE_LAYERS=0
SAVE_STRATEGY="epoch"

# eval setup
TEST_FILE="../data_creator/test_data_4types"
EVAL_OUTPUT="$OUTPUT_DIR/eval_outputs/predictions.jsonl"
EVAL_BATCH=10
LIMIT_EVAL=2000

# ==========================================
# STEP 1: DPO
# ==========================================
echo ""
echo "========== Step 1: DPO training =========="

python tune_w_rlhf.py \
    --dpo_data_dir "$DPO_DATA_DIR" \
    --sft_model_dir "$SFT_MODEL_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --accum_steps "$ACCUM_STEPS" \
    --beta "$BETA" \
    --max_length "$MAX_LENGTH" \
    --max_prompt_length "$MAX_PROMPT_LENGTH" \
    --warmup_ratio "$WARMUP_RATIO" \
    --weight_decay "$WEIGHT_DECAY" \
    --freeze_layers "$FREEZE_LAYERS"

echo ""
echo "DPO training completed at $(date)"

# ==========================================
# STEP 2: eval DPO model on test set
# ==========================================
echo ""
echo "========== Step 2: Evaluating DPO model =========="

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
echo "  - $DPO_DATA_DIR/"
echo "  - $OUTPUT_DIR/final_model/"
echo "  - $EVAL_OUTPUT"
