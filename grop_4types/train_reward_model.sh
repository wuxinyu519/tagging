#!/bin/bash
#PBS -N reward_model
#PBS -o reward_model.out
#PBS -e reward_model.err
#PBS -l walltime=12:00:00
#PBS -q instructional
#PBS -l select=1:ncpus=8:ngpus=1:host=gpu011

cd $PBS_O_WORKDIR

# ============ ENV SETUP ============
module load cuda12.6/toolkit
eval "$(conda shell.bash hook)"
conda activate gemma

export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=false

echo "Reward Model Training started at $(date)"
nvidia-smi

## 路径配置
DPO_DATA_DIR="../data_creator/dpo_data_final"
BASE_MODEL_DIR="../sft/finetuned_gemma1b_tagging_4types_best/final_model"
OUTPUT_DIR="./reward_model_output"

## 训练超参数
EPOCHS=3
BATCH_SIZE=4
LR=1e-5
MAX_LENGTH=1024
WARMUP_RATIO=0.05
WEIGHT_DECAY=0.01

# ==========================================
# Train Reward Model
# ==========================================
echo ""
echo "========== Training Reward Model =========="

python train_reward_model.py \
    --dpo_data_dir "$DPO_DATA_DIR" \
    --base_model_dir "$BASE_MODEL_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --max_length "$MAX_LENGTH" \
    --warmup_ratio "$WARMUP_RATIO" \
    --weight_decay "$WEIGHT_DECAY"

echo ""
echo "Reward Model training completed at $(date)"
echo "=========================================="
echo "Model saved in: $OUTPUT_DIR/final_model/"