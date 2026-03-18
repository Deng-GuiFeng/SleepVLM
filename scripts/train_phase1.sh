#!/usr/bin/env bash
# Phase 1: Waveform-Perceptual Pre-training (WPT)
#
# Train the model to perceive PSG waveform morphology by predicting
# per-second spectral and amplitude features from single epoch images.
#
# Data: MASS-SS2/4/5 (85 subjects)
# Vision encoder: UNFROZEN (LoRA applied to all nn.Linear except lm_head)
#
# Usage:
#   bash scripts/train_phase1.sh
#   EPOCHS=2 LR=1e-4 bash scripts/train_phase1.sh

set -euo pipefail

# Auto-detect available GPUs
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l)
    if [ "$GPU_COUNT" -gt 0 ]; then
        export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((GPU_COUNT - 1)))
    else
        echo "Error: No GPU detected"
        exit 1
    fi
fi
export TORCHDYNAMO_DISABLE=${TORCHDYNAMO_DISABLE:-1}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# Paths
MODEL_PATH=${MODEL_PATH:-"models/Qwen2.5-VL-3B-Instruct"}
TRAIN_FILE=${TRAIN_FILE:-"data/phase1_wpt/train.jsonl"}
IMAGE_ROOT=${IMAGE_ROOT:-"data"}
OUTPUT_DIR=${OUTPUT_DIR:-"outputs/phase1_wpt"}

# Training hyperparameters (paper: Table S7)
PER_DEVICE_TRAIN_BS=${PER_DEVICE_TRAIN_BS:-4}
GRAD_ACC=${GRAD_ACC:-1}
EPOCHS=${EPOCHS:-2}
LR=${LR:-1e-4}
WD=${WD:-0.1}
WARMUP=${WARMUP:-0.03}
SAVE_STEPS=${SAVE_STEPS:-1000}
LOG_STEPS=${LOG_STEPS:-10}
SEED=${SEED:-42}
NUM_WORKERS=${NUM_WORKERS:-4}

# LoRA config
LORA_R=${LORA_R:-16}
LORA_ALPHA=${LORA_ALPHA:-32}
LORA_DROPOUT=${LORA_DROPOUT:-0.05}

# Visual token control
MIN_PIXELS=${MIN_PIXELS:-3136}
MAX_PIXELS=${MAX_PIXELS:-1003520}

NUM_PROCS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | sed '/^$/d' | wc -l)
if [ "$NUM_PROCS" -le 1 ]; then
    ACCELERATE_ARGS="--num_processes 1"
else
    ACCELERATE_ARGS="--multi_gpu --num_processes $NUM_PROCS"
fi

echo "========================================"
echo "Phase 1: Waveform-Perceptual Pre-training"
echo "========================================"
echo "GPUs: $CUDA_VISIBLE_DEVICES ($NUM_PROCS processes)"
echo "Model: $MODEL_PATH"
echo "Train data: $TRAIN_FILE"
echo "Output: $OUTPUT_DIR"
echo "Epochs: $EPOCHS, LR: $LR, Batch: $PER_DEVICE_TRAIN_BS"
echo "Vision encoder: UNFROZEN"
echo "========================================"

mkdir -p "${OUTPUT_DIR}"

accelerate launch $ACCELERATE_ARGS \
    sleepvlm/training/train.py \
    --model_path "$MODEL_PATH" \
    --train_file "$TRAIN_FILE" \
    --image_root "$IMAGE_ROOT" \
    --output_dir "$OUTPUT_DIR" \
    --epochs $EPOCHS \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BS \
    --gradient_accumulation_steps $GRAD_ACC \
    --learning_rate $LR \
    --weight_decay $WD \
    --warmup_ratio $WARMUP \
    --save_steps $SAVE_STEPS \
    --log_every $LOG_STEPS \
    --seed $SEED \
    --num_workers $NUM_WORKERS \
    --min_pixels $MIN_PIXELS \
    --max_pixels $MAX_PIXELS \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --tensorboard
