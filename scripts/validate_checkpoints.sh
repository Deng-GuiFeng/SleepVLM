#!/usr/bin/env bash
# Validate all saved checkpoints on the MASS-SS3 validation set.
#
# After Phase 2 SFT training, this script evaluates every checkpoint
# against the 12-subject validation split to select the best one.
# Each checkpoint is:
#   1. Merged with the base model (LoRA adapter -> full model)
#   2. Served via vLLM (one server per GPU)
#   3. Evaluated on the validation set
#   4. Results saved; the checkpoint is skipped if already evaluated
#
# Usage:
#   bash scripts/validate_checkpoints.sh
#
#   # With custom paths:
#   CHECKPOINT_DIR=outputs/phase2_sft \
#   BASE_MODEL=outputs/phase1_wpt/merged \
#   bash scripts/validate_checkpoints.sh

set -euo pipefail

# --- Configuration ---
CHECKPOINT_DIR="${CHECKPOINT_DIR:-outputs/phase2_sft}"
OUTPUT_DIR="${OUTPUT_DIR:-${CHECKPOINT_DIR}/val}"
MERGE_DIR="${MERGE_DIR:-${CHECKPOINT_DIR}/lora_merge}"
BASE_MODEL="${BASE_MODEL:-models/Qwen2.5-VL-3B-Instruct}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-sleepvlm/prompts/phase2_sft_fine.md}"
SPLIT_JSON="${SPLIT_JSON:-split.json}"
IMG_DIR="${IMG_DIR:-data/MASS}"
ANNOTATION_DIR="${ANNOTATION_DIR:-MASS-EX/annotations}"

TEMPERATURE=${TEMPERATURE:-1e-6}
TOP_P=${TOP_P:-0.8}
MAX_TOKENS=${MAX_TOKENS:-1024}
NUM_THREADS=${NUM_THREADS:-64}
BASE_PORT=${BASE_PORT:-6002}
TP_SIZE=${TP_SIZE:-1}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-5000}

mkdir -p "$OUTPUT_DIR" "$MERGE_DIR"

GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ "$GPU_COUNT" -eq 0 ]; then
    echo "Error: No GPUs detected"
    exit 1
fi
echo "Detected $GPU_COUNT GPUs"

# --- Helper: wait for vLLM server ---
wait_for_ready() {
    local port="$1"
    local timeout="${SERVER_READY_TIMEOUT:-300}"
    local waited=0
    until curl --noproxy '*' -s "http://127.0.0.1:${port}/health" >/dev/null 2>&1; do
        sleep 5
        waited=$((waited + 5))
        if (( waited % 30 == 0 )); then
            echo "    ... waiting (${waited}s)"
        fi
        if (( waited >= timeout )); then
            echo "ERROR: Server on port ${port} not ready within ${timeout}s" >&2
            return 1
        fi
    done
}

# --- Collect checkpoints ---
CHECKPOINTS=()
for ckpt in "$CHECKPOINT_DIR"/checkpoint-* "$CHECKPOINT_DIR"/epoch-*; do
    [ -d "$ckpt" ] && CHECKPOINTS+=("$ckpt")
done

if [ ${#CHECKPOINTS[@]} -eq 0 ]; then
    echo "No checkpoints found in $CHECKPOINT_DIR"
    exit 1
fi
echo "Found ${#CHECKPOINTS[@]} checkpoints to validate"
echo ""

# --- Evaluate each checkpoint ---
for ckpt_path in "${CHECKPOINTS[@]}"; do
    ckpt_name=$(basename "$ckpt_path")
    result_file="${OUTPUT_DIR}/${ckpt_name}_results.jsonl"

    # Skip if already evaluated
    if [ -f "$result_file" ]; then
        echo "[SKIP] $ckpt_name (already evaluated)"
        continue
    fi

    echo "============================================================"
    echo "Evaluating: $ckpt_name"
    echo "============================================================"

    # Step 1: Merge LoRA adapter
    merged_path="${MERGE_DIR}/${ckpt_name}"
    if [ ! -d "$merged_path" ]; then
        echo "  Merging LoRA adapter..."
        python scripts/merge_lora.py \
            --base_model_path "$BASE_MODEL" \
            --adapter_path "$ckpt_path" \
            --output_path "$merged_path"
    else
        echo "  Merged model exists: $merged_path"
    fi

    # Step 2: Launch vLLM servers (in a subshell for clean process management)
    (
        # Cleanup lingering servers
        pkill -9 -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
        sleep 3

        SERVER_PIDS=()
        cleanup() {
            set +e
            for pid in "${SERVER_PIDS[@]}"; do
                kill "$pid" 2>/dev/null || true
            done
            wait "${SERVER_PIDS[@]}" 2>/dev/null || true
        }
        trap cleanup EXIT

        # Auto-detect quantized models
        QUANT_ARGS=""
        if [ -f "${merged_path}/config.json" ]; then
            QUANT_METHOD=$(python3 -c "
import json
with open('${merged_path}/config.json') as f:
    cfg = json.load(f)
print(cfg.get('quantization_config', {}).get('quant_method', ''))
" 2>/dev/null)
            if [ -n "$QUANT_METHOD" ]; then
                QUANT_ARGS="--quantization $QUANT_METHOD --dtype float16"
            fi
        fi

        successful_servers=0
        LOG_DIR="/tmp/vllm_val_logs_$$"
        mkdir -p "$LOG_DIR"

        for gpu_id in $(seq 0 $((GPU_COUNT - 1))); do
            PORT=$((BASE_PORT + successful_servers))
            export CUDA_VISIBLE_DEVICES=$gpu_id

            python -m vllm.entrypoints.openai.api_server \
                --model "$merged_path" \
                --host 127.0.0.1 \
                --port "$PORT" \
                --tensor-parallel-size "$TP_SIZE" \
                --served-model-name "$merged_path" \
                --max-model-len "$MAX_MODEL_LEN" \
                --gpu-memory-utilization 0.9 \
                --trust-remote-code \
                ${QUANT_ARGS} \
                --mm-processor-kwargs '{"min_pixels":3136,"max_pixels":1003520}' \
                > "$LOG_DIR/gpu${gpu_id}.log" 2>&1 &
            pid=$!

            if wait_for_ready "$PORT"; then
                echo "  [OK] GPU $gpu_id -> port $PORT"
                SERVER_PIDS+=("$pid")
                successful_servers=$((successful_servers + 1))
            else
                echo "  [WARN] GPU $gpu_id failed" >&2
                kill "$pid" 2>/dev/null || true
                wait "$pid" 2>/dev/null || true
                sleep 2
            fi
        done

        if (( successful_servers == 0 )); then
            echo "ERROR: No servers started for $ckpt_name" >&2
            exit 1
        fi

        # Step 3: Run inference on validation set
        python -m sleepvlm.inference.predict \
            --model_name "$merged_path" \
            --system_prompt "$SYSTEM_PROMPT" \
            --split_json "$SPLIT_JSON" \
            --split val \
            --img_dir "$IMG_DIR" \
            --annotation_dir "$ANNOTATION_DIR" \
            --output_dir "$OUTPUT_DIR" \
            --base_port "$BASE_PORT" \
            --num_gpus "$successful_servers" \
            --num_threads "$NUM_THREADS" \
            --temperature "$TEMPERATURE" \
            --top_p "$TOP_P" \
            --max_tokens "$MAX_TOKENS"

        # Rename output to include checkpoint name
        mv "${OUTPUT_DIR}/results.jsonl" "$result_file"
    )

    # Step 4: Evaluate
    echo "  Computing metrics..."
    PYTHONPATH=. python scripts/evaluate.py \
        --results_jsonl "$result_file" \
        --output_dir "$OUTPUT_DIR" 2>&1 | grep -E "Accuracy|Macro|Kappa" || true

    echo "  Done: $ckpt_name"
    echo ""
done

# --- Summary: find best checkpoint ---
echo "============================================================"
echo "Validation Summary"
echo "============================================================"
python3 -c "
import json, os, sys

output_dir = '$OUTPUT_DIR'
best_kappa = -1
best_ckpt = None

for fname in sorted(os.listdir(output_dir)):
    if not fname.endswith('_results.jsonl'):
        continue
    ckpt_name = fname.replace('_results.jsonl', '')
    fpath = os.path.join(output_dir, fname)
    results = [json.loads(l) for l in open(fpath)]
    valid = [r for r in results if r.get('pred', -1) >= 0]
    if not valid:
        continue
    correct = sum(1 for r in valid if r['pred'] == r['label'])
    acc = correct / len(valid)

    from sklearn.metrics import cohen_kappa_score
    y_true = [r['label'] for r in valid]
    y_pred = [r['pred'] for r in valid]
    kappa = cohen_kappa_score(y_true, y_pred)

    print(f'  {ckpt_name}: acc={acc:.4f}, kappa={kappa:.4f} ({len(valid)} epochs)')
    if kappa > best_kappa:
        best_kappa = kappa
        best_ckpt = ckpt_name

if best_ckpt:
    print(f'')
    print(f'Best checkpoint: {best_ckpt} (kappa={best_kappa:.4f})')
    print(f'Merge command:')
    print(f'  python scripts/merge_lora.py \\\\')
    print(f'      --adapter_path $CHECKPOINT_DIR/{best_ckpt} \\\\')
    print(f'      --output_path $CHECKPOINT_DIR/merged')
else:
    print('No valid results found.')
"
echo "============================================================"
