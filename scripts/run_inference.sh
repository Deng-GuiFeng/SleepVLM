#!/usr/bin/env bash
# Run inference on the test set using vLLM.
#
# This script:
#   1. Launches vLLM servers (one per available GPU)
#   2. Runs parallel inference on all test subjects
#   3. Computes and saves evaluation metrics
#
# Usage:
#   bash scripts/run_inference.sh
#   MODEL_PATH=outputs/phase2_sft/lora_merge/best bash scripts/run_inference.sh

set -euo pipefail

# Configuration
MODEL_PATH="${MODEL_PATH:-"outputs/phase2_sft/lora_merge/best"}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-"sleepvlm/prompts/phase2_sft_fine.md"}"
SPLIT_JSON="${SPLIT_JSON:-"split.json"}"
IMG_DIR="${IMG_DIR:-"data/MASS"}"
OUTPUT_DIR="${OUTPUT_DIR:-"outputs/eval"}"
ANNOTATION_DIR="${ANNOTATION_DIR:-"MASS-EX/annotations"}"

# Inference parameters
TEMPERATURE=${TEMPERATURE:-1e-6}
TOP_P=${TOP_P:-0.8}
MAX_TOKENS=${MAX_TOKENS:-1024}
NUM_THREADS=${NUM_THREADS:-64}

# vLLM server config
BASE_PORT=${BASE_PORT:-6002}
TP_SIZE=${TP_SIZE:-1}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-5000}

mkdir -p "$OUTPUT_DIR"

# --- Auto-detect GPUs ---
GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ "$GPU_COUNT" -eq 0 ]; then
    echo "Error: No GPUs detected"
    exit 1
fi

# --- Cleanup any lingering vLLM processes ---
cleanup_before_start() {
    echo "[CLEANUP] Checking for lingering processes..."
    for offset in $(seq 0 $((GPU_COUNT - 1))); do
        local port=$((BASE_PORT + offset))
        local pid=$(lsof -t -i :$port 2>/dev/null || true)
        if [[ -n "$pid" ]]; then
            echo "[CLEANUP] Port $port occupied by PID $pid, killing..."
            kill -9 $pid 2>/dev/null || true
        fi
    done
    pkill -9 -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
    sleep 2
}
cleanup_before_start

# --- Launch vLLM servers ---
SERVER_PIDS=()
SERVER_PORTS=()

cleanup() {
    set +e
    if [[ ${#SERVER_PIDS[@]} -gt 0 ]]; then
        echo "Shutting down vLLM servers (PIDs: ${SERVER_PIDS[*]})..."
        kill "${SERVER_PIDS[@]}" 2>/dev/null || true
        wait "${SERVER_PIDS[@]}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

wait_for_ready() {
    local port="$1"
    local timeout="${SERVER_READY_TIMEOUT:-300}"
    local waited=0
    echo "  Waiting for server on port ${port} (timeout: ${timeout}s)..."
    until curl -s "http://127.0.0.1:${port}/health" >/dev/null 2>&1; do
        sleep 5
        waited=$((waited + 5))
        if (( waited % 30 == 0 )); then
            echo "  ... still waiting (${waited}s elapsed)"
        fi
        if (( waited >= timeout )); then
            echo "ERROR: Server on port ${port} not ready within ${timeout}s" >&2
            return 1
        fi
    done
    echo "  Server on port ${port} ready after ${waited}s."
}

LOG_DIR="/tmp/vllm_sleepvlm_logs_$$"
mkdir -p "$LOG_DIR"
echo "Server logs: $LOG_DIR"

successful_servers=0
for gpu_id in $(seq 0 $((GPU_COUNT - 1))); do
    PORT=$((BASE_PORT + successful_servers))
    echo "[TRY] GPU ${gpu_id} -> port ${PORT}"
    CUDA_VISIBLE_DEVICES=$gpu_id python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_PATH" \
        --host 127.0.0.1 \
        --port "$PORT" \
        --tensor-parallel-size "$TP_SIZE" \
        --served-model-name "$MODEL_PATH" \
        --max-model-len "$MAX_MODEL_LEN" \
        --gpu-memory-utilization 0.9 \
        --trust-remote-code \
        --mm-processor-kwargs '{"min_pixels":3136,"max_pixels":1003520}' \
        > "$LOG_DIR/gpu${gpu_id}_port${PORT}.log" 2>&1 &
    pid=$!

    if wait_for_ready "$PORT"; then
        echo "[OK] Server ready: GPU ${gpu_id} at port ${PORT} (PID ${pid})"
        SERVER_PIDS+=("$pid")
        SERVER_PORTS+=("$PORT")
        successful_servers=$((successful_servers + 1))
    else
        echo "[WARN] GPU ${gpu_id} failed. See: $LOG_DIR/gpu${gpu_id}_port${PORT}.log" >&2
        kill "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
        sleep 2
    fi
done

if (( successful_servers == 0 )); then
    echo "ERROR: No vLLM servers started. Check GPU availability." >&2
    exit 1
fi
echo "Launched ${successful_servers} server(s)"

# --- Run inference ---
echo ""
echo "========================================"
echo "Running Inference"
echo "========================================"
echo "Model: $MODEL_PATH"
echo "Output: $OUTPUT_DIR"
echo "Temperature: $TEMPERATURE, Top-p: $TOP_P"
echo "========================================"

python -m sleepvlm.inference.predict \
    --model_name "$MODEL_PATH" \
    --system_prompt "$SYSTEM_PROMPT" \
    --split_json "$SPLIT_JSON" \
    --split "test" \
    --img_dir "$IMG_DIR" \
    --annotation_dir "$ANNOTATION_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --base_port "$BASE_PORT" \
    --num_gpus "$successful_servers" \
    --num_threads "$NUM_THREADS" \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    --max_tokens "$MAX_TOKENS"

echo ""
echo "========================================"
echo "Inference Complete"
echo "========================================"
echo "Results: ${OUTPUT_DIR}/"
echo "========================================"
