#!/usr/bin/env bash
# Project 1 - Start vLLM OpenAI-compatible server and wait until ready.
# Usage: ./serve_vllm.sh <model> <port> <extra_args>
set -euo pipefail

MODEL="${1:?model required}"
PORT="${2:?port required}"
EXTRA_ARGS="${3:-}"

# Preflight: verify torch can see the GPU BEFORE starting the server.
# This fails fast (2s) instead of a 30s stack trace if versions are mismatched.
echo "[vllm] preflight check..."
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(f'[vllm] torch {torch.__version__} | cuda {torch.version.cuda} | {torch.cuda.get_device_name(0)}')" || {
    echo "[vllm] PREFLIGHT FAILED - torch cannot see GPU. Aborting."
    exit 1
}

echo "[vllm] launching model=$MODEL port=$PORT"
vllm serve "$MODEL" \
    --port "$PORT" \
    --host 0.0.0.0 \
    $EXTRA_ARGS &

VLLM_PID=$!
echo "[vllm] pid=$VLLM_PID"

# Wait for /v1/models to respond
echo "[vllm] waiting for readiness..."
for i in $(seq 1 120); do
    if curl -s "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then
        echo "[vllm] ready after ${i}s"
        echo "$VLLM_PID"
        exit 0
    fi
    sleep 2
done
echo "[vllm] FAILED to become ready in 240s"
kill "$VLLM_PID" 2>/dev/null || true
exit 1
