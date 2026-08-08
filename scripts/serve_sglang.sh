#!/usr/bin/env bash
# Project 1 - Start SGLang OpenAI-compatible server and wait until ready.
# Usage: ./serve_sglang.sh <model> <port> <extra_args>
set -euo pipefail

MODEL="${1:?model required}"
PORT="${2:?port required}"
EXTRA_ARGS="${3:-}"

echo "[sglang] launching model=$MODEL port=$PORT"
python -m sglang.launch_server \
    --model-path "$MODEL" \
    --port "$PORT" \
    --host 0.0.0.0 \
    $EXTRA_ARGS &

SGLANG_PID=$!
echo "[sglang] pid=$SGLANG_PID"

echo "[sglang] waiting for readiness..."
for i in $(seq 1 120); do
    if curl -s "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then
        echo "[sglang] ready after ${i}s"
        echo "$SGLANG_PID"
        exit 0
    fi
    sleep 2
done
echo "[sglang] FAILED to become ready in 240s"
kill "$SGLANG_PID" 2>/dev/null || true
exit 1
