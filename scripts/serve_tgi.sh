#!/usr/bin/env bash
# Project 1 - Start TGI server and wait until ready.
# TGI exposes an OpenAI-compatible /v1/completions route via text-generation-router.
# Usage: ./serve_tgi.sh <model> <port> <extra_args>
set -euo pipefail

MODEL="${1:?model required}"
PORT="${2:?port required}"
EXTRA_ARGS="${3:-}"

echo "[tgi] launching model=$MODEL port=$PORT"
# TGI uses --model-id, not --model
text-generation-launcher \
    --model-id "$MODEL" \
    --port "$PORT" \
    --hostname 0.0.0.0 \
    $EXTRA_ARGS &

TGI_PID=$!
echo "[tgi] pid=$TGI_PID"

echo "[tgi] waiting for readiness..."
# TGI health endpoint is /health
for i in $(seq 1 180); do
    if curl -s "http://localhost:$PORT/health" >/dev/null 2>&1; then
        echo "[tgi] ready after ${i}s"
        echo "$TGI_PID"
        exit 0
    fi
    sleep 2
done
echo "[tgi] FAILED to become ready in 360s"
kill "$TGI_PID" 2>/dev/null || true
exit 1
