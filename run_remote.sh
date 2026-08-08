#!/usr/bin/env bash
# Benchmark against already-deployed engine pods (Option A).
#
# The engine pods run the official Docker images (vLLM / SGLang / TGI) on
# RunPod. Each exposes its OpenAI-compatible API at a public proxy URL:
#   https://<pod-id>-8000.proxy.runpod.net
#
# This script runs ONLY the load generator + aggregation on your laptop:
# install is not needed (just httpx), and no engine is installed here.
#
# Usage:
#   ENDPOINTS="vllm=https://abc-8000.proxy.runpod.net sglang=https://def-8000.proxy.runpod.net" \
#       ./run_remote.sh
#
#   # subset / single engine
#   ENDPOINTS="vllm=https://abc-8000.proxy.runpod.net" ./run_remote.sh vllm
#
#   # override sweep
#   MODEL=meta-llama/Meta-Llama-3-8B-Instruct CONCURRENCY="1 8 32" \
#       PROMPT_LENS="128 512" PROMPTS_PER_CELL=10 \
#       ENDPOINTS="vllm=https://abc-8000.proxy.runpod.net" ./run_remote.sh
#
# The `ENDPOINTS` value is a space-separated list of `engine=url` pairs.
set -euo pipefail

cd "$(dirname "$0")"

# --- Config (overridable via env) ---
MODEL="${MODEL:-meta-llama/Meta-Llama-3-8B-Instruct}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-128}"
CONCURRENCY="${CONCURRENCY:-1 2 4 8 16 32}"
PROMPT_LENS="${PROMPT_LENS:-32 128 512 2048}"
PROMPTS_PER_CELL="${PROMPTS_PER_CELL:-40}"
ENDPOINTS="${ENDPOINTS:-}"

# Engines to bench (from positional args) or all declared in ENDPOINTS
declare -A EP=()
for pair in $ENDPOINTS; do
    name="${pair%%=*}"
    url="${pair#*=}"
    EP["$name"]="$url"
done

if [ ${#EP[@]} -eq 0 ]; then
    echo "ERROR: set ENDPOINTS, e.g."
    echo '  ENDPOINTS="vllm=https://abc-8000.proxy.runpod.net sglang=https://def-8000.proxy.runpod.net" ./run_remote.sh'
    exit 1
fi

if [ $# -gt 0 ]; then
    ENGINES=("$@")
else
    ENGINES=("${!EP[@]}")
fi

mkdir -p results

echo "============================================"
echo " inference-bench — remote engine benchmark"
echo "============================================"
echo "Model:            $MODEL"
echo "Endpoints:        ${ENDPOINTS}"
echo "Engines:          ${ENGINES[*]}"
echo "Concurrency:      $CONCURRENCY"
echo "Prompt lens:      $PROMPT_LENS"
echo "Output tokens:    $OUTPUT_TOKENS"
echo "Prompts/cell:     $PROMPTS_PER_CELL"
echo "============================================"
echo

# --- Dependency check (httpx is the only client dep) ---
python -c "import httpx" 2>/dev/null || {
    echo "[setup] installing httpx..."
    pip install -q httpx
}

for engine in "${ENGINES[@]}"; do
    url="${EP[$engine]:-}"
    if [ -z "$url" ]; then
        echo "[$engine] SKIPPED — no URL given for $engine"
        continue
    fi

    echo "=========================================="
    echo " [$engine] benchmarking $url"
    echo "=========================================="
    python scripts/bench_client.py \
        --endpoint "$url" \
        --engine "$engine" \
        --model "$MODEL" \
        --concurrency $CONCURRENCY \
        --prompt-len $PROMPT_LENS \
        --output-tokens "$OUTPUT_TOKENS" \
        --prompts-per-cell "$PROMPTS_PER_CELL" \
        --out "results/${engine}_raw.jsonl"
    echo
done

echo "============================================"
echo " Aggregating results"
echo "============================================"
python scripts/collect_results.py --in results/ --out results/

echo
echo "============================================"
echo " DONE. Results in: results/"
echo "============================================"
echo "  - results/summary.csv         (one row per cell)"
echo "  - results/summary.json        (Grafana-ready)"
echo "  - results/summary_table.txt   (human-readable)"
echo "  - results/<engine>_raw.jsonl  (per-request raw data)"
