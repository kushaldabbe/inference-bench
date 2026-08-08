#!/usr/bin/env bash
# Project 1 - Full benchmark orchestrator.
#
# Installs one engine at a time, serves the model, runs the bench client,
# collects results, tears down the server, then moves to the next engine.
# Run this ON the RunPod pod.
#
# Usage:
#   ./run_all.sh                      # bench all 3 engines with default config
#   ./run_all.sh vllm                 # bench only vllm
#   ./run_all.sh vllm sglang          # bench vllm + sglang
#   MODEL=... PORT=8000 ./run_all.sh  # override config via env
set -euo pipefail

cd "$(dirname "$0")"
PROJECT_DIR="$(pwd)"

# --- Config (overridable via env) ---
MODEL="${MODEL:-meta-llama/Meta-Llama-3-8B-Instruct}"
PORT="${PORT:-8000}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-128}"
CONCURRENCY="${CONCURRENCY:-1 2 4 8 16 32}"
PROMPT_LENS="${PROMPT_LENS:-32 128 512 2048}"
PROMPTS_PER_CELL="${PROMPTS_PER_CELL:-20}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:---gpu-memory-utilization 0.9 --max-model-len 4096}"
SGLANG_EXTRA_ARGS="${SGLANG_EXTRA_ARGS:---mem-fraction-static 0.9 --context-length 4096}"
TGI_EXTRA_ARGS="${TGI_EXTRA_ARGS:---max-total-tokens 4096 --max-batch-size 256}"

# Engines to bench (from positional args, or all 3)
if [ $# -gt 0 ]; then
    ENGINES=("$@")
else
    ENGINES=(vllm sglang tgi)
fi

mkdir -p results

echo "============================================"
echo " Project 1 - LLM Inference Engine Benchmark"
echo "============================================"
echo "Project dir:      $PROJECT_DIR"
echo "Model:            $MODEL"
echo "Port:             $PORT"
echo "Engines:          ${ENGINES[*]}"
echo "Concurrency:      $CONCURRENCY"
echo "Prompt lens:      $PROMPT_LENS"
echo "Output tokens:    $OUTPUT_TOKENS"
echo "Prompts/cell:     $PROMPTS_PER_CELL"
echo "============================================"
echo

# --- Dependency check ---
python -c "import httpx" 2>/dev/null || {
    echo "[setup] installing httpx..."
    pip install -q httpx
}

bench_engine() {
    local engine="$1"
    echo "=========================================="
    echo " [$engine] phase 1: install"
    echo "=========================================="
    case "$engine" in
        vllm)
            # Pin: vLLM 0.8.5.post1 requires torch==2.6.0 (cu126 wheel).
            # RunPod driver 570 supports CUDA <=12.8, so cu126 works.
            # Newer vLLM needs torch 2.7+ (cu128/cu130) which driver 570 can't run.
            pip install -q "vllm==0.8.5.post1"
            ;;
        sglang)
            pip install -q "sglang[all]"
            ;;
        tgi)
            # TGI ships as a bundled Rust+Python package
            pip install -q text-generation-inference
            ;;
        *)
            echo "Unknown engine: $engine"
            return 1
            ;;
    esac

    echo "=========================================="
    echo " [$engine] phase 2: serve"
    echo "=========================================="
    local serve_script="scripts/serve_${engine}.sh"
    local extra_args_var="${engine^^}_EXTRA_ARGS"
    local extra_args="${!extra_args_var}"

    # Start server; script prints PID on last line
    local serve_log
    serve_log=$(bash "$serve_script" "$MODEL" "$PORT" "$extra_args")
    local server_pid
    server_pid=$(echo "$serve_log" | tail -1)
    echo "[$engine] server pid=$server_pid"

    # Cleanup trap - ensure server dies even on error
    trap "kill $server_pid 2>/dev/null || true" EXIT

    echo "=========================================="
    echo " [$engine] phase 3: benchmark"
    echo "=========================================="
    python scripts/bench_client.py \
        --endpoint "http://localhost:$PORT" \
        --engine "$engine" \
        --model "$MODEL" \
        --concurrency $CONCURRENCY \
        --prompt-len $PROMPT_LENS \
        --output-tokens "$OUTPUT_TOKENS" \
        --prompts-per-cell "$PROMPTS_PER_CELL" \
        --out "results/${engine}_raw.jsonl"

    echo "=========================================="
    echo " [$engine] phase 4: teardown"
    echo "=========================================="
    kill "$server_pid" 2>/dev/null || true
    trap - EXIT
    # Wait for GPU memory to free
    echo "[$engine] waiting 15s for GPU memory to free..."
    sleep 15
    echo
}

for engine in "${ENGINES[@]}"; do
    bench_engine "$engine" || {
        echo "[$engine] FAILED - continuing to next engine"
    }
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
echo
echo "Next: import results/summary.json into Grafana, or view summary_table.txt"
