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
PROMPTS_PER_CELL="${PROMPTS_PER_CELL:-40}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:---gpu-memory-utilization 0.9 --max-model-len 4096}"
SGLANG_EXTRA_ARGS="${SGLANG_EXTRA_ARGS:---mem-fraction-static 0.9 --context-length 4096}"
TGI_EXTRA_ARGS="${TGI_EXTRA_ARGS:---max-total-tokens 4096 --max-batch-size 256}"

# Engine install pins. Bump deliberately; results are not comparable across
# engine versions. Each engine gets its OWN venv (see bench_engine) so pip
# never mutates the pod template's preinstalled torch.
#
# Confirmed-working stack on RunPod (RTX 4090, driver 570/580, CUDA 13.0,
# Python 3.12, template torch 2.8.0+cu128):
#   - venv with --system-site-packages reuses the template torch (no torch
#     download) and it sees the GPU (torch.cuda.is_available() == True).
#   - vLLM 0.11.0 + transformers==4.55.2: vLLM 0.11.0 is NOT compatible with
#     transformers 5.x (LlamaTokenizer.all_special_tokens_extended error), so
#     transformers is pinned below 5.
VLLM_PIN="${VLLM_PIN:-vllm==0.11.0}"
VLLM_TRANSFORMERS_PIN="${VLLM_TRANSFORMERS_PIN:-transformers==4.55.2 tokenizers>=0.21.1}"
SGLANG_PIN="${SGLANG_PIN:-sglang[all]}"
TGI_PIN="${TGI_PIN:-text-generation-inference}"

# --- Pod preflight (fails fast, before burning GPU hours) ---
echo "=== Pod preflight ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || echo "WARN: nvidia-smi not found"
python3 -V
python3 -c "import torch; print('template torch', torch.__version__, '| cuda build', torch.version.cuda, '| visible', torch.cuda.is_available())" 2>/dev/null || echo "note: template has no torch (fine — engines install their own in venvs)"
ls -la "$HOME/.cache/huggingface/token" >/dev/null 2>&1 && echo "HF token present" || echo "WARN: no HF token at ~/.cache/huggingface/token — run: huggingface-cli login"
echo

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

# --- Dependency check (client dep; installed into the first venv as needed) ---
# httpx is installed into each engine venv during setup, so nothing is needed
# at the system level.

bench_engine() {
    local engine="$1"
    local venv_dir="$PROJECT_DIR/venvs/$engine"
    local venv_py="$venv_dir/bin/python"
    local venv_bin="$venv_dir/bin"
    local venv_marker="$venv_dir/.ready"

    echo "=========================================="
    echo " [$engine] phase 1: install (isolated venv)"
    echo "=========================================="
    # Isolated venv per engine with --system-site-packages: pip never touches
    # the template's torch (it is reused from the system site-packages), and
    # engines cannot conflict with each other.
    if [ ! -d "$venv_dir" ]; then
        python3 -m venv --system-site-packages "$venv_dir"
    fi

    if [ ! -f "$venv_marker" ]; then
        "$venv_py" -m pip install --upgrade pip
        case "$engine" in
            vllm)
                # Template torch (2.8.0+cu128) is reused via system-site-packages.
                # transformers must stay < 5.x — vLLM 0.11.0 breaks with 5.x
                # (LlamaTokenizer.all_special_tokens_extended AttributeError).
                # hf_transfer: RunPod templates set HF_HUB_ENABLE_HF_TRANSFER=1
                # but don't ship the package — model download fails without it.
                "$venv_py" -m pip install "$VLLM_PIN" $VLLM_TRANSFORMERS_PIN httpx hf_transfer
                ;;
            sglang)
                # FlashInfer JIT-compiles kernels at server startup via ninja;
                # without it, launch fails with 'No such file or directory: ninja'.
                "$venv_py" -m pip install "$SGLANG_PIN" httpx ninja
                ;;
            tgi)
                # TGI ships as a bundled Rust+Python package; install into venv.
                "$venv_py" -m pip install "$TGI_PIN" httpx
                ;;
            *)
                echo "Unknown engine: $engine"
                return 1
                ;;
        esac
        touch "$venv_marker"
    else
        echo "[$engine] venv already provisioned — skipping install"
    fi

    # Preflight inside the venv: verify torch sees the GPU and prints versions.
    echo "[$engine] venv preflight:"
    "$venv_py" -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('  torch', torch.__version__, '| cuda build', torch.version.cuda, '| gpu', torch.cuda.get_device_name(0))" \
        || { echo "[$engine] PREFLIGHT FAILED — aborting this engine"; return 1; }

    echo "=========================================="
    echo " [$engine] phase 2: serve"
    echo "=========================================="
    local serve_script="scripts/serve_${engine}.sh"
    local extra_args_var="${engine^^}_EXTRA_ARGS"
    local extra_args="${!extra_args_var}"

    # Prepend the engine venv's bin dir so `vllm`, `python`, and the launcher
    # resolve inside the venv.
    local old_path="$PATH"
    export PATH="$venv_bin:$PATH"

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
    "$venv_py" scripts/bench_client.py \
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
    export PATH="$old_path"
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
# Use the first available venv python (all have httpx + stdlib needed).
if [ -x "$PROJECT_DIR/venvs/vllm/bin/python" ]; then
    "$PROJECT_DIR/venvs/vllm/bin/python" scripts/collect_results.py --in results/ --out results/
else
    python3 scripts/collect_results.py --in results/ --out results/
fi

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
