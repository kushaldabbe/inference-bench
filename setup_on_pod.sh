#!/usr/bin/env bash
# =============================================================================
# Project 1 - Pod Bootstrap Script
# =============================================================================
# Upload this ONE file to the pod (via Jupyter file browser), then run:
#   bash setup_on_pod.sh
# It recreates the full inference-bench/ directory with all scripts.
#
# CAVEAT: the canonical, lint-clean source of each script is the loose file in
# scripts/ + run_all.sh at the repo root. The copies embedded below are a
# convenience snapshot for single-file upload. If you edit a script, regenerate
# this file (or just upload the whole repo folder instead - see README Option A).
# =============================================================================
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/workspace/inference-bench}"
mkdir -p "$PROJECT_DIR/scripts" "$PROJECT_DIR/results"

cd "$PROJECT_DIR"

echo "=== Creating project files in $PROJECT_DIR ==="

# --- bench_client.py ---
cat > scripts/bench_client.py << 'BENCH_CLIENT_EOF'
"""Project 1 - LLM Inference Engine Benchmark.

Async streaming load generator. Measures TTFT, ITL, and throughput for an
OpenAI-compatible inference endpoint (vLLM, SGLang, or TGI in OpenAI-compat mode).
"""
import argparse
import asyncio
import json
import random
import time
from pathlib import Path

import httpx

SEED = 42


def make_prompts(n: int, target_tokens: int) -> list[str]:
    rng = random.Random(SEED + target_tokens)
    target_chars = int(target_tokens * 1.3)
    words = (
        "the system must process incoming requests under variable load "
        "conditions while maintaining low latency and high throughput "
        "across different batch sizes and sequence lengths "
    ).split()
    prompts = []
    for _ in range(n):
        buf = []
        cur = 0
        while cur < target_chars:
            w = rng.choice(words)
            buf.append(w)
            cur += len(w) + 1
        prompt = " ".join(buf) + "\n\nSummarize the above in one sentence."
        prompts.append(prompt)
    return prompts


async def stream_one(
    client: httpx.AsyncClient,
    endpoint: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    req_id: int,
) -> dict:
    t_start = time.perf_counter()
    ttft = None
    token_times: list[float] = []
    n_tokens = 0
    error = None

    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }

    try:
        async with client.stream(
            "POST",
            f"{endpoint}/v1/completions",
            json=payload,
            timeout=httpx.Timeout(connect=10.0, read=300.0, write=10.0, pool=10.0),
        ) as resp:
            if resp.status_code != 200:
                error = f"http {resp.status_code}: {await resp.aread()}"
            else:
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    data = line[len("data: "):]
                    if data.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    choices = chunk.get("choices", [])
                    if not choices:
                        continue
                    text = choices[0].get("text", "")
                    if text:
                        t_now = time.perf_counter() - t_start
                        if ttft is None:
                            ttft = t_now
                        token_times.append(t_now)
                        n_tokens += 1
    except Exception as e:
        error = f"{type(e).__name__}: {e}"

    t_end = time.perf_counter()
    e2e = t_end - t_start

    itls = []
    if len(token_times) >= 2:
        itls = [token_times[i] - token_times[i - 1] for i in range(1, len(token_times))]

    return {
        "req_id": req_id,
        "engine": None,
        "concurrency": None,
        "prompt_len_target": None,
        "ttft_s": ttft,
        "itls_s": itls,
        "e2e_s": e2e,
        "n_output_tokens": n_tokens,
        "error": error,
    }


async def run_cell(
    endpoint: str,
    model: str,
    engine: str,
    concurrency: int,
    prompt_len: int,
    output_tokens: int,
    temperature: float,
    prompts_per_cell: int,
    out_path: Path,
):
    prompts = make_prompts(prompts_per_cell, prompt_len)
    limits = httpx.Limits(max_connections=concurrency * 2, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(limits=limits) as client:
        tasks = [
            stream_one(client, endpoint, model, p, output_tokens, temperature, i)
            for i, p in enumerate(prompts)
        ]
        results = await asyncio.gather(*tasks)

    with out_path.open("a") as f:
        for r in results:
            r["engine"] = engine
            r["concurrency"] = concurrency
            r["prompt_len_target"] = prompt_len
            r["temperature"] = temperature
            r["max_tokens"] = output_tokens
            f.write(json.dumps(r) + "\n")
            f.flush()

    ok = [r for r in results if r["error"] is None]
    fail = len(results) - len(ok)
    if ok:
        toks = sum(r["n_output_tokens"] for r in ok)
        wall = max(r["e2e_s"] for r in ok)
        throughput = toks / wall if wall > 0 else 0
        ttfts = [r["ttft_s"] for r in ok if r["ttft_s"] is not None]
        ttft_mean = sum(ttfts) / len(ttfts) if ttfts else 0
        print(
            f"  [{engine}] c={concurrency:<3} plen={prompt_len:<5} "
            f"ok={len(ok)}/{len(results)} "
            f"throughput={throughput:.1f} tok/s "
            f"ttft_mean={ttft_mean*1000:.0f}ms "
            f"e2e_mean={sum(r['e2e_s'] for r in ok)/len(ok)*1000:.0f}ms"
            + (f"  ERRORS={fail}" if fail else "")
        )
    else:
        print(f"  [{engine}] c={concurrency:<3} plen={prompt_len:<5} ALL FAILED")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://localhost:8000")
    ap.add_argument("--engine", required=True, choices=["vllm", "sglang", "tgi"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--prompt-len", type=int, nargs="+", default=[32, 128, 512, 2048])
    ap.add_argument("--output-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--prompts-per-cell", type=int, default=20)
    ap.add_argument("--out", default="results/raw.jsonl")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.touch()

    print(f"=== Benchmarking {args.engine} @ {args.endpoint} ===")
    print(f"Model: {args.model}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Prompt lens: {args.prompt_len}")
    print(f"Output tokens: {args.output_tokens}")
    print(f"Prompts per cell: {args.prompts_per_cell}")
    print()

    for plen in args.prompt_len:
        for c in args.concurrency:
            await run_cell(
                endpoint=args.endpoint,
                model=args.model,
                engine=args.engine,
                concurrency=c,
                prompt_len=plen,
                output_tokens=args.output_tokens,
                temperature=args.temperature,
                prompts_per_cell=args.prompts_per_cell,
                out_path=out_path,
            )
    print(f"\nRaw results -> {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
BENCH_CLIENT_EOF

# --- collect_results.py ---
cat > scripts/collect_results.py << 'COLLECT_EOF'
"""Project 1 - Aggregate raw JSONL benchmark results into CSV + summary JSON."""
import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def percentile(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = (len(xs) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if f == c:
        return xs[f]
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="indir", default="results")
    ap.add_argument("--out", dest="outdir", default="results")
    args = ap.parse_args()

    indir = Path(args.indir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    raw_files = sorted(indir.glob("*_raw.jsonl"))
    if not raw_files:
        print(f"No *_raw.jsonl files in {indir}")
        return

    all_rows = []
    for rf in raw_files:
        all_rows.extend(load_jsonl(rf))
    print(f"Loaded {len(all_rows)} raw requests from {len(raw_files)} files")

    cells = defaultdict(list)
    for r in all_rows:
        key = (r["engine"], r["concurrency"], r["prompt_len_target"])
        cells[key].append(r)

    summary_rows = []
    for (engine, conc, plen), rows in sorted(cells.items()):
        ok = [r for r in rows if r["error"] is None and r["ttft_s"] is not None]
        total = len(rows)
        if not ok:
            continue
        ttfts = [r["ttft_s"] * 1000 for r in ok]
        e2es = [r["e2e_s"] * 1000 for r in ok]
        all_itls = []
        for r in ok:
            all_itls.extend([x * 1000 for x in r["itls_s"]])
        total_out_toks = sum(r["n_output_tokens"] for r in ok)
        wall = max(r["e2e_s"] for r in ok)
        throughput = total_out_toks / wall if wall > 0 else 0
        summary_rows.append({
            "engine": engine,
            "concurrency": conc,
            "prompt_len": plen,
            "n_ok": len(ok),
            "n_total": total,
            "error_rate": (total - len(ok)) / total,
            "ttft_mean_ms": statistics.mean(ttfts),
            "ttft_p50_ms": percentile(ttfts, 50),
            "ttft_p99_ms": percentile(ttfts, 99),
            "itl_mean_ms": statistics.mean(all_itls) if all_itls else 0,
            "itl_p50_ms": percentile(all_itls, 50),
            "itl_p99_ms": percentile(all_itls, 99),
            "e2e_mean_ms": statistics.mean(e2es),
            "e2e_p50_ms": percentile(e2es, 50),
            "e2e_p99_ms": percentile(e2es, 99),
            "throughput_tok_s": throughput,
            "total_output_tokens": total_out_toks,
        })

    csv_path = outdir / "summary.csv"
    if summary_rows:
        cols = list(summary_rows[0].keys())
        with csv_path.open("w") as f:
            f.write(",".join(cols) + "\n")
            for row in summary_rows:
                f.write(
                    ",".join(
                        f"{row[c]:.4f}" if isinstance(row[c], float) else str(row[c])
                        for c in cols
                    )
                    + "\n"
                )
    print(f"Wrote {csv_path}")

    nested: dict = defaultdict(lambda: defaultdict(dict))
    for row in summary_rows:
        nested[row["engine"]][row["concurrency"]][row["prompt_len"]] = row
    json_path = outdir / "summary.json"
    with json_path.open("w") as f:
        json.dump(nested, f, indent=2)
    print(f"Wrote {json_path}")

    txt_path = outdir / "summary_table.txt"
    with txt_path.open("w") as f:
        f.write("=== Throughput (tok/s) ===\n")
        f.write(f"{'engine':<10} {'conc':<5} {'plen':<6} {'tok/s':<10} "
                f"{'ttft_p50':<10} {'ttft_p99':<10} {'itl_p50':<10} "
                f"{'e2e_p99':<10} {'err%':<6}\n")
        for row in summary_rows:
            f.write(
                f"{row['engine']:<10} {row['concurrency']:<5} {row['prompt_len']:<6} "
                f"{row['throughput_tok_s']:<10.1f} "
                f"{row['ttft_p50_ms']:<10.0f} {row['ttft_p99_ms']:<10.0f} "
                f"{row['itl_p50_ms']:<10.0f} {row['e2e_p99_ms']:<10.0f} "
                f"{row['error_rate']*100:<6.1f}\n"
            )
    print(f"Wrote {txt_path}")
    print("\n" + txt_path.read_text())


if __name__ == "__main__":
    main()
COLLECT_EOF

# --- serve_vllm.sh ---
cat > scripts/serve_vllm.sh << 'SERVE_VLLM_EOF'
#!/usr/bin/env bash
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
vllm serve "$MODEL" --port "$PORT" --host 0.0.0.0 $EXTRA_ARGS &

VLLM_PID=$!
echo "[vllm] pid=$VLLM_PID"

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
SERVE_VLLM_EOF

# --- serve_sglang.sh ---
cat > scripts/serve_sglang.sh << 'SERVE_SGLANG_EOF'
#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:?model required}"
PORT="${2:?port required}"
EXTRA_ARGS="${3:-}"

echo "[sglang] launching model=$MODEL port=$PORT"
python -m sglang.launch_server \
    --model-path "$MODEL" --port "$PORT" --host 0.0.0.0 $EXTRA_ARGS &

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
SERVE_SGLANG_EOF

# --- serve_tgi.sh ---
cat > scripts/serve_tgi.sh << 'SERVE_TGI_EOF'
#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:?model required}"
PORT="${2:?port required}"
EXTRA_ARGS="${3:-}"

echo "[tgi] launching model=$MODEL port=$PORT"
text-generation-launcher \
    --model-id "$MODEL" --port "$PORT" --hostname 0.0.0.0 $EXTRA_ARGS &

TGI_PID=$!
echo "[tgi] pid=$TGI_PID"

echo "[tgi] waiting for readiness..."
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
SERVE_TGI_EOF

# --- run_all.sh ---
cat > run_all.sh << 'RUN_ALL_EOF'
#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

MODEL="${MODEL:-meta-llama/Meta-Llama-3-8B-Instruct}"
PORT="${PORT:-8000}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-128}"
CONCURRENCY="${CONCURRENCY:-1 2 4 8 16 32}"
PROMPT_LENS="${PROMPT_LENS:-32 128 512 2048}"
PROMPTS_PER_CELL="${PROMPTS_PER_CELL:-20}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:---gpu-memory-utilization 0.9 --max-model-len 4096}"
SGLANG_EXTRA_ARGS="${SGLANG_EXTRA_ARGS:---mem-fraction-static 0.9 --context-length 4096}"
TGI_EXTRA_ARGS="${TGI_EXTRA_ARGS:---max-total-tokens 4096 --max-batch-size 256}"

if [ $# -gt 0 ]; then
    ENGINES=("$@")
else
    ENGINES=(vllm sglang tgi)
fi

mkdir -p results

echo "============================================"
echo " Project 1 - LLM Inference Engine Benchmark"
echo "============================================"
echo "Model:            $MODEL"
echo "Port:             $PORT"
echo "Engines:          ${ENGINES[*]}"
echo "Concurrency:      $CONCURRENCY"
echo "Prompt lens:      $PROMPT_LENS"
echo "Output tokens:    $OUTPUT_TOKENS"
echo "Prompts/cell:     $PROMPTS_PER_CELL"
echo "============================================"
echo

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

    local serve_log
    serve_log=$(bash "$serve_script" "$MODEL" "$PORT" "$extra_args")
    local server_pid
    server_pid=$(echo "$serve_log" | tail -1)
    echo "[$engine] server pid=$server_pid"

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
RUN_ALL_EOF

# --- Make scripts executable ---
chmod +x run_all.sh scripts/*.sh

echo
echo "=== Setup complete! ==="
echo "Files created in $PROJECT_DIR:"
find . -type f | sort
echo
echo "Next steps:"
echo "  1. huggingface-cli login   # paste your HF token"
echo "  2. ./run_all.sh vllm       # start with vLLM only (~20 min)"
echo "  3. ./run_all.sh sglang     # then SGLang"
echo "  4. ./run_all.sh tgi        # then TGI"
echo "  5. ./run_all.sh            # or all 3 at once"
