# inference-bench — vLLM vs SGLang vs TGI

> Benchmark **vLLM vs SGLang vs TGI** on a single GPU. Measures throughput,
> TTFT, ITL, and p99 latency across a concurrency × prompt-length sweep.
>
> Part of an LLM inference learning journey. This project establishes baseline
> fluency with inference engines and produces a public, reproducible artifact
> (results + dashboard) you can link on a resume.

## What it measures

For each `(engine, concurrency, prompt_len)` cell:

| Metric | How |
|---|---|
| **TTFT** (time to first token) | First streamed chunk timestamp − request start |
| **ITL** (inter-token latency) | Delta between consecutive streamed tokens |
| **E2E latency** | Last token timestamp − request start |
| **Throughput** | Σ output tokens / max(e2e) across the cell |
| **Error rate** | Failed / total requests |

Latencies reported as mean / p50 / p99.

## Cost on RunPod

| GPU | $/hr | Time for full sweep | Est. cost |
|---|---|---|---|
| RTX 4090 (24GB) | ~$0.40 | ~2 hr | **~$0.80** |
| A100 (40GB) | ~$1.50 | ~2 hr | ~$3.00 |

Fits well within a $15 credit. Use the RTX 4090.

## Files

```
inference-bench/
├── configs/
│   └── bench_config.yaml      # single source of truth for params
├── scripts/
│   ├── bench_client.py        # async streaming load generator
│   ├── collect_results.py     # raw JSONL -> CSV + Grafana JSON + table
│   ├── serve_vllm.sh          # start vLLM, wait for readiness
│   ├── serve_sglang.sh        # start SGLang, wait for readiness
│   └── serve_tgi.sh           # start TGI, wait for readiness
├── results/                   # output (created on run, gitignored except .gitkeep)
├── dashboard/
│   └── grafana-dashboard.json # empty scaffold - build panels from your data
├── setup_on_pod.sh            # one-file bootstrap: recreates this dir on a pod
└── run_all.sh                 # orchestrator: install -> serve -> bench -> teardown
```

## Quick start (on RunPod)

### 1. Spin up a pod

- Template: **PyTorch 2.5 + CUDA 12.4** (RunPod's stock template)
- GPU: **RTX 4090 (24GB)**
- Disk: 50 GB container disk (for model weights)

### 2. Get this project onto the pod

Two options:

**Option A — upload the whole folder** (e.g. via `scp`, rsync, or the Jupyter
file browser) to `/workspace/inference-bench/`.

```bash
# From your machine:
scp -r ./inference-bench root@<POD_IP>:/workspace/
```

**Option B — upload only `setup_on_pod.sh`**, then run it to recreate everything:

```bash
bash setup_on_pod.sh    # recreates /workspace/inference-bench/ with all scripts
```

### 3. Set HF token (for Llama-3 gated access)

```bash
export HF_TOKEN=hf_xxx   # your token from https://huggingface.co/settings/tokens
# Accept Llama-3 license at:
# https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
```

### 4. Run

```bash
cd /workspace/inference-bench
chmod +x run_all.sh scripts/*.sh

# Full sweep (all 3 engines, ~2 hr, ~$0.80 on RTX 4090)
./run_all.sh

# Or bench one engine at a time (faster feedback, cheaper iteration)
./run_all.sh vllm
./run_all.sh sglang
./run_all.sh tgi
```

### 5. Pull results back to your machine

```bash
scp -r root@<POD_IP>:/workspace/inference-bench/results/ ./inference-bench/
```

### 6. (Optional) Tear down the pod

Stop the pod in the RunPod UI to stop billing. Disk storage costs ~$0.10/GB/mo
if you want to keep the model cached.

## Configuration

All knobs live in [`configs/bench_config.yaml`](configs/bench_config.yaml). The
orchestrator also accepts env-var overrides:

```bash
# Smaller / faster sweep for quick iteration
CONCURRENCY="1 8 32" PROMPT_LENS="128 512" PROMPTS_PER_CELL=10 \
    ./run_all.sh vllm

# Different model
MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0" ./run_all.sh

# Override vLLM server args
VLLM_EXTRA_ARGS="--gpu-memory-utilization 0.85 --max-model-len 2048" \
    ./run_all.sh vllm
```

## Output

After `run_all.sh` completes, `results/` contains:

| File | Description |
|---|---|
| `vllm_raw.jsonl` | One JSON line per request (TTFT, ITL list, e2e, token count) |
| `sglang_raw.jsonl` | Same for SGLang |
| `tgi_raw.jsonl` | Same for TGI |
| `summary.csv` | One row per `(engine, concurrency, prompt_len)` cell with aggregated stats |
| `summary.json` | Same data, nested `engine -> concurrency -> prompt_len` for Grafana import |
| `summary_table.txt` | Human-readable comparison table (printed to stdout) |

## Building the Grafana dashboard

`dashboard/grafana-dashboard.json` is an empty scaffold — the best dashboard is
one you design from your own data. Recommended panels:

1. **Throughput vs concurrency** (line chart, one series per engine)
2. **TTFT p99 vs prompt length** (line chart, one series per engine)
3. **ITL p50 vs concurrency** (line chart, one series per engine)
4. **E2E p99 vs concurrency** (bar chart grouped by engine)
5. **Throughput vs prompt length** (heatmap)

Load `results/summary.json` into Grafana via the **JSON API** data source, or
convert to SQLite/Postgres for a permanent dashboard.

## Troubleshooting

**vLLM OOM on load:** lower `--gpu-memory-utilization` to 0.85, or use a smaller
model (`MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0`).

**vLLM / torch CUDA mismatch:** vLLM is pinned to `0.8.5.post1` (needs
`torch==2.6.0`, cu126 wheel). RunPod's driver 570 supports CUDA ≤12.8, so cu126
works. Newer vLLM needs torch 2.7+ (cu128/cu130) which driver 570 can't run —
stick with the pin or upgrade the pod template's driver.

**TGI install fails:** TGI's Rust build can be fragile. Try
`pip install text-generation-inference==0.10.0` for a pinned, known-good
version, or skip TGI and run `./run_all.sh vllm sglang`.

**SGLang `--port` already in use:** ensure no leftover process:
`pkill -f sglang` and `pkill -f vllm`.

**Llama-3 gated access:** You must (1) have an HF token, and (2) accept the
model license at the HF model page. Otherwise download fails with 401.

**Slow model download:** pre-download weights once, save the pod disk, and
reuse: `huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct`.

## What to write in the resume / blog post

After collecting results, write a short blog post or README section covering:

1. **Setup** — single RTX 4090, Llama-3-8B-Instruct, 3 engines, default configs.
2. **Methodology** — streaming OpenAI-compat API, concurrency sweep {1,2,4,8,16,32} × prompt lengths {32,128,512,2048}, 20 prompts/cell, greedy decoding.
3. **Results** — 3-4 plots: throughput vs concurrency, TTFT vs prompt length, ITL vs concurrency, p99 latency.
4. **Analysis** — *why* one engine wins where. Tie back to: continuous batching implementation, prefix caching default-on/off, KV-cache memory layout, scheduling policy.
5. **Conclusion** — recommendation per workload (interactive vs batch, short vs long context).

This analysis is what separates "I ran a benchmark" from "I understand inference systems."

## Local development (optional)

The harness only needs `httpx` to run the client locally (for linting/formatting).
The heavy engines are installed on the pod by `run_all.sh`, not locally.

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
ruff check .
```
