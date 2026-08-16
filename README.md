# inference-bench

Benchmark and compare LLM serving engines — **vLLM**, **SGLang**, and **TGI** —
on a single NVIDIA GPU. `inference-bench` drives each engine's OpenAI-compatible
streaming API with a load generator and records per-request latency and
throughput across a concurrency × prompt-length sweep, so all engines see an
identical, reproducible workload.

## How it works (one pod, isolated engine venvs)

Everything runs on **one** RunPod pod. Each engine is installed into its **own
virtualenv** (`venvs/vllm`, `venvs/sglang`, `venvs/tgi`), so:

- pip never touches the pod template's preinstalled torch (the source of the
  classic CUDA/torch/vLLM mismatch), and
- the three engines cannot conflict with each other.

`run_all.sh` installs → serves → benchmarks → tears down each engine in turn
on the same GPU.

```
one RunPod pod (RTX 4090)
run_all.sh
  ├─ venvs/vllm    →  vllm serve  :8000  → bench_client.py  → results/vllm_raw.jsonl
  ├─ venvs/sglang  →  sglang      :8000  → bench_client.py  → results/sglang_raw.jsonl
  └─ venvs/tgi     →  tgi         :8000  → bench_client.py  → results/tgi_raw.jsonl
                                  └→ collect_results.py → summary.csv/json/txt
```

## What it measures

Each `(engine, concurrency, prompt_len)` cell produces:

| Metric | Definition |
|---|---|
| TTFT | Time to first token — first streamed chunk minus request start |
| ITL | Inter-token latency — deltas between consecutive tokens |
| E2E | End-to-end latency — last token minus request start |
| Throughput | Total output tokens / max(E2E) across the cell |
| Error rate | Failed requests / total requests |

Latencies are reported as mean, p50, and p99.

Concurrency is enforced with an `asyncio.Semaphore` (in-flight request gate),
not an httpx connection-pool side effect, so the sweep measures true server
concurrency. `PROMPTS_PER_CELL` defaults to 40, above the max concurrency of 32,
so the top of the sweep is meaningful.

## Requirements

- One RunPod pod: RTX 4090 (24 GB), any PyTorch/CUDA template (the template's
  stack is irrelevant — each engine installs its own).
- Python 3.11+ and `venv` available on the pod (standard on RunPod templates).
- A Hugging Face token with access to any gated model (e.g. Llama-3), via
  `huggingface-cli login` on the pod.

## Quick start (single pod)

```bash
git clone https://github.com/kushaldabbe/inference-bench.git
cd inference-bench
chmod +x run_all.sh scripts/*.sh

# ONE-TIME: log in to HF (writes ~/.cache/huggingface/token, shared by all venvs)
huggingface-cli login

# Smoke test first (fast, cheap): TinyLlama, 1 engine, small sweep
MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
CONCURRENCY="1 8 32" PROMPT_LENS="128 512" PROMPTS_PER_CELL=5 \
    ./run_all.sh vllm

# Full sweep: Llama-3-8B, all three engines
./run_all.sh
# ...or one at a time
./run_all.sh vllm
./run_all.sh sglang
./run_all.sh tgi
```

`run_all.sh` prints a **preflight** (nvidia-smi, template torch, HF token
presence) before installing anything, so a bad pod fails in seconds, not after
billing hours.

### Engine installs (in each venv)

Each venv is created with `--system-site-packages`, so the pod template's
preinstalled torch is **reused, not re-downloaded** — this is what makes the
install fast and version-safe.

| Engine | Pin | Notes |
|---|---|---|
| vLLM | `vllm==0.11.0` + `transformers==4.55.2` | ⚠️ vLLM 0.11.0 breaks with transformers 5.x (`LlamaTokenizer.all_special_tokens_extended` error); transformers is pinned below 5 |
| SGLang | `sglang[all]` | |
| TGI | `text-generation-inference` | ⚠️ **TGI is archived** (HF maintenance mode, Mar 2026); still runs, treat as legacy |

Pins are overridable: `VLLM_PIN=... SGLANG_PIN=... TGI_PIN=... ./run_all.sh`.
Venvs are cached per engine (a `.ready` marker skips reinstall); delete
`venvs/<engine>` to force a clean install. Bump versions deliberately —
results are not comparable across engine versions.

## Configuration

Overrides are environment variables:

```bash
# Smaller sweep for quick iteration
MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
CONCURRENCY="1 8 32" PROMPT_LENS="128 512" PROMPTS_PER_CELL=10 ./run_all.sh vllm

# Pass extra flags to the server
VLLM_EXTRA_ARGS="--gpu-memory-utilization 0.85 --max-model-len 2048" ./run_all.sh vllm
```

Sweep defaults (`configs/bench_config.yaml`): concurrency `1 2 4 8 16 32`,
prompt lens `32 128 512 2048`, 128 output tokens, greedy (`temperature=0`),
40 prompts per cell.


## Output

| File | Description |
|---|---|
| `results/<engine>_raw.jsonl` | One JSON object per request (TTFT, ITL list, E2E, token count, error) |
| `results/summary.csv` | One row per cell with aggregated stats |
| `results/summary.json` | Same data nested as `engine → concurrency → prompt_len` for tooling |
| `results/summary_table.txt` | Human-readable comparison, also printed to stdout |

`results/` is gitignored except for `.gitkeep`.

## Dashboard

`dashboard/grafana-dashboard.json` is an empty scaffold. Feed
`results/summary.json` to Grafana (JSON API data source, or load it into
SQLite/Postgres) and build panels such as:

- Throughput vs concurrency (per engine)
- TTFT p99 vs prompt length (per engine)
- ITL p50 vs concurrency (per engine)
- E2E p99 vs concurrency (per engine)

## Repository layout

```
inference-bench/
├── configs/bench_config.yaml   # sweep defaults + engine pins
├── scripts/
│   ├── bench_client.py         # async streaming load generator (semaphore-gated concurrency)
│   ├── collect_results.py      # raw JSONL → CSV + JSON + table
│   ├── serve_vllm.sh           # start engine + readiness probe
│   ├── serve_sglang.sh
│   └── serve_tgi.sh
├── dashboard/
│   └── grafana-dashboard.json  # empty scaffold
├── results/                    # output (gitignored)
├── venvs/                      # per-engine virtualenvs (gitignored, created at runtime)
└── run_all.sh                  # orchestrator: preflight → venv install → serve → bench → teardown
```

## Troubleshooting

**Preflight fails (CUDA not available in a venv).** The venv's torch couldn't
see the GPU. Most likely pip pulled a CPU torch — the script installs torch from
the PyTorch cu126 index specifically to avoid this. Delete `venvs/<engine>` and
re-run; confirm `nvidia-smi` works first.

**torch/CUDA mismatch during install.** Because each engine installs into an
isolated venv, the template's preinstalled torch is never touched. If a version
conflict still appears, delete `venvs/<engine>` and check the engine pin
(`VLLM_PIN`, `SGLANG_PIN`, `TGI_PIN`).

**Gated model 401.** Run `huggingface-cli login` on the pod (writes
`~/.cache/huggingface/token`, shared by all venvs) and accept the model license
on its HF page. The preflight checks for the token file before running.

**Timeouts on long prompts.** Default client read timeout is 300 s; the load
generator is not the bottleneck. First check GPU utilization and the server
log (`serve_*.sh` output goes to the terminal).

**vLLM OOM on load.** Lower `--gpu-memory-utilization` (e.g. 0.85) or benchmark
a smaller model (`MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0`).

## Notes

- Prompts are generated from a fixed seed, so every engine sees the same
  workload and results are directly comparable.
- Decoding is greedy (`temperature=0`) by default for reproducibility.
- Engine versions are pinned (see `configs/bench_config.yaml` and the `*_PIN`
  env vars); bump versions deliberately and re-benchmark, since results are not
  comparable across engine versions.
- Each engine runs on the same GPU sequentially, so hardware conditions are
  identical across engines.
- TGI is archived (HF maintenance mode, Mar 2026): it still runs, but treat
  vLLM vs SGLang as the headline comparison and label TGI as legacy in any
  writeup.
