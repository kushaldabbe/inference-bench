# inference-bench

Benchmark and compare LLM serving engines — **vLLM**, **SGLang**, and **TGI** —
on a single NVIDIA GPU. `inference-bench` drives each engine's OpenAI-compatible
streaming API with a load generator and records per-request latency and
throughput across a concurrency × prompt-length sweep, so all engines see an
identical, reproducible workload.

## How it works (engine pods + laptop client)

Engines are **not** installed here. Each engine runs as a RunPod pod from its
official Docker image (the image *is* the environment — no CUDA/torch version
mismatches). The laptop only runs the load generator (`httpx`) and hits each
pod's public proxy URL:

```
your laptop                RunPod pods (RTX 4090)
bench_client.py   ──────►  https://<pod>-8000.proxy.runpod.net  (vLLM)
collect_results.py         https://<pod>-8000.proxy.runpod.net  (SGLang)
                           https://<pod>-8000.proxy.runpod.net  (TGI)
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

- One RTX 4090 (24 GB) pod per engine on RunPod, deployed from the images below.
- `httpx` on the laptop (auto-installed by `run_remote.sh` if missing).
- A Hugging Face token with access to any gated model (e.g. Llama-3) — set as
  `HF_TOKEN` in each pod's environment.

## Deploy engine pods (RunPod)

For each engine, create a pod with **Custom Container**, GPU **RTX 4090**, and
set `HF_TOKEN` in the pod environment. Confirmed images and server args:

| Engine | Image | Server args |
|---|---|---|
| vLLM | `vllm/vllm-openai:v0.26.0` | `--model meta-llama/Meta-Llama-3-8B-Instruct --host 0.0.0.0 --port 8000 --gpu-memory-utilization 0.9 --max-model-len 4096` |
| SGLang | `lmsysorg/sglang:v0.5.17-cu129-runtime` | `--model-path meta-llama/Meta-Llama-3-8B-Instruct --host 0.0.0.0 --port 8000 --mem-fraction-static 0.9 --context-length 4096` |
| TGI | `ghcr.io/huggingface/text-generation-inference:3.3.7` | `--model-id meta-llama/Meta-Llama-3-8B-Instruct --port 8000 --hostname 0.0.0.0 --max-total-tokens 4096 --max-batch-size 256` |

Notes:

- Deploy all three at once, or one at a time and reuse the pod slot (stop +
  change image). RunPod bills per running pod.
- The pod's port 8000 is exposed as `https://<pod-id>-8000.proxy.runpod.net`.
- The host driver is **not** a constraint: these images run on any driver
  supporting CUDA 12.x/13.x (confirmed on RunPod driver 580 / CUDA 13.0).
- **TGI is archived** (Hugging Face put it in maintenance mode, Mar 2026). It
  still runs and is a valid data point; treat vLLM vs SGLang as the headline
  comparison and label TGI as legacy in any writeup.

## Run the benchmark (laptop)

```bash
git clone https://github.com/kushaldabbe/inference-bench.git
cd inference-bench

ENDPOINTS="vllm=https://<vllm-pod>-8000.proxy.runpod.net \
           sglang=https://<sglang-pod>-8000.proxy.runpod.net \
           tgi=https://<tgi-pod>-8000.proxy.runpod.net" \
    ./run_remote.sh
```

Run a subset, or a single engine:

```bash
ENDPOINTS="vllm=https://<vllm-pod>-8000.proxy.runpod.net" ./run_remote.sh vllm
```

### Sanity-check a pod before the full sweep

```bash
curl -s https://<pod>-8000.proxy.runpod.net/v1/models
```

## Configuration

Overrides are environment variables:

```bash
# Smaller sweep for quick iteration
MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
CONCURRENCY="1 8 32" PROMPT_LENS="128 512" PROMPTS_PER_CELL=10 \
ENDPOINTS="vllm=https://<vllm-pod>-8000.proxy.runpod.net" ./run_remote.sh
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
├── configs/bench_config.yaml   # sweep defaults + engine images
├── scripts/
│   ├── bench_client.py         # async streaming load generator (semaphore-gated concurrency)
│   ├── collect_results.py      # raw JSONL → CSV + JSON + table
│   ├── serve_vllm.sh           # (legacy, single-pod Option B)
│   ├── serve_sglang.sh
│   └── serve_tgi.sh
├── dashboard/
│   └── grafana-dashboard.json  # empty scaffold
├── results/                    # output (gitignored)
├── run_remote.sh               # laptop orchestrator against deployed pods (Option A)
└── run_all.sh                  # (legacy, install-and-run on one pod)
```

## Troubleshooting

**Pod not reachable.** Confirm the pod is running and the proxy URL is the
`<pod-id>-8000` form. A 404 on `/v1/models` usually means the image's args
weren't accepted — check the pod logs.

**Gated model 401.** Set `HF_TOKEN` in the pod environment and accept the
model license on its HF page.

**Timeouts on long prompts.** Default client read timeout is 300 s; the load
generator is not the bottleneck. First check GPU utilization and the pod logs.

**vLLM OOM on load.** Lower `--gpu-memory-utilization` (e.g. 0.85) or benchmark
a smaller model (`MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0`).

## Notes

- Prompts are generated from a fixed seed, so every engine sees the same
  workload and results are directly comparable.
- Decoding is greedy (`temperature=0`) by default for reproducibility.
- The client runs from the laptop, so reported latencies include public-network
  latency (roughly constant across engines). For sub-millisecond-accurate TTFT,
  run the client on a CPU pod in the same region — the relative comparison is
  unaffected.
- Engine versions are pinned at deploy time (see `configs/bench_config.yaml`);
  bump versions deliberately and re-benchmark, since results are not comparable
  across engine versions.
