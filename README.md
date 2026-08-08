# inference-bench

Benchmark and compare LLM serving engines — **vLLM**, **SGLang**, and **TGI** —
on a single GPU. `inference-bench` drives each engine's OpenAI-compatible
streaming API with a load generator and records per-request latency and
throughput across a concurrency × prompt-length sweep, so the three engines see
an identical, reproducible workload.

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

## Requirements

- Linux with one NVIDIA GPU (~24 GB VRAM; tested on RTX 4090 / A10G). Llama-3-8B
  in fp16 needs ~16 GB, smaller models need less.
- `curl` for the readiness probes.
- A Hugging Face token with access to any gated model you benchmark (e.g. Llama-3).

The engines themselves are installed into the active Python environment by
`run_all.sh` — there is no separate install step. The load generator only
depends on `httpx`, so it can also run from a separate machine and point
`--endpoint` at the server.

## Usage

```bash
git clone https://github.com/kushaldabbe/inference-bench.git
cd inference-bench
chmod +x run_all.sh scripts/*.sh

export HF_TOKEN=hf_xxx            # required for gated models
./run_all.sh                      # benchmark all three engines
./run_all.sh vllm                 # benchmark one engine
./run_all.sh vllm sglang          # benchmark a subset
```

Each engine is installed, served, benchmarked, then torn down before the next
starts. Per-request results stream to `results/<engine>_raw.jsonl`; a summary is
written at the end.

To run on a fresh GPU box with only one file uploaded, `setup_on_pod.sh`
reconstructs the whole project:

```bash
bash setup_on_pod.sh
```

## Configuration

Defaults live in [`configs/bench_config.yaml`](configs/bench_config.yaml) and can
be overridden with environment variables:

```bash
# Smaller sweep for quick iteration
CONCURRENCY="1 8 32" PROMPT_LENS="128 512" PROMPTS_PER_CELL=10 ./run_all.sh vllm

# Different model
MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0" ./run_all.sh

# Pass extra flags to the server
VLLM_EXTRA_ARGS="--gpu-memory-utilization 0.85 --max-model-len 2048" ./run_all.sh vllm
```

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
├── configs/bench_config.yaml   # default sweep + per-engine flags
├── scripts/
│   ├── bench_client.py         # async streaming load generator
│   ├── collect_results.py      # raw JSONL → CSV + JSON + table
│   ├── serve_vllm.sh           # serve + readiness probe
│   ├── serve_sglang.sh
│   └── serve_tgi.sh
├── dashboard/
│   └── grafana-dashboard.json  # empty scaffold
├── results/                    # output (gitignored)
├── setup_on_pod.sh             # one-file bootstrap for a fresh host
└── run_all.sh                  # orchestrator: install → serve → bench → teardown
```

## Troubleshooting

**vLLM OOM on load.** Lower `--gpu-memory-utilization` (e.g. 0.85) or benchmark
a smaller model.

**vLLM / torch CUDA mismatch.** vLLM is pinned to `0.8.5.post1`, which requires
`torch==2.6.0` (cu126 wheel). Newer vLLM pulls torch 2.7+ (cu128/cu130) and may
not run on older drivers — keep the pin or upgrade the host driver.

**TGI install fails.** The Rust build can be fragile; pin a known-good version
(`pip install text-generation-inference==0.10.0`) or skip TGI
(`./run_all.sh vllm sglang`).

**Port already in use.** Clear leftover servers: `pkill -f sglang; pkill -f vllm`.

**Gated model 401.** Set `HF_TOKEN` and accept the model license on its HF page.

## Notes

- Prompts are generated from a fixed seed, so every engine sees the same
  workload and results are directly comparable.
- Decoding is greedy (`temperature=0`) by default for reproducibility.
- Each engine is version-pinned in `run_all.sh`; bump versions deliberately and
  re-benchmark, since results are not comparable across engine versions.
