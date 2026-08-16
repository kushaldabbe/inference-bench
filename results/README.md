# Benchmark Results

Curated, reproducible results from `inference-bench`. One folder per engine.

```
results/
├── README.md
├── vllm/              # vLLM 0.11.0 (torch 2.8.0+cu128, transformers 4.55.2)
│   ├── vllm_raw.jsonl        # per-request raw data (960 requests)
│   ├── summary.csv           # one row per (concurrency, prompt_len) cell
│   ├── summary.json          # same data nested for tooling / Grafana
│   └── summary_table.txt     # human-readable comparison table
└── sglang/            # SGLang (venv pin; FlashInfer attention backend)
    ├── sglang_raw.jsonl      # per-request raw data (960 requests)
    ├── summary.csv
    ├── summary.json
    └── summary_table.txt
```

Old or intermediate runs (e.g. the early short-output attempt) live in
`results_archive/`, which is gitignored — kept for reference, not published.

## Methodology

- **Hardware:** NVIDIA RTX 4090 (24 GB), RunPod, driver 570–580 / CUDA 13.0.
- **Model:** `meta-llama/Meta-Llama-3-8B-Instruct`, fp16, greedy decoding
  (`temperature=0`), `max_tokens=128`.
- **Workload:** fixed-seed synthetic prompts; concurrency `{1,2,4,8,16,32}` ×
  prompt lengths `{32,128,512,2048}` × 40 requests per cell = 960 requests.
- **API:** `/v1/chat/completions` with streaming (OpenAI-compatible).
- **Metrics:** TTFT (time to first token), ITL (inter-token latency), E2E,
  throughput (total tokens ÷ cell wall time), error rate. Reported as
  mean / p50 / p99.
- **Concurrency:** enforced with an `asyncio.Semaphore` (true in-flight gate),
  not an httpx connection-pool side effect.
- **Throughput:** total output tokens ÷ wall time of the whole cell (all
  requests in a cell ran concurrently), so it reflects real batch throughput
  under continuous batching — not tokens ÷ slowest single request.

## Methodology notes (measurement fixes)

1. **Prompt length vs output length.** An early run used a
   "Summarize the above in one sentence" instruction, which made the model stop
   early via EOS (~30–44 tokens), so throughput compared short answers. The
   published runs use a "write a long, detailed analysis" instruction, so every
   request generates the full `max_tokens=128`, giving a fixed-length,
   comparable workload. (The abandoned run is archived, not published.)
2. **Throughput formula.** v1 computed throughput as
   `total_tokens / slowest_single_request_e2e`, which over-counts under
   concurrency. v2 records `cell_wall_s` (first request start → last request
   finish) and uses that. All published numbers are v2.

## Results summary (so far)

Both engines show the same continuous-batching curve: ~45–60 tok/s at
concurrency 1 rising to ~1000–1080 tok/s at concurrency 32 (Llama-3-8B fp16,
RTX 4090). vLLM and SGLang are effectively tied on this workload.

| Concurrency | vLLM tok/s (plen=128) | SGLang tok/s (plen=128) |
|---|---|---|
| 1 | 58.6 | 60.2 |
| 2 | 115.4 | 118.1 |
| 4 | 230.3 | 234.9 |
| 8 | 454.5 | 464.8 |
| 16 | 738.6 | 762.3 |
| 32 | 1061.4 | 1077.7 |

Notes:
- SGLang's `c=1, plen=32` cell (44.7 tok/s, 772 ms TTFT) is a cold-start
  warmup artifact — the first cell after server startup; all later cells are
  clean (TTFT 36–64 ms).
- Full per-cell stats (TTFT/ITL/E2E p50/p99) are in each engine's
  `summary_table.txt`.

## How to regenerate

```bash
python scripts/collect_results.py --in results/vllm --out results/vllm
python scripts/collect_results.py --in results/sglang --out results/sglang
```

Or re-run the sweep (see root README).
