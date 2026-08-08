"""Project 1 - LLM Inference Engine Benchmark.

Async streaming load generator. Measures TTFT, ITL, and throughput for an
OpenAI-compatible inference endpoint (vLLM, SGLang, or TGI in OpenAI-compat mode).

Usage:
    python scripts/bench_client.py \\
        --endpoint http://localhost:8000 \\
        --engine vllm \\
        --model meta-llama/Meta-Llama-3-8B-Instruct \\
        --concurrency 1 2 4 8 16 32 \\
        --prompt-len 32 128 512 2048 \\
        --output-tokens 128 \\
        --prompts-per-cell 20 \\
        --out results/vllm_raw.jsonl
"""

import argparse
import asyncio
import json
import random
import time
from pathlib import Path

import httpx

# Deterministic seed so all engines see the same prompt set
SEED = 42


def make_prompts(n: int, target_tokens: int) -> list[str]:
    """Generate n synthetic prompts of roughly target_tokens length.

    Uses a repeatable lorem-style fill so length is predictable and content
    is neutral (no model-specific bias).
    """
    rng = random.Random(SEED + target_tokens)
    # ~1.3 chars per token for English text
    target_chars = int(target_tokens * 1.3)
    words = (  # noqa: SIM905 - intentional: readable sentence split into a word list
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
    """Send one streaming completion request; record TTFT + per-token ITL."""
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
                    data = line[len("data: ") :]
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
        "engine": None,  # filled by caller
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
    """Run one (engine, concurrency, prompt_len) cell."""
    prompts = make_prompts(prompts_per_cell, prompt_len)
    # Open enough connections for the concurrency level
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
            f"ttft_mean={ttft_mean * 1000:.0f}ms "
            f"e2e_mean={sum(r['e2e_s'] for r in ok) / len(ok) * 1000:.0f}ms"
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
    # Truncate to start fresh for this engine run
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
