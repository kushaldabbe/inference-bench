"""Project 1 - Aggregate raw JSONL benchmark results into CSV + summary JSON.

Reads results/*_raw.jsonl, computes per-cell stats (mean/p50/p99 TTFT, ITL,
e2e, throughput), writes:
  - results/summary.csv       (one row per cell)
  - results/summary.json      (Grafana-ready: nested by engine/concurrency/plen)
  - results/summary_table.txt (human-readable comparison)

Usage:
    python scripts/collect_results.py --in results/ --out results/
"""

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

    # Group by (engine, concurrency, prompt_len)
    cells = defaultdict(list)
    for r in all_rows:
        key = (r["engine"], r["concurrency"], r["prompt_len_target"])
        cells[key].append(r)

    # Build summary rows
    summary_rows = []
    for (engine, conc, plen), rows in sorted(cells.items()):
        ok = [r for r in rows if r["error"] is None and r["ttft_s"] is not None]
        total = len(rows)
        if not ok:
            continue
        ttfts = [r["ttft_s"] * 1000 for r in ok]  # ms
        e2es = [r["e2e_s"] * 1000 for r in ok]
        all_itls = []
        for r in ok:
            all_itls.extend([x * 1000 for x in r["itls_s"]])
        total_out_toks = sum(r["n_output_tokens"] for r in ok)
        wall = max(r["e2e_s"] for r in ok)
        throughput = total_out_toks / wall if wall > 0 else 0
        summary_rows.append(
            {
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
            }
        )

    # Write CSV
    csv_path = outdir / "summary.csv"
    if summary_rows:
        cols = list(summary_rows[0].keys())
        with csv_path.open("w") as f:
            f.write(",".join(cols) + "\n")
            for row in summary_rows:
                f.write(
                    ",".join(
                        f"{row[c]:.4f}" if isinstance(row[c], float) else str(row[c]) for c in cols
                    )
                    + "\n"
                )
    print(f"Wrote {csv_path}")

    # Write Grafana-ready nested JSON
    nested: dict = defaultdict(lambda: defaultdict(dict))
    for row in summary_rows:
        nested[row["engine"]][row["concurrency"]][row["prompt_len"]] = row
    json_path = outdir / "summary.json"
    with json_path.open("w") as f:
        json.dump(nested, f, indent=2)
    print(f"Wrote {json_path}")

    # Human-readable comparison table
    txt_path = outdir / "summary_table.txt"
    with txt_path.open("w") as f:
        f.write("=== Throughput (tok/s) ===\n")
        f.write(
            f"{'engine':<10} {'conc':<5} {'plen':<6} {'tok/s':<10} "
            f"{'ttft_p50':<10} {'ttft_p99':<10} {'itl_p50':<10} "
            f"{'e2e_p99':<10} {'err%':<6}\n"
        )
        for row in summary_rows:
            f.write(
                f"{row['engine']:<10} {row['concurrency']:<5} {row['prompt_len']:<6} "
                f"{row['throughput_tok_s']:<10.1f} "
                f"{row['ttft_p50_ms']:<10.0f} {row['ttft_p99_ms']:<10.0f} "
                f"{row['itl_p50_ms']:<10.0f} {row['e2e_p99_ms']:<10.0f} "
                f"{row['error_rate'] * 100:<6.1f}\n"
            )
    print(f"Wrote {txt_path}")
    print("\n" + txt_path.read_text())


if __name__ == "__main__":
    main()
