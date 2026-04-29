from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from benchmarker import Benchmarker
from config import BenchmarkConfig


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="bench",
        description="vLLM Performance Profiler — benchmarks TTFT, TPS, and VRAM for Gemma 4 models",
    )
    sub = p.add_subparsers(dest="command")

    # ── run ───────────────────────────────────────────────────────────────────
    run = sub.add_parser("run", help="Run a benchmark against a live vLLM endpoint")
    run.add_argument("--model", required=True, help="HuggingFace model ID (e.g. google/gemma-4-E2B-it)")
    run.add_argument("--url", default="http://localhost:8000/v1", help="vLLM base URL (default: localhost)")
    run.add_argument("--api-key", default="EMPTY", help="API key (use EMPTY for local vLLM)")
    run.add_argument("--requests", type=int, default=40, help="Total number of requests (default: 40)")
    run.add_argument("--concurrency", type=int, default=8, help="Max concurrent requests (default: 8)")
    run.add_argument("--max-tokens", type=int, default=150, help="Max completion tokens per request (default: 150)")
    run.add_argument("--output-dir", default="results", help="Directory to write JSON results (default: results/)")
    run.add_argument("--prompts-file", help="Path to a JSON file with a list of prompt strings")
    run.add_argument("-v", "--verbose", action="store_true")

    # ── compare ───────────────────────────────────────────────────────────────
    cmp = sub.add_parser("compare", help="Print a comparison table from saved result files")
    cmp.add_argument("files", nargs="+", metavar="RESULT_JSON")

    return p


def _print_run_summary(s: dict) -> None:
    width = 54
    print(f"\n{'─' * width}")
    print(f"  Model      : {s['model']}")
    print(f"  Requests   : {s['successful_requests']}/{s['total_requests']} succeeded")
    print(f"  Wall time  : {s['wall_time_s']}s")
    print(f"  TTFT mean  : {s['ttft']['mean_s']}s  (p95: {s['ttft']['p95_s']}s)")
    print(f"  TPS mean   : {s['tps']['mean']}")
    print(f"  Throughput : {s['throughput_tokens_per_s']} tok/s")
    print(f"  Peak VRAM  : {s['gpu']['peak_vram_mb']} MB")
    print(f"{'─' * width}\n")


def _print_comparison(files: list[str]) -> None:
    rows = []
    for f in files:
        data = json.loads(Path(f).read_text())
        rows.append(data["summary"])

    header = (
        f"{'Model':<36} {'TTFT mean':>10} {'TTFT p95':>10} "
        f"{'TPS mean':>10} {'Tok/s':>8} {'Peak VRAM':>11}"
    )
    sep = "─" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")
    for r in rows:
        if "error" in r:
            print(f"{r['model']:<36}  {'ERROR':>10}")
            continue
        print(
            f"{r['model']:<36} "
            f"{r['ttft']['mean_s']:>9.3f}s "
            f"{r['ttft']['p95_s']:>9.3f}s "
            f"{r['tps']['mean']:>10.1f} "
            f"{r['throughput_tokens_per_s']:>8.1f} "
            f"{r['gpu']['peak_vram_mb']:>8d} MB"
        )
    print(f"{sep}\n")


async def _run(args: argparse.Namespace) -> None:
    prompts = None
    if args.prompts_file:
        prompts = json.loads(Path(args.prompts_file).read_text())

    config = BenchmarkConfig(
        model=args.model,
        base_url=args.url,
        api_key=args.api_key,
        max_tokens=args.max_tokens,
        concurrency=args.concurrency,
        num_requests=args.requests,
        output_dir=args.output_dir,
        **({"prompts": prompts} if prompts else {}),
    )

    bench = Benchmarker(config)
    results = await bench.run()
    outfile = results.save(args.output_dir)
    _print_run_summary(results.summary())
    print(f"  Full results saved to: {outfile}\n")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "compare":
        _print_comparison(args.files)
        return

    if args.command == "run":
        setup_logging(getattr(args, "verbose", False))
        asyncio.run(_run(args))
        return

    parser.print_help()
    sys.exit(1)


if __name__ == "__main__":
    main()
