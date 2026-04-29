from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from statistics import mean, median
from typing import Any

from config import BenchmarkConfig
from gpu_monitor import GPUMonitor, GPUSummary
from profiler import InferenceProfiler, RequestMetrics

logger = logging.getLogger(__name__)


class BenchmarkResults:
    def __init__(
        self,
        model: str,
        metrics: list[RequestMetrics],
        gpu_summary: GPUSummary,
        config: BenchmarkConfig,
        wall_time: float,
    ) -> None:
        self.model = model
        self.metrics = metrics
        self.gpu_summary = gpu_summary
        self.config = config
        self.wall_time = wall_time

    @staticmethod
    def _percentile(values: list[float], pct: float) -> float:
        if not values:
            return 0.0
        sorted_vals = sorted(values)
        idx = int(len(sorted_vals) * pct / 100)
        return sorted_vals[min(idx, len(sorted_vals) - 1)]

    def summary(self) -> dict[str, Any]:
        successful = [m for m in self.metrics if m.success]
        failed = [m for m in self.metrics if not m.success]

        if not successful:
            return {"error": "All requests failed", "model": self.model}

        ttfts = [m.ttft for m in successful]
        tpss = [m.tps for m in successful]
        latencies = [m.total_latency for m in successful]
        total_tokens = sum(m.completion_tokens for m in successful)

        return {
            "model": self.model,
            "total_requests": len(self.metrics),
            "successful_requests": len(successful),
            "failed_requests": len(failed),
            "concurrency": self.config.concurrency,
            "wall_time_s": round(self.wall_time, 3),
            "throughput_req_per_s": round(len(successful) / self.wall_time, 2),
            "throughput_tokens_per_s": round(total_tokens / self.wall_time, 1),
            "ttft": {
                "mean_s": round(mean(ttfts), 4),
                "median_s": round(median(ttfts), 4),
                "p95_s": round(self._percentile(ttfts, 95), 4),
                "p99_s": round(self._percentile(ttfts, 99), 4),
                "min_s": round(min(ttfts), 4),
                "max_s": round(max(ttfts), 4),
            },
            "tps": {
                "mean": round(mean(tpss), 2),
                "median": round(median(tpss), 2),
                "p95": round(self._percentile(tpss, 95), 2),
                "min": round(min(tpss), 2),
                "max": round(max(tpss), 2),
            },
            "latency": {
                "mean_s": round(mean(latencies), 4),
                "p95_s": round(self._percentile(latencies, 95), 4),
                "p99_s": round(self._percentile(latencies, 99), 4),
            },
            "gpu": self.gpu_summary.to_dict(),
        }

    def save(self, output_dir: str = "results") -> Path:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        model_slug = self.model.replace("/", "_").replace("-", "_")
        outfile = out / f"{model_slug}.json"

        payload = {
            "summary": self.summary(),
            "config": {
                "model": self.config.model,
                "base_url": self.config.base_url,
                "max_tokens": self.config.max_tokens,
                "concurrency": self.config.concurrency,
                "num_requests": self.config.num_requests,
                "temperature": self.config.temperature,
            },
            "per_request_metrics": [
                {
                    "ttft": m.ttft,
                    "tps": m.tps,
                    "total_latency": m.total_latency,
                    "completion_tokens": m.completion_tokens,
                    "success": m.success,
                    "error": m.error,
                }
                for m in self.metrics
            ],
        }

        outfile.write_text(json.dumps(payload, indent=2))
        logger.info("Results saved to %s", outfile)
        return outfile


class Benchmarker:
    """Orchestrates concurrent inference requests against a vLLM endpoint
    and collects TTFT, TPS, throughput, and VRAM metrics.
    """

    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config
        self.profiler = InferenceProfiler(config)
        self.gpu_monitor = GPUMonitor(poll_interval=config.gpu_poll_interval)

    async def _run_with_semaphore(
        self,
        sem: asyncio.Semaphore,
        prompt: str,
    ) -> RequestMetrics:
        async with sem:
            return await self.profiler.profile_request(prompt)

    async def run(self) -> BenchmarkResults:
        """Run the full benchmark suite and return results."""
        # Cycle through prompts to fill num_requests
        prompts = (self.config.prompts * self.config.num_requests)[: self.config.num_requests]
        sem = asyncio.Semaphore(self.config.concurrency)

        logger.info(
            "Starting benchmark | model=%s | requests=%d | concurrency=%d",
            self.config.model,
            self.config.num_requests,
            self.config.concurrency,
        )

        self.gpu_monitor.start()
        wall_start = time.perf_counter()

        tasks = [self._run_with_semaphore(sem, p) for p in prompts]
        metrics: tuple[RequestMetrics, ...] = await asyncio.gather(*tasks)

        wall_time = time.perf_counter() - wall_start
        gpu_summary = await self.gpu_monitor.stop()
        await self.profiler.close()

        results = BenchmarkResults(
            model=self.config.model,
            metrics=list(metrics),
            gpu_summary=gpu_summary,
            config=self.config,
            wall_time=wall_time,
        )

        s = results.summary()
        logger.info("Benchmark complete in %.1fs", wall_time)
        logger.info(
            "TTFT mean=%.3fs p95=%.3fs | TPS mean=%.1f | Throughput=%.1f tok/s | Peak VRAM=%d MB",
            s["ttft"]["mean_s"],
            s["ttft"]["p95_s"],
            s["tps"]["mean"],
            s["throughput_tokens_per_s"],
            s["gpu"]["peak_vram_mb"],
        )

        return results
