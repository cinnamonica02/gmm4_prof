from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest

from benchmarker import Benchmarker, BenchmarkResults
from config import BenchmarkConfig
from gpu_monitor import GPUMonitor, GPUSummary, GPUSnapshot
from profiler import InferenceProfiler, RequestMetrics


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_config(**kwargs) -> BenchmarkConfig:
    defaults = dict(
        model="test-model",
        base_url="http://localhost:8000/v1",
        num_requests=5,
        concurrency=2,
        max_tokens=50,
    )
    defaults.update(kwargs)
    return BenchmarkConfig(**defaults)


def make_metric(ttft: float = 0.1, tps: float = 50.0, success: bool = True) -> RequestMetrics:
    return RequestMetrics(
        ttft=ttft,
        tps=tps,
        total_latency=ttft + 0.4,
        completion_tokens=25,
        prompt="test prompt",
        success=success,
        error=None if success else "simulated error",
    )


def make_gpu_summary(vram_used: int = 10000) -> GPUSummary:
    s = GPUSummary()
    s.snapshots = [
        GPUSnapshot(timestamp=0.0, vram_used_mb=vram_used, vram_total_mb=81920, gpu_utilization_pct=70),
        GPUSnapshot(timestamp=0.5, vram_used_mb=vram_used + 2000, vram_total_mb=81920, gpu_utilization_pct=85),
    ]
    return s


# ── BenchmarkConfig ───────────────────────────────────────────────────────────

class TestBenchmarkConfig:
    def test_defaults(self):
        cfg = BenchmarkConfig(model="google/gemma-4-E2B-it")
        assert cfg.base_url == "http://localhost:8000/v1"
        assert cfg.api_key == "EMPTY"
        assert cfg.concurrency == 8
        assert cfg.num_requests == 40
        assert len(cfg.prompts) > 0

    def test_custom_overrides(self):
        cfg = BenchmarkConfig(model="m", concurrency=16, num_requests=100, max_tokens=512)
        assert cfg.concurrency == 16
        assert cfg.num_requests == 100
        assert cfg.max_tokens == 512

    def test_prompts_are_independent_across_instances(self):
        """Each instance should get its own prompts list, not share a mutable default."""
        cfg_a = BenchmarkConfig(model="a")
        cfg_b = BenchmarkConfig(model="b")
        cfg_a.prompts.append("extra")
        assert "extra" not in cfg_b.prompts


# ── RequestMetrics ────────────────────────────────────────────────────────────

class TestRequestMetrics:
    def test_success_fields(self):
        m = make_metric(ttft=0.05, tps=120.0)
        assert m.success is True
        assert m.error is None
        assert m.ttft == pytest.approx(0.05)

    def test_failure_fields(self):
        m = make_metric(success=False)
        assert m.success is False
        assert m.error is not None


# ── GPUSummary ────────────────────────────────────────────────────────────────

class TestGPUSummary:
    def test_peak_vram(self):
        s = make_gpu_summary(vram_used=10000)
        assert s.peak_vram_mb == 12000

    def test_mean_vram(self):
        s = make_gpu_summary(vram_used=10000)
        assert s.mean_vram_mb == pytest.approx(11000.0)

    def test_mean_gpu_utilization(self):
        s = make_gpu_summary()
        assert s.mean_gpu_utilization == pytest.approx(77.5)

    def test_empty_summary_returns_zeros(self):
        s = GPUSummary()
        assert s.peak_vram_mb == 0
        assert s.mean_vram_mb == 0.0
        assert s.mean_gpu_utilization == 0.0

    def test_to_dict_has_expected_keys(self):
        keys = make_gpu_summary().to_dict().keys()
        assert keys == {"mean_vram_mb", "peak_vram_mb", "mean_gpu_utilization_pct", "num_snapshots"}


# ── BenchmarkResults ──────────────────────────────────────────────────────────

class TestBenchmarkResults:
    def _make_results(self, n_success: int = 5, n_fail: int = 0) -> BenchmarkResults:
        cfg = make_config()
        metrics = [make_metric(ttft=0.1 * (i + 1)) for i in range(n_success)]
        metrics += [make_metric(success=False) for _ in range(n_fail)]
        return BenchmarkResults(
            model="test-model",
            metrics=metrics,
            gpu_summary=make_gpu_summary(),
            config=cfg,
            wall_time=3.0,
        )

    def test_summary_structure(self):
        s = self._make_results().summary()
        for key in ("ttft", "tps", "latency", "gpu", "throughput_tokens_per_s", "wall_time_s"):
            assert key in s, f"Missing key: {key}"

    def test_success_and_fail_counts(self):
        s = self._make_results(n_success=4, n_fail=1).summary()
        assert s["successful_requests"] == 4
        assert s["failed_requests"] == 1
        assert s["total_requests"] == 5

    def test_all_failed_returns_error_dict(self):
        s = self._make_results(n_success=0, n_fail=3).summary()
        assert "error" in s

    def test_throughput_is_positive(self):
        s = self._make_results().summary()
        assert s["throughput_tokens_per_s"] > 0
        assert s["throughput_req_per_s"] > 0

    def test_save_writes_valid_json(self, tmp_path):
        results = self._make_results()
        outfile = results.save(str(tmp_path))
        assert outfile.exists()
        data = json.loads(outfile.read_text())
        assert "summary" in data
        assert "per_request_metrics" in data
        assert "config" in data
        assert len(data["per_request_metrics"]) == 5

    def test_save_filename_uses_model_slug(self, tmp_path):
        results = self._make_results()
        outfile = results.save(str(tmp_path))
        assert "test" in outfile.name


# ── Benchmarker (mocked) ──────────────────────────────────────────────────────

class TestBenchmarker:
    @pytest.mark.asyncio
    async def test_run_returns_correct_request_count(self):
        cfg = make_config(num_requests=4, concurrency=2)
        mock_metric = make_metric()

        with (
            patch.object(InferenceProfiler, "profile_request", new_callable=AsyncMock, return_value=mock_metric),
            patch.object(InferenceProfiler, "close", new_callable=AsyncMock),
            patch.object(GPUMonitor, "_query_nvidia_smi", return_value=None),
        ):
            results = await Benchmarker(cfg).run()

        assert len(results.metrics) == 4
        assert all(m.success for m in results.metrics)

    @pytest.mark.asyncio
    async def test_concurrency_limit_is_respected(self):
        """Semaphore must prevent more than `concurrency` simultaneous requests."""
        cfg = make_config(num_requests=8, concurrency=3)
        active = 0
        max_active = 0

        async def fake_profile(prompt: str) -> RequestMetrics:
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0.01)
            active -= 1
            return make_metric()

        with (
            patch.object(InferenceProfiler, "profile_request", new_callable=AsyncMock, side_effect=fake_profile),
            patch.object(InferenceProfiler, "close", new_callable=AsyncMock),
            patch.object(GPUMonitor, "_query_nvidia_smi", return_value=None),
        ):
            await Benchmarker(cfg).run()

        assert max_active <= cfg.concurrency

    @pytest.mark.asyncio
    async def test_failed_requests_are_included_in_results(self):
        cfg = make_config(num_requests=4, concurrency=2)
        responses = [make_metric(success=True), make_metric(success=False)] * 2

        with (
            patch.object(InferenceProfiler, "profile_request", new_callable=AsyncMock, side_effect=responses),
            patch.object(InferenceProfiler, "close", new_callable=AsyncMock),
            patch.object(GPUMonitor, "_query_nvidia_smi", return_value=None),
        ):
            results = await Benchmarker(cfg).run()

        assert sum(1 for m in results.metrics if m.success) == 2
        assert sum(1 for m in results.metrics if not m.success) == 2
