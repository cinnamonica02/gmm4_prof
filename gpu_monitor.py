from __future__ import annotations




import asyncio
import logging
import subprocess
import time
from dataclasses import dataclass, field
from typing import Optional



logger = logging.getLogger(__name__)







@dataclass
class GPUSnapshot:
    timestamp: float
    vram_used_mb: int
    vram_total_mb: int
    gpu_utilization_pct: int






@dataclass
class GPUSummary:
    snapshots: list[GPUSnapshot] = field(default_factory=list)

    @property
    def mean_vram_mb(self) -> float:
        if not self.snapshots:
            return 0.0
        return sum(s.vram_used_mb for s in self.snapshots) / len(self.snapshots)

    @property
    def peak_vram_mb(self) -> int:
        if not self.snapshots:
            return 0
        return max(s.vram_used_mb for s in self.snapshots)

    @property
    def mean_gpu_utilization(self) -> float:
        if not self.snapshots:
            return 0.0
        return sum(s.gpu_utilization_pct for s in self.snapshots) / len(self.snapshots)

    def to_dict(self) -> dict:
        return {
            "mean_vram_mb": round(self.mean_vram_mb, 1),
            "peak_vram_mb": self.peak_vram_mb,
            "mean_gpu_utilization_pct": round(self.mean_gpu_utilization, 1),
            "num_snapshots": len(self.snapshots),
        }








class GPUMonitor:
    def __init__(self, poll_interval: float = 0.5) -> None:
        self.poll_interval = poll_interval
        self._task: Optional[asyncio.Task] = None
        self._running: bool = False
        self.summary = GPUSummary()

    @staticmethod
    def _query_nvidia_smi() -> Optional[GPUSnapshot]:
        try:
            raw = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used,memory.total,utilization.gpu",
                    "--format=csv,nounits,noheader",
                ],
                encoding="utf-8",
                timeout=2,
            )
            parts = [p.strip() for p in raw.strip().split(",")]
            return GPUSnapshot(
                timestamp=time.perf_counter(),
                vram_used_mb=int(parts[0]),
                vram_total_mb=int(parts[1]),
                gpu_utilization_pct=int(parts[2]),
            )
        except Exception as exc:
            logger.warning("nvidia-smi query failed: %s", exc)
            return None



    async def _poll_loop(self) -> None:
        loop = asyncio.get_event_loop()
        while self._running:
            snapshot = await loop.run_in_executor(None, self._query_nvidia_smi)
            if snapshot:
                self.summary.snapshots.append(snapshot)
            await asyncio.sleep(self.poll_interval)




    def start(self) -> None:
        self._running = True
        self.summary = GPUSummary()
        self._task = asyncio.get_event_loop().create_task(self._poll_loop())
        logger.debug("GPU monitor started (poll_interval=%.1fs)", self.poll_interval)




    async def stop(self) -> GPUSummary:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.debug("GPU monitor stopped | peak VRAM=%d MB", self.summary.peak_vram_mb)
        return self.summary
