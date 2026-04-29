from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

from openai import AsyncOpenAI

from config import BenchmarkConfig




logger = logging.getLogger(__name__)


@dataclass
class RequestMetrics:
    ttft: float             # Time to First Token (seconds)
    tps: float              # Tokens Per Second
    total_latency: float    # Wall time for complete response (seconds)
    completion_tokens: int
    prompt: str
    success: bool
    error: Optional[str] = None


class InferenceProfiler:

    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config
        self.client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
        )



    async def profile_request(self, prompt: str) -> RequestMetrics:
        start_time = time.perf_counter()
        ttft: Optional[float] = None
        completion_tokens: int = 0

        try:
            stream = await self.client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                stream=True,
                stream_options={"include_usage": True},
            )

            async for chunk in stream:
                if (
                    ttft is None
                    and chunk.choices
                    and chunk.choices[0].delta.content
                ):
                    ttft = time.perf_counter() - start_time



                # vLLM emits usage in the final chunk when include_usage=True
                if chunk.usage and chunk.usage.completion_tokens:
                    completion_tokens = chunk.usage.completion_tokens






            total_latency = time.perf_counter() - start_time
            tps = completion_tokens / total_latency if total_latency > 0 else 0.0

            logger.debug(
                "Request done | TTFT=%.3fs | TPS=%.1f | tokens=%d",
                ttft or 0.0,
                tps,
                completion_tokens,
            )

            return RequestMetrics(
                ttft=ttft or total_latency,
                tps=tps,
                total_latency=total_latency,
                completion_tokens=completion_tokens,
                prompt=prompt,
                success=True,
            )

        except Exception as exc:
            logger.error("Request failed: %s", exc)
            return RequestMetrics(
                ttft=0.0,
                tps=0.0,
                total_latency=time.perf_counter() - start_time,
                completion_tokens=0,
                prompt=prompt,
                success=False,
                error=str(exc),
            )
            

    async def close(self) -> None:
        await self.client.close()
