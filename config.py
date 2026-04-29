from __future__ import annotations

from dataclasses import dataclass, field

DEFAULT_PROMPTS: list[str] = [
    "Analyze the sentiment of this financial headline: "
    "'Tech stocks surge as Fed signals rate cuts ahead.' "
    "Reply with Positive, Negative, or Neutral and a one-sentence justification.",
    "Analyze the sentiment of this financial headline: "
    "'Major bank reports record losses amid rising loan defaults.' "
    "Reply with Positive, Negative, or Neutral and a one-sentence justification.",
    "Analyze the sentiment of this financial headline: "
    "'Inflation cools to 2-year low, equity markets rally strongly.' "
    "Reply with Positive, Negative, or Neutral and a one-sentence justification.",
    "Analyze the sentiment of this financial headline: "
    "'Earnings miss expectations by 20%, shares plunge after hours.' "
    "Reply with Positive, Negative, or Neutral and a one-sentence justification.",
    "Analyze the sentiment of this financial headline: "
    "'Central bank raises rates 50bps to combat persistent inflation.' "
    "Reply with Positive, Negative, or Neutral and a one-sentence justification.",
]


@dataclass
class BenchmarkConfig:
    model: str
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"
    max_tokens: int = 150
    concurrency: int = 8
    num_requests: int = 40
    temperature: float = 0.0
    gpu_poll_interval: float = 0.5
    output_dir: str = "results"
    prompts: list[str] = field(default_factory=lambda: list(DEFAULT_PROMPTS))
