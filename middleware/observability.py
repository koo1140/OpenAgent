from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Optional

try:
    import tiktoken  # type: ignore
except Exception:  # pragma: no cover
    tiktoken = None


@dataclass
class LayerMetrics:
    layer_name: str
    model_used: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    success: bool = True
    error: Optional[str] = None
    retries: int = 0


@dataclass
class PipelineMetrics:
    request_id: str
    start_time: float = field(default_factory=time.time)
    layers: list[LayerMetrics] = field(default_factory=list)
    execution_path: str = ""
    memory_tokens_used: int = 0
    total_tool_calls: int = 0
    sub_agents_spawned: int = 0
    escalations: int = 0

    @property
    def total_tokens(self) -> int:
        return sum(layer.input_tokens + layer.output_tokens for layer in self.layers)

    @property
    def total_latency_ms(self) -> float:
        return (time.time() - self.start_time) * 1000.0

    @property
    def estimated_cost_usd(self) -> float:
        pricing = {
            "gpt-4": (0.03, 0.06),
            "gpt-3.5": (0.001, 0.002),
            "claude-3-5-sonnet": (0.003, 0.015),
            "llama-3": (0.0, 0.0),
            "mixtral": (0.0, 0.0),
        }
        cost = 0.0
        for layer in self.layers:
            model = (layer.model_used or "").lower()
            model_key = next((key for key in pricing if key in model), "gpt-3.5")
            inp, out = pricing[model_key]
            cost += (layer.input_tokens / 1000.0 * inp) + (layer.output_tokens / 1000.0 * out)
        return round(cost, 6)

    def to_log_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "execution_path": self.execution_path,
            "total_tokens": self.total_tokens,
            "total_latency_ms": round(self.total_latency_ms, 1),
            "estimated_cost_usd": self.estimated_cost_usd,
            "memory_tokens": self.memory_tokens_used,
            "tool_calls": self.total_tool_calls,
            "sub_agents": self.sub_agents_spawned,
            "escalations": self.escalations,
            "layers": [
                {
                    "name": layer.layer_name,
                    "model": layer.model_used,
                    "tokens": layer.input_tokens + layer.output_tokens,
                    "latency_ms": round(layer.latency_ms, 1),
                    "success": layer.success,
                    "retries": layer.retries,
                    "error": layer.error,
                }
                for layer in self.layers
            ],
        }


class TokenCounter:
    _encoders: dict[str, Any] = {}

    @classmethod
    def _get_encoder(cls, model: str):
        model_key = model or "gpt-4"
        if model_key in cls._encoders:
            return cls._encoders[model_key]
        if tiktoken is None:
            cls._encoders[model_key] = None
            return None
        try:
            encoder = tiktoken.encoding_for_model(model_key)
        except Exception:
            encoder = tiktoken.get_encoding("cl100k_base")
        cls._encoders[model_key] = encoder
        return encoder

    @classmethod
    def count(cls, text: str, model: str = "gpt-4") -> int:
        if not text:
            return 0
        encoder = cls._get_encoder(model)
        if encoder is None:
            # Fallback heuristic when tiktoken is unavailable.
            return max(1, len(text) // 4)
        try:
            return len(encoder.encode(text))
        except Exception:
            return max(1, len(text) // 4)

    @classmethod
    def count_messages(cls, messages: list[dict[str, Any]], model: str = "gpt-4") -> int:
        total = 0
        for msg in messages:
            total += 4
            for _, value in msg.items():
                total += cls.count(str(value), model=model)
        return total


class PipelineTracker:
    def __init__(self, request_id: str):
        self.metrics = PipelineMetrics(request_id=request_id)

    @contextmanager
    def track_layer(self, layer_name: str, model: str):
        metric = LayerMetrics(layer_name=layer_name, model_used=model)
        start = time.time()
        try:
            yield metric
        except Exception as exc:
            metric.success = False
            metric.error = str(exc)
            raise
        finally:
            metric.latency_ms = (time.time() - start) * 1000.0
            self.metrics.layers.append(metric)

    def record_memory_tokens(self, token_count: int) -> None:
        self.metrics.memory_tokens_used = token_count

    def increment_tool_calls(self, amount: int = 1) -> None:
        self.metrics.total_tool_calls += max(amount, 0)

    def finalize(self) -> dict[str, Any]:
        return self.metrics.to_log_dict()
