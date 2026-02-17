from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from engine.memory_loader import MemoryLoadingStrategy
from schemas.meta_analysis import ComplexityScore, MetaAnalysis


@dataclass
class ExecutionPath:
    name: str
    layers: list[str]
    memory_strategy: MemoryLoadingStrategy
    retry_budget: int
    sub_agents_allowed: bool
    enable_context_bus: bool = False


class AdaptiveRouter:
    PATH_DEFINITIONS = {
        "fast": ExecutionPath(
            name="fast",
            layers=["meta", "executor"],
            memory_strategy=MemoryLoadingStrategy.SKELETON,
            retry_budget=1,
            sub_agents_allowed=False,
        ),
        "standard": ExecutionPath(
            name="standard",
            layers=["meta", "planner", "executor", "gatekeeper"],
            memory_strategy=MemoryLoadingStrategy.SELECTIVE,
            retry_budget=3,
            sub_agents_allowed=False,
        ),
        "deep": ExecutionPath(
            name="deep",
            layers=["meta", "planner", "executor", "orchestrator", "gatekeeper"],
            memory_strategy=MemoryLoadingStrategy.FULL,
            retry_budget=5,
            sub_agents_allowed=True,
            enable_context_bus=True,
        ),
    }

    def route(self, analysis: MetaAnalysis) -> ExecutionPath:
        if analysis.tone_shift_detected or analysis.mental_health_update:
            return self._escalate_for_safety(analysis.complexity_score)

        score = analysis.complexity_score
        if score.value <= ComplexityScore.SIMPLE.value:
            return self.PATH_DEFINITIONS["fast"]
        if score.value <= ComplexityScore.MODERATE.value:
            return self.PATH_DEFINITIONS["standard"]
        return self.PATH_DEFINITIONS["deep"]

    def escalate(self, current_path: ExecutionPath, reason: str = "") -> Optional[ExecutionPath]:
        del reason
        escalation_map = {
            "fast": "standard",
            "standard": "deep",
            "deep": None,
        }
        next_name = escalation_map.get(current_path.name)
        if next_name is None:
            return None
        return self.PATH_DEFINITIONS[next_name]

    def _escalate_for_safety(self, score: ComplexityScore) -> ExecutionPath:
        if score.value >= ComplexityScore.COMPLEX.value:
            return self.PATH_DEFINITIONS["deep"]
        return self.PATH_DEFINITIONS["standard"]
