from __future__ import annotations

from dataclasses import dataclass, field

from schemas.meta_analysis import MetaAnalysis


@dataclass
class ValidationResult:
    passed: bool
    reinvoke_prompt: str = ""
    warnings: list[str] = field(default_factory=list)


class PipelineCheckpoint:
    REQUIREMENTS = {
        "planner": {
            "critical": ["user_intent", "plan"],
            "important": ["knowledge_needed", "knowledge_gaps"],
            "optional": ["aha_moments", "learning_opportunities"],
        },
        "executor": {
            "critical": ["user_intent", "plan"],
            "important": ["knowledge_needed"],
            "optional": ["known_data"],
        },
        "orchestrator": {
            "critical": ["user_intent", "plan", "sub_agents_needed"],
            "important": ["knowledge_needed", "knowledge_gaps"],
            "optional": ["aha_moments"],
        },
        "gatekeeper": {
            "critical": ["user_intent"],
            "important": ["memory_actions", "mental_health_update"],
            "optional": ["tone_shift_detected"],
        },
    }

    def validate(self, source: MetaAnalysis, target: str) -> ValidationResult:
        reqs = self.REQUIREMENTS.get(target)
        if reqs is None:
            return ValidationResult(passed=True)

        missing_critical: list[str] = []
        missing_important: list[str] = []

        for field_name in reqs["critical"]:
            if self._is_empty(getattr(source, field_name, None)):
                missing_critical.append(field_name)

        for field_name in reqs["important"]:
            if self._is_empty(getattr(source, field_name, None)):
                missing_important.append(field_name)

        if missing_critical:
            return ValidationResult(
                passed=False,
                reinvoke_prompt=self._build_prompt(missing_critical, missing_important, target),
                warnings=[f"Critical fields missing for {target}: {missing_critical}"],
            )

        if missing_important and len(missing_important) >= len(reqs["important"]):
            return ValidationResult(
                passed=False,
                reinvoke_prompt=self._build_prompt([], missing_important, target),
                warnings=[f"Important fields all empty for {target}: {missing_important}"],
            )

        warnings: list[str] = []
        if missing_important:
            warnings.append(f"Important fields missing for {target}: {missing_important}")

        return ValidationResult(passed=True, warnings=warnings)

    def _is_empty(self, value) -> bool:
        if value is None:
            return True
        if isinstance(value, str):
            return value.strip().lower() in {"", "none", "n/a", "null"}
        if isinstance(value, list):
            return len(value) == 0
        if isinstance(value, bool):
            return False
        return False

    def _build_prompt(self, critical: list[str], important: list[str], target: str) -> str:
        lines = [
            "Your previous analysis was incomplete.",
            f"The {target} layer requires these fields:",
        ]
        if critical:
            lines.append(f"CRITICAL: {', '.join(critical)}")
        if important:
            lines.append(f"IMPORTANT: {', '.join(important)}")
        lines.append("Re-analyze and provide specific values. Avoid 'none' and 'n/a'.")
        return "\n".join(lines)
