from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from engine.adaptive_router import ExecutionPath
from schemas.meta_analysis import MetaAnalysis


@dataclass
class GatekeeperDecision:
    action: str
    reason: str = ""
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "reason": self.reason,
            "issues": self.issues,
            "warnings": self.warnings,
        }


class GatekeeperV2:
    def validate(
        self,
        result: dict,
        analysis: MetaAnalysis,
        path: ExecutionPath,
    ) -> GatekeeperDecision:
        issues: list[str] = []

        execution = result.get("execution", {})
        if not execution or execution.get("response") is None:
            issues.append("No execution output produced")

        tool_results = result.get("tool_results", [])
        failed_tools = [
            tool for tool in tool_results if tool.get("status") in {"rejected", "error", "truncated", "unknown_tool"}
        ]
        if tool_results and len(failed_tools) > len(tool_results) * 0.5:
            issues.append(f"{len(failed_tools)}/{len(tool_results)} tool calls failed")

        if analysis.memory_actions and not result.get("memory_updated"):
            issues.append(f"{len(analysis.memory_actions)} memory actions pending but not applied")

        if analysis.mental_health_update and not result.get("memory_updated"):
            issues.append("Mental health signal detected but memory was not updated")

        if analysis.knowledge_gaps:
            sub_results = result.get("sub_agent_results", [])
            addressed: set[str] = set()
            for item in sub_results:
                item_text = str(item).lower()
                for gap in analysis.knowledge_gaps:
                    if gap.lower() in item_text:
                        addressed.add(gap)
            unaddressed = [gap for gap in analysis.knowledge_gaps if gap not in addressed]
            if unaddressed and path.name != "deep":
                issues.append(f"Knowledge gaps unaddressed: {', '.join(unaddressed)}")

        drift = self._check_persona_drift(result.get("sub_agent_results", []))
        if drift:
            issues.append(f"Persona drift detected: {drift}")

        if analysis.analysis_confidence < 0.4:
            issues.append(f"Low analysis confidence ({analysis.analysis_confidence})")

        severity = len(issues)
        if severity == 0:
            return GatekeeperDecision(action="finalize")

        if severity >= 2 and path.name in {"fast", "standard"}:
            return GatekeeperDecision(
                action="escalate",
                reason="; ".join(issues),
                issues=issues,
            )

        return GatekeeperDecision(action="finalize_with_warnings", warnings=issues)

    def _check_persona_drift(self, sub_results: list) -> Optional[str]:
        drift_indicators = [
            "i cannot",
            "as an ai",
            "i don't have opinions",
            "i'm just a",
            "i apologize but",
        ]
        for item in sub_results:
            lowered = str(item).lower()
            for token in drift_indicators:
                if token in lowered:
                    return f"generic AI language found: '{token}'"
        return None
