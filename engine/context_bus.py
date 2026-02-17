from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


@dataclass
class Discovery:
    source_agent: str
    discovery_type: str
    content: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class SubAgentContext:
    persona_guidelines: str
    task_description: str
    tools_allowed: list[str]
    sibling_discoveries: list[Discovery]
    global_plan_summary: str


class ContextBus:
    def __init__(self) -> None:
        self._persona_prompt: str = ""
        self._discoveries: list[Discovery] = []
        self._plan_summary: str = ""
        self._lock = threading.Lock()

    def initialize_from_session(
        self,
        identity_summary: str,
        soul_principles: list[str],
        user_context: str,
        task_plan: dict | list | str,
    ) -> None:
        self._persona_prompt = self._build_persona(identity_summary, soul_principles, user_context)
        self._plan_summary = self._summarize_plan(task_plan)

    def get_sub_agent_context(self, task_description: str, tools_allowed: list[str]) -> SubAgentContext:
        with self._lock:
            return SubAgentContext(
                persona_guidelines=self._persona_prompt,
                task_description=task_description,
                tools_allowed=list(tools_allowed),
                sibling_discoveries=self._discoveries.copy(),
                global_plan_summary=self._plan_summary,
            )

    def report_discovery(self, source_agent: str, discovery_type: str, content: str) -> None:
        with self._lock:
            self._discoveries.append(
                Discovery(
                    source_agent=source_agent,
                    discovery_type=discovery_type,
                    content=content,
                )
            )

    def get_all_discoveries(self) -> list[Discovery]:
        with self._lock:
            return self._discoveries.copy()

    def _build_persona(self, identity: str, principles: list[str], user_ctx: str) -> str:
        principle_lines = "\n".join([f"- {item}" for item in principles]) if principles else "- Keep answers grounded and helpful"
        return (
            "## Identity\n"
            f"{identity}\n\n"
            "## Principles\n"
            f"{principle_lines}\n\n"
            "## User Context\n"
            f"{user_ctx}"
        )

    def _summarize_plan(self, plan: dict | list | str) -> str:
        if isinstance(plan, dict):
            steps = plan.get("steps") or plan.get("plan") or []
        elif isinstance(plan, list):
            steps = plan
        else:
            steps = [str(plan)]
        return "Plan: " + " -> ".join(str(step) for step in steps)
