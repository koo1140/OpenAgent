from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

from engine.adaptive_router import AdaptiveRouter, ExecutionPath
from engine.context_bus import ContextBus
from engine.gatekeeper import GatekeeperV2
from engine.memory_loader import MemoryLoader
from engine.pipeline_checkpoint import PipelineCheckpoint
from middleware.observability import PipelineTracker, TokenCounter
from schemas.meta_analysis import MetaAnalysis

PlannerCallback = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]
ExecutorCallback = Callable[[dict[str, Any], int], Awaitable[dict[str, Any]]]
OrchestratorCallback = Callable[[dict[str, Any], MetaAnalysis, int, Optional[ContextBus]], Awaitable[list[dict[str, Any]]]]
ReinvokeMetaCallback = Callable[[str, str], Awaitable[MetaAnalysis]]
StageCallback = Callable[[str], Awaitable[None]]
ModelLookup = Callable[[str], str]


@dataclass
class ExecutionDependencies:
    planner: Optional[PlannerCallback] = None
    executor: Optional[ExecutorCallback] = None
    orchestrator: Optional[OrchestratorCallback] = None
    reinvoke_meta: Optional[ReinvokeMetaCallback] = None
    queue_stage: Optional[StageCallback] = None
    model_lookup: Optional[ModelLookup] = None


class ExecutionEngine:
    def __init__(self) -> None:
        self.memory_loader = MemoryLoader()
        self.checkpoint = PipelineCheckpoint()
        self.router = AdaptiveRouter()
        self.gatekeeper = GatekeeperV2()
        self._max_escalations = 1

    async def execute(
        self,
        path: ExecutionPath,
        analysis: MetaAnalysis,
        message: str,
        tracker: PipelineTracker,
        warnings: list[str],
        deps: ExecutionDependencies,
        carry_forward: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        if deps.executor is None:
            raise ValueError("Execution callback is required")

        accumulated_warnings = list(warnings)
        escalation_count = 0
        current_path = path
        current_analysis = analysis
        carried = carry_forward

        while True:
            memory_context = self.memory_loader.load(
                strategy=current_path.memory_strategy,
                analysis=current_analysis,
            )
            tracker.record_memory_tokens(memory_context.token_count)

            context: dict[str, Any] = {
                "analysis": current_analysis,
                "message": message,
                "memory": memory_context,
                "carry_forward": carried,
                "tool_results": [],
                "tool_events": [],
                "sub_agent_results": [],
                "warnings": [],
            }
            if carried:
                context.update(carried)

            for layer_name in current_path.layers:
                if layer_name == "meta":
                    continue

                if deps.queue_stage is not None:
                    await deps.queue_stage(layer_name)

                if layer_name == "planner":
                    validation = self.checkpoint.validate(current_analysis, "planner")
                    accumulated_warnings.extend(validation.warnings)
                    if not validation.passed and deps.reinvoke_meta is not None:
                        current_analysis = await deps.reinvoke_meta(validation.reinvoke_prompt, message)
                        context["analysis"] = current_analysis

                    if deps.planner is not None:
                        planner_model = self._model_name(deps, "planner")
                        with tracker.track_layer("planner", planner_model) as metric:
                            metric.input_tokens = TokenCounter.count(message, planner_model)
                            plan_result = await deps.planner(context)
                            metric.output_tokens = TokenCounter.count(str(plan_result), planner_model)
                        context["plan"] = plan_result

                elif layer_name == "executor":
                    executor_model = self._model_name(deps, "main")
                    with tracker.track_layer("executor", executor_model) as metric:
                        metric.input_tokens = TokenCounter.count(message, executor_model)
                        execution_result = await deps.executor(context, current_path.retry_budget)
                        metric.output_tokens = TokenCounter.count(str(execution_result), executor_model)

                    if execution_result.get("status") == "needs_approval":
                        execution_result.setdefault("execution_path", current_path.name)
                        execution_result.setdefault("meta_analysis", current_analysis.model_dump())
                        execution_result.setdefault("warnings", accumulated_warnings)
                        return execution_result

                    context.update(execution_result)
                    tracker.increment_tool_calls(len(execution_result.get("tool_events", [])))

                elif layer_name == "orchestrator":
                    if not current_path.sub_agents_allowed:
                        continue
                    if not current_analysis.sub_agents_needed:
                        continue
                    if deps.orchestrator is None:
                        continue

                    bus: Optional[ContextBus] = None
                    if current_path.enable_context_bus:
                        bus = ContextBus()
                        bus.initialize_from_session(
                            identity_summary=memory_context.identity_skeleton,
                            soul_principles=memory_context.soul_principles,
                            user_context=memory_context.user_relevant_chunks,
                            task_plan=context.get("plan", {}),
                        )

                    orchestrator_model = self._model_name(deps, "orchestrator")
                    with tracker.track_layer("orchestrator", orchestrator_model) as metric:
                        metric.input_tokens = TokenCounter.count(str(context.get("plan", "")), orchestrator_model)
                        sub_results = await deps.orchestrator(context, current_analysis, current_path.retry_budget, bus)
                        metric.output_tokens = TokenCounter.count(str(sub_results), orchestrator_model)

                    context["sub_agent_results"] = sub_results
                    tracker.metrics.sub_agents_spawned = len(sub_results)

            if "gatekeeper" in current_path.layers:
                decision = self.gatekeeper.validate(
                    result={
                        "execution": context.get("execution", {}),
                        "tool_results": context.get("tool_results", []),
                        "sub_agent_results": context.get("sub_agent_results", []),
                        "memory_updated": False,
                    },
                    analysis=current_analysis,
                    path=current_path,
                )
                context["gatekeeper"] = decision.to_dict()

                if decision.action == "escalate":
                    if escalation_count >= self._max_escalations:
                        accumulated_warnings.extend(decision.issues)
                        return self._finalize(context, current_analysis, current_path, accumulated_warnings)
                    next_path = self.router.escalate(current_path, decision.reason)
                    if next_path is None:
                        accumulated_warnings.extend(decision.issues)
                        return self._finalize(context, current_analysis, current_path, accumulated_warnings)
                    current_path = next_path
                    escalation_count += 1
                    tracker.metrics.escalations += 1
                    carried = context
                    accumulated_warnings.extend(decision.issues)
                    continue

                if decision.action == "finalize_with_warnings":
                    accumulated_warnings.extend(decision.warnings)

            return self._finalize(context, current_analysis, current_path, accumulated_warnings)

    def _finalize(
        self,
        context: dict[str, Any],
        analysis: MetaAnalysis,
        path: ExecutionPath,
        warnings: list[str],
    ) -> dict[str, Any]:
        memory_updated = False
        if analysis.memory_actions:
            memory_updated = self.memory_loader.apply_memory_actions(analysis.memory_actions)

        meta = {
            "analysis": analysis.model_dump(),
            "plan": context.get("plan", {}),
            "gatekeeper": context.get("gatekeeper", {}),
            "warnings": warnings + context.get("warnings", []),
        }

        return {
            "reply": context.get("reply", ""),
            "status": context.get("status", "ok"),
            "tool_events": context.get("tool_events", []),
            "subagent_outputs": context.get("sub_agent_results", []),
            "meta": meta,
            "execution": context.get("execution", {}),
            "tool_results": context.get("tool_results", []),
            "memory_updated": memory_updated,
            "execution_path": path.name,
        }

    def _model_name(self, deps: ExecutionDependencies, layer: str) -> str:
        if deps.model_lookup is None:
            return "unknown"
        try:
            return deps.model_lookup(layer)
        except Exception:
            return "unknown"
