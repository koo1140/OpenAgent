from __future__ import annotations

import json
import re
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class UserTone(str, Enum):
    FRUSTRATED = "frustrated"
    CURIOUS = "curious"
    ENTHUSIASTIC = "enthusiastic"
    NEUTRAL = "neutral"
    STRESSED = "stressed"
    CONFUSED = "confused"


class ComplexityScore(int, Enum):
    TRIVIAL = 1
    SIMPLE = 2
    MODERATE = 3
    COMPLEX = 4
    CRITICAL = 5


class SubAgentSpec(BaseModel):
    role: str = Field(..., description="e.g. backend_engineer")
    task: str = Field(..., description="Specific task description")
    tools_required: list[str] = Field(default_factory=list)

    @field_validator("tools_required", mode="before")
    @classmethod
    def _coerce_tools(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            parts = [part.strip() for part in value.split(",") if part.strip()]
            return parts
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        return []


class MetaAnalysis(BaseModel):
    # User understanding
    user_intent: str = Field("Respond to the user request", min_length=3)
    chat_subject: str = Field("general", min_length=1)
    user_tone: UserTone = Field(default=UserTone.NEUTRAL)

    # Information state
    known_data: list[str] = Field(default_factory=list)
    missing_data: list[str] = Field(default_factory=list)

    # Knowledge assessment
    knowledge_needed: list[str] = Field(default_factory=list)
    knowledge_gaps: list[str] = Field(default_factory=list)

    # Learning and insight
    learning_opportunities: list[str] = Field(default_factory=list)
    aha_moments: list[str] = Field(default_factory=list)

    # Memory directives
    memory_actions: list[str] = Field(default_factory=list)

    # Strategy and delegation
    plan: list[str] = Field(default_factory=lambda: ["Respond to user query directly"])
    sub_agents_needed: list[SubAgentSpec] = Field(default_factory=list)

    # Adaptive routing fields
    complexity_score: ComplexityScore = Field(default=ComplexityScore.MODERATE)
    keep_memory_loaded: bool = Field(default=True)
    mental_health_update: Optional[str] = Field(default=None)
    tone_shift_detected: bool = Field(default=False)
    analysis_confidence: float = Field(default=0.8, ge=0.0, le=1.0)

    @field_validator(
        "known_data",
        "missing_data",
        "knowledge_needed",
        "knowledge_gaps",
        "learning_opportunities",
        "aha_moments",
        "memory_actions",
        "plan",
        mode="before",
    )
    @classmethod
    def _coerce_list_of_text(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            text = value.strip()
            return [text] if text else []
        if isinstance(value, list):
            result: list[str] = []
            for item in value:
                if isinstance(item, dict):
                    if "content" in item:
                        text = str(item.get("content", "")).strip()
                    elif "summary" in item:
                        text = str(item.get("summary", "")).strip()
                    else:
                        text = json.dumps(item, ensure_ascii=False)
                else:
                    text = str(item).strip()
                if text:
                    result.append(text)
            return result
        return [str(value).strip()] if str(value).strip() else []

    @field_validator("user_intent", "chat_subject", mode="before")
    @classmethod
    def _coerce_non_empty_text(cls, value: Any) -> str:
        text = str(value or "").strip()
        return text or "unknown"

    @field_validator("user_tone", mode="before")
    @classmethod
    def _coerce_user_tone(cls, value: Any) -> UserTone:
        if isinstance(value, UserTone):
            return value
        text = str(value or "").strip().lower()
        mapping = {
            "calm": UserTone.NEUTRAL,
            "positive": UserTone.ENTHUSIASTIC,
            "excited": UserTone.ENTHUSIASTIC,
            "anxious": UserTone.STRESSED,
            "upset": UserTone.FRUSTRATED,
        }
        if text in mapping:
            return mapping[text]
        for tone in UserTone:
            if tone.value == text:
                return tone
        return UserTone.NEUTRAL

    @field_validator("complexity_score", mode="before")
    @classmethod
    def _coerce_complexity(cls, value: Any) -> ComplexityScore:
        if isinstance(value, ComplexityScore):
            return value
        if isinstance(value, int):
            bounded = min(max(value, 1), 5)
            return ComplexityScore(bounded)
        text = str(value or "").strip().lower()
        label_map = {
            "trivial": 1,
            "simple": 2,
            "moderate": 3,
            "complex": 4,
            "critical": 5,
        }
        if text in label_map:
            return ComplexityScore(label_map[text])
        if text.isdigit():
            bounded = min(max(int(text), 1), 5)
            return ComplexityScore(bounded)
        return ComplexityScore.MODERATE

    @field_validator("analysis_confidence", mode="before")
    @classmethod
    def _coerce_confidence(cls, value: Any) -> float:
        try:
            number = float(value)
        except Exception:
            return 0.8
        return min(max(number, 0.0), 1.0)

    @model_validator(mode="before")
    @classmethod
    def _map_legacy_payload(cls, raw: Any) -> Any:
        if not isinstance(raw, dict):
            return raw

        payload = dict(raw)

        # Map current legacy/meta output keys into v3 schema.
        intent_obj = payload.get("intent")
        subject_obj = payload.get("subject")
        tone_obj = payload.get("tone")
        needs_obj = payload.get("needs")
        user_obj = payload.get("user")
        patterns_obj = payload.get("patterns")
        plan_obj = payload.get("plan")

        if "user_intent" not in payload and isinstance(intent_obj, dict):
            payload["user_intent"] = (
                intent_obj.get("user_intent")
                or intent_obj.get("intent")
                or intent_obj.get("summary")
            )
        if "chat_subject" not in payload and isinstance(subject_obj, dict):
            payload["chat_subject"] = (
                subject_obj.get("chat_subject")
                or subject_obj.get("subject")
                or subject_obj.get("summary")
            )
        if "user_tone" not in payload and isinstance(tone_obj, dict):
            payload["user_tone"] = (
                tone_obj.get("user_tone")
                or tone_obj.get("tone")
                or tone_obj.get("summary")
            )

        if isinstance(needs_obj, dict):
            if "knowledge_needed" not in payload:
                payload["knowledge_needed"] = (
                    needs_obj.get("knowledge_needed")
                    or needs_obj.get("skills_needed")
                    or []
                )
            if "knowledge_gaps" not in payload:
                payload["knowledge_gaps"] = (
                    needs_obj.get("knowledge_missing")
                    or needs_obj.get("knowledge_gaps")
                    or []
                )

        if isinstance(user_obj, dict):
            if "known_data" not in payload:
                known = user_obj.get("known_data") or user_obj.get("facts") or user_obj.get("summary")
                payload["known_data"] = known
            if "missing_data" not in payload:
                missing = user_obj.get("missing_data") or user_obj.get("unknowns")
                payload["missing_data"] = missing

        if isinstance(patterns_obj, dict):
            if "learning_opportunities" not in payload:
                payload["learning_opportunities"] = patterns_obj.get("learning_opportunities")
            if "aha_moments" not in payload:
                payload["aha_moments"] = patterns_obj.get("aha_moments")

        if "keep_memory_loaded" not in payload:
            if "keepMemoryFilesLoaded" in payload:
                payload["keep_memory_loaded"] = payload.get("keepMemoryFilesLoaded")
            elif isinstance(plan_obj, dict) and "load_memories" in plan_obj:
                payload["keep_memory_loaded"] = bool(plan_obj.get("load_memories"))

        if "tone_shift_detected" not in payload and "tone_changes_noticed" in payload:
            payload["tone_shift_detected"] = bool(payload.get("tone_changes_noticed"))

        if "plan" not in payload:
            recommended = payload.get("recommended_plan")
            if recommended:
                payload["plan"] = [str(recommended)]
        elif isinstance(plan_obj, dict):
            steps = plan_obj.get("steps") or plan_obj.get("plan")
            if isinstance(steps, str):
                payload["plan"] = [steps]
            elif isinstance(steps, list):
                payload["plan"] = steps
            else:
                payload["plan"] = [json.dumps(plan_obj, ensure_ascii=False)]

        if "sub_agents_needed" not in payload:
            suggestions = payload.get("subagent_suggestions") or payload.get("sub_agents_needed")
            if isinstance(suggestions, list):
                converted = []
                for item in suggestions:
                    if not isinstance(item, dict):
                        continue
                    converted.append(
                        {
                            "role": item.get("role", "subagent"),
                            "task": item.get("task") or item.get("description") or "",
                            "tools_required": item.get("tools") or item.get("tools_required") or [],
                        }
                    )
                payload["sub_agents_needed"] = converted

        if "memory_actions" not in payload and isinstance(plan_obj, dict):
            payload["memory_actions"] = plan_obj.get("memory_actions", [])

        if "complexity_score" not in payload:
            payload["complexity_score"] = _compute_complexity_from_payload(payload)

        return payload

    @model_validator(mode="after")
    def _guarantee_minimums(self) -> "MetaAnalysis":
        if not self.plan:
            self.plan = ["Respond to user query directly"]
        if len(self.user_intent.strip()) < 3:
            self.user_intent = "Respond to the user request"
        if len(self.chat_subject.strip()) < 1:
            self.chat_subject = "general"
        return self


def _compute_complexity_from_payload(payload: dict[str, Any]) -> int:
    score = 1
    gaps = payload.get("knowledge_gaps") or []
    sub_agents = payload.get("sub_agents_needed") or payload.get("subagent_suggestions") or []
    tone_shift = bool(payload.get("tone_shift_detected") or payload.get("tone_changes_noticed"))
    plan = payload.get("plan") or []

    if gaps:
        score += 1
    if sub_agents:
        score += 1
    if tone_shift:
        score += 1
    if isinstance(plan, list) and len(plan) > 3:
        score += 1
    if isinstance(plan, dict):
        score += 1

    return min(score, 5)


class MetaAnalysisParser:
    @staticmethod
    def parse(raw_json: Any) -> tuple[MetaAnalysis, list[str]]:
        warnings: list[str] = []

        parsed: Any = raw_json
        if isinstance(raw_json, str):
            parsed = MetaAnalysisParser._try_parse_text(raw_json, warnings)
            if parsed is None:
                warnings.append("CRITICAL: Layer 1 returned non-JSON output")
                return MetaAnalysisParser._create_fallback(), warnings

        if not isinstance(parsed, dict):
            warnings.append("Layer 1 output is not a JSON object")
            return MetaAnalysisParser._create_fallback(), warnings

        try:
            return MetaAnalysis.model_validate(parsed), warnings
        except Exception as exc:
            warnings.append(f"Schema validation failed: {exc}")

        # Partial salvage: keep only fields that individually validate.
        safe_fields: dict[str, Any] = {}
        for field_name in MetaAnalysis.model_fields.keys():
            if field_name not in parsed:
                continue
            attempt = {field_name: parsed[field_name]}
            if field_name not in {"user_intent", "chat_subject", "plan"}:
                attempt["user_intent"] = parsed.get("user_intent", "Respond to the user request")
                attempt["chat_subject"] = parsed.get("chat_subject", "general")
                attempt["plan"] = parsed.get("plan", ["Respond to user query directly"])
            try:
                MetaAnalysis.model_validate(attempt)
                safe_fields[field_name] = parsed[field_name]
            except Exception:
                warnings.append(f"Field '{field_name}' invalid, using default")

        try:
            merged = {
                "user_intent": safe_fields.get("user_intent", "Unable to parse intent"),
                "chat_subject": safe_fields.get("chat_subject", "general"),
                "plan": safe_fields.get("plan", ["Respond to user query directly"]),
            }
            merged.update({k: v for k, v in safe_fields.items() if k not in merged})
            return MetaAnalysis.model_validate(merged), warnings
        except Exception:
            warnings.append("Fallback analysis used after partial salvage failure")
            return MetaAnalysisParser._create_fallback(), warnings

    @staticmethod
    def _create_fallback() -> MetaAnalysis:
        return MetaAnalysis(
            user_intent="Parse failure: respond directly to user message",
            chat_subject="unknown",
            plan=["Respond directly without strategic planning"],
            complexity_score=ComplexityScore.SIMPLE,
            analysis_confidence=0.1,
            keep_memory_loaded=True,
        )

    @staticmethod
    def _try_parse_text(text: str, warnings: list[str]) -> Optional[dict[str, Any]]:
        stripped = text.strip()
        if not stripped:
            return None

        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        block = MetaAnalysisParser._extract_json_from_markdown(stripped)
        if block is not None:
            warnings.append("Recovered JSON from markdown code fence")
            return block

        candidate = MetaAnalysisParser._extract_json_candidate(stripped)
        if candidate is not None:
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    warnings.append("Recovered JSON from mixed text payload")
                    return parsed
            except json.JSONDecodeError:
                return None

        return None

    @staticmethod
    def _extract_json_from_markdown(text: str) -> Optional[dict[str, Any]]:
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
        if not match:
            return None
        candidate = match.group(1)
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None

    @staticmethod
    def _extract_json_candidate(text: str) -> Optional[str]:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        return text[start : end + 1]
