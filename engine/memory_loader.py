from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from middleware.observability import TokenCounter


class MemoryLoadingStrategy(Enum):
    NONE = "none"
    SKELETON = "skeleton"
    SELECTIVE = "selective"
    FULL = "full"


@dataclass
class MemoryContext:
    content: str
    token_count: int
    strategy_used: str
    identity_skeleton: str
    soul_principles: list[str]
    user_relevant_chunks: str


class SemanticMemoryRetriever:
    def __init__(self, chunk_size: int = 400, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def retrieve_relevant_chunks(self, content: str, query: str, max_chunks: int = 3) -> list[str]:
        chunks = self._split_by_sections(content)
        if not chunks:
            return []

        query_terms = {term.strip().lower() for term in query.split() if term.strip()}
        if not query_terms:
            return chunks[:max_chunks]

        scored: list[tuple[int, int, str]] = []
        for idx, chunk in enumerate(chunks):
            words = {term.strip().lower() for term in chunk.split() if term.strip()}
            overlap_score = len(words & query_terms)
            recency_bonus = idx
            scored.append((overlap_score, recency_bonus, chunk))

        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        selected = [chunk for overlap, _, chunk in scored if overlap > 0][:max_chunks]
        return selected

    def _split_by_sections(self, content: str) -> list[str]:
        sections = re.split(r"\n(?=## )", content)
        chunks: list[str] = []
        for section in sections:
            section = section.strip()
            if not section:
                continue
            words = section.split()
            if len(words) <= self.chunk_size:
                chunks.append(section)
                continue
            step = max(self.chunk_size - self.overlap, 1)
            for index in range(0, len(words), step):
                chunk = " ".join(words[index : index + self.chunk_size]).strip()
                if chunk:
                    chunks.append(chunk)
        return chunks


class MemoryLoader:
    MEMORY_DIR = Path(__file__).resolve().parents[1] / "memory"
    SUMMARY_PATTERN = re.compile(r"<!-- SUMMARY_START -->(.*?)<!-- SUMMARY_END -->", flags=re.DOTALL)

    def __init__(self) -> None:
        self.retriever = SemanticMemoryRetriever()

    def load(
        self,
        strategy: MemoryLoadingStrategy,
        analysis: Any = None,
        context_budget_tokens: int = 4000,
    ) -> MemoryContext:
        files = {
            "identity": self._read_file("identity.md"),
            "soul": self._read_file("soul.md"),
            "user": self._read_file("user.md"),
        }

        if strategy == MemoryLoadingStrategy.NONE:
            return self._build_context("", files, strategy)
        if strategy == MemoryLoadingStrategy.SKELETON:
            content = self._extract_skeletons(files)
            return self._build_context(content, files, strategy)
        if strategy == MemoryLoadingStrategy.SELECTIVE:
            content = self._selective_load(files, analysis, context_budget_tokens)
            return self._build_context(content, files, strategy)

        content = self._full_load(files, context_budget_tokens)
        return self._build_context(content, files, strategy)

    def apply_memory_actions(self, actions: list[Any]) -> bool:
        if not actions:
            return False

        user_path = self.MEMORY_DIR / "user.md"
        current = user_path.read_text(encoding="utf-8") if user_path.exists() else ""

        bullet_lines: list[str] = []
        for action in actions:
            normalized = self._normalize_action(action)
            if normalized:
                bullet_lines.append(f"- {normalized}")

        if not bullet_lines:
            return False

        update_block = (
            f"\n\n## Session Update ({datetime.now().strftime('%Y-%m-%d %H:%M')})\n"
            + "\n".join(bullet_lines)
        )
        updated = current + update_block
        updated = self._regenerate_summary(updated)

        self.MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        user_path.write_text(updated, encoding="utf-8")
        return True

    def _extract_skeletons(self, files: dict[str, str]) -> str:
        parts: list[str] = []
        for name, content in files.items():
            skeleton = self._extract_skeleton_single(content)
            label = "Summary" if self._summary_exists(content) else "Preview"
            parts.append(f"## {name.title()} ({label})\n{skeleton}")
        return "\n\n".join(parts)

    def _selective_load(self, files: dict[str, str], analysis: Any, budget: int) -> str:
        parts: list[str] = []
        identity_summary = self._extract_skeleton_single(files.get("identity", ""))
        soul_summary = self._extract_skeleton_single(files.get("soul", ""))
        parts.append(f"## Identity\n{identity_summary}")
        parts.append(f"## Soul\n{soul_summary}")

        user_chunks: list[str] = []
        if analysis is not None:
            query_bits = [
                str(getattr(analysis, "user_intent", "") or ""),
                str(getattr(analysis, "chat_subject", "") or ""),
                " ".join(getattr(analysis, "knowledge_needed", []) or []),
            ]
            query = " ".join(bit for bit in query_bits if bit).strip()
            user_chunks = self.retriever.retrieve_relevant_chunks(
                files.get("user", ""), query, max_chunks=3
            )

        if user_chunks:
            parts.append("## User Context (Relevant)\n" + "\n---\n".join(user_chunks))

        assembled = "\n\n".join(parts)
        token_count = TokenCounter.count(assembled)
        while token_count > budget and user_chunks:
            user_chunks.pop()
            if user_chunks:
                parts[-1] = "## User Context (Relevant)\n" + "\n---\n".join(user_chunks)
            else:
                parts = parts[:2]
            assembled = "\n\n".join(parts)
            token_count = TokenCounter.count(assembled)

        return assembled

    def _full_load(self, files: dict[str, str], budget: int) -> str:
        assembled = "\n\n".join([f"## {name.title()}\n{content}" for name, content in files.items()])
        token_count = TokenCounter.count(assembled)
        if token_count <= budget:
            return assembled

        identity_and_soul = (
            f"## Identity\n{files.get('identity', '')}\n\n"
            f"## Soul\n{files.get('soul', '')}"
        )
        remaining = max(budget - TokenCounter.count(identity_and_soul), 0)
        user_content = files.get("user", "")

        if remaining < 200:
            user_trimmed = self._extract_skeleton_single(user_content)
            return f"{identity_and_soul}\n\n## User\n{user_trimmed}"

        lines = user_content.splitlines()
        selected: list[str] = []
        running = 0
        for line in lines:
            line_tokens = TokenCounter.count(line)
            if running + line_tokens > remaining:
                omitted = max(len(lines) - len(selected), 0)
                selected.append(f"[... user.md truncated; {omitted} lines omitted ...]")
                break
            selected.append(line)
            running += line_tokens

        return f"{identity_and_soul}\n\n## User\n" + "\n".join(selected)

    def _build_context(
        self,
        content: str,
        files: dict[str, str],
        strategy: MemoryLoadingStrategy,
    ) -> MemoryContext:
        return MemoryContext(
            content=content,
            token_count=TokenCounter.count(content) if content else 0,
            strategy_used=strategy.value,
            identity_skeleton=self._extract_skeleton_single(files.get("identity", "")),
            soul_principles=self._extract_principles(files.get("soul", "")),
            user_relevant_chunks=content,
        )

    def _summary_exists(self, content: str) -> bool:
        return bool(self.SUMMARY_PATTERN.search(content or ""))

    def _extract_skeleton_single(self, content: str) -> str:
        if not content:
            return ""
        match = self.SUMMARY_PATTERN.search(content)
        if match:
            return match.group(1).strip()
        return "\n".join(content.strip().splitlines()[:10])

    def _extract_principles(self, soul_content: str) -> list[str]:
        principles: list[str] = []
        for line in soul_content.splitlines():
            stripped = line.strip()
            if stripped.startswith(("- ", "* ")):
                principles.append(stripped.lstrip("-* ").strip())
        return principles[:10]

    def _normalize_action(self, action: Any) -> str:
        if isinstance(action, str):
            return action.strip()
        if isinstance(action, dict):
            if action.get("content"):
                return str(action.get("content")).strip()
            if action.get("reason"):
                return str(action.get("reason")).strip()
            return json_safe_dump(action)
        return str(action).strip()

    def _regenerate_summary(self, full_content: str) -> str:
        headers = [line.strip() for line in full_content.splitlines() if line.strip().startswith("## ")]
        recent_bullets = [line for line in full_content.splitlines()[-30:] if line.strip().startswith("-")]
        summary = (
            "<!-- SUMMARY_START -->\n"
            f"Sections: {', '.join(header.lstrip('# ').strip() for header in headers[:10])}\n"
            f"Recent updates: {len(recent_bullets)}\n"
            "<!-- SUMMARY_END -->"
        )
        if self._summary_exists(full_content):
            return self.SUMMARY_PATTERN.sub(summary, full_content)
        return summary + "\n\n" + full_content

    def _read_file(self, filename: str) -> str:
        path = self.MEMORY_DIR / filename
        if path.exists():
            return path.read_text(encoding="utf-8")
        return ""


def json_safe_dump(value: Any) -> str:
    try:
        raw = str(value) if not isinstance(value, (dict, list)) else json.dumps(value, ensure_ascii=False)
        return re.sub(r"\s+", " ", raw).strip()
    except Exception:
        return str(value)
