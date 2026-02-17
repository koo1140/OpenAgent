from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class FileMetadata:
    total_lines: int
    showing: str
    truncated: bool
    remaining_lines: int = 0
    navigation_hint: str = ""
    file_type: str = ""


@dataclass
class ReadResult:
    content: str
    metadata: FileMetadata


class WindowedFileReader:
    DEFAULT_WINDOW = 200

    def read(
        self,
        path: str,
        start_line: int = 0,
        end_line: Optional[int] = None,
        mode: str = "smart",
    ) -> ReadResult:
        file_path = Path(path)
        if not file_path.exists():
            return ReadResult(
                content=f"Error: File '{path}' not found.",
                metadata=FileMetadata(total_lines=0, showing="N/A", truncated=False),
            )
        if file_path.is_dir():
            return ReadResult(
                content=(
                    f"Tool validation error: read_file path '{path}' is a directory; "
                    "pass a file path"
                ),
                metadata=FileMetadata(total_lines=0, showing="N/A", truncated=False),
            )

        lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
        total = len(lines)
        ext = file_path.suffix.lower()

        resolved_mode = mode
        if mode == "smart":
            resolved_mode = self._auto_mode(ext, total, start_line)

        if resolved_mode == "tail":
            result = self._read_tail(lines, total)
        elif resolved_mode == "structure":
            result = self._read_structure(lines, total, ext)
        elif resolved_mode == "full" and total <= self.DEFAULT_WINDOW:
            result = ReadResult(
                content="\n".join(lines),
                metadata=FileMetadata(
                    total_lines=total,
                    showing=f"1-{total}",
                    truncated=False,
                    file_type=ext,
                ),
            )
        else:
            result = self._read_window(lines, total, start_line, end_line, ext)

        return result

    def render(self, result: ReadResult) -> str:
        meta = result.metadata
        header = (
            "[read_file] "
            f"{meta.showing}; total_lines={meta.total_lines}; "
            f"truncated={str(meta.truncated).lower()}"
        )
        if meta.navigation_hint:
            header += f"; hint={meta.navigation_hint}"
        if result.content:
            return f"{header}\n\n{result.content}"
        return header

    def _auto_mode(self, ext: str, total: int, start_line: int) -> str:
        if total <= self.DEFAULT_WINDOW:
            return "full"
        if ext in {".log", ".err", ".out"}:
            return "tail"
        if start_line == 0 and total > self.DEFAULT_WINDOW * 2:
            return "structure"
        return "window"

    def _read_window(
        self,
        lines: list[str],
        total: int,
        start: int,
        end: Optional[int],
        ext: str,
    ) -> ReadResult:
        safe_start = max(start, 0)
        safe_end = min(end if end is not None else safe_start + self.DEFAULT_WINDOW, total)
        if safe_end < safe_start:
            safe_end = safe_start

        selected = lines[safe_start:safe_end]
        numbered = [f"{safe_start + i + 1:>6} | {line}" for i, line in enumerate(selected)]

        return ReadResult(
            content="\n".join(numbered),
            metadata=FileMetadata(
                total_lines=total,
                showing=f"lines {safe_start + 1}-{safe_end} of {total}",
                truncated=(safe_start > 0 or safe_end < total),
                remaining_lines=max(total - safe_end, 0),
                navigation_hint=self._hint(safe_start, safe_end, total),
                file_type=ext,
            ),
        )

    def _read_tail(self, lines: list[str], total: int) -> ReadResult:
        start = max(total - self.DEFAULT_WINDOW, 0)
        selected = lines[start:]
        numbered = [f"{start + i + 1:>6} | {line}" for i, line in enumerate(selected)]
        hint = (
            "Showing last window; use mode='window' with start_line=0 for beginning."
            if start > 0
            else "Showing complete file."
        )
        return ReadResult(
            content="\n".join(numbered),
            metadata=FileMetadata(
                total_lines=total,
                showing=f"lines {start + 1}-{total} (TAIL)",
                truncated=start > 0,
                navigation_hint=hint,
            ),
        )

    def _read_structure(self, lines: list[str], total: int, ext: str) -> ReadResult:
        patterns = {
            ".py": r"^\s*(class |def |async def |@\w+|import |from )",
            ".js": r"^\s*(export |const |let |var |function |class |import )",
            ".ts": r"^\s*(export |const |let |var |function |class |interface |type |import )",
            ".tsx": r"^\s*(export |const |let |var |function |class |interface |type |import )",
            ".go": r"^\s*(func |type |package |import )",
            ".rs": r"^\s*(fn |struct |enum |impl |mod |use |pub )",
            ".java": r"^\s*(public |private |protected |class |interface |import )",
            ".json": r'^\s*[{}\[\]]|^\s*"[^"]+":',
            ".yaml": r"^[a-zA-Z_]",
            ".yml": r"^[a-zA-Z_]",
            ".md": r"^#{1,3} ",
            ".toml": r"^\[",
        }
        regex = re.compile(patterns.get(ext, r"^[a-zA-Z_]"))

        landmarks = [f"L{i + 1:>5} | {line.rstrip()}" for i, line in enumerate(lines) if regex.match(line)]
        if not landmarks:
            first = [f"L{i + 1:>5} | {line.rstrip()}" for i, line in enumerate(lines[:20])]
            tail_start = max(total - 20, 0)
            last = [
                f"L{tail_start + i + 1:>5} | {line.rstrip()}"
                for i, line in enumerate(lines[tail_start:])
            ]
            omitted = max(total - len(first) - len(last), 0)
            landmarks = first + [f"... ({omitted} lines omitted) ..."] + last

        return ReadResult(
            content="\n".join(landmarks),
            metadata=FileMetadata(
                total_lines=total,
                showing=f"structural summary ({len(landmarks)} landmarks)",
                truncated=True,
                remaining_lines=total,
                navigation_hint=(
                    "Use mode='window' with start_line=N for specific ranges; "
                    "use mode='tail' for file end."
                ),
                file_type=ext,
            ),
        )

    def _hint(self, start: int, end: int, total: int) -> str:
        hints: list[str] = []
        if start > 0:
            hints.append(f"Previous: start_line={max(start - self.DEFAULT_WINDOW, 0)}")
        if end < total:
            hints.append(f"Next: start_line={end}")
            hints.append("Jump to end: mode='tail'")
            hints.append("Overview: mode='structure'")
        else:
            hints.append("End of file reached")
        return " | ".join(hints)
