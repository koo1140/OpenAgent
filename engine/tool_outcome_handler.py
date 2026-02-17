from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RetryAction:
    should_retry: bool
    strategy: str = ""
    attempts_remaining: int = 0
    fallback_suggestion: str = ""


class ToolOutcomeHandler:
    TERMINAL = {
        "denied": "User denied permission. Do not retry.",
        "not_found": "Resource does not exist at this path.",
        "unknown_tool": "Requested tool is unavailable.",
    }

    RECOVERABLE = {
        "rejected": {
            "max_retries": 2,
            "strategy": "Check tool name and parameters; correct malformed inputs.",
        },
        "timeout": {
            "max_retries": 1,
            "strategy": "Narrow command scope or split into smaller operations.",
        },
        "truncated": {
            "max_retries": 3,
            "strategy": "Use read_file mode='window' or mode='tail' with explicit ranges.",
        },
        "error": {
            "max_retries": 1,
            "strategy": "Adjust approach and retry once with safer parameters.",
        },
    }

    def __init__(self) -> None:
        self._attempts: dict[str, int] = {}

    def handle(self, tool_name: str, outcome: str, error_msg: str, remaining_budget: int) -> RetryAction:
        key = f"{tool_name}:{outcome}"
        self._attempts[key] = self._attempts.get(key, 0) + 1
        attempt = self._attempts[key]

        if outcome in self.TERMINAL:
            return RetryAction(should_retry=False, fallback_suggestion=self.TERMINAL[outcome])

        if remaining_budget <= 0:
            return RetryAction(
                should_retry=False,
                fallback_suggestion=f"Retry budget exhausted after outcome '{outcome}'.",
            )

        spec = self.RECOVERABLE.get(outcome)
        if spec and attempt <= spec["max_retries"]:
            return RetryAction(
                should_retry=True,
                strategy=spec["strategy"],
                attempts_remaining=max(spec["max_retries"] - attempt, 0),
            )

        return RetryAction(
            should_retry=False,
            fallback_suggestion=(
                f"Tool '{tool_name}' failed with '{outcome}' after {attempt} attempt(s): {error_msg}"
            ),
        )

    def reset(self) -> None:
        self._attempts.clear()
