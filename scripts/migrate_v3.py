from __future__ import annotations

import json
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.json"
DB_PATH = ROOT / "sessions.db"
MEMORY_DIR = ROOT / "memory"
SUMMARY_START = "<!-- SUMMARY_START -->"
SUMMARY_END = "<!-- SUMMARY_END -->"


def ensure_config() -> None:
    if not CONFIG_PATH.exists():
        return
    data = json.loads(CONFIG_PATH.read_text(encoding="utf-8-sig"))
    if "architecture_mode" in data:
        data.pop("architecture_mode", None)
        CONFIG_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print("[migrate_v3] Removed architecture_mode from config.json")


def ensure_summary_markers(path: Path) -> None:
    if not path.exists():
        return
    content = path.read_text(encoding="utf-8-sig")
    if SUMMARY_START in content and SUMMARY_END in content:
        return

    lines = [line.strip() for line in content.splitlines() if line.strip()][:10]
    summary = (
        f"{SUMMARY_START}\n"
        f"Preview: {' | '.join(lines[:3])}\n"
        f"{SUMMARY_END}\n\n"
    )
    path.write_text(summary + content, encoding="utf-8")
    print(f"[migrate_v3] Added summary block to {path.name}")


def ensure_memory_summaries() -> None:
    for name in ["identity.md", "soul.md", "user.md"]:
        ensure_summary_markers(MEMORY_DIR / name)


def ensure_db_columns() -> None:
    if not DB_PATH.exists():
        return

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(turns)")
    existing = {row[1] for row in cur.fetchall()}

    if "execution_path" not in existing:
        cur.execute("ALTER TABLE turns ADD COLUMN execution_path TEXT")
        print("[migrate_v3] Added turns.execution_path")

    if "metrics_json" not in existing:
        cur.execute("ALTER TABLE turns ADD COLUMN metrics_json TEXT")
        print("[migrate_v3] Added turns.metrics_json")

    conn.commit()
    conn.close()


def run() -> None:
    ensure_config()
    ensure_memory_summaries()
    ensure_db_columns()
    print("[migrate_v3] Migration complete")


if __name__ == "__main__":
    run()
