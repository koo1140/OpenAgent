from pathlib import Path

from engine.adaptive_router import AdaptiveRouter
from engine.file_reader import WindowedFileReader
from engine.memory_loader import MemoryLoader, MemoryLoadingStrategy
from engine.pipeline_checkpoint import PipelineCheckpoint
from schemas.meta_analysis import ComplexityScore, MetaAnalysisParser


def test_meta_analysis_parser_valid_json_object():
    raw = {
        "user_intent": "Implement a router",
        "chat_subject": "backend",
        "plan": ["analyze", "implement"],
    }
    analysis, warnings = MetaAnalysisParser.parse(raw)
    assert analysis.user_intent == "Implement a router"
    assert analysis.chat_subject == "backend"
    assert analysis.plan
    assert warnings == []


def test_meta_analysis_parser_recovers_markdown_json():
    raw = """before\n```json\n{\"user_intent\":\"Need help debugging\",\"chat_subject\":\"python\",\"plan\":[\"inspect logs\"]}\n```\nafter"""
    analysis, warnings = MetaAnalysisParser.parse(raw)
    assert analysis.chat_subject == "python"
    assert any("Recovered JSON" in warning for warning in warnings)


def test_meta_analysis_parser_fallback_on_non_json():
    analysis, warnings = MetaAnalysisParser.parse("not json at all")
    assert analysis.analysis_confidence <= 0.1
    assert any("CRITICAL" in warning for warning in warnings)


def test_router_selects_paths_by_complexity_and_safety():
    router = AdaptiveRouter()

    low, _ = MetaAnalysisParser.parse({"user_intent": "Quick answer", "chat_subject": "general", "plan": ["reply"], "complexity_score": 1})
    moderate, _ = MetaAnalysisParser.parse({"user_intent": "Explain and implement", "chat_subject": "code", "plan": ["analyze", "implement"], "complexity_score": 3})
    high, _ = MetaAnalysisParser.parse({"user_intent": "Refactor system", "chat_subject": "architecture", "plan": ["analyze"], "complexity_score": ComplexityScore.CRITICAL.value})
    safety, _ = MetaAnalysisParser.parse({"user_intent": "Need emotional support", "chat_subject": "wellbeing", "plan": ["respond"], "complexity_score": 2, "tone_shift_detected": True})

    assert router.route(low).name == "fast"
    assert router.route(moderate).name == "standard"
    assert router.route(high).name == "deep"
    assert router.route(safety).name in {"standard", "deep"}


def test_memory_loader_skeleton_and_selective(tmp_path):
    mem = tmp_path / "memory"
    mem.mkdir()
    (mem / "identity.md").write_text("<!-- SUMMARY_START -->\nidentity summary\n<!-- SUMMARY_END -->\n\nfull")
    (mem / "soul.md").write_text("<!-- SUMMARY_START -->\nsoul summary\n<!-- SUMMARY_END -->\n\nfull")
    (mem / "user.md").write_text("## Profile\nUser likes python and rust.\n## Recent\nAsks about fastapi and routers.")

    loader = MemoryLoader()
    loader.MEMORY_DIR = mem

    skeleton = loader.load(MemoryLoadingStrategy.SKELETON)
    assert "identity summary" in skeleton.content
    assert "soul summary" in skeleton.content

    analysis, _ = MetaAnalysisParser.parse(
        {
            "user_intent": "Improve FastAPI router",
            "chat_subject": "fastapi",
            "knowledge_needed": ["routing", "python"],
            "plan": ["inspect", "refactor"],
        }
    )
    selective = loader.load(MemoryLoadingStrategy.SELECTIVE, analysis=analysis, context_budget_tokens=600)
    assert "Identity" in selective.content
    assert selective.token_count > 0


def test_windowed_file_reader_modes(tmp_path):
    path = tmp_path / "sample.py"
    lines = [f"line {i}" for i in range(1, 401)]
    path.write_text("\n".join(lines), encoding="utf-8")

    reader = WindowedFileReader()
    smart = reader.read(str(path), mode="smart")
    assert smart.metadata.total_lines == 400

    tail = reader.read(str(path), mode="tail")
    assert "TAIL" in tail.metadata.showing

    window = reader.read(str(path), mode="window", start_line=100, end_line=120)
    assert "lines 101-120" in window.metadata.showing
    assert window.metadata.truncated is True


def test_pipeline_checkpoint_detects_missing_critical():
    checkpoint = PipelineCheckpoint()
    analysis, _ = MetaAnalysisParser.parse(
        {
            "user_intent": "none",
            "chat_subject": "general",
            "plan": ["respond"],
        }
    )
    result = checkpoint.validate(analysis, "planner")
    assert result.passed is False
    assert "CRITICAL" in result.reinvoke_prompt
