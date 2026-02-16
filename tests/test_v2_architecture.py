import hashlib
from pathlib import Path

import pytest

from main import (
    AgentConfig,
    ChatRequest,
    ProviderConfig,
    execute_v2_tool,
    parse_agent_action,
    parse_subagent_tag,
    parse_tool_tag,
    resolve_agent_config,
    run_v2_orchestrator,
    run_v2_pipeline,
    run_v2_subagent,
    should_allow_memory_write,
    user_requested_memory_write,
)


def _sha256(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()


def test_parse_tool_tag_valid():
    parsed = parse_tool_tag('[TOOL]orchestrator(tasks="1. A", context="B")[/TOOL]')
    assert parsed is not None
    assert parsed["tool_name"] == "orchestrator"
    assert parsed["params"]["tasks"] == "1. A"
    assert parsed["params"]["context"] == "B"


def test_parse_tool_tag_case_sensitive_rejects_lowercase_tag():
    assert parse_tool_tag('[tool]orchestrator(tasks="1. A")[/tool]') is None


def test_parse_subagent_tag_valid():
    parsed = parse_subagent_tag('[SUBAGENT task="Collect data" tools="read_file, regex_search"]')
    assert parsed is not None
    assert parsed["task"] == "Collect data"
    assert parsed["tools"] == "read_file, regex_search"


@pytest.mark.asyncio
async def test_parse_agent_action_repairs_malformed_output(monkeypatch):
    cfg = ProviderConfig(provider="OpenAI", model="x", api_key="k")

    async def fake_repair(*args, **kwargs):
        return '[TOOL]summarizer(query="Summarize last 2 messages")[/TOOL]'

    monkeypatch.setattr("main.repair_agent_action_output", fake_repair)
    parsed = await parse_agent_action(
        role="main",
        text='[TOOL]summarizer(query=Summarize last 2 messages)[/TOOL]',
        provider_cfg=cfg,
        allowed_tool_names=["summarizer", "orchestrator"],
        allow_subagent=False,
        allow_direct_text=True,
    )
    assert parsed["type"] == "tool"
    assert parsed["tool_name"] == "summarizer"
    assert parsed["repaired"] is True


@pytest.mark.asyncio
async def test_parse_agent_action_unrecoverable_nonfatal(monkeypatch):
    cfg = ProviderConfig(provider="OpenAI", model="x", api_key="k")

    async def fake_repair(*args, **kwargs):
        return ""

    monkeypatch.setattr("main.repair_agent_action_output", fake_repair)
    parsed = await parse_agent_action(
        role="orchestrator",
        text="bad syntax",
        provider_cfg=cfg,
        allowed_tool_names=["FINISH", "read_file"],
        allow_subagent=True,
        allow_direct_text=False,
    )
    assert parsed["type"] == "invalid"


def test_resolve_agent_config_legacy_back_compat_mapping():
    legacy_cfg = AgentConfig(
        main=ProviderConfig(provider="OpenAI", model="m", api_key="k"),
        sub=ProviderConfig(provider="OpenAI", model="s", api_key="k"),
        meta=ProviderConfig(provider="OpenAI", model="meta", api_key="k"),
    )
    resolved = resolve_agent_config(legacy_cfg)
    assert resolved.architecture_mode == "legacy"
    assert resolved.orchestrator is not None
    assert resolved.summarizer is not None


def test_v2_source_prompts_integrity_hashes():
    expected = {
        "prompts/v2/source/main_agent_full.txt": "9BD56841407E0574647F23C7AAFB29F893E46EB993E40AD6262BC432B221F3B0",
        "prompts/v2/source/orchestrator_full.txt": "3A4846D53C13F11B5D6CA294BA1B5C63D2A06602E8C28D234EB783DD969EFA67",
        "prompts/v2/source/sub_agent_full.txt": "8FD751C23A3A320F96A3647C294843730BE64D4D4975E09ACDCF066640F7F925",
        "prompts/v2/source/summarizer_full.txt": "908776B27A3799283B2FC3C7401170E83B29A3FE4EF5CA9F9031D9047BADD89A",
        "prompts/v2/source/flow_events_full.json": "700B6D5E9AC3754E94A6589344ECCE9CCEE259DA7D376EB6A0A569AB8DA8999B",
    }
    for path, hash_value in expected.items():
        assert _sha256(path) == hash_value


def test_memory_write_explicit_request_allows_short_content():
    assert user_requested_memory_write("please remember this")
    assert should_allow_memory_write({"content": "short"}, explicit_request=True)


@pytest.mark.asyncio
async def test_execute_v2_tool_explicit_short_memory_succeeds(monkeypatch):
    async def fake_execute_tool(name, arguments):
        return "Memory saved (id: 1)"

    monkeypatch.setattr("main.execute_tool", fake_execute_tool)
    outcome = await execute_v2_tool(
        "memory_create",
        {"content": "short"},
        user_message="remember this",
        allow_shell=False,
    )
    assert outcome["status"] == "ok"
    assert outcome["terminal"] is False


@pytest.mark.asyncio
async def test_execute_v2_tool_implicit_low_value_memory_rejected():
    outcome = await execute_v2_tool(
        "memory_create",
        {"content": "tiny"},
        user_message="hi",
        allow_shell=False,
    )
    assert outcome["status"] == "rejected"
    assert outcome["terminal"] is True


@pytest.mark.asyncio
async def test_execute_v2_tool_shell_denied_terminal():
    outcome = await execute_v2_tool(
        "shell_command",
        {"command": "dir"},
        user_message="just help me",
        allow_shell=False,
    )
    assert outcome["status"] in {"rejected", "denied"}
    assert outcome["terminal"] is True


@pytest.mark.asyncio
async def test_execute_v2_tool_unknown_tool_classified(monkeypatch):
    async def fake_execute_tool(name, arguments):
        return "Unknown tool: mystery"

    monkeypatch.setattr("main.execute_tool", fake_execute_tool)
    outcome = await execute_v2_tool(
        "memory_search",
        {"query": "x"},
        user_message="x",
        allow_shell=False,
    )
    assert outcome["status"] == "unknown_tool"
    assert outcome["terminal"] is True


@pytest.mark.asyncio
async def test_orchestrator_loop_guard_repeated_failure(monkeypatch):
    responses = [
        {"choices": [{"message": {"content": '[TOOL]memory_search(query="abc")[/TOOL]'}}]},
        {"choices": [{"message": {"content": '[TOOL]memory_search(query="abc")[/TOOL]'}}]},
    ]

    async def fake_call(*args, **kwargs):
        return responses.pop(0)

    async def fake_execute_tool(name, arguments):
        return "Tool execution error: boom"

    monkeypatch.setattr("main.LLMProvider.call", fake_call)
    monkeypatch.setattr("main.execute_tool", fake_execute_tool)

    result = await run_v2_orchestrator(
        tasks="1. Search memory",
        context="",
        user_message="do it",
        allow_shell=False,
    )
    assert "auto-finish" in result["result"].lower()
    assert any("loop guard" in w.lower() for w in result["warnings"])


@pytest.mark.asyncio
async def test_subagent_loop_guard_repeated_failure(monkeypatch):
    responses = [
        {"choices": [{"message": {"content": '[TOOL]memory_search(query="abc")[/TOOL]'}}]},
        {"choices": [{"message": {"content": '[TOOL]memory_search(query="abc")[/TOOL]'}}]},
    ]

    async def fake_call(*args, **kwargs):
        return responses.pop(0)

    async def fake_execute_tool(name, arguments):
        return "Tool execution error: boom"

    monkeypatch.setattr("main.LLMProvider.call", fake_call)
    monkeypatch.setattr("main.execute_tool", fake_execute_tool)

    result = await run_v2_subagent(
        task="search memory",
        tools=["memory_search"],
        context="",
        user_message="do it",
        allow_shell=False,
    )
    assert "auto-finish" in result["result"].lower()
    assert any("loop guard" in w.lower() for w in result["warnings"])


@pytest.mark.asyncio
async def test_subagent_invalid_tool_list_auto_finishes():
    result = await run_v2_subagent(
        task="invalid tools",
        tools=["not_a_real_tool"],
        context="",
        user_message="do it",
        allow_shell=False,
    )
    assert "no valid tools" in result["result"].lower()
    assert any("no valid tools" in w.lower() for w in result["warnings"])


@pytest.mark.asyncio
async def test_main_forces_finalization_after_orchestrator(monkeypatch):
    from main import create_temp_session, ensure_session_policy

    session = create_temp_session("v2-finalization")
    ensure_session_policy(session["id"])

    main_responses = [
        {"choices": [{"message": {"content": '[TOOL]orchestrator(tasks="1. do thing", context="")[/TOOL]'}}]},
        {"choices": [{"message": {"content": '[TOOL]summarizer(query="extra")[/TOOL]'}}]},
    ]

    async def fake_call(*args, **kwargs):
        return main_responses.pop(0)

    async def fake_orchestrator(**kwargs):
        return {"result": "orchestrator done", "tool_events": [], "subagent_outputs": [], "warnings": []}

    monkeypatch.setattr("main.LLMProvider.call", fake_call)
    monkeypatch.setattr("main.run_v2_orchestrator", fake_orchestrator)

    result = await run_v2_pipeline(
        request=ChatRequest(message="please do this"),
        session_id=session["id"],
        persistent=False,
    )
    assert result["reply"] == "orchestrator done"
    assert any("bypassed" in w.lower() for w in result["meta"].get("warnings", []))
