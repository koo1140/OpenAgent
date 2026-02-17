import hashlib
from pathlib import Path

import pytest

from main import (
    AgentConfig,
    ChatRequest,
    ProviderConfig,
    TOOLS_SPEC,
    execute_v2_tool,
    parse_agent_action,
    parse_subagent_tag,
    parse_tool_tag,
    resolve_agent_config,
    run_tool_loop,
    run_v2_orchestrator,
    run_v2_pipeline,
    run_v2_subagent,
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
async def test_execute_v2_tool_memory_write_not_semantically_rejected(monkeypatch):
    async def fake_execute_tool(name, arguments):
        return "Memory saved (id: 2)"

    monkeypatch.setattr("main.execute_tool", fake_execute_tool)
    outcome = await execute_v2_tool(
        "memory_create",
        {"content": "tiny"},
        user_message="hi",
        allow_shell=False,
    )
    assert outcome["status"] == "ok"
    assert outcome["terminal"] is False


@pytest.mark.asyncio
async def test_execute_v2_tool_shell_denied_terminal():
    outcome = await execute_v2_tool(
        "shell_command",
        {"command": "dir"},
        user_message="just help me",
        allow_shell=False,
        approval_mode="auto_deny",
    )
    assert outcome["status"] == "denied"
    assert outcome["terminal"] is True


@pytest.mark.asyncio
async def test_execute_v2_tool_shell_needs_approval_in_ask_mode():
    outcome = await execute_v2_tool(
        "shell_command",
        {"command": "dir"},
        user_message="run it",
        allow_shell=False,
        approval_mode="ask",
    )
    assert outcome["status"] == "needs_approval"
    assert outcome["terminal"] is False
    assert outcome.get("pending_tools")


@pytest.mark.asyncio
async def test_execute_v2_tool_rejects_invalid_edit_file_shape():
    outcome = await execute_v2_tool(
        "edit_file",
        {"file": "identity.md", "search": "x", "replace": "y"},
        user_message="x",
        allow_shell=False,
    )
    assert outcome["status"] == "rejected"
    assert "Tool validation error" in outcome["message"]
    assert outcome["terminal"] is True


@pytest.mark.asyncio
async def test_execute_v2_tool_read_file_directory_path_rejected():
    outcome = await execute_v2_tool(
        "read_file",
        {"path": "."},
        user_message="x",
        allow_shell=False,
    )
    assert outcome["status"] == "rejected"
    assert "is a directory" in outcome["message"]
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
async def test_orchestrator_shell_approval_pause_and_resume(monkeypatch):
    responses = [
        {"choices": [{"message": {"content": '[TOOL]shell_command(command="echo hi")[/TOOL]'}}]},
        {"choices": [{"message": {"content": '[TOOL]FINISH(result="{\\"summary\\":\\"done\\",\\"critical_facts\\":[],\\"artifact_ids\\":[] }")[/TOOL]'}}]},
    ]

    async def fake_call(*args, **kwargs):
        return responses.pop(0)

    async def fake_execute_tool(name, arguments):
        assert name == "shell_command"
        return "STDOUT:\nhi\nSTDERR:\n"

    monkeypatch.setattr("main.LLMProvider.call", fake_call)
    monkeypatch.setattr("main.execute_tool", fake_execute_tool)

    first = await run_v2_orchestrator(
        tasks="1. run shell",
        context="",
        user_message="do it",
        allow_shell=False,
        approval_mode="ask",
    )
    assert first["status"] == "needs_approval"
    assert first.get("pending_tools")

    resumed = await run_v2_orchestrator(
        tasks="",
        context="",
        user_message="do it",
        allow_shell=False,
        state=first["state"],
        shell_approval=True,
        approval_mode="ask",
    )
    assert resumed["status"] == "ok"
    assert resumed["result"] == "done"
    assert resumed["tool_events"]


@pytest.mark.asyncio
async def test_orchestrator_finish_requires_structured_payload(monkeypatch):
    responses = [
        {"choices": [{"message": {"content": '[TOOL]FINISH(result="done")[/TOOL]'}}]},
    ]

    async def fake_call(*args, **kwargs):
        return responses.pop(0)

    monkeypatch.setattr("main.LLMProvider.call", fake_call)
    result = await run_v2_orchestrator(
        tasks="1. finish",
        context="",
        user_message="u",
        allow_shell=False,
    )
    assert "invalid FINISH payload" in result["result"]
    assert any("invalid FINISH payload" in w for w in result["warnings"])


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
async def test_legacy_tool_loop_shell_always_needs_approval(monkeypatch):
    cfg = ProviderConfig(provider="OpenAI", model="x", api_key="k")
    messages = [{"role": "user", "content": "Do this"}]

    async def fake_call(*args, **kwargs):
        return {
            "choices": [{
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "shell_command", "arguments": "{\"command\":\"echo hi\"}"},
                        }
                    ],
                }
            }]
        }

    monkeypatch.setattr("main.LLMProvider.call", fake_call)
    result = await run_tool_loop(
        cfg,
        messages,
        TOOLS_SPEC,
        approval_mode="ask",
        allow_shell=False,
        user_message="hello",
    )
    assert result["status"] == "needs_approval"
    assert result["pending_tool_calls"]


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


@pytest.mark.asyncio
async def test_v2_pipeline_shell_pause_and_resume(monkeypatch):
    from main import create_temp_session, ensure_session_policy

    session = create_temp_session("v2-approval-resume")
    ensure_session_policy(session["id"])

    responses = [
        {"choices": [{"message": {"content": '[TOOL]orchestrator(tasks="1. run shell", context="")[/TOOL]'}}]},
        {"choices": [{"message": {"content": '[TOOL]shell_command(command="echo hi")[/TOOL]'}}]},
        {"choices": [{"message": {"content": '[TOOL]FINISH(result="{\\"summary\\":\\"orchestrator done\\",\\"critical_facts\\":[\\"shell returned hi\\"],\\"artifact_ids\\":[] }")[/TOOL]'}}]},
        {"choices": [{"message": {"content": "Final user reply"}}]},
    ]

    async def fake_call(*args, **kwargs):
        return responses.pop(0)

    async def fake_execute_tool(name, arguments):
        return "STDOUT:\nhi\nSTDERR:\n"

    monkeypatch.setattr("main.LLMProvider.call", fake_call)
    monkeypatch.setattr("main.execute_tool", fake_execute_tool)

    first = await run_v2_pipeline(
        request=ChatRequest(message="please run shell"),
        session_id=session["id"],
        persistent=False,
    )
    assert first["status"] == "needs_approval"
    assert first.get("v2_resume_state")
    assert first.get("pending_tools")

    resumed = await run_v2_pipeline(
        request=ChatRequest(message="please run shell"),
        session_id=session["id"],
        persistent=False,
        state=first["v2_resume_state"],
        shell_approval=True,
    )
    assert resumed["status"] == "ok"
    assert resumed["reply"] == "Final user reply"
