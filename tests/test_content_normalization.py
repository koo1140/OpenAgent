import pytest

from main import (
    ProviderConfig,
    content_to_text,
    create_temp_session,
    robust_json_loads,
    run_tool_loop,
    store_turn,
    temp_sessions,
)


def test_content_to_text_plain_string():
    assert content_to_text("hello") == "hello"


def test_content_to_text_list_of_strings():
    assert content_to_text(["a", "b", "c"]) == "abc"


def test_content_to_text_list_of_dicts_text():
    assert content_to_text([{"text": "alpha"}, {"text": "beta"}]) == "alphabeta"


def test_content_to_text_mixed_list():
    value = ["x", {"content": "y"}, 12, None, {"value": "z"}]
    assert content_to_text(value) == "xyz"


def test_content_to_text_dict_with_text():
    assert content_to_text({"text": "hello"}) == "hello"


def test_content_to_text_dict_without_text_key():
    assert content_to_text({"type": "tool_use", "id": "123"}) == ""


def test_content_to_text_none():
    assert content_to_text(None) == ""


def test_robust_json_loads_valid_json_string():
    assert robust_json_loads('{"ok": true}') == {"ok": True}


def test_robust_json_loads_list_wrapped_json():
    assert robust_json_loads(['{"a": 1}']) == {"a": 1}


def test_robust_json_loads_non_json_list_returns_raw():
    parsed = robust_json_loads([{"text": "not-json"}])
    assert parsed == {"raw": "not-json"}


def test_robust_json_loads_empty_non_text_returns_empty_dict():
    assert robust_json_loads([{"type": "tool_use", "id": "x"}]) == {}


@pytest.mark.asyncio
async def test_run_tool_loop_normalizes_list_content_no_tool_calls(monkeypatch):
    cfg = ProviderConfig(provider="OpenAI", model="x", api_key="k")

    async def fake_call(*args, **kwargs):
        return {"choices": [{"message": {"content": [{"text": "hello"}, {"text": " world"}]}}]}

    monkeypatch.setattr("main.LLMProvider.call", fake_call)

    result = await run_tool_loop(
        cfg,
        [{"role": "user", "content": "hi"}],
        tools=[],
        approval_mode="allow",
        allow_shell=False,
        user_message="hi",
    )
    assert result["status"] == "ok"
    assert result["content"] == "hello world"


@pytest.mark.asyncio
async def test_run_tool_loop_normalizes_list_content_with_tool_calls(monkeypatch):
    cfg = ProviderConfig(provider="OpenAI", model="x", api_key="k")
    responses = [
        {
            "choices": [
                {
                    "message": {
                        "content": [{"text": "prep"}],
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "function": {
                                    "name": "read_file",
                                    "arguments": '{"path":"main.py"}',
                                },
                            }
                        ],
                    }
                }
            ]
        },
        {"choices": [{"message": {"content": [{"text": "done"}]}}]},
    ]

    async def fake_call(*args, **kwargs):
        return responses.pop(0)

    async def fake_execute_tool(name, arguments):
        return "file-content"

    monkeypatch.setattr("main.LLMProvider.call", fake_call)
    monkeypatch.setattr("main.execute_tool", fake_execute_tool)

    input_messages = [{"role": "user", "content": "hi"}]
    result = await run_tool_loop(
        cfg,
        input_messages,
        tools=[],
        approval_mode="allow",
        allow_shell=False,
        user_message="hi",
    )

    assert result["status"] == "ok"
    assert result["content"] == "done"
    assert any(m["role"] == "assistant" and m["content"] == "prep" for m in input_messages)


def test_store_turn_coerces_list_assistant_reply_for_temp_session():
    session = create_temp_session("test")
    store_turn(
        session_id=session["id"],
        persistent=False,
        user_message="u",
        assistant_reply=[{"text": "alpha"}, {"text": "beta"}],
        meta_json={},
        tool_events=[],
        subagent_outputs=[],
    )
    turns = temp_sessions[session["id"]]["turns"]
    assert turns[-1]["assistant_reply"] == "alphabeta"
