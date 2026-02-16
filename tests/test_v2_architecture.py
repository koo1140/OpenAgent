import hashlib
from pathlib import Path

import pytest

from main import (
    AgentConfig,
    ProviderConfig,
    parse_agent_action,
    parse_subagent_tag,
    parse_tool_tag,
    resolve_agent_config,
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
