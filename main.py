"""
Agentic Gateway Backend
Multi-provider LLM orchestration with Meta, Main, and Sub agents
"""

from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict, Any, Literal, Tuple, Callable, Awaitable
import asyncio
import httpx
import json
import logging
import re
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path

app = FastAPI()

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==========================================================================================================================================================
# CONSTANTS
# =============================================================================

DB_PATH = Path("sessions.db")
CONFIG_PATH = Path("config.json")
MEMORY_DIR = Path("memory")
PROMPTS_DIR = Path("prompts")
SKILLS_DIR = Path("skills")

PERSISTENT_MEMORY_PATH = MEMORY_DIR / "persistent.json"
SKILLS_INDEX_PATH = SKILLS_DIR / "skills.json"

MAX_HISTORY_TURNS = 10
MAX_META_TURNS = 50
MAX_TOOL_ROUNDS = 3
TOOL_OUTPUT_LIMIT = 1200

# Default meta output for fallback scenarios
DEFAULT_META_OUTPUT = {
    "user_intent": "unknown",
    "chat_subject": "general",
    "user_tone": "neutral",
    "recommended_plan": "respond conversationally",
    "keepMemoryFilesLoaded": False,
    "memory_files_to_load": None,
    "skills_to_load": None,
    "extra_instructions": None,
    "subagent_suggestions": []
}

DEFAULT_PROVIDERS: Dict[str, Dict[str, Any]] = {
    "OpenAI": {
        "provider_type": "openai_compatible",
        "base_url": "https://api.openai.com/v1/chat/completions",
        "supports_tools": True,
        "supports_response_format": True,
    },
    "Groq": {
        "provider_type": "openai_compatible",
        "base_url": "https://api.groq.com/openai/v1/chat/completions",
        "supports_tools": True,
        "supports_response_format": False,
    },
    "Mistral": {
        "provider_type": "openai_compatible",
        "base_url": "https://api.mistral.ai/v1/chat/completions",
        "supports_tools": True,
        "supports_response_format": True,
    },
    "OpenRouter": {
        "provider_type": "openai_compatible",
        "base_url": "https://openrouter.ai/api/v1/chat/completions",
        "supports_tools": True,
        "supports_response_format": True,
    },
    "Together": {
        "provider_type": "openai_compatible",
        "base_url": "https://api.together.xyz/v1/chat/completions",
        "supports_tools": False,
        "supports_response_format": False,
    },
    "DeepSeek": {
        "provider_type": "openai_compatible",
        "base_url": "https://api.deepseek.com/v1/chat/completions",
        "supports_tools": False,
        "supports_response_format": False,
    },
    "Anthropic": {
        "provider_type": "anthropic",
        "base_url": "https://api.anthropic.com/v1/messages",
        "supports_tools": True,
        "supports_response_format": False,
    },
    "Custom": {
        "provider_type": "openai_compatible",
        "base_url": None,
        "supports_tools": True,
        "supports_response_format": True,
    },
}


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def truncate(text: str, limit: int = TOOL_OUTPUT_LIMIT) -> str:
    if text is None:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... (truncated, {len(text)} chars)"


def serialize_tool_arguments(arguments: Dict[str, Any]) -> str:
    try:
        return json.dumps(arguments or {}, ensure_ascii=True)
    except Exception:
        return "{}"


def build_pending_tool(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": f"call_{uuid.uuid4()}",
        "name": name,
        "arguments": serialize_tool_arguments(arguments),
    }


def sse_event(event: str, data: Dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def list_available_skills() -> List[str]:
    if not SKILLS_DIR.exists():
        return []
    return sorted([path.stem for path in SKILLS_DIR.glob("*.md")])


def list_available_memory_files() -> List[str]:
    if not MEMORY_DIR.exists():
        return []
    return sorted([path.name for path in MEMORY_DIR.glob("*.md")])


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class ProviderConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    provider: str
    provider_type: Literal["openai_compatible", "anthropic"] = "openai_compatible"
    model: str
    api_key: str
    base_url: Optional[str] = None
    headers: Optional[Dict[str, str]] = None
    supports_tools: Optional[bool] = None
    supports_response_format: Optional[bool] = None


class AgentConfig(BaseModel):
    architecture_mode: Literal["legacy", "hierarchical_v2"] = "legacy"
    meta: Optional[ProviderConfig] = None
    main: ProviderConfig
    sub: ProviderConfig
    orchestrator: Optional[ProviderConfig] = None
    summarizer: Optional[ProviderConfig] = None


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class MetaOutput(BaseModel):
    model_config = ConfigDict(extra="allow")
    # Layer 1: Meta Agents
    intent: Optional[Dict[str, Any]] = None
    tone: Optional[Dict[str, Any]] = None
    user: Optional[Dict[str, Any]] = None
    subject: Optional[Dict[str, Any]] = None
    needs: Optional[Dict[str, Any]] = None
    patterns: Optional[Dict[str, Any]] = None

    # Layer 2: Planner
    plan: Optional[Dict[str, Any]] = None

    # Layer 5: Gatekeeper (stored from previous turn or current)
    gatekeeper: Optional[Dict[str, Any]] = None

    # For raw text fallback
    raw: Optional[str] = None

    # Legacy fields (for compatibility if needed, but we'll try to move away)
    user_intent: str = "unknown"
    chat_subject: str = "general"
    user_tone: str = "neutral"
    recommended_plan: str = "respond conversationally"
    extra_instructions: Optional[str] = None
    memory_actions: List[Dict[str, Any]] = Field(default_factory=list)


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    reply: str
    meta: Optional[Dict[str, Any]] = None
    tool_events: List[Dict[str, Any]] = Field(default_factory=list)
    subagent_outputs: List[Dict[str, Any]] = Field(default_factory=list)
    status: str = "ok"
    pending_tools: Optional[List[Dict[str, Any]]] = None
    session_id: Optional[str] = None
    architecture_mode: Literal["legacy", "hierarchical_v2"] = "legacy"


class SessionCreateRequest(BaseModel):
    title: Optional[str] = None
    persistent: bool = True


class SessionUpdateRequest(BaseModel):
    title: str


class ToolApprovalRequest(BaseModel):
    session_id: str
    decision: Literal["run_once", "allow_session", "deny"]


# =============================================================================
# GLOBAL STATE
# =============================================================================

config: Optional[AgentConfig] = None
pending_approvals: Dict[str, Dict[str, Any]] = {}
rate_limit_decisions: Dict[str, asyncio.Queue] = {}
session_policies: Dict[str, Dict[str, Any]] = {}
temp_sessions: Dict[str, Dict[str, Any]] = {}

MEMORY_DIR.mkdir(exist_ok=True)
SKILLS_DIR.mkdir(exist_ok=True)

if not PERSISTENT_MEMORY_PATH.exists():
    PERSISTENT_MEMORY_PATH.write_text(json.dumps({"memories": []}, indent=2))
if not SKILLS_INDEX_PATH.exists():
    SKILLS_INDEX_PATH.write_text(json.dumps({"skills": []}, indent=2))


# =============================================================================
# DATABASE
# =============================================================================

def db_connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    conn = db_connect()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            title TEXT,
            created_at TEXT,
            updated_at TEXT,
            persistent INTEGER,
            working_memory TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS turns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            user_message TEXT,
            assistant_reply TEXT,
            meta_json TEXT,
            tool_json TEXT,
            subagent_json TEXT,
            architecture_mode TEXT,
            created_at TEXT,
            FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS tool_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            turn_id INTEGER,
            name TEXT,
            args_json TEXT,
            result_text TEXT,
            status TEXT,
            created_at TEXT,
            FOREIGN KEY(turn_id) REFERENCES turns(id) ON DELETE CASCADE
        )
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_tool_events_turn ON tool_events(turn_id)")

    # Migration: Add working_memory to sessions if it doesn't exist
    try:
        cur.execute("ALTER TABLE sessions ADD COLUMN working_memory TEXT")
    except sqlite3.OperationalError:
        pass # Already exists
    try:
        cur.execute("ALTER TABLE turns ADD COLUMN architecture_mode TEXT")
    except sqlite3.OperationalError:
        pass # Already exists

    conn.commit()
    conn.close()


init_db()


# =============================================================================
# PROVIDER RESOLUTION
# =============================================================================

def resolve_provider_config(cfg: ProviderConfig) -> ProviderConfig:
    reg = DEFAULT_PROVIDERS.get(cfg.provider)
    updates: Dict[str, Any] = {}
    if reg:
        updates["provider_type"] = reg.get("provider_type", cfg.provider_type)
        updates["base_url"] = cfg.base_url or reg.get("base_url")
        if cfg.supports_tools is None:
            updates["supports_tools"] = reg.get("supports_tools", True)
        if cfg.supports_response_format is None:
            updates["supports_response_format"] = reg.get("supports_response_format", False)
    return cfg.model_copy(update=updates)


def resolve_agent_config(agent_cfg: AgentConfig) -> AgentConfig:
    resolved_meta = resolve_provider_config(agent_cfg.meta) if agent_cfg.meta else None
    resolved_main = resolve_provider_config(agent_cfg.main)
    resolved_sub = resolve_provider_config(agent_cfg.sub)
    resolved_orchestrator = resolve_provider_config(agent_cfg.orchestrator) if agent_cfg.orchestrator else None
    resolved_summarizer = resolve_provider_config(agent_cfg.summarizer) if agent_cfg.summarizer else None

    # Backward-compatible defaults:
    # - legacy mode relies on meta/main/sub
    # - hierarchical_v2 can fall back to existing fields if not explicitly set
    if resolved_meta is None:
        resolved_meta = resolved_orchestrator or resolved_main
    if resolved_orchestrator is None:
        resolved_orchestrator = resolved_meta or resolved_main
    if resolved_summarizer is None:
        resolved_summarizer = resolved_meta or resolved_main

    return AgentConfig(
        architecture_mode=agent_cfg.architecture_mode,
        meta=resolved_meta,
        main=resolved_main,
        sub=resolved_sub,
        orchestrator=resolved_orchestrator,
        summarizer=resolved_summarizer,
    )


def load_config() -> Optional[AgentConfig]:
    if not CONFIG_PATH.exists():
        return None
    try:
        data = json.loads(CONFIG_PATH.read_text())
        loaded = AgentConfig(**data)
        return resolve_agent_config(loaded)
    except Exception as exc:
        print(f"⚠ Failed to load config.json: {exc}")
        return None


config = load_config()
if config:
    print("✅ Loaded agent config from config.json")
else:
    print("⚠ No config.json found, you need to POST /api/config at least once")


def get_architecture_mode() -> Literal["legacy", "hierarchical_v2"]:
    if not config:
        return "legacy"
    return config.architecture_mode


def get_orchestrator_cfg() -> ProviderConfig:
    if not config:
        raise ValueError("Configuration not set")
    return config.orchestrator or config.main


def get_summarizer_cfg() -> ProviderConfig:
    if not config:
        raise ValueError("Configuration not set")
    return config.summarizer or config.main


# =============================================================================
# PROVIDER CALLS
# =============================================================================

def openai_tools_to_anthropic(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    converted: List[Dict[str, Any]] = []
    for tool in tools or []:
        fn = tool.get("function", {})
        converted.append({
            "name": fn.get("name"),
            "description": fn.get("description", ""),
            "input_schema": fn.get("parameters", {}),
        })
    return converted


def openai_messages_to_anthropic(messages: List[Dict[str, Any]]) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    system_parts: List[str] = []
    result_messages: List[Dict[str, Any]] = []
    pending_tool_results: List[Dict[str, Any]] = []

    for msg in messages:
        role = msg.get("role")
        if role == "system":
            system_parts.append(msg.get("content", ""))
            continue
        if role == "tool":
            pending_tool_results.append({
                "type": "tool_result",
                "tool_use_id": msg.get("tool_call_id"),
                "content": msg.get("content", ""),
            })
            continue

        if pending_tool_results:
            result_messages.append({"role": "user", "content": pending_tool_results})
            pending_tool_results = []

        if role == "assistant":
            content_blocks: List[Dict[str, Any]] = []
            if msg.get("content"):
                content_blocks.append({"type": "text", "text": msg.get("content")})
            for tool_call in msg.get("tool_calls", []) or []:
                fn = tool_call.get("function", {})
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except Exception:
                    args = {}
                content_blocks.append({
                    "type": "tool_use",
                    "id": tool_call.get("id"),
                    "name": fn.get("name"),
                    "input": args,
                })
            if not content_blocks:
                content_blocks = [{"type": "text", "text": ""}]
            result_messages.append({"role": "assistant", "content": content_blocks})
        else:
            result_messages.append({
                "role": "user",
                "content": [{"type": "text", "text": msg.get("content", "")}],
            })

    if pending_tool_results:
        result_messages.append({"role": "user", "content": pending_tool_results})

    system = "\n\n".join([part for part in system_parts if part])
    return system or None, result_messages


def anthropic_response_to_openai(data: Dict[str, Any]) -> Dict[str, Any]:
    content_blocks = data.get("content", []) or []
    text_parts: List[str] = []
    tool_calls: List[Dict[str, Any]] = []
    for block in content_blocks:
        if block.get("type") == "text":
            text_parts.append(block.get("text", ""))
        if block.get("type") == "tool_use":
            tool_calls.append({
                "id": block.get("id"),
                "type": "function",
                "function": {
                    "name": block.get("name"),
                    "arguments": json.dumps(block.get("input", {})),
                }
            })
    message: Dict[str, Any] = {
        "role": "assistant",
        "content": "".join(text_parts),
    }
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {"choices": [{"message": message}]}


class LLMProvider:
    @staticmethod
    async def call(
        provider_config: ProviderConfig,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        response_format: Optional[Dict[str, Any]] = None,
        on_rate_limit: Optional[Callable[[int], Awaitable[None]]] = None,
        on_user_decision_required: Optional[Callable[[], Awaitable[str]]] = None
    ) -> Dict[str, Any]:
        # Small delay to avoid aggressive rate limits
        await asyncio.sleep(0.5)

        try:
            if provider_config.provider_type == "openai_compatible":
                return await LLMProvider._call_openai_compatible(provider_config, messages, tools, response_format)
            else:
                return await LLMProvider._call_anthropic(provider_config, messages, tools)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 429:
                logger.warning(f"Rate limit hit for provider {provider_config.provider}. Retrying in 30s...")
                if on_rate_limit:
                    await on_rate_limit(30)

                await asyncio.sleep(30)

                try:
                    if provider_config.provider_type == "openai_compatible":
                        return await LLMProvider._call_openai_compatible(provider_config, messages, tools, response_format)
                    else:
                        return await LLMProvider._call_anthropic(provider_config, messages, tools)
                except httpx.HTTPStatusError as exc2:
                    if exc2.response.status_code == 429:
                        if on_user_decision_required:
                            logger.info("Rate limit hit again. Waiting for user decision.")
                            decision = await on_user_decision_required()
                            if decision == "continue":
                                return await LLMProvider.call(provider_config, messages, tools, response_format, on_rate_limit, on_user_decision_required)
                            else:
                                raise exc2
                        else:
                            raise exc2
                    else:
                        raise exc2
            else:
                raise exc

    @staticmethod
    async def _call_openai_compatible(
        cfg: ProviderConfig,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        response_format: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        endpoint = cfg.base_url or DEFAULT_PROVIDERS.get(cfg.provider, {}).get("base_url")
        if not endpoint:
            raise ValueError(f"Missing base_url for provider {cfg.provider}")

        headers = {
            "Authorization": f"Bearer {cfg.api_key}",
            "Content-Type": "application/json",
        }
        if cfg.headers:
            headers.update(cfg.headers)

        payload: Dict[str, Any] = {
            "model": cfg.model,
            "messages": messages,
        }

        if tools and cfg.supports_tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        if response_format and cfg.supports_response_format:
            payload["response_format"] = response_format

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(endpoint, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()

    @staticmethod
    async def _call_anthropic(
        cfg: ProviderConfig,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        system, conversation = openai_messages_to_anthropic(messages)
        headers = {
            "x-api-key": cfg.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        if cfg.headers:
            headers.update(cfg.headers)

        payload: Dict[str, Any] = {
            "model": cfg.model,
            "max_tokens": 2048,
            "messages": conversation,
        }
        if system:
            payload["system"] = system
        if tools and cfg.supports_tools:
            payload["tools"] = openai_tools_to_anthropic(tools)

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(cfg.base_url or DEFAULT_PROVIDERS["Anthropic"]["base_url"], headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            return anthropic_response_to_openai(data)


# =============================================================================
# TOOLS
# =============================================================================

TOOLS_SPEC: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "shell_command",
            "description": "Execute a shell command",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The shell command to execute"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file. Returns full content if <200 lines, otherwise first 60 lines",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to read"},
                    "file_path": {"type": "string", "description": "Alias for path"},
                    "file": {"type": "string", "description": "Alias for path"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Edit a file. Can write full content if <200 lines, otherwise limited operations",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to edit"},
                    "file_path": {"type": "string", "description": "Alias for path"},
                    "file": {"type": "string", "description": "Alias for path"},
                    "content": {"type": "string", "description": "New content to write"},
                    "search": {"type": "string", "description": "Text or regex pattern to find"},
                    "replace": {"type": "string", "description": "Replacement text for search"},
                    "regex": {"type": "boolean", "description": "Whether search is regex (default true)"},
                    "max_replacements": {"type": "integer", "description": "Maximum number of replacements; default 1"},
                    "mode": {"type": "string", "enum": ["write", "append"], "description": "Write mode"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "regex_search",
            "description": "Search for text using regex in files or directory",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex pattern"},
                    "path": {"type": "string", "description": "File or directory path"},
                    "recursive": {"type": "boolean", "description": "Search recursively in directories"},
                },
                "required": ["pattern", "path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "curl_request",
            "description": "Make HTTP requests (GET, POST, etc.)",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to request"},
                    "method": {"type": "string", "enum": ["GET", "POST", "PUT", "DELETE", "PATCH"]},
                    "headers": {"type": "object", "description": "HTTP headers"},
                    "body": {"type": "string", "description": "Request body"},
                },
                "required": ["url", "method"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spawn_subagents",
            "description": "Spawn focused sub-agents for parallel work",
            "parameters": {
                "type": "object",
                "properties": {
                    "subagents": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "role": {"type": "string"},
                                "task": {"type": "string"},
                                "tools": {
                                    "type": "array",
                                    "items": {"type": "string"}
                                },
                                "context": {"type": "string"}
                            },
                            "required": ["role", "task"]
                        }
                    }
                },
                "required": ["subagents"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "memory_create",
            "description": "Save a piece of information to persistent memory",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "The content to save"},
                    "category": {"type": "string", "enum": ["episodic", "semantic", "procedural"], "description": "Memory category"},
                    "tags": {"type": "array", "items": {"type": "string"}, "description": "Tags for the memory"},
                },
                "required": ["content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_search",
            "description": "Search for information in persistent memory",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"},
                    "tags": {"type": "array", "items": {"type": "string"}, "description": "Tags to filter by"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "skill_create",
            "description": "Create a new skill and save it",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Name of the skill"},
                    "description": {"type": "string", "description": "Brief description of the skill"},
                    "content": {"type": "string", "description": "The content/logic of the skill"},
                },
                "required": ["name", "description", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "skill_load",
            "description": "Load an existing skill's content",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Name of the skill to load"},
                },
                "required": ["name"],
            },
        },
    },
]


def tool_schema_by_name() -> Dict[str, Dict[str, Any]]:
    schemas: Dict[str, Dict[str, Any]] = {}
    for entry in TOOLS_SPEC:
        fn = entry.get("function", {})
        name = fn.get("name")
        params = fn.get("parameters", {})
        if isinstance(name, str):
            schemas[name] = params if isinstance(params, dict) else {}
    return schemas


def coerce_tool_value(value: Any, schema: Dict[str, Any]) -> Tuple[Any, Optional[str]]:
    expected_type = schema.get("type")
    if expected_type == "string":
        if isinstance(value, str):
            return value, None
        return str(value), None
    if expected_type == "boolean":
        if isinstance(value, bool):
            return value, None
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1", "yes"}:
                return True, None
            if lowered in {"false", "0", "no"}:
                return False, None
        return None, "must be a boolean"
    if expected_type == "integer":
        if isinstance(value, int) and not isinstance(value, bool):
            return value, None
        if isinstance(value, str):
            text = value.strip()
            if text.isdigit() or (text.startswith("-") and text[1:].isdigit()):
                return int(text), None
        return None, "must be an integer"
    if expected_type == "object":
        if isinstance(value, dict):
            return value, None
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, dict):
                    return parsed, None
            except Exception:
                pass
        return None, "must be an object (JSON object string accepted)"
    if expected_type == "array":
        if isinstance(value, list):
            return value, None
        if isinstance(value, str):
            raw = value.strip()
            if not raw:
                return [], None
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    return parsed, None
            except Exception:
                pass
            # Best-effort fallback for comma-separated list.
            return [part.strip() for part in raw.split(",") if part.strip()], None
        return None, "must be an array (JSON array string accepted)"
    return value, None


def normalize_and_validate_tool_arguments(name: str, arguments: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    schemas = tool_schema_by_name()
    if name not in schemas:
        # Unknown tools are handled by execute_tool itself.
        return arguments if isinstance(arguments, dict) else {}, None

    params_schema = schemas.get(name, {}) or {}
    props = params_schema.get("properties", {}) or {}
    required = params_schema.get("required", []) or []
    args = dict(arguments) if isinstance(arguments, dict) else {}

    if name in {"read_file", "edit_file"}:
        # Compatibility aliases used by older prompts and model outputs.
        if "path" not in args:
            alias_path = args.get("file_path")
            if alias_path is None:
                alias_path = args.get("file")
            if alias_path is not None:
                args["path"] = alias_path

    unknown = sorted([key for key in args.keys() if key not in props])
    if unknown:
        return None, f"unexpected parameter(s) for {name}: {', '.join(unknown)}"

    normalized: Dict[str, Any] = {}
    for key, schema in props.items():
        if key not in args:
            continue
        coerced, err = coerce_tool_value(args.get(key), schema if isinstance(schema, dict) else {})
        if err:
            return None, f"parameter '{key}' {err}"
        enum_values = (schema or {}).get("enum")
        if enum_values and coerced not in enum_values:
            return None, f"parameter '{key}' must be one of: {', '.join(map(str, enum_values))}"
        normalized[key] = coerced

    missing = [key for key in required if key not in normalized]
    if missing:
        return None, f"missing required parameter(s) for {name}: {', '.join(missing)}"

    for key in required:
        value = normalized.get(key)
        if isinstance(value, str) and not value.strip():
            return None, f"parameter '{key}' must be non-empty"

    if name in {"read_file", "edit_file"}:
        normalized.pop("file_path", None)
        normalized.pop("file", None)

    if name == "edit_file":
        has_content = "content" in normalized
        has_search = "search" in normalized or "replace" in normalized
        if has_content and has_search:
            return None, "edit_file must use either 'content' or ('search' and 'replace'), not both"
        if not has_content and not has_search:
            return None, "edit_file requires either 'content' or both 'search' and 'replace'"
        if has_search:
            if "search" not in normalized or "replace" not in normalized:
                return None, "edit_file requires both 'search' and 'replace' when using replacement mode"
            search_value = normalized.get("search")
            if isinstance(search_value, str) and not search_value:
                return None, "parameter 'search' must be non-empty"
            max_replacements = normalized.get("max_replacements")
            if isinstance(max_replacements, int) and max_replacements < 1:
                return None, "parameter 'max_replacements' must be >= 1"

    return normalized, None


def create_artifact(
    artifacts: List[Dict[str, Any]],
    owner: str,
    tool_name: str,
    args: Dict[str, Any],
    status: str,
    content: str,
) -> Dict[str, Any]:
    artifact_id = f"artifact_{uuid.uuid4().hex[:12]}"
    text = content if isinstance(content, str) else str(content)
    artifact = {
        "id": artifact_id,
        "owner": owner,
        "tool": tool_name,
        "args": args,
        "status": status,
        "created_at": now_iso(),
        "content": truncate(text, limit=8000),
    }
    artifacts.append(artifact)
    return artifact


def render_artifact_context(
    artifacts: List[Dict[str, Any]],
    artifact_ids: Optional[List[str]] = None,
    max_items: int = 4,
    max_chars_per_item: int = 2500,
) -> str:
    if not artifacts:
        return ""
    selected: List[Dict[str, Any]] = []
    if artifact_ids:
        wanted = set(artifact_ids)
        selected = [artifact for artifact in artifacts if artifact.get("id") in wanted]
    if not selected:
        selected = artifacts[-max_items:]
    selected = selected[:max_items]
    blocks: List[str] = []
    for artifact in selected:
        art_id = artifact.get("id", "")
        tool_name = artifact.get("tool", "")
        status = artifact.get("status", "")
        content = artifact.get("content", "")
        blocks.append(
            f"[{art_id}] tool={tool_name} status={status}\n{truncate(content, limit=max_chars_per_item)}"
        )
    return "\n\n".join(blocks)


def parse_finish_payload(raw_result: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    text = (raw_result or "").strip()
    if not text:
        return None, "FINISH result is empty"

    candidate = text
    if not text.startswith("{"):
        extracted = extract_json_candidate(text)
        if extracted:
            candidate = extracted
    try:
        payload = json.loads(candidate)
    except Exception:
        return None, "FINISH result must be valid JSON object string"

    if not isinstance(payload, dict):
        return None, "FINISH payload must be a JSON object"

    summary = payload.get("summary")
    critical_facts = payload.get("critical_facts", [])
    artifact_ids = payload.get("artifact_ids", [])

    if not isinstance(summary, str) or not summary.strip():
        return None, "FINISH payload missing non-empty 'summary'"
    if not isinstance(critical_facts, list) or any(not isinstance(item, str) for item in critical_facts):
        return None, "FINISH payload 'critical_facts' must be a list of strings"
    if not isinstance(artifact_ids, list) or any(not isinstance(item, str) for item in artifact_ids):
        return None, "FINISH payload 'artifact_ids' must be a list of strings"

    return {
        "summary": summary.strip(),
        "critical_facts": [item.strip() for item in critical_facts if item.strip()],
        "artifact_ids": [item.strip() for item in artifact_ids if item.strip()],
    }, None


def filter_tools(allowed: Optional[List[str]]) -> List[Dict[str, Any]]:
    if not allowed:
        return []
    allowed_set = set(allowed)
    return [tool for tool in TOOLS_SPEC if tool["function"]["name"] in allowed_set]


def resolve_tool_path(raw_path: str) -> Path:
    path = Path((raw_path or "").strip())
    if path.is_absolute():
        return path
    if path.parent != Path("."):
        return path

    memory_candidate = MEMORY_DIR / path.name
    if memory_candidate.exists() or path.name in {"identity.md", "user.md", "soul.md"}:
        return memory_candidate
    return path


async def execute_tool(name: str, arguments: Dict[str, Any]) -> str:
    try:
        normalized_args, validation_error = normalize_and_validate_tool_arguments(name, arguments)
        if validation_error:
            return f"Tool validation error: {validation_error}"
        arguments = normalized_args or {}

        if name == "shell_command":
            proc = await asyncio.create_subprocess_shell(
                arguments.get("command", ""),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(Path.cwd()),
            )
            stdout, stderr = await proc.communicate()
            return f"STDOUT:\n{stdout.decode()}\nSTDERR:\n{stderr.decode()}"

        if name == "read_file":
            raw_path = (arguments.get("path") or "").strip()
            if not raw_path:
                return "Tool validation error: read_file requires non-empty 'path'"
            path = resolve_tool_path(raw_path)
            if not path.exists():
                return f"Error: File {path} does not exist"
            if path.is_dir():
                return f"Tool validation error: read_file path '{path}' is a directory; pass a file path"
            lines = path.read_text().splitlines()
            if len(lines) <= 200:
                return "\n".join(lines)
            return "\n".join(lines[:60]) + f"\n... (truncated, {len(lines)} total lines)"

        if name == "edit_file":
            raw_path = (arguments.get("path") or "").strip()
            if not raw_path:
                return "Tool validation error: edit_file requires non-empty 'path'"
            path = resolve_tool_path(raw_path)
            if path.exists() and path.is_dir():
                return f"Tool validation error: edit_file path '{path}' is a directory; pass a file path"

            if "search" in arguments or "replace" in arguments:
                if not path.exists():
                    return f"Error: File {path} does not exist"

                source_text = path.read_text()
                pattern_text = arguments.get("search", "")
                replacement = arguments.get("replace", "")
                use_regex = arguments.get("regex", True)
                max_replacements = arguments.get("max_replacements", 1)

                if use_regex:
                    try:
                        pattern = re.compile(pattern_text)
                    except re.error as exc:
                        return f"Tool validation error: parameter 'search' invalid regex: {exc}"
                    updated, count = pattern.subn(replacement, source_text, count=max_replacements)
                else:
                    if pattern_text not in source_text:
                        count = 0
                        updated = source_text
                    else:
                        updated = source_text.replace(pattern_text, replacement, max_replacements)
                        count = source_text.count(pattern_text) if max_replacements <= 0 else min(source_text.count(pattern_text), max_replacements)

                if count == 0:
                    return "Tool validation error: edit_file replacement found no matches"

                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(updated)
                return f"Successfully edited {path} ({count} replacement{'s' if count != 1 else ''})"

            path.parent.mkdir(parents=True, exist_ok=True)
            mode = arguments.get("mode", "write")
            content = arguments.get("content", "")
            if mode == "append":
                with open(path, "a") as file:
                    file.write(content)
            else:
                path.write_text(content)
            return f"Successfully wrote to {path}"

        if name == "regex_search":
            pattern = re.compile(arguments.get("pattern", ""))
            path = Path(arguments.get("path", ""))
            recursive = arguments.get("recursive", False)
            results: List[str] = []
            if path.is_file():
                content = path.read_text()
                matches = pattern.findall(content)
                if matches:
                    results.append(f"{path}: {matches}")
            elif path.is_dir():
                glob_pattern = "**/*" if recursive else "*"
                for file in path.glob(glob_pattern):
                    if file.is_file():
                        try:
                            content = file.read_text()
                            matches = pattern.findall(content)
                            if matches:
                                results.append(f"{file}: {matches}")
                        except Exception:
                            pass
            return "\n".join(results) if results else "No matches found"

        if name == "curl_request":
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.request(
                    method=arguments.get("method", "GET"),
                    url=arguments.get("url", ""),
                    headers=arguments.get("headers", {}),
                    content=arguments.get("body"),
                )
                return f"Status: {response.status_code}\n{response.text}"

        if name == "memory_create":
            import time
            content = PERSISTENT_MEMORY_PATH.read_text()
            data = json.loads(content)
            memory = {
                "id": len(data["memories"]),
                "content": arguments.get("content", ""),
                "category": arguments.get("category", "semantic"),
                "tags": arguments.get("tags", []),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            data["memories"].append(memory)
            PERSISTENT_MEMORY_PATH.write_text(json.dumps(data, indent=2))
            return f"Memory saved (id: {memory['id']})"

        if name == "memory_search":
            query = arguments.get("query", "").lower()
            tags = arguments.get("tags", [])
            content = PERSISTENT_MEMORY_PATH.read_text()
            data = json.loads(content)
            results = []
            for m in data["memories"]:
                score = 0
                if query in m["content"].lower():
                    score += 1
                if tags:
                    for t in tags:
                        if t in m.get("tags", []):
                            score += 1
                if score > 0:
                    results.append(m)

            res_list = results[:10]
            if not res_list:
                return "No memories found."
            return json.dumps(res_list, indent=2)

        if name == "skill_create":
            name_val = arguments.get("name", "")
            description = arguments.get("description", "")
            content_val = arguments.get("content", "")

            idx_content = SKILLS_INDEX_PATH.read_text()
            index = json.loads(idx_content)

            file_name = f"{name_val}.md"
            index["skills"].append({
                "name": name_val,
                "description": description,
                "file": file_name
            })
            SKILLS_INDEX_PATH.write_text(json.dumps(index, indent=2))
            (SKILLS_DIR / file_name).write_text(content_val)
            return f"Skill '{name_val}' created and saved to {file_name}."

        if name == "skill_load":
            name_val = arguments.get("name", "")
            # Try to find in index first
            idx_content = SKILLS_INDEX_PATH.read_text()
            index = json.loads(idx_content)
            skill_info = next((s for s in index["skills"] if s["name"] == name_val), None)

            file_path = None
            if skill_info:
                file_path = SKILLS_DIR / skill_info["file"]
            else:
                # Fallback to direct filename check
                for ext in [".md", ".txt", ""]:
                    p = SKILLS_DIR / (name_val + ext)
                    if p.exists() and p.is_file():
                        file_path = p
                        break

            if file_path and file_path.exists():
                return file_path.read_text()
            return f"Skill '{name_val}' not found."

        return f"Unknown tool: {name}"
    except Exception as exc:
        return f"Tool execution error: {exc}"


# =============================================================================
# PROMPTS AND MEMORY
# =============================================================================

def load_prompt(path: Path, default: str) -> str:
    if path.exists():
        return path.read_text()
    return default


def load_meta_agent_prompt() -> str:
    return load_prompt(PROMPTS_DIR / "meta_agent.txt", "You are a meta agent.")


def load_main_agent_prompt() -> str:
    return load_prompt(PROMPTS_DIR / "main_agent.txt", "You are a helpful assistant.")


def load_memory_files(files: List[str]) -> str:
    content: List[str] = []
    for filename in files:
        path = MEMORY_DIR / filename
        if path.exists():
            content.append(f"=== {filename.upper()} ===\n{path.read_text()}\n")
    return "\n".join(content)


def load_skills(files: List[str]) -> str:
    content: List[str] = []
    for filename in files:
        name = filename if filename.endswith(".md") else f"{filename}.md"
        path = SKILLS_DIR / name
        if path.exists():
            content.append(f"=== SKILL: {name} ===\n{path.read_text()}\n")
    return "\n".join(content)


def apply_memory_actions(memory_actions: List[Dict[str, Any]]) -> None:
    for action in memory_actions:
        action_type = (action.get("type") or "append").lower()
        file_name = action.get("file") or "user.md"
        content = action.get("content") or ""
        path = MEMORY_DIR / file_name
        path.parent.mkdir(parents=True, exist_ok=True)
        if action_type == "replace":
            path.write_text(content)
        else:
            if path.exists():
                existing = path.read_text()
                if existing and not existing.endswith("\n"):
                    existing += "\n"
                path.write_text(existing + content)
            else:
                path.write_text(content)


# =============================================================================
# JSON PARSING
# =============================================================================

def extract_json_candidate(text: str) -> Optional[str]:
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start:end + 1]


async def repair_json_with_llm(provider_cfg: ProviderConfig, raw_text: str) -> Optional[Dict[str, Any]]:
    repair_prompt = (
        "Fix the following into valid JSON. Output ONLY JSON with no extra text. "
        "If any fields are missing, add best-effort defaults."
    )
    messages = [
        {"role": "system", "content": repair_prompt},
        {"role": "user", "content": raw_text},
    ]
    try:
        response = await LLMProvider.call(provider_cfg, messages, response_format={"type": "json_object"})
        content = content_to_text(response["choices"][0]["message"]["content"])
        return json.loads(content)
    except Exception:
        return None


def content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content

    text_keys = ("text", "content", "value")

    def extract_from_dict(item: Dict[str, Any]) -> str:
        for key in text_keys:
            value = item.get(key)
            if isinstance(value, str):
                return value
        return ""

    if isinstance(content, dict):
        extracted = extract_from_dict(content)
        logger.debug("Normalized non-string provider content from dict to text (%d chars).", len(extracted))
        return extracted

    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                extracted = extract_from_dict(item)
                if extracted:
                    parts.append(extracted)
        normalized = "".join(parts)
        logger.debug("Normalized non-string provider content from list to text (%d chars).", len(normalized))
        return normalized

    if content is None:
        logger.debug("Normalized provider content from None to empty string.")
    else:
        logger.debug("Normalized provider content from %s to empty string.", type(content).__name__)
    return ""


def robust_json_loads(text: Any) -> Dict[str, Any]:
    """
    Attempts to parse JSON from the given text.
    If direct parsing fails, tries to extract a JSON candidate.
    If all fails, returns the raw text in a dict with a 'raw' key.
    """
    normalized_text = content_to_text(text)
    if not normalized_text:
        return {}

    cleaned_text = normalized_text.strip()

    # Try direct parse
    try:
        return json.loads(cleaned_text)
    except json.JSONDecodeError:
        pass

    # Try to extract candidate
    candidate = extract_json_candidate(cleaned_text)
    if candidate:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # Fallback to raw text
    return {"raw": cleaned_text}


# =============================================================================
# SESSION HELPERS
# =============================================================================

def ensure_session_policy(session_id: str) -> Dict[str, Any]:
    if session_id not in session_policies:
        session_policies[session_id] = {"shell_command_allowed": False}
    return session_policies[session_id]


def create_temp_session(title: str) -> Dict[str, Any]:
    session_id = str(uuid.uuid4())
    temp_sessions[session_id] = {
        "id": session_id,
        "title": title,
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "persistent": False,
        "turns": [],
    }
    ensure_session_policy(session_id)
    return temp_sessions[session_id]


def create_persistent_session(title: str) -> Dict[str, Any]:
    session_id = str(uuid.uuid4())
    conn = db_connect()
    conn.execute(
        "INSERT INTO sessions (id, title, created_at, updated_at, persistent) VALUES (?, ?, ?, ?, ?)",
        (session_id, title, now_iso(), now_iso(), 1),
    )
    conn.commit()
    conn.close()
    ensure_session_policy(session_id)
    return {"id": session_id, "title": title, "persistent": True}


def session_exists(session_id: str) -> bool:
    if session_id in temp_sessions:
        return True
    conn = db_connect()
    row = conn.execute("SELECT id FROM sessions WHERE id = ?", (session_id,)).fetchone()
    conn.close()
    return bool(row)


def get_session_turns(session_id: str) -> List[Dict[str, Any]]:
    if session_id in temp_sessions:
        return temp_sessions[session_id]["turns"]
    conn = db_connect()
    rows = conn.execute(
        "SELECT * FROM turns WHERE session_id = ? ORDER BY id ASC",
        (session_id,),
    ).fetchall()
    turns: List[Dict[str, Any]] = []
    for row in rows:
        tool_rows = conn.execute(
            "SELECT * FROM tool_events WHERE turn_id = ? ORDER BY id ASC",
            (row["id"],),
        ).fetchall()
        tool_events = []
        for tool_row in tool_rows:
            tool_events.append({
                "id": tool_row["id"],
                "name": tool_row["name"],
                "args": json.loads(tool_row["args_json"] or "{}"),
                "result": truncate(tool_row["result_text"]),
                "status": tool_row["status"],
                "created_at": tool_row["created_at"],
            })
        turns.append({
            "id": row["id"],
            "user_message": row["user_message"],
            "assistant_reply": row["assistant_reply"],
            "meta": json.loads(row["meta_json"] or "{}"),
            "tool_events": tool_events,
            "subagent_outputs": json.loads(row["subagent_json"] or "[]"),
            "architecture_mode": row["architecture_mode"] or "legacy",
            "created_at": row["created_at"],
        })
    conn.close()
    return turns


def get_recent_turns(session_id: str, limit: int) -> List[Dict[str, Any]]:
    if session_id in temp_sessions:
        return temp_sessions[session_id]["turns"][-limit:]
    conn = db_connect()
    rows = conn.execute(
        "SELECT * FROM turns WHERE session_id = ? ORDER BY id DESC LIMIT ?",
        (session_id, limit),
    ).fetchall()
    conn.close()
    return list(reversed([
        {
            "user_message": row["user_message"],
            "assistant_reply": row["assistant_reply"],
        }
        for row in rows
    ]))


def get_recent_tool_logs(session_id: str, limit: int = 20) -> List[Dict[str, Any]]:
    if session_id in temp_sessions:
        turns = temp_sessions[session_id]["turns"]
        tool_logs: List[Dict[str, Any]] = []
        for turn in turns:
            tool_logs.extend(turn.get("tool_events", []))
        return tool_logs[-limit:]
    conn = db_connect()
    rows = conn.execute(
        """
        SELECT te.* FROM tool_events te
        JOIN turns t ON t.id = te.turn_id
        WHERE t.session_id = ?
        ORDER BY te.id DESC
        LIMIT ?
        """,
        (session_id, limit),
    ).fetchall()
    conn.close()
    return [
        {
            "id": row["id"],
            "name": row["name"],
            "args": json.loads(row["args_json"] or "{}"),
            "result": truncate(row["result_text"]),
            "status": row["status"],
            "created_at": row["created_at"],
        }
        for row in rows
    ]


def build_session_context(session_id: str) -> Tuple[str, str]:
    turns = get_recent_turns(session_id, MAX_META_TURNS)
    conversation: List[Dict[str, Any]] = []
    for turn in turns:
        conversation.append({"role": "user", "content": turn["user_message"]})
        conversation.append({"role": "assistant", "content": turn["assistant_reply"]})
    tool_logs = get_recent_tool_logs(session_id, 20)

    working_memory = ""
    if session_id in temp_sessions:
        working_memory = temp_sessions[session_id].get("working_memory", "")
    else:
        conn = db_connect()
        row = conn.execute("SELECT working_memory FROM sessions WHERE id = ?", (session_id,)).fetchone()
        conn.close()
        if row:
            working_memory = row["working_memory"] or ""

    session_context = {
        "conversation": conversation,
        "tool_logs": tool_logs,
        "working_memory": working_memory
    }
    return json.dumps(session_context, indent=2), working_memory


def store_turn(
    session_id: str,
    persistent: bool,
    user_message: str,
    assistant_reply: str,
    meta_json: Dict[str, Any],
    tool_events: List[Dict[str, Any]],
    subagent_outputs: List[Dict[str, Any]],
    architecture_mode: str = "legacy",
) -> None:
    canonical_reply = content_to_text(assistant_reply)
    if not persistent:
        temp_sessions[session_id]["turns"].append({
            "user_message": user_message,
            "assistant_reply": canonical_reply,
            "meta": meta_json,
            "tool_events": [
                {
                    "id": ev.get("id"),
                    "name": ev["name"],
                    "args": ev["args"],
                    "result": truncate(ev["result"]),
                    "status": ev["status"],
                    "created_at": ev["created_at"],
                }
                for ev in tool_events
            ],
            "subagent_outputs": subagent_outputs,
            "architecture_mode": architecture_mode,
            "created_at": now_iso(),
        })
        temp_sessions[session_id]["updated_at"] = now_iso()
        return

    conn = db_connect()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO turns (session_id, user_message, assistant_reply, meta_json, tool_json, subagent_json, architecture_mode, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            session_id,
            user_message,
            canonical_reply,
            json.dumps(meta_json),
            json.dumps([]),
            json.dumps(subagent_outputs),
            architecture_mode,
            now_iso(),
        ),
    )
    turn_id = cur.lastrowid
    for ev in tool_events:
        cur.execute(
            """
            INSERT INTO tool_events (turn_id, name, args_json, result_text, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                turn_id,
                ev["name"],
                json.dumps(ev["args"]),
                ev["result"],
                ev["status"],
                ev["created_at"],
            ),
        )
        ev["id"] = cur.lastrowid
    tool_summary = [
        {
            "id": ev.get("id"),
            "name": ev["name"],
            "args": ev["args"],
            "status": ev["status"],
            "result": truncate(ev["result"]),
            "created_at": ev["created_at"],
        }
        for ev in tool_events
    ]
    cur.execute("UPDATE turns SET tool_json = ? WHERE id = ?", (json.dumps(tool_summary), turn_id))
    cur.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (now_iso(), session_id))
    conn.commit()
    conn.close()


# =============================================================================
# AGENT ORCHESTRATION
# =============================================================================

async def run_meta_layer(
    session_context: str,
    skills_index: str,
    memory_context: str,
    working_memory: str,
    on_agent_complete: Optional[Any] = None,
    **kwargs
) -> Dict[str, Any]:
    meta_agents = {
        "intent": "prompts/meta_intent.txt",
        "tone": "prompts/meta_tone.txt",
        "user": "prompts/meta_user.txt",
        "subject": "prompts/meta_subject.txt",
        "needs": "prompts/meta_needs.txt",
        "patterns": "prompts/meta_patterns.txt",
    }
    
    planner_prompt = Path("prompts/planner.txt").read_text()

    combined_prompt = (
        "You are a meta-analysis and planning system.\n"
        "Perform 6 analyses and generate a planner decision in one response.\n"
        "Output a single JSON object with keys: intent, tone, user, subject, needs, patterns, plan.\n"
        "The 'plan' key must be an object.\n\n"
    )
    for name, path in meta_agents.items():
        prompt = Path(path).read_text()
        if name == "needs":
            prompt = prompt.replace("{skills_index}", skills_index)
        combined_prompt += f"--- {name.upper()} ---\n{prompt}\n\n"

    combined_prompt += (
        "--- PLANNER ---\n"
        f"{planner_prompt}\n\n"
        "Planner constraints:\n"
        "1. Base decisions on the 6 analyses.\n"
        "2. Keep plan fields compatible with existing schema.\n"
        "3. Prefer just_chat=true unless tools/subagents are clearly needed.\n"
    )

    system_prompt = f"{combined_prompt}\n\n{memory_context}"
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"{session_context}\n\nWORKING MEMORY:\n{working_memory}\n\nAVAILABLE SKILLS:\n{skills_index}"
        }
    ]

    try:
        # Removed strict JSON requirement
        resp = await LLMProvider.call(config.meta, messages, **kwargs)
        content = content_to_text(resp["choices"][0]["message"]["content"])
        results = robust_json_loads(content)
        if not isinstance(results, dict):
            results = {"raw": content}
        meta_dict_keys = ("intent", "tone", "user", "subject", "needs", "patterns")
        for key in meta_dict_keys:
            value = results.get(key)
            if value is None or isinstance(value, dict):
                continue
            # MetaOutput expects dict sections; wrap scalar/list outputs safely.
            results[key] = {"summary": content_to_text(value)}

        if not isinstance(results.get("plan"), dict):
            inferred_plan = {}
            planner_keys = (
                "plan",
                "notes_for_main",
                "load_memories",
                "load_skills",
                "use_sub_agents",
                "sub_agent_tasks",
                "just_chat",
            )
            for key in planner_keys:
                if key in results:
                    inferred_plan[key] = results.get(key)
            if inferred_plan and isinstance(inferred_plan.get("plan"), str):
                results["plan"] = inferred_plan
            else:
                results["plan"] = {"just_chat": True, "plan": "respond conversationally"}
        if on_agent_complete:
            for name, res in results.items():
                await on_agent_complete(name, res)
        return results
    except Exception as e:
        logger.error(f"Error in combined meta agent call: {e}")
        results = {"error": str(e)}
        if on_agent_complete:
            await on_agent_complete("error", results)
        return results

async def run_planner_layer(meta_output: Dict[str, Any], working_memory: str, skills_index: str, **kwargs) -> Dict[str, Any]:
    prompt = Path("prompts/planner.txt").read_text()
    context = f"META ANALYSIS:\n{json.dumps(meta_output, indent=2)}\n\nWORKING MEMORY:\n{working_memory}\n\nAVAILABLE SKILLS:\n{skills_index}"

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": context}
    ]

    try:
        # Removed strict JSON requirement
        resp = await LLMProvider.call(config.meta, messages, **kwargs)
        content = content_to_text(resp["choices"][0]["message"]["content"])
        return robust_json_loads(content)
    except Exception as e:
        logger.error(f"Error in planner layer: {e}")
        return {"error": str(e), "just_chat": True, "plan": "respond conversationally"}

async def run_gatekeeper_layer(conversation: str, agent_response: str, user_md: str, identity_md: str, **kwargs) -> Dict[str, Any]:
    prompt = Path("prompts/gatekeeper.txt").read_text()
    context = f"CONVERSATION:\n{conversation}\n\nAGENT RESPONSE:\n{agent_response}\n\nCURRENT USER.MD:\n{user_md}\n\nCURRENT IDENTITY.MD:\n{identity_md}"

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": context}
    ]

    try:
        # Removed strict JSON requirement
        resp = await LLMProvider.call(config.meta, messages, **kwargs)
        content = content_to_text(resp["choices"][0]["message"]["content"])
        return robust_json_loads(content)
    except Exception as e:
        logger.error(f"Error in gatekeeper layer: {e}")
        return {"error": str(e)}

async def apply_gatekeeper(session_id: str, persistent: bool, result: Dict[str, Any]) -> None:
    if not result:
        return

    if result.get("save_memory") and result.get("memory"):
        m = result["memory"]
        # Use our memory_create tool logic (to be implemented/updated)
        await execute_tool("memory_create", {
            "content": m.get("content", ""),
            "category": m.get("category", ""),
            "tags": m.get("tags", [])
        })
        logger.info(f"  [GATE] Memory saved for session {session_id}")

    if result.get("update_user") and result.get("user_update"):
        path = MEMORY_DIR / "user.md"
        current = path.read_text() if path.exists() else ""
        path.write_text(current + "\n" + result["user_update"])
        logger.info(f"  [GATE] User updated")

    if result.get("update_identity") and result.get("identity_update"):
        path = MEMORY_DIR / "identity.md"
        current = path.read_text() if path.exists() else ""
        path.write_text(current + "\n" + result["identity_update"])
        logger.info(f"  [GATE] Identity updated")

    if "working_memory_summary" in result:
        working_mem = result["working_memory_summary"]
        conn = db_connect()
        conn.execute("UPDATE sessions SET working_memory = ? WHERE id = ?", (working_mem, session_id))
        conn.commit()
        conn.close()
        logger.info(f"  [GATE] Working memory updated for session {session_id}")


def build_main_messages(
    user_message: str,
    meta_output: MetaOutput,
    history_turns: List[Dict[str, Any]],
    subagent_outputs: List[Dict[str, Any]],
    loaded_context: str = ""
) -> List[Dict[str, Any]]:
    main_prompt = load_main_agent_prompt()

    available_memory = list_available_memory_files()
    available_skills = list_available_skills()

    # Planner's notes are now our extra instructions
    extra_instructions = ""
    if meta_output.plan:
        extra_instructions = meta_output.plan.get("notes_for_main", "")

    memory_text = ", ".join(available_memory) if available_memory else "none"
    skills_text = ", ".join(available_skills) if available_skills else "none"

    system_parts = [
        main_prompt,
        f"Available memory files: {memory_text}. Use read_file to load if needed.",
        f"Available skills: {skills_text}. Use read_file to load if needed.",
    ]
    if extra_instructions:
        system_parts.append(f"Instruction from Planner:\n{extra_instructions}")

    if loaded_context:
        system_parts.append(f"LOADED CONTEXT:\n{loaded_context}")

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": "\n\n".join(system_parts)},
        {"role": "system", "content": f"Meta analysis and Plan JSON:\n{meta_output.model_dump_json(indent=2)}"},
    ]

    if subagent_outputs:
        messages.append({
            "role": "system",
            "content": f"Subagent results:\n{json.dumps(subagent_outputs, indent=2)}"
        })

    for turn in history_turns:
        messages.append({"role": "user", "content": turn["user_message"]})
        messages.append({"role": "assistant", "content": turn["assistant_reply"]})

    messages.append({"role": "user", "content": user_message})
    return messages


async def run_tool_loop(
    provider_cfg: ProviderConfig,
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    approval_mode: Literal["ask", "allow", "auto_deny"],
    allow_shell: bool,
    user_message: str,
    **kwargs
) -> Dict[str, Any]:
    tool_events: List[Dict[str, Any]] = []
    subagent_outputs: List[Dict[str, Any]] = []
    for _ in range(MAX_TOOL_ROUNDS):
        response = await LLMProvider.call(provider_cfg, messages, tools=tools, **kwargs)
        assistant_msg = response["choices"][0]["message"]
        assistant_content = content_to_text(assistant_msg.get("content", ""))
        tool_calls = assistant_msg.get("tool_calls")
        if tool_calls:
            for tool_call in tool_calls:
                if not tool_call.get("id"):
                    tool_call["id"] = f"call_{uuid.uuid4()}"

            def needs_approval(tc: Dict[str, Any]) -> bool:
                fn = tc.get("function", {})
                if fn.get("name") != "shell_command":
                    return False
                return approval_mode == "ask"

            if approval_mode == "ask" and any(needs_approval(tc) for tc in tool_calls):
                pending_messages = messages + [{
                    "role": "assistant",
                    "content": assistant_content,
                    "tool_calls": tool_calls,
                }]
                return {
                    "status": "needs_approval",
                    "pending_messages": pending_messages,
                    "pending_tool_calls": tool_calls,
                    "tool_events": tool_events,
                    "subagent_outputs": subagent_outputs,
                }

            messages.append({
                "role": "assistant",
                "content": assistant_content,
                "tool_calls": tool_calls,
            })

            for tool_call in tool_calls:
                fn = tool_call.get("function", {})
                name = fn.get("name")
                args_raw = fn.get("arguments", "{}")
                try:
                    args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                except Exception:
                    args = {}

                if name == "shell_command" and approval_mode == "auto_deny":
                    result = "DENIED: shell_command is not allowed for this agent."
                    status = "denied"
                elif name == "shell_command" and approval_mode == "ask":
                    result = "DENIED: shell_command requires explicit approval."
                    status = "denied"
                elif name == "spawn_subagents":
                    subagents_spec = args.get("subagents", [])
                    outputs = await run_subagents(subagents_spec)
                    subagent_outputs.extend(outputs)
                    result = json.dumps(outputs, indent=2)
                    status = "ok"
                else:
                    result = await execute_tool(name, args)
                    status = "error" if result.startswith("Tool execution error") else "ok"

                tool_events.append({
                    "id": None,
                    "name": name,
                    "args": args,
                    "result": result,
                    "status": status,
                    "created_at": now_iso(),
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.get("id"),
                    "content": result,
                })
            continue

        return {
            "status": "ok",
            "content": assistant_content,
            "tool_events": tool_events,
            "subagent_outputs": subagent_outputs,
        }

    return {
        "status": "ok",
        "content": "",
        "tool_events": tool_events,
        "subagent_outputs": subagent_outputs,
    }


async def run_subagents(subagents_spec: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
    if not subagents_spec:
        return []

    async def run_single(spec: Dict[str, Any]) -> Dict[str, Any]:
        role = spec.get("role", "subagent")
        task = spec.get("task", "")
        context = spec.get("context", "")
        allowed_tools = spec.get("tools") or []
        tools = filter_tools(allowed_tools)

        system_prompt = (
            "You are a focused sub-agent. "
            "Do the task and return a concise result. "
            "Use tools when needed to inspect or change files. "
            "Do not chat with the user."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Role: {role}\nTask: {task}\nContext: {context}"},
        ]
        loop_result = await run_tool_loop(
            config.sub,
            messages,
            tools,
            approval_mode="auto_deny",
            allow_shell=False,
            user_message="",
            **kwargs
        )
        return {
            "role": role,
            "task": task,
            "result": loop_result.get("content", ""),
            "tool_events": [
                {
                    "name": ev["name"],
                    "args": ev["args"],
                    "status": ev["status"],
                    "result": truncate(ev["result"]),
                }
                for ev in loop_result.get("tool_events", [])
            ],
        }

    results = await asyncio.gather(*[run_single(spec) for spec in subagents_spec])
    return results


# =============================================================================
# V2 STRICT-SYNTAX ARCHITECTURE
# =============================================================================

V2_MAX_MAIN_ROUNDS = 4
V2_MAX_ORCHESTRATOR_ROUNDS = 10
V2_MAX_SUBAGENT_ROUNDS = 6


def load_v2_prompt(path: str, default: str) -> str:
    return load_prompt(PROMPTS_DIR / "v2" / path, default)


def unescape_tag_value(value: str) -> str:
    try:
        return bytes(value, "utf-8").decode("unicode_escape")
    except Exception:
        return value.replace('\\"', '"').replace("\\\\", "\\")


def parse_param_list(params_text: str) -> Optional[Dict[str, str]]:
    text = (params_text or "").strip()
    if not text:
        return {}

    pattern = re.compile(
        r'\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*"((?:[^"\\]|\\.)*)"\s*(,|$)',
        re.DOTALL,
    )
    pos = 0
    params: Dict[str, str] = {}
    while pos < len(text):
        match = pattern.match(text, pos)
        if not match:
            return None
        key = match.group(1)
        value = unescape_tag_value(match.group(2))
        params[key] = value
        pos = match.end()
        if match.group(3) == "":
            break
    if text[pos:].strip():
        return None
    return params


def parse_attr_list(attrs_text: str) -> Optional[Dict[str, str]]:
    text = attrs_text or ""
    pattern = re.compile(
        r'\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*"((?:[^"\\]|\\.)*)"\s*',
        re.DOTALL,
    )
    pos = 0
    attrs: Dict[str, str] = {}
    while pos < len(text):
        match = pattern.match(text, pos)
        if not match:
            if text[pos:].strip():
                return None
            break
        key = match.group(1)
        value = unescape_tag_value(match.group(2))
        attrs[key] = value
        pos = match.end()
    return attrs


def parse_tool_tag(text: str) -> Optional[Dict[str, Any]]:
    if not isinstance(text, str):
        return None
    stripped = text.strip()
    match = re.fullmatch(r"\[TOOL\](.+)\[/TOOL\]", stripped, flags=re.DOTALL)
    if not match:
        return None

    invocation = match.group(1).strip()
    call_match = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)\((.*)\)", invocation, flags=re.DOTALL)
    if not call_match:
        return None

    tool_name = call_match.group(1)
    params_text = call_match.group(2).strip()
    params = parse_param_list(params_text)
    if params is None:
        return None

    return {
        "type": "tool",
        "tool_name": tool_name,
        "params": params,
        "raw": stripped,
    }


def parse_subagent_tag(text: str) -> Optional[Dict[str, Any]]:
    if not isinstance(text, str):
        return None
    stripped = text.strip()
    match = re.fullmatch(r"\[SUBAGENT\s+(.+)\]", stripped, flags=re.DOTALL)
    if not match:
        return None

    attrs = parse_attr_list(match.group(1))
    if attrs is None:
        return None
    if "task" not in attrs or "tools" not in attrs:
        return None

    return {
        "type": "subagent",
        "task": attrs["task"],
        "tools": attrs["tools"],
        "attrs": attrs,
        "raw": stripped,
    }


def parse_tools_csv(tools_text: str) -> List[str]:
    raw = (tools_text or "").strip()
    if not raw:
        return []
    if raw == "*":
        return [tool["function"]["name"] for tool in TOOLS_SPEC]
    if "," in raw:
        return [part.strip() for part in raw.split(",") if part.strip()]
    return [part.strip() for part in raw.split() if part.strip()]


def v2_known_tool_names() -> List[str]:
    # In v2, sub-agent spawning is explicit via [SUBAGENT], not a callable tool.
    return [tool["function"]["name"] for tool in TOOLS_SPEC if tool["function"]["name"] != "spawn_subagents"]


def sanitize_subagent_tools(tools: List[str]) -> List[str]:
    allowed = set(v2_known_tool_names())
    seen = set()
    result: List[str] = []
    for tool_name in tools or []:
        normalized = (tool_name or "").strip()
        if not normalized or normalized in seen:
            continue
        if normalized in allowed:
            seen.add(normalized)
            result.append(normalized)
    return result


def normalize_action_signature(action_type: str, action_name: str, payload: Dict[str, Any]) -> str:
    try:
        payload_json = json.dumps(payload or {}, sort_keys=True, ensure_ascii=True)
    except Exception:
        payload_json = str(payload)
    return f"{action_type}:{action_name}:{payload_json}"


def tool_outcome(status: str, message: str, terminal: bool, retryable: bool) -> Dict[str, Any]:
    return {
        "status": status,
        "message": message,
        "terminal": terminal,
        "retryable": retryable,
    }


def looks_like_action_markup(text: str) -> bool:
    if not isinstance(text, str):
        return False
    return "[TOOL" in text or "[SUBAGENT" in text


async def repair_agent_action_output(
    provider_cfg: ProviderConfig,
    raw_text: str,
    role: str,
    allowed_tool_names: List[str],
    allow_subagent: bool,
    allow_direct_text: bool,
    **kwargs
) -> str:
    tools_text = ", ".join(allowed_tool_names) if allowed_tool_names else "none"
    system_prompt = (
        "Repair the user's malformed agent action output.\n"
        "Rules:\n"
        "1. Output exactly one action in case-sensitive syntax, or plain text if allowed.\n"
        "2. Valid tool syntax: [TOOL]tool_name(param=\"value\")[/TOOL]\n"
        "3. Valid subagent syntax: [SUBAGENT task=\"...\" tools=\"...\"]\n"
        "4. FINISH must be [TOOL]FINISH(result=\"...\")[/TOOL]\n"
        "5. Do not explain.\n"
        f"Role: {role}\n"
        f"Allowed tools: {tools_text}\n"
        f"Allow [SUBAGENT]: {'yes' if allow_subagent else 'no'}\n"
        f"Allow direct text: {'yes' if allow_direct_text else 'no'}\n"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": raw_text},
    ]
    try:
        response = await LLMProvider.call(provider_cfg, messages, **kwargs)
        return content_to_text(response["choices"][0]["message"]["content"]).strip()
    except Exception as exc:
        logger.warning("Action repair failed for role %s: %s", role, exc)
        return ""


async def parse_agent_action(
    role: str,
    text: str,
    provider_cfg: ProviderConfig,
    allowed_tool_names: List[str],
    allow_subagent: bool,
    allow_direct_text: bool,
    **kwargs
) -> Dict[str, Any]:
    def strict_parse(candidate: str) -> Dict[str, Any]:
        parsed_tool = parse_tool_tag(candidate)
        if parsed_tool:
            tool_name = parsed_tool["tool_name"]
            if tool_name not in allowed_tool_names:
                return {"type": "invalid", "reason": f"Tool '{tool_name}' is not allowed", "raw": candidate}
            return {
                "type": "tool",
                "tool_name": tool_name,
                "params": parsed_tool["params"],
                "raw": parsed_tool["raw"],
            }

        if allow_subagent:
            parsed_sub = parse_subagent_tag(candidate)
            if parsed_sub:
                return {
                    "type": "subagent",
                    "task": parsed_sub["task"],
                    "tools": parse_tools_csv(parsed_sub["tools"]),
                    "attrs": parsed_sub["attrs"],
                    "raw": parsed_sub["raw"],
                }

        if allow_direct_text and not looks_like_action_markup(candidate):
            return {"type": "direct_text", "content": candidate.strip(), "raw": candidate}

        return {"type": "invalid", "reason": "No valid action found", "raw": candidate}

    first = strict_parse(text)
    if first["type"] != "invalid":
        first["repaired"] = False
        return first

    repaired_output = await repair_agent_action_output(
        provider_cfg,
        text,
        role,
        allowed_tool_names,
        allow_subagent,
        allow_direct_text,
        **kwargs,
    )
    if repaired_output:
        second = strict_parse(repaired_output)
        if second["type"] != "invalid":
            second["repaired"] = True
            second["repair_source"] = text
            return second

    return {
        "type": "invalid",
        "reason": first.get("reason", "Invalid action"),
        "raw": text,
        "repaired": False,
    }


async def execute_v2_tool(
    name: str,
    arguments: Dict[str, Any],
    user_message: str,
    allow_shell: bool,
    approval_mode: Literal["ask", "allow", "auto_deny"] = "auto_deny",
    shell_approved: bool = False,
) -> Dict[str, Any]:
    del user_message
    del allow_shell

    normalized_args, validation_error = normalize_and_validate_tool_arguments(name, arguments)
    if validation_error:
        return tool_outcome(
            "rejected",
            f"Tool validation error: {validation_error}",
            terminal=True,
            retryable=False,
        )
    arguments = normalized_args or {}

    if name == "FINISH":
        return tool_outcome("ok", "FINISH intercepted.", terminal=False, retryable=False)

    if name == "shell_command" and approval_mode == "ask" and not shell_approved:
        pending = build_pending_tool(name, arguments)
        return {
            **tool_outcome(
                "needs_approval",
                "Tool approval required to continue.",
                terminal=False,
                retryable=False,
            ),
            "pending_tools": [pending],
        }

    if name == "shell_command" and approval_mode == "auto_deny":
        return tool_outcome(
            "denied",
            "DENIED: shell_command is not allowed for this session.",
            terminal=True,
            retryable=False,
        )
    if name == "shell_command" and not shell_approved and approval_mode != "allow":
        return tool_outcome(
            "denied",
            "DENIED: shell_command was not approved.",
            terminal=True,
            retryable=False,
        )

    result = await execute_tool(name, arguments)
    if result.startswith("Tool validation error:"):
        return tool_outcome("rejected", result, terminal=True, retryable=False)
    if result.startswith("Unknown tool:"):
        return tool_outcome("unknown_tool", result, terminal=True, retryable=False)
    if result.startswith("Tool execution error"):
        return tool_outcome("error", result, terminal=False, retryable=True)
    return tool_outcome("ok", result, terminal=False, retryable=False)


def build_transcript(turns: List[Dict[str, Any]], current_user_message: str) -> str:
    lines: List[str] = []
    for turn in turns:
        lines.append(f"User: {turn.get('user_message', '')}")
        lines.append(f"Assistant: {turn.get('assistant_reply', '')}")
    if current_user_message:
        lines.append(f"User: {current_user_message}")
    return "\n".join(lines)


async def run_v2_summarizer(query: str, transcript: str, **kwargs) -> str:
    prompt = load_v2_prompt(
        "summarizer.txt",
        "You are Summarizer Agent. Return concise plain text only."
    )
    messages = [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": f"Query:\n{query}\n\nConversation History:\n{transcript}",
        },
    ]
    response = await LLMProvider.call(get_summarizer_cfg(), messages, **kwargs)
    summary = content_to_text(response["choices"][0]["message"]["content"]).strip()
    return summary or "Not found in conversation."


def format_tool_events_for_response(tool_events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "id": ev.get("id"),
            "name": ev.get("name"),
            "args": ev.get("args", {}),
            "status": ev.get("status", "ok"),
            "result": truncate(ev.get("result", "")),
            "created_at": ev.get("created_at"),
            "artifact_id": ev.get("artifact_id"),
        }
        for ev in tool_events
    ]


async def run_v2_subagent(
    task: str,
    tools: List[str],
    context: str,
    user_message: str,
    allow_shell: bool,
    queue_event: Optional[Callable[[str, Dict[str, Any]], Awaitable[None]]] = None,
    state: Optional[Dict[str, Any]] = None,
    shell_approval: Optional[bool] = None,
    approval_mode: Literal["ask", "allow", "auto_deny"] = "ask",
    **kwargs
) -> Dict[str, Any]:
    if state is None:
        prompt = load_v2_prompt(
            "sub_agent.txt",
            "You are a Sub-Agent. Use allowed tools and finish with [TOOL]FINISH(result=\"...\")[/TOOL]."
        )
        allowed_tools = sanitize_subagent_tools(tools)
        if not allowed_tools:
            result_text = "Sub-agent auto-finish: no valid tools were provided for this task."
            return {
                "status": "ok",
                "role": "subagent",
                "task": task,
                "result": result_text,
                "output": result_text,
                "tool_events": [],
                "warnings": ["no valid tools after sanitization"],
            }
        if "FINISH" not in allowed_tools:
            allowed_tools.append("FINISH")

        system_prompt = (
            f"{prompt}\n\n"
            "Syntax is case-sensitive.\n"
            f"Allowed tools for this task: {', '.join(allowed_tools)}"
        )
        state = {
            "task": task,
            "context": context,
            "user_message": user_message,
            "allow_shell": allow_shell,
            "allowed_tools": allowed_tools,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"task={task}\ncontext={context}"},
            ],
            "tool_events": [],
            "partial_notes": [],
            "warnings": [],
            "action_attempts": {},
            "action_failures": {},
            "round": 0,
            "pending_tool": None,
            "artifacts": [],
            "finish_payload": None,
        }
    else:
        task = state.get("task", task)
        context = state.get("context", context)
        user_message = state.get("user_message", user_message)
        allow_shell = bool(state.get("allow_shell", allow_shell))

    messages = state["messages"]
    tool_events = state["tool_events"]
    partial_notes = state["partial_notes"]
    warnings = state["warnings"]
    action_attempts = state["action_attempts"]
    action_failures = state["action_failures"]
    allowed_tools = state["allowed_tools"]
    artifacts = state["artifacts"]

    def finish_result(result_text: str) -> Dict[str, Any]:
        return {
            "status": "ok",
            "role": "subagent",
            "task": task,
            "result": result_text,
            "output": result_text,
            "tool_events": tool_events,
            "warnings": warnings,
            "artifacts": artifacts,
            "finish_payload": state.get("finish_payload"),
        }

    async def apply_outcome(tool_name: str, args: Dict[str, Any], outcome: Dict[str, Any], signature: str) -> Optional[Dict[str, Any]]:
        result = outcome["message"]
        status = outcome["status"]
        terminal = bool(outcome.get("terminal"))
        action_attempts[signature] = action_attempts.get(signature, 0) + 1
        if status != "ok":
            action_failures[signature] = action_failures.get(signature, 0) + 1

        event_status = status
        if terminal and status in {"rejected", "denied"}:
            event_status = f"terminal_{status}"

        artifact = create_artifact(
            artifacts,
            owner="subagent",
            tool_name=tool_name,
            args=args,
            status=event_status,
            content=result,
        )

        if queue_event:
            await queue_event("tool_result", {"name": tool_name, "status": event_status, "result": truncate(result)})
        tool_events.append({
            "name": tool_name,
            "args": args,
            "status": event_status,
            "result": truncate(result),
            "artifact_id": artifact.get("id"),
        })
        partial_notes.append(f"{tool_name}:{event_status}")
        messages.append({
            "role": "system",
            "content": (
                f"Tool result ({tool_name}, {event_status}):\n{result}\n"
                f"Artifact ID: {artifact.get('id')}"
            ),
        })

        if terminal:
            warnings.append(f"terminal failure on '{tool_name}' ({status})")
            auto_finish = (
                f"Sub-agent auto-finish: terminal failure on '{tool_name}' ({status}). "
                "Returning partial progress."
            )
            if partial_notes:
                auto_finish += f" Notes: {' | '.join(partial_notes)}"
            return finish_result(auto_finish)

        if action_failures.get(signature, 0) >= 2:
            warnings.append(f"loop guard triggered for repeated failing action '{tool_name}'")
            auto_finish = (
                f"Sub-agent auto-finish: repeated failure on '{tool_name}'. "
                "Returning partial progress."
            )
            if partial_notes:
                auto_finish += f" Notes: {' | '.join(partial_notes)}"
            return finish_result(auto_finish)
        return None

    while state["round"] < V2_MAX_SUBAGENT_ROUNDS:
        pending_tool = state.get("pending_tool")
        if pending_tool:
            tool_name = pending_tool["tool_name"]
            args = pending_tool["args"]
            signature = pending_tool["signature"]
            if shell_approval is None:
                return {
                    "status": "needs_approval",
                    "pending_tools": [build_pending_tool(tool_name, args)],
                    "state": state,
                    "role": "subagent",
                    "task": task,
                }
            if shell_approval:
                outcome = await execute_v2_tool(
                    tool_name,
                    args,
                    user_message=user_message,
                    allow_shell=allow_shell,
                    approval_mode="allow",
                    shell_approved=True,
                )
            else:
                outcome = tool_outcome(
                    "denied",
                    "DENIED: shell_command was not approved.",
                    terminal=True,
                    retryable=False,
                )
            shell_approval = None
            state["pending_tool"] = None
            final = await apply_outcome(tool_name, args, outcome, signature)
            if final:
                return final
            continue

        response = await LLMProvider.call(config.sub, messages, **kwargs)
        state["round"] += 1
        assistant_raw = content_to_text(response["choices"][0]["message"]["content"]).strip()
        parsed = await parse_agent_action(
            role="subagent",
            text=assistant_raw,
            provider_cfg=config.sub,
            allowed_tool_names=allowed_tools,
            allow_subagent=False,
            allow_direct_text=False,
            **kwargs,
        )

        if parsed["type"] == "invalid":
            warnings.append(parsed.get("reason", "invalid action"))
            fallback = (
                "Sub-agent auto-finish: malformed syntax after repair attempt. "
                "Returning partial progress."
            )
            if partial_notes:
                fallback += f" Notes: {' | '.join(partial_notes)}"
            return finish_result(fallback)

        if parsed["type"] != "tool":
            warnings.append("sub-agent returned non-tool action")
            return finish_result("Sub-agent auto-finish: invalid action type.")

        tool_name = parsed["tool_name"]
        args = parsed["params"]
        if tool_name == "FINISH":
            payload, payload_error = parse_finish_payload(args.get("result", ""))
            if payload_error:
                warnings.append(f"invalid FINISH payload: {payload_error}")
                return finish_result(f"Sub-agent auto-finish: invalid FINISH payload ({payload_error}).")
            missing = [art_id for art_id in payload.get("artifact_ids", []) if not any(a.get("id") == art_id for a in artifacts)]
            if missing:
                warnings.append(f"FINISH referenced missing artifact ids: {', '.join(missing)}")
            state["finish_payload"] = payload
            result_text = payload.get("summary", "").strip()
            return finish_result(result_text)

        signature = normalize_action_signature("tool", tool_name, args)
        messages.append({"role": "assistant", "content": parsed.get("raw", assistant_raw)})
        if queue_event:
            await queue_event("tool_call", {"name": tool_name, "args": args})
        outcome = await execute_v2_tool(
            tool_name,
            args,
            user_message=user_message,
            allow_shell=allow_shell,
            approval_mode=approval_mode,
            shell_approved=False,
        )
        if outcome["status"] == "needs_approval":
            state["pending_tool"] = {
                "tool_name": tool_name,
                "args": args,
                "signature": signature,
            }
            return {
                "status": "needs_approval",
                "pending_tools": outcome.get("pending_tools", [build_pending_tool(tool_name, args)]),
                "state": state,
                "role": "subagent",
                "task": task,
            }
        final = await apply_outcome(tool_name, args, outcome, signature)
        if final:
            return final

    warnings.append("max rounds reached")
    return finish_result("Sub-agent auto-finish: max rounds reached without FINISH.")


async def run_v2_orchestrator(
    tasks: str,
    context: str,
    user_message: str,
    allow_shell: bool,
    queue_event: Optional[Callable[[str, Dict[str, Any]], Awaitable[None]]] = None,
    state: Optional[Dict[str, Any]] = None,
    shell_approval: Optional[bool] = None,
    approval_mode: Literal["ask", "allow", "auto_deny"] = "ask",
    **kwargs
) -> Dict[str, Any]:
    if state is None:
        prompt = load_v2_prompt(
            "orchestrator.txt",
            "You are Orchestrator Agent. Execute one action per turn and finish with FINISH."
        )
        state = {
            "tasks": tasks,
            "context": context,
            "user_message": user_message,
            "allow_shell": allow_shell,
            "allowed_tool_names": v2_known_tool_names() + ["FINISH"],
            "messages": [
                {"role": "system", "content": f"{prompt}\n\nSyntax is case-sensitive."},
                {"role": "user", "content": f"tasks:\n{tasks}\n\ncontext:\n{context or ''}"},
            ],
            "tool_events": [],
            "subagent_outputs": [],
            "warnings": [],
            "action_attempts": {},
            "action_failures": {},
            "round": 0,
            "pending_tool": None,
            "pending_subagent": None,
            "artifacts": [],
            "finish_payload": None,
        }
    else:
        tasks = state.get("tasks", tasks)
        context = state.get("context", context)
        user_message = state.get("user_message", user_message)
        allow_shell = bool(state.get("allow_shell", allow_shell))

    messages = state["messages"]
    tool_events = state["tool_events"]
    subagent_outputs = state["subagent_outputs"]
    warnings = state["warnings"]
    action_attempts = state["action_attempts"]
    action_failures = state["action_failures"]
    allowed_tool_names = state["allowed_tool_names"]
    artifacts = state["artifacts"]

    def finish_result(result_text: str) -> Dict[str, Any]:
        return {
            "status": "ok",
            "result": result_text,
            "tool_events": tool_events,
            "subagent_outputs": subagent_outputs,
            "warnings": warnings,
            "artifacts": artifacts,
            "finish_payload": state.get("finish_payload"),
        }

    async def apply_outcome(tool_name: str, args: Dict[str, Any], outcome: Dict[str, Any], signature: str) -> Optional[Dict[str, Any]]:
        result = outcome["message"]
        status = outcome["status"]
        terminal = bool(outcome.get("terminal"))
        action_attempts[signature] = action_attempts.get(signature, 0) + 1
        if status != "ok":
            action_failures[signature] = action_failures.get(signature, 0) + 1

        event_status = status
        if terminal and status in {"rejected", "denied"}:
            event_status = f"terminal_{status}"

        artifact = create_artifact(
            artifacts,
            owner="orchestrator",
            tool_name=tool_name,
            args=args,
            status=event_status,
            content=result,
        )

        if queue_event:
            await queue_event("tool_result", {"name": tool_name, "status": event_status, "result": truncate(result)})
        tool_events.append({
            "id": None,
            "name": tool_name,
            "args": args,
            "status": event_status,
            "result": result,
            "created_at": now_iso(),
            "artifact_id": artifact.get("id"),
        })
        messages.append({
            "role": "system",
            "content": (
                f"Tool result ({tool_name}, {event_status}):\n{result}\n"
                f"Artifact ID: {artifact.get('id')}"
            ),
        })

        if terminal:
            warnings.append(f"terminal failure on '{tool_name}' ({status})")
            return finish_result(
                f"Orchestrator auto-finish: terminal failure on '{tool_name}' ({status}). Returning partial results."
            )

        if action_failures.get(signature, 0) >= 2:
            warnings.append(f"loop guard triggered for repeated failing action '{tool_name}'")
            return finish_result(
                f"Orchestrator auto-finish: repeated failure on '{tool_name}'. Returning partial results."
            )
        return None

    while state["round"] < V2_MAX_ORCHESTRATOR_ROUNDS:
        pending_tool = state.get("pending_tool")
        if pending_tool:
            tool_name = pending_tool["tool_name"]
            args = pending_tool["args"]
            signature = pending_tool["signature"]
            if shell_approval is None:
                return {
                    "status": "needs_approval",
                    "pending_tools": [build_pending_tool(tool_name, args)],
                    "state": state,
                }
            if shell_approval:
                outcome = await execute_v2_tool(
                    tool_name,
                    args,
                    user_message=user_message,
                    allow_shell=allow_shell,
                    approval_mode="allow",
                    shell_approved=True,
                )
            else:
                outcome = tool_outcome(
                    "denied",
                    "DENIED: shell_command was not approved.",
                    terminal=True,
                    retryable=False,
                )
            shell_approval = None
            state["pending_tool"] = None
            final = await apply_outcome(tool_name, args, outcome, signature)
            if final:
                return final
            continue

        pending_subagent = state.get("pending_subagent")
        if pending_subagent:
            sub_result = await run_v2_subagent(
                task=pending_subagent["task"],
                tools=[],
                context=context,
                user_message=user_message,
                allow_shell=allow_shell,
                queue_event=queue_event,
                state=pending_subagent["state"],
                shell_approval=shell_approval,
                approval_mode=approval_mode,
                **kwargs,
            )
            shell_approval = None
            if sub_result.get("status") == "needs_approval":
                pending_subagent["state"] = sub_result["state"]
                return {
                    "status": "needs_approval",
                    "pending_tools": sub_result["pending_tools"],
                    "state": state,
                }

            subagent_outputs.append(sub_result)
            for artifact in sub_result.get("artifacts", []):
                if isinstance(artifact, dict):
                    artifacts.append(artifact)
            if queue_event:
                await queue_event("subagent", sub_result)
            if sub_result.get("warnings"):
                signature = pending_subagent["signature"]
                action_failures[signature] = action_failures.get(signature, 0) + 1
                if action_failures.get(signature, 0) >= 2:
                    warnings.append("loop guard triggered for repeated failing sub-agent action")
                    return finish_result("Orchestrator auto-finish: repeated failing sub-agent action.")
            messages.append({"role": "system", "content": f"Sub-agent result:\n{sub_result.get('result', '')}"})
            state["pending_subagent"] = None
            continue

        response = await LLMProvider.call(get_orchestrator_cfg(), messages, **kwargs)
        state["round"] += 1
        assistant_raw = content_to_text(response["choices"][0]["message"]["content"]).strip()
        parsed = await parse_agent_action(
            role="orchestrator",
            text=assistant_raw,
            provider_cfg=get_orchestrator_cfg(),
            allowed_tool_names=allowed_tool_names,
            allow_subagent=True,
            allow_direct_text=False,
            **kwargs,
        )

        if parsed["type"] == "invalid":
            warning = parsed.get("reason", "invalid action")
            warnings.append(warning)
            safe_finish = "Orchestrator auto-finish: malformed syntax after repair attempt. Returning partial results."
            if subagent_outputs:
                safe_finish += " Sub-agent outputs included."
            return finish_result(safe_finish)

        messages.append({"role": "assistant", "content": parsed.get("raw", assistant_raw)})

        if parsed["type"] == "subagent":
            sub_task = parsed.get("task", "")
            sub_tools = sanitize_subagent_tools(parsed.get("tools", []))
            sub_signature = normalize_action_signature("subagent", sub_task, {"tools": sub_tools, "context": context})
            action_attempts[sub_signature] = action_attempts.get(sub_signature, 0) + 1
            if not sub_tools:
                action_failures[sub_signature] = action_failures.get(sub_signature, 0) + 1
                warnings.append("subagent request had no valid tools; auto-finished that branch")
                sub_result = {
                    "status": "ok",
                    "role": "subagent",
                    "task": sub_task,
                    "result": "Sub-agent auto-finish: no valid tools after sanitization.",
                    "output": "Sub-agent auto-finish: no valid tools after sanitization.",
                    "tool_events": [],
                    "warnings": ["no valid tools after sanitization"],
                    "artifacts": [],
                    "finish_payload": None,
                }
                subagent_outputs.append(sub_result)
                if queue_event:
                    await queue_event("subagent", sub_result)
                messages.append({"role": "system", "content": f"Sub-agent result:\n{sub_result.get('result', '')}"})
                if action_failures.get(sub_signature, 0) >= 2:
                    warnings.append("loop guard triggered for repeated failing sub-agent action")
                    return finish_result("Orchestrator auto-finish: repeated failing sub-agent action.")
                continue

            if queue_event:
                await queue_event("stage", {"name": "subagent"})
            sub_result = await run_v2_subagent(
                task=sub_task,
                tools=sub_tools,
                context=context,
                user_message=user_message,
                allow_shell=allow_shell,
                queue_event=queue_event,
                approval_mode=approval_mode,
                **kwargs,
            )
            if sub_result.get("status") == "needs_approval":
                state["pending_subagent"] = {
                    "signature": sub_signature,
                    "task": sub_task,
                    "state": sub_result["state"],
                }
                return {
                    "status": "needs_approval",
                    "pending_tools": sub_result["pending_tools"],
                    "state": state,
                }

            subagent_outputs.append(sub_result)
            for artifact in sub_result.get("artifacts", []):
                if isinstance(artifact, dict):
                    artifacts.append(artifact)
            if queue_event:
                await queue_event("subagent", sub_result)
            if sub_result.get("warnings"):
                action_failures[sub_signature] = action_failures.get(sub_signature, 0) + 1
            if action_failures.get(sub_signature, 0) >= 2:
                warnings.append("loop guard triggered for repeated failing sub-agent action")
                return finish_result("Orchestrator auto-finish: repeated failing sub-agent action.")
            messages.append({"role": "system", "content": f"Sub-agent result:\n{sub_result.get('result', '')}"})
            continue

        if parsed["type"] == "tool":
            tool_name = parsed["tool_name"]
            args = parsed["params"]
            if tool_name == "FINISH":
                payload, payload_error = parse_finish_payload(args.get("result", ""))
                if payload_error:
                    warnings.append(f"invalid FINISH payload: {payload_error}")
                    return finish_result(f"Orchestrator auto-finish: invalid FINISH payload ({payload_error}).")
                missing = [art_id for art_id in payload.get("artifact_ids", []) if not any(a.get("id") == art_id for a in artifacts)]
                if missing:
                    warnings.append(f"FINISH referenced missing artifact ids: {', '.join(missing)}")
                state["finish_payload"] = payload
                final_result = payload.get("summary", "").strip()
                return finish_result(final_result)

            signature = normalize_action_signature("tool", tool_name, args)
            if queue_event:
                await queue_event("tool_call", {"name": tool_name, "args": args})
            outcome = await execute_v2_tool(
                tool_name,
                args,
                user_message=user_message,
                allow_shell=allow_shell,
                approval_mode=approval_mode,
                shell_approved=False,
            )
            if outcome["status"] == "needs_approval":
                state["pending_tool"] = {
                    "tool_name": tool_name,
                    "args": args,
                    "signature": signature,
                }
                return {
                    "status": "needs_approval",
                    "pending_tools": outcome.get("pending_tools", [build_pending_tool(tool_name, args)]),
                    "state": state,
                }
            final = await apply_outcome(tool_name, args, outcome, signature)
            if final:
                return final
            continue

    warnings.append("max rounds reached")
    return finish_result("Orchestrator auto-finish: max rounds reached without FINISH.")


async def run_v2_pipeline(
    request: ChatRequest,
    session_id: str,
    persistent: bool,
    queue_event: Optional[Callable[[str, Dict[str, Any]], Awaitable[None]]] = None,
    state: Optional[Dict[str, Any]] = None,
    shell_approval: Optional[bool] = None,
    **kwargs
) -> Dict[str, Any]:
    if state is None:
        if queue_event:
            await queue_event("stage", {"name": "main_thinking"})

        history_turns = get_recent_turns(session_id, MAX_HISTORY_TURNS)
        transcript_turns = get_recent_turns(session_id, MAX_META_TURNS)
        transcript = build_transcript(transcript_turns, request.message)

        memory_files = ["soul.md", "user.md"]
        if (MEMORY_DIR / "identity.md").exists():
            memory_files.append("identity.md")
        memory_context = load_memory_files(memory_files)

        main_prompt = load_v2_prompt("main_agent.txt", "You are Main Agent.")
        messages: List[Dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    f"{main_prompt}\n\n"
                    "Syntax is case-sensitive.\n"
                    "Main can only call summarizer or orchestrator.\n\n"
                    f"Memory context:\n{memory_context}"
                ),
            }
        ]
        for turn in history_turns:
            messages.append({"role": "user", "content": turn["user_message"]})
            messages.append({"role": "assistant", "content": turn["assistant_reply"]})
        messages.append({"role": "user", "content": request.message})

        state = {
            "request_message": request.message,
            "messages": messages,
            "transcript": transcript,
            "tool_events": [],
            "subagent_outputs": [],
            "summaries": [],
            "orchestrator_result": "",
            "warnings": [],
            "orchestrator_invoked": False,
            "force_direct_finalization": False,
            "round": 0,
            "pending_orchestrator": None,
            "artifacts": [],
            "orchestrator_finish_payload": None,
        }
    else:
        request = ChatRequest(message=state.get("request_message", request.message), session_id=session_id)

    messages = state["messages"]
    tool_events = state["tool_events"]
    subagent_outputs = state["subagent_outputs"]
    summaries = state["summaries"]
    warnings = state["warnings"]
    artifacts = state["artifacts"]

    def build_meta() -> Dict[str, Any]:
        return {
            "architecture_mode": "hierarchical_v2",
            "warnings": warnings,
            "summaries": summaries,
            "orchestrator_result": state["orchestrator_result"],
            "orchestrator_finish_payload": state.get("orchestrator_finish_payload"),
            "artifact_index": [
                {
                    "id": artifact.get("id"),
                    "owner": artifact.get("owner"),
                    "tool": artifact.get("tool"),
                    "status": artifact.get("status"),
                }
                for artifact in artifacts
            ],
        }

    def response_payload(reply: str, status: str = "ok", pending_tools: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        payload = {
            "reply": reply,
            "meta": build_meta(),
            "tool_events": format_tool_events_for_response(tool_events),
            "subagent_outputs": subagent_outputs,
            "status": status,
            "session_id": session_id,
            "architecture_mode": "hierarchical_v2",
        }
        if pending_tools is not None:
            payload["pending_tools"] = pending_tools
        return payload

    def persist_and_return(reply: str) -> Dict[str, Any]:
        meta = build_meta()
        store_turn(
            session_id=session_id,
            persistent=persistent,
            user_message=request.message,
            assistant_reply=reply,
            meta_json=meta,
            tool_events=tool_events,
            subagent_outputs=subagent_outputs,
            architecture_mode="hierarchical_v2",
        )
        return response_payload(reply, status="ok")

    while state["round"] < V2_MAX_MAIN_ROUNDS:
        pending_orchestrator = state.get("pending_orchestrator")
        if pending_orchestrator:
            if queue_event:
                await queue_event("stage", {"name": "orchestrator"})
            orchestrator_output = await run_v2_orchestrator(
                tasks="",
                context="",
                user_message=request.message,
                allow_shell=False,
                queue_event=queue_event,
                state=pending_orchestrator,
                shell_approval=shell_approval,
                approval_mode="ask",
                **kwargs,
            )
            shell_approval = None
            if orchestrator_output.get("status") == "needs_approval":
                state["pending_orchestrator"] = orchestrator_output["state"]
                result = response_payload("Tool approval required to continue.", status="needs_approval", pending_tools=orchestrator_output["pending_tools"])
                result["v2_resume_state"] = state
                return result

            state["pending_orchestrator"] = None
            state["orchestrator_result"] = orchestrator_output.get("result", "")
            tool_events.extend(orchestrator_output.get("tool_events", []))
            subagent_outputs.extend(orchestrator_output.get("subagent_outputs", []))
            warnings.extend(orchestrator_output.get("warnings", []))
            artifacts.extend(orchestrator_output.get("artifacts", []))
            state["orchestrator_finish_payload"] = orchestrator_output.get("finish_payload")
            messages.append({"role": "system", "content": f"Orchestrator result:\n{state['orchestrator_result']}"})
            finish_payload = orchestrator_output.get("finish_payload") or {}
            facts = finish_payload.get("critical_facts", []) if isinstance(finish_payload, dict) else []
            if facts:
                messages.append({
                    "role": "system",
                    "content": "Orchestrator critical facts:\n" + "\n".join([f"- {fact}" for fact in facts]),
                })
            artifact_ids = finish_payload.get("artifact_ids", []) if isinstance(finish_payload, dict) else []
            artifact_context = render_artifact_context(orchestrator_output.get("artifacts", []), artifact_ids=artifact_ids)
            if artifact_context:
                messages.append({
                    "role": "system",
                    "content": f"Referenced artifacts from orchestrator:\n{artifact_context}",
                })
            messages.append({
                "role": "system",
                "content": (
                    "Orchestrator has completed. Produce a direct user-facing reply now. "
                    "Do not call any more tools for this turn."
                ),
            })
            state["orchestrator_invoked"] = True
            state["force_direct_finalization"] = True
            continue

        response = await LLMProvider.call(config.main, messages, **kwargs)
        state["round"] += 1
        assistant_raw = content_to_text(response["choices"][0]["message"]["content"]).strip()
        parsed = await parse_agent_action(
            role="main",
            text=assistant_raw,
            provider_cfg=config.main,
            allowed_tool_names=["summarizer", "orchestrator"],
            allow_subagent=False,
            allow_direct_text=True,
            **kwargs,
        )

        if parsed["type"] == "invalid":
            if state["force_direct_finalization"] and state["orchestrator_result"]:
                safe_reply = state["orchestrator_result"]
                warnings.append("main produced invalid action during finalization; using orchestrator result directly")
            else:
                safe_reply = (
                    "I hit an internal action-format issue while processing your request. "
                    "I can still help directly if you want me to continue step by step."
                )
                warnings.append(parsed.get("reason", "invalid action"))
            if queue_event:
                await queue_event("stage", {"name": "finish"})
            return persist_and_return(safe_reply)

        if parsed["type"] == "direct_text":
            reply = parsed.get("content", assistant_raw)
            if queue_event:
                await queue_event("stage", {"name": "finish"})
            return persist_and_return(reply)

        if state["force_direct_finalization"] and parsed["type"] == "tool":
            warnings.append("main attempted extra tool call after orchestrator completion; bypassed")
            reply = state["orchestrator_result"] or "I completed the orchestration and here is the result."
            if queue_event:
                await queue_event("stage", {"name": "finish"})
            return persist_and_return(reply)

        if parsed["type"] != "tool":
            warnings.append("main produced unsupported action type")
            if queue_event:
                await queue_event("stage", {"name": "finish"})
            return persist_and_return("I ran into an internal workflow issue and stopped safely.")

        tool_name = parsed["tool_name"]
        params = parsed["params"]
        messages.append({"role": "assistant", "content": parsed.get("raw", assistant_raw)})

        if tool_name == "summarizer":
            if queue_event:
                await queue_event("stage", {"name": "summarizer"})
            query = params.get("query", "Summarize relevant context.")
            summary = await run_v2_summarizer(query, state["transcript"], **kwargs)
            summaries.append({"query": query, "summary": summary})
            messages.append({"role": "system", "content": f"Summarizer output:\n{summary}"})
            continue

        if tool_name == "orchestrator":
            if state["orchestrator_invoked"]:
                warnings.append("orchestrator was already invoked once; forcing final response")
                reply = state["orchestrator_result"] or "I have already completed the orchestration for this request."
                if queue_event:
                    await queue_event("stage", {"name": "finish"})
                return persist_and_return(reply)

            if queue_event:
                await queue_event("stage", {"name": "orchestrator"})
            tasks = params.get("tasks", "")
            context = params.get("context", "")
            orchestrator_output = await run_v2_orchestrator(
                tasks=tasks,
                context=context,
                user_message=request.message,
                allow_shell=False,
                queue_event=queue_event,
                approval_mode="ask",
                **kwargs,
            )
            if orchestrator_output.get("status") == "needs_approval":
                state["pending_orchestrator"] = orchestrator_output["state"]
                result = response_payload(
                    "Tool approval required to continue.",
                    status="needs_approval",
                    pending_tools=orchestrator_output["pending_tools"],
                )
                result["v2_resume_state"] = state
                return result

            state["orchestrator_result"] = orchestrator_output.get("result", "")
            tool_events.extend(orchestrator_output.get("tool_events", []))
            subagent_outputs.extend(orchestrator_output.get("subagent_outputs", []))
            warnings.extend(orchestrator_output.get("warnings", []))
            artifacts.extend(orchestrator_output.get("artifacts", []))
            state["orchestrator_finish_payload"] = orchestrator_output.get("finish_payload")
            messages.append({"role": "system", "content": f"Orchestrator result:\n{state['orchestrator_result']}"})
            finish_payload = orchestrator_output.get("finish_payload") or {}
            facts = finish_payload.get("critical_facts", []) if isinstance(finish_payload, dict) else []
            if facts:
                messages.append({
                    "role": "system",
                    "content": "Orchestrator critical facts:\n" + "\n".join([f"- {fact}" for fact in facts]),
                })
            artifact_ids = finish_payload.get("artifact_ids", []) if isinstance(finish_payload, dict) else []
            artifact_context = render_artifact_context(orchestrator_output.get("artifacts", []), artifact_ids=artifact_ids)
            if artifact_context:
                messages.append({
                    "role": "system",
                    "content": f"Referenced artifacts from orchestrator:\n{artifact_context}",
                })
            messages.append({
                "role": "system",
                "content": (
                    "Orchestrator has completed. Produce a direct user-facing reply now. "
                    "Do not call any more tools for this turn."
                ),
            })
            state["orchestrator_invoked"] = True
            state["force_direct_finalization"] = True
            continue

    warnings.append("main max rounds reached")
    fallback_reply = state["orchestrator_result"] or "I completed internal processing but could not finalize a stable response."
    if queue_event:
        await queue_event("stage", {"name": "finish"})
    return persist_and_return(fallback_reply)


async def run_legacy_pipeline(request: ChatRequest, session_id: str, persistent: bool) -> ChatResponse:
    session_history, working_memory = build_session_context(session_id)
    session_context = f"{session_history}\n\nCURRENT USER MESSAGE: {request.message}"

    skills_list = list_available_skills()
    skills_index = ", ".join(skills_list) if skills_list else "none"
    memory_context = load_memory_files(["identity.md", "soul.md", "user.md"])

    meta_results = await run_meta_layer(session_context, skills_index, memory_context, working_memory)
    plan_result = meta_results.get("plan")
    if not isinstance(plan_result, dict):
        plan_result = {"just_chat": True, "plan": "respond conversationally"}

    meta_output = MetaOutput(
        intent=meta_results.get("intent"),
        tone=meta_results.get("tone"),
        user=meta_results.get("user"),
        subject=meta_results.get("subject"),
        needs=meta_results.get("needs"),
        patterns=meta_results.get("patterns"),
        plan=plan_result,
        raw=meta_results.get("raw")
    )

    loaded_context = ""
    if plan_result.get("load_memories"):
        for query in plan_result["load_memories"]:
            results = await execute_tool("memory_search", {"query": query})
            loaded_context += f"Memory search '{query}': {results}\n"

    if plan_result.get("load_skills"):
        for skill in plan_result["load_skills"]:
            content = await execute_tool("skill_load", {"name": skill})
            loaded_context += f"Skill '{skill}': {content}\n"

    subagent_outputs: List[Dict[str, Any]] = []
    if plan_result.get("use_sub_agents"):
        sub_tasks = plan_result.get("sub_agent_tasks", [])
        sub_specs = [{"role": "subagent", "task": t, "tools": ["shell_command", "memory_search", "memory_create", "read_file"]} for t in sub_tasks]
        outputs = await run_subagents(sub_specs)
        subagent_outputs.extend(outputs)
        for output in outputs:
            loaded_context += f"Sub agent task '{output['task']}' result: {output['result']}\n"

    history_turns = get_recent_turns(session_id, MAX_HISTORY_TURNS)
    messages = build_main_messages(request.message, meta_output, history_turns, subagent_outputs, loaded_context=loaded_context)

    shell_allowed = session_policies[session_id].get("shell_command_allowed", False)
    loop_result = await run_tool_loop(
        config.main,
        messages,
        TOOLS_SPEC,
        approval_mode="ask",
        allow_shell=shell_allowed,
        user_message=request.message,
    )

    if loop_result["status"] == "needs_approval":
        pending_approvals[session_id] = {
            "messages": loop_result["pending_messages"],
            "tool_calls": loop_result["pending_tool_calls"],
            "tool_events": loop_result["tool_events"],
            "user_message": request.message,
            "meta_json": meta_output.model_dump(),
            "subagent_outputs": loop_result.get("subagent_outputs", []),
            "persistent": persistent,
            "architecture_mode": "legacy",
        }
        pending_tools = [
            {
                "id": tc.get("id"),
                "name": tc.get("function", {}).get("name"),
                "arguments": tc.get("function", {}).get("arguments", "{}"),
            }
            for tc in loop_result["pending_tool_calls"]
        ]
        return ChatResponse(
            reply="Tool approval required to continue.",
            meta=meta_output.model_dump(),
            tool_events=[
                {
                    "id": ev.get("id"),
                    "name": ev["name"],
                    "args": ev["args"],
                    "status": ev["status"],
                    "result": truncate(ev["result"]),
                    "created_at": ev["created_at"],
                }
                for ev in loop_result.get("tool_events", [])
            ],
            subagent_outputs=loop_result.get("subagent_outputs", []),
            status="needs_approval",
            pending_tools=pending_tools,
            session_id=session_id,
            architecture_mode="legacy",
        )

    reply = loop_result.get("content", "")
    tool_events = loop_result.get("tool_events", [])
    subagent_outputs = loop_result.get("subagent_outputs", [])

    store_turn(
        session_id=session_id,
        persistent=persistent,
        user_message=request.message,
        assistant_reply=reply,
        meta_json=meta_output.model_dump(),
        tool_events=tool_events,
        subagent_outputs=subagent_outputs,
        architecture_mode="legacy",
    )

    user_md = (MEMORY_DIR / "user.md").read_text() if (MEMORY_DIR / "user.md").exists() else ""
    identity_md = (MEMORY_DIR / "identity.md").read_text() if (MEMORY_DIR / "identity.md").exists() else ""
    gate_result = await run_gatekeeper_layer(session_context, reply, user_md, identity_md)
    await apply_gatekeeper(session_id, persistent, gate_result)
    meta_output.gatekeeper = gate_result

    return ChatResponse(
        reply=reply,
        meta=meta_output.model_dump(),
        tool_events=[
            {
                "id": ev.get("id"),
                "name": ev["name"],
                "args": ev["args"],
                "status": ev["status"],
                "result": truncate(ev["result"]),
                "created_at": ev["created_at"],
            }
            for ev in tool_events
        ],
        subagent_outputs=subagent_outputs,
        status="ok",
        session_id=session_id,
        architecture_mode="legacy",
    )


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/api/providers")
async def get_providers():
    return DEFAULT_PROVIDERS


@app.get("/api/config")
async def get_config():
    if not config:
        raise HTTPException(status_code=404, detail="Configuration not set")
    return config.model_dump()


@app.post("/api/config")
async def save_config(agent_config: AgentConfig):
    global config
    config = resolve_agent_config(agent_config)
    CONFIG_PATH.write_text(config.model_dump_json(indent=2))
    return {"status": "success"}


@app.get("/api/sessions")
async def list_sessions():
    conn = db_connect()
    rows = conn.execute(
        "SELECT id, title, created_at, updated_at FROM sessions WHERE persistent = 1 ORDER BY updated_at DESC"
    ).fetchall()
    conn.close()
    return [
        {
            "id": row["id"],
            "title": row["title"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "persistent": True,
        }
        for row in rows
    ]


@app.post("/api/sessions")
async def create_session(request: SessionCreateRequest):
    title = request.title or "New chat"
    if request.persistent:
        session = create_persistent_session(title)
    else:
        session = create_temp_session(title)
    return session


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    if not session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "id": session_id,
        "persistent": session_id not in temp_sessions,
        "turns": get_session_turns(session_id),
    }


@app.patch("/api/sessions/{session_id}")
async def rename_session(session_id: str, request: SessionUpdateRequest):
    if session_id in temp_sessions:
        temp_sessions[session_id]["title"] = request.title
        temp_sessions[session_id]["updated_at"] = now_iso()
        return {"status": "success"}
    conn = db_connect()
    cur = conn.execute("UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?", (request.title, now_iso(), session_id))
    conn.commit()
    conn.close()
    if cur.rowcount == 0:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "success"}


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    if session_id in temp_sessions:
        temp_sessions.pop(session_id, None)
        return {"status": "deleted"}
    conn = db_connect()
    cur = conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
    conn.commit()
    conn.close()
    if cur.rowcount == 0:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "deleted"}


@app.get("/api/tool_events/{event_id}")
async def get_tool_event(event_id: int):
    conn = db_connect()
    row = conn.execute("SELECT * FROM tool_events WHERE id = ?", (event_id,)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Tool event not found")
    return {
        "id": row["id"],
        "name": row["name"],
        "args": json.loads(row["args_json"] or "{}"),
        "result": row["result_text"],
        "status": row["status"],
        "created_at": row["created_at"],
    }


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    logger.debug(f"Config in endpoint: {config}")
    if not config:
        logger.error("Configuration not set in endpoint")
        raise HTTPException(status_code=400, detail="Configuration not set")

    if request.session_id and session_exists(request.session_id):
        session_id = request.session_id
    else:
        session = create_persistent_session("New chat")
        session_id = session["id"]

    ensure_session_policy(session_id)
    persistent = session_id not in temp_sessions

    event_queue = asyncio.Queue()

    async def queue_event(event, data):
        await event_queue.put((event, data))

    async def on_rate_limit_callback(seconds):
        await queue_event("rate_limit_retry", {"seconds": seconds})

    async def on_user_decision_callback():
        await queue_event("rate_limit_error", {"session_id": session_id})
        if session_id not in rate_limit_decisions:
            rate_limit_decisions[session_id] = asyncio.Queue()
        decision = await rate_limit_decisions[session_id].get()
        return decision

    call_kwargs = {
        "on_rate_limit": on_rate_limit_callback,
        "on_user_decision_required": on_user_decision_callback
    }

    async def run_orchestration():
        try:
            architecture_mode = get_architecture_mode()
            if architecture_mode == "hierarchical_v2":
                await queue_event("meta", {"architecture_mode": "hierarchical_v2"})
                v2_result = await run_v2_pipeline(
                    request=request,
                    session_id=session_id,
                    persistent=persistent,
                    queue_event=queue_event,
                    **call_kwargs,
                )
                if "meta" not in v2_result or not isinstance(v2_result.get("meta"), dict):
                    v2_result["meta"] = {"architecture_mode": "hierarchical_v2"}
                if v2_result.get("status") == "needs_approval":
                    pending_approvals[session_id] = {
                        "architecture_mode": "hierarchical_v2",
                        "v2_resume_state": v2_result.get("v2_resume_state"),
                        "user_message": request.message,
                        "persistent": persistent,
                    }
                    await queue_event("needs_approval", {
                        "pending_tools": v2_result.get("pending_tools", []),
                        "meta": v2_result.get("meta"),
                        "tool_events": v2_result.get("tool_events", []),
                        "subagent_outputs": v2_result.get("subagent_outputs", []),
                        "session_id": session_id,
                        "architecture_mode": "hierarchical_v2",
                    })
                    await event_queue.put(None)
                    return
                await queue_event("assistant", {
                    "content": v2_result.get("reply", ""),
                    "meta": v2_result.get("meta"),
                    "tool_events": v2_result.get("tool_events", []),
                    "subagent_outputs": v2_result.get("subagent_outputs", []),
                    "session_id": session_id,
                    "architecture_mode": "hierarchical_v2",
                })
                await queue_event("done", {"session_id": session_id, "architecture_mode": "hierarchical_v2"})
                await event_queue.put(None)
                return

            await queue_event("stage", {"name": "meta_start"})
            session_history, working_memory = build_session_context(session_id)
            subagent_outputs: List[Dict[str, Any]] = []
            # Include current message in context for analysis
            session_context = f"{session_history}\n\nCURRENT USER MESSAGE: {request.message}"

            skills_list = list_available_skills()
            skills_index = ", ".join(skills_list) if skills_list else "none"
            memory_context = load_memory_files(["identity.md", "soul.md", "user.md"])

            # Layer 1+2: Combined Meta + Planner
            meta_results = await run_meta_layer(
                session_context,
                skills_index,
                memory_context,
                working_memory,
                **call_kwargs,
            )
            await queue_event("meta_partial", meta_results)

            plan_result = meta_results.get("plan")
            if not isinstance(plan_result, dict):
                plan_result = {"just_chat": True, "plan": "respond conversationally"}

            meta_output = MetaOutput(
                intent=meta_results.get("intent"),
                tone=meta_results.get("tone"),
                user=meta_results.get("user"),
                subject=meta_results.get("subject"),
                needs=meta_results.get("needs"),
                patterns=meta_results.get("patterns"),
                plan=plan_result,
                raw=meta_results.get("raw")
            )
            await queue_event("meta", meta_output.model_dump())

            # Load requested context from planner
            loaded_context = ""
            if plan_result.get("load_memories"):
                for query in plan_result["load_memories"]:
                    results = await execute_tool("memory_search", {"query": query})
                    loaded_context += f"Memory search '{query}': {results}\n"

            if plan_result.get("load_skills"):
                for skill in plan_result["load_skills"]:
                    content = await execute_tool("skill_load", {"name": skill})
                    loaded_context += f"Skill '{skill}': {content}\n"

            # Sub agents triggered by planner
            if plan_result.get("use_sub_agents"):
                await queue_event("stage", {"name": "subagents_start"})
                sub_tasks = plan_result.get("sub_agent_tasks", [])
                sub_specs = [{"role": "subagent", "task": t, "tools": ["shell_command", "memory_search", "memory_create", "read_file"]} for t in sub_tasks]
                outputs = await run_subagents(sub_specs, **call_kwargs)
                subagent_outputs.extend(outputs)
                for output in outputs:
                    await queue_event("subagent", output)
                    loaded_context += f"Sub agent task '{output['task']}' result: {output['result']}\n"

            history_turns = get_recent_turns(session_id, MAX_HISTORY_TURNS)
            messages = build_main_messages(request.message, meta_output, history_turns, subagent_outputs, loaded_context=loaded_context)

            shell_allowed = session_policies[session_id].get("shell_command_allowed", False)
            tool_events: List[Dict[str, Any]] = []
            tools_started = False

            for _ in range(MAX_TOOL_ROUNDS):
                response = await LLMProvider.call(config.main, messages, tools=TOOLS_SPEC, **call_kwargs)
                assistant_msg = response["choices"][0]["message"]
                assistant_content = content_to_text(assistant_msg.get("content", ""))
                tool_calls = assistant_msg.get("tool_calls")

                if tool_calls:
                    for tool_call in tool_calls:
                        if not tool_call.get("id"):
                            tool_call["id"] = f"call_{uuid.uuid4()}"

                    def needs_approval(tc: Dict[str, Any]) -> bool:
                        fn = tc.get("function", {})
                        return fn.get("name") == "shell_command"

                    if any(needs_approval(tc) for tc in tool_calls):
                        pending_messages = messages + [{
                            "role": "assistant",
                            "content": assistant_content,
                            "tool_calls": tool_calls,
                        }]
                        pending_approvals[session_id] = {
                            "messages": pending_messages,
                            "tool_calls": tool_calls,
                            "tool_events": tool_events,
                            "user_message": request.message,
                            "meta_json": meta_output.model_dump(),
                            "subagent_outputs": subagent_outputs,
                            "persistent": persistent,
                            "architecture_mode": "legacy",
                        }
                        pending_tools = [
                            {
                                "id": tc.get("id"),
                                "name": tc.get("function", {}).get("name"),
                                "arguments": tc.get("function", {}).get("arguments", "{}"),
                            }
                            for tc in tool_calls
                        ]
                        await queue_event("needs_approval", {
                            "pending_tools": pending_tools,
                            "meta": meta_output.model_dump(),
                            "tool_events": [
                                {
                                    "id": ev.get("id"),
                                    "name": ev["name"],
                                    "args": ev["args"],
                                    "status": ev["status"],
                                    "result": truncate(ev["result"]),
                                    "created_at": ev["created_at"],
                                }
                                for ev in tool_events
                            ],
                            "session_id": session_id,
                            "architecture_mode": "legacy",
                        })
                        await event_queue.put(None)
                        return

                    if not tools_started:
                        await queue_event("stage", {"name": "tools_start"})
                        tools_started = True

                    messages.append({
                        "role": "assistant",
                        "content": assistant_content,
                        "tool_calls": tool_calls,
                    })

                    for tool_call in tool_calls:
                        fn = tool_call.get("function", {})
                        name = fn.get("name")
                        args_raw = fn.get("arguments", "{}")
                        try:
                            args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                        except Exception:
                            args = {}

                        await queue_event("tool_call", {"name": name, "args": args})

                        if name == "shell_command" and not shell_allowed:
                            result = "DENIED: shell_command is not allowed for this agent."
                            status = "denied"
                        elif name == "shell_command":
                            result = await execute_tool(name, args)
                            status = "error" if result.startswith("Tool execution error") else "ok"
                        elif name == "spawn_subagents":
                            await queue_event("stage", {"name": "subagents_start"})
                            subagents_spec = args.get("subagents", [])
                            outputs = await run_subagents(subagents_spec, **call_kwargs)
                            subagent_outputs.extend(outputs)
                            for output in outputs:
                                await queue_event("subagent", output)
                            result = json.dumps(outputs, indent=2)
                            status = "ok"
                        else:
                            result = await execute_tool(name, args)
                            status = "error" if result.startswith("Tool execution error") else "ok"

                        tool_events.append({
                            "id": None,
                            "name": name,
                            "args": args,
                            "result": result,
                            "status": status,
                            "created_at": now_iso(),
                        })
                        await queue_event("tool_result", {
                            "name": name,
                            "status": status,
                            "result": truncate(result),
                        })
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.get("id"),
                            "content": result,
                        })
                    continue

                reply = assistant_content
                store_turn(
                    session_id=session_id,
                    persistent=persistent,
                    user_message=request.message,
                    assistant_reply=reply,
                    meta_json=meta_output.model_dump(),
                    tool_events=tool_events,
                    subagent_outputs=subagent_outputs,
                )
                # Layer 5: Gatekeeper
                await queue_event("stage", {"name": "gatekeeper_start"})
                user_md = (MEMORY_DIR / "user.md").read_text() if (MEMORY_DIR / "user.md").exists() else ""
                identity_md = (MEMORY_DIR / "identity.md").read_text() if (MEMORY_DIR / "identity.md").exists() else ""
                gate_result = await run_gatekeeper_layer(session_context, reply, user_md, identity_md, **call_kwargs)
                await apply_gatekeeper(session_id, persistent, gate_result)
                meta_output.gatekeeper = gate_result

                await queue_event("assistant", {
                    "content": reply,
                    "meta": meta_output.model_dump(),
                    "tool_events": [
                        {
                            "id": ev.get("id"),
                            "name": ev["name"],
                            "args": ev["args"],
                            "status": ev["status"],
                            "result": truncate(ev["result"]),
                            "created_at": ev["created_at"],
                        }
                        for ev in tool_events
                    ],
                    "subagent_outputs": subagent_outputs,
                    "session_id": session_id,
                    "architecture_mode": "legacy",
                })
                await queue_event("done", {"session_id": session_id, "architecture_mode": "legacy"})
                await event_queue.put(None)
                return

            reply = ""
            store_turn(
                session_id=session_id,
                persistent=persistent,
                user_message=request.message,
                assistant_reply=reply,
                meta_json=meta_output.model_dump(),
                tool_events=tool_events,
                subagent_outputs=subagent_outputs,
            )
            await queue_event("assistant", {
                "content": reply,
                "meta": meta_output.model_dump(),
                "tool_events": [
                    {
                        "id": ev.get("id"),
                        "name": ev["name"],
                        "args": ev["args"],
                        "status": ev["status"],
                        "result": truncate(ev["result"]),
                        "created_at": ev["created_at"],
                    }
                    for ev in tool_events
                ],
                "subagent_outputs": subagent_outputs,
                "session_id": session_id,
                "architecture_mode": "legacy",
            })
            await queue_event("done", {"session_id": session_id, "architecture_mode": "legacy"})
            await event_queue.put(None)
        except Exception as e:
            logger.error(f"Error in orchestration: {e}")
            await queue_event("error", {"detail": str(e)})
            await event_queue.put(None)

    async def event_generator():
        orchestration_task = asyncio.create_task(run_orchestration())
        try:
            while True:
                item = await event_queue.get()
                if item is None:
                    break
                event, data = item
                yield sse_event(event, data)
        finally:
            if not orchestration_task.done():
                orchestration_task.cancel()
            rate_limit_decisions.pop(session_id, None)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/api/chat")
async def chat(request: ChatRequest) -> ChatResponse:
    if not config:
        raise HTTPException(status_code=400, detail="Configuration not set")

    if request.session_id and session_exists(request.session_id):
        session_id = request.session_id
    else:
        session = create_persistent_session("New chat")
        session_id = session["id"]

    ensure_session_policy(session_id)
    persistent = session_id not in temp_sessions
    if get_architecture_mode() == "hierarchical_v2":
        v2_result = await run_v2_pipeline(
            request=request,
            session_id=session_id,
            persistent=persistent,
        )
        if v2_result.get("status") == "needs_approval":
            pending_approvals[session_id] = {
                "architecture_mode": "hierarchical_v2",
                "v2_resume_state": v2_result.get("v2_resume_state"),
                "user_message": request.message,
                "persistent": persistent,
            }
        return ChatResponse(
            reply=v2_result.get("reply", ""),
            meta=v2_result.get("meta"),
            tool_events=v2_result.get("tool_events", []),
            subagent_outputs=v2_result.get("subagent_outputs", []),
            status=v2_result.get("status", "ok"),
            pending_tools=v2_result.get("pending_tools"),
            session_id=v2_result.get("session_id", session_id),
            architecture_mode="hierarchical_v2",
        )

    return await run_legacy_pipeline(request, session_id, persistent)


@app.post("/api/tools/approve")
async def approve_tools(request: ToolApprovalRequest) -> ChatResponse:
    if request.session_id not in pending_approvals:
        raise HTTPException(status_code=404, detail="No pending tool approval")

    pending = pending_approvals.pop(request.session_id)
    original_user_message = pending.get("user_message", "(tool approval continuation)")
    persistent = pending.get("persistent", request.session_id not in temp_sessions)
    architecture_mode = pending.get("architecture_mode", "legacy")

    allow_shell = request.decision in ["run_once", "allow_session"]

    if architecture_mode == "hierarchical_v2":
        resume_state = pending.get("v2_resume_state")
        if not isinstance(resume_state, dict):
            raise HTTPException(status_code=400, detail="Missing v2 resume state")

        v2_result = await run_v2_pipeline(
            request=ChatRequest(message=original_user_message, session_id=request.session_id),
            session_id=request.session_id,
            persistent=persistent,
            state=resume_state,
            shell_approval=allow_shell,
        )

        if v2_result.get("status") == "needs_approval":
            pending_approvals[request.session_id] = {
                "architecture_mode": "hierarchical_v2",
                "v2_resume_state": v2_result.get("v2_resume_state"),
                "user_message": original_user_message,
                "persistent": persistent,
            }
            return ChatResponse(
                reply=v2_result.get("reply", "Tool approval required to continue."),
                meta=v2_result.get("meta"),
                tool_events=v2_result.get("tool_events", []),
                subagent_outputs=v2_result.get("subagent_outputs", []),
                status="needs_approval",
                pending_tools=v2_result.get("pending_tools", []),
                session_id=request.session_id,
                architecture_mode="hierarchical_v2",
            )

        return ChatResponse(
            reply=v2_result.get("reply", ""),
            meta=v2_result.get("meta"),
            tool_events=v2_result.get("tool_events", []),
            subagent_outputs=v2_result.get("subagent_outputs", []),
            status=v2_result.get("status", "ok"),
            pending_tools=v2_result.get("pending_tools"),
            session_id=request.session_id,
            architecture_mode="hierarchical_v2",
        )

    messages = pending["messages"]
    tool_calls = pending["tool_calls"]
    tool_events = pending["tool_events"]
    meta_json = pending.get("meta_json", {})
    subagent_outputs = pending.get("subagent_outputs", [])

    for tool_call in tool_calls:
        fn = tool_call.get("function", {})
        name = fn.get("name")
        args_raw = fn.get("arguments", "{}")
        try:
            args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
        except Exception:
            args = {}

        if name == "shell_command" and not allow_shell:
            result = "DENIED: user requested manual steps instead of running shell_command."
            status = "denied"
        elif name == "shell_command":
            result = await execute_tool(name, args)
            status = "error" if result.startswith("Tool execution error") else "ok"
        elif name == "spawn_subagents":
            outputs = await run_subagents(args.get("subagents", []))
            subagent_outputs = subagent_outputs + outputs
            result = json.dumps(outputs, indent=2)
            status = "ok"
        else:
            result = await execute_tool(name, args)
            status = "error" if result.startswith("Tool execution error") else "ok"

        tool_events.append({
            "id": None,
            "name": name,
            "args": args,
            "result": result,
            "status": status,
            "created_at": now_iso(),
        })
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.get("id"),
            "content": result,
        })

    loop_result = await run_tool_loop(
        config.main,
        messages,
        TOOLS_SPEC,
        approval_mode="ask",
        allow_shell=allow_shell,
        user_message=original_user_message,
    )

    if loop_result["status"] == "needs_approval":
        pending_approvals[request.session_id] = {
            "messages": loop_result["pending_messages"],
            "tool_calls": loop_result["pending_tool_calls"],
            "tool_events": tool_events + loop_result.get("tool_events", []),
            "user_message": original_user_message,
            "meta_json": meta_json,
            "subagent_outputs": subagent_outputs + loop_result.get("subagent_outputs", []),
            "persistent": persistent,
            "architecture_mode": architecture_mode,
        }
        pending_tools = [
            {
                "id": tc.get("id"),
                "name": tc.get("function", {}).get("name"),
                "arguments": tc.get("function", {}).get("arguments", "{}"),
            }
            for tc in loop_result["pending_tool_calls"]
        ]
        return ChatResponse(
            reply="Tool approval required to continue.",
            meta=meta_json,
            tool_events=[
                {
                    "id": ev.get("id"),
                    "name": ev["name"],
                    "args": ev["args"],
                    "status": ev["status"],
                    "result": truncate(ev["result"]),
                    "created_at": ev["created_at"],
                }
                for ev in tool_events
            ],
            subagent_outputs=subagent_outputs + loop_result.get("subagent_outputs", []),
            status="needs_approval",
            pending_tools=pending_tools,
            session_id=request.session_id,
            architecture_mode=architecture_mode,
        )

    reply = loop_result.get("content", "")
    tool_events.extend(loop_result.get("tool_events", []))
    subagent_outputs = subagent_outputs + loop_result.get("subagent_outputs", [])

    store_turn(
        session_id=request.session_id,
        persistent=persistent,
        user_message=original_user_message,
        assistant_reply=reply,
        meta_json=meta_json,
        tool_events=tool_events,
        subagent_outputs=subagent_outputs,
        architecture_mode=architecture_mode,
    )

    if architecture_mode == "legacy":
        # Layer 5: Gatekeeper (legacy only)
        session_history, _ = build_session_context(request.session_id)
        session_context = f"{session_history}\n\nCURRENT USER MESSAGE: {original_user_message}"
        user_md = (MEMORY_DIR / "user.md").read_text() if (MEMORY_DIR / "user.md").exists() else ""
        identity_md = (MEMORY_DIR / "identity.md").read_text() if (MEMORY_DIR / "identity.md").exists() else ""
        gate_result = await run_gatekeeper_layer(session_context, reply, user_md, identity_md)
        await apply_gatekeeper(request.session_id, persistent, gate_result)

        if isinstance(meta_json, dict):
            meta_json["gatekeeper"] = gate_result

    return ChatResponse(
        reply=reply,
        meta=meta_json,
        tool_events=[
            {
                "id": ev.get("id"),
                "name": ev["name"],
                "args": ev["args"],
                "status": ev["status"],
                "result": truncate(ev["result"]),
                "created_at": ev["created_at"],
            }
            for ev in tool_events
        ],
        subagent_outputs=subagent_outputs,
        status="ok",
        session_id=request.session_id,
        architecture_mode=architecture_mode,
    )


@app.post("/api/chat/decision")
async def chat_decision(session_id: str = Body(embed=True), decision: str = Body(embed=True)):
    if session_id in rate_limit_decisions:
        await rate_limit_decisions[session_id].put(decision)
        return {"status": "success"}
    raise HTTPException(status_code=404, detail="No pending rate limit decision for this session")


@app.get("/")
async def serve_frontend():
    return FileResponse("index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
