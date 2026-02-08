"""
Agentic Gateway Backend
Multi-provider LLM orchestration with Meta, Main, and Sub agents
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict, Any, Literal, Tuple
import asyncio
import httpx
import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

app = FastAPI()

# =============================================================================
# CONSTANTS
# =============================================================================

DB_PATH = Path("sessions.db")
CONFIG_PATH = Path("config.json")
MEMORY_DIR = Path("memory")
PROMPTS_DIR = Path("prompts")
SKILLS_DIR = Path("skills")

MAX_HISTORY_TURNS = 10
MAX_META_TURNS = 50
MAX_TOOL_ROUNDS = 3
TOOL_OUTPUT_LIMIT = 1200

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
    return datetime.utcnow().isoformat()


def truncate(text: str, limit: int = TOOL_OUTPUT_LIMIT) -> str:
    if text is None:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... (truncated, {len(text)} chars)"


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
    meta: ProviderConfig
    main: ProviderConfig
    sub: ProviderConfig


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class MetaOutput(BaseModel):
    model_config = ConfigDict(extra="allow")
    user_intent: str = "unknown"
    chat_subject: str = "general"
    user_tone: str = "neutral"
    recommended_plan: str = "respond conversationally"
    keepMemoryFilesLoaded: bool = False
    memory_files_to_load: Optional[List[str]] = None
    skills_to_load: Optional[List[str]] = None
    extra_instructions: Optional[str] = None
    subagent_suggestions: List[Dict[str, Any]] = Field(default_factory=list)
    subagents_needed: List[Dict[str, Any]] = Field(default_factory=list)
    knowledge_needed: List[str] = Field(default_factory=list)
    knowledge_gaps: List[str] = Field(default_factory=list)
    learning_opportunities: List[str] = Field(default_factory=list)
    aha_moments: List[str] = Field(default_factory=list)
    memory_actions: List[Dict[str, Any]] = Field(default_factory=list)
    tone_changes_noticed: Optional[str] = None
    mental_health_update: Optional[str] = None


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
session_policies: Dict[str, Dict[str, Any]] = {}
temp_sessions: Dict[str, Dict[str, Any]] = {}

MEMORY_DIR.mkdir(exist_ok=True)
SKILLS_DIR.mkdir(exist_ok=True)


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
            persistent INTEGER
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
    return AgentConfig(
        meta=resolve_provider_config(agent_cfg.meta),
        main=resolve_provider_config(agent_cfg.main),
        sub=resolve_provider_config(agent_cfg.sub),
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
    ) -> Dict[str, Any]:
        cfg = resolve_provider_config(provider_config)
        if cfg.provider_type == "anthropic":
            return await LLMProvider._call_anthropic(cfg, messages, tools)
        return await LLMProvider._call_openai_compatible(cfg, messages, tools, response_format)

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
                    "content": {"type": "string", "description": "New content to write"},
                    "mode": {"type": "string", "enum": ["write", "append"], "description": "Write mode"},
                },
                "required": ["path", "content"],
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
]


def filter_tools(allowed: Optional[List[str]]) -> List[Dict[str, Any]]:
    if not allowed:
        return []
    allowed_set = set(allowed)
    return [tool for tool in TOOLS_SPEC if tool["function"]["name"] in allowed_set]


async def execute_tool(name: str, arguments: Dict[str, Any]) -> str:
    try:
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
            path = Path(arguments.get("path", ""))
            if not path.exists():
                return f"Error: File {path} does not exist"
            lines = path.read_text().splitlines()
            if len(lines) <= 200:
                return "\n".join(lines)
            return "\n".join(lines[:60]) + f"\n... (truncated, {len(lines)} total lines)"

        if name == "edit_file":
            path = Path(arguments.get("path", ""))
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
            import re
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

        return f"Unknown tool: {name}"
    except Exception as exc:
        return f"Tool execution error: {exc}"


def user_requested_shell(user_message: str) -> bool:
    if not user_message:
        return False
    lower = user_message.lower()
    keywords = [
        "run ",
        "execute",
        "shell",
        "command",
        "terminal",
        "cmd",
        "powershell",
        "bash",
    ]
    return any(keyword in lower for keyword in keywords)


def is_text_only_shell(command: str) -> bool:
    cmd = (command or "").strip()
    lower = cmd.lower()
    if not cmd:
        return True
    text_only_prefixes = ["echo ", "printf ", "write-output", "write-host"]
    if any(lower.startswith(prefix) for prefix in text_only_prefixes):
        if ">" in cmd or "|" in cmd:
            return False
        return True
    if "-command" in lower and ("echo " in lower or "write-output" in lower or "write-host" in lower):
        if ">" in cmd or "|" in cmd:
            return False
        return True
    return False


def should_allow_shell(command: str, user_message: str) -> bool:
    if not command:
        return False
    if user_requested_shell(user_message):
        return True
    if is_text_only_shell(command):
        return False
    return True


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
        content = response["choices"][0]["message"]["content"]
        return json.loads(content)
    except Exception:
        return None


async def parse_meta_json(provider_cfg: ProviderConfig, content: str) -> Dict[str, Any]:
    try:
        return json.loads(content)
    except Exception:
        candidate = extract_json_candidate(content)
        if candidate:
            try:
                return json.loads(candidate)
            except Exception:
                pass
    repaired = await repair_json_with_llm(provider_cfg, content)
    if repaired is not None:
        return repaired
    return {}


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


def build_session_context(session_id: str) -> str:
    turns = get_recent_turns(session_id, MAX_META_TURNS)
    conversation: List[Dict[str, Any]] = []
    for turn in turns:
        conversation.append({"role": "user", "content": turn["user_message"]})
        conversation.append({"role": "assistant", "content": turn["assistant_reply"]})
    tool_logs = get_recent_tool_logs(session_id, 20)
    session_context = {
        "conversation": conversation,
        "tool_logs": tool_logs,
    }
    return json.dumps(session_context, indent=2)


def store_turn(
    session_id: str,
    persistent: bool,
    user_message: str,
    assistant_reply: str,
    meta_json: Dict[str, Any],
    tool_events: List[Dict[str, Any]],
    subagent_outputs: List[Dict[str, Any]],
) -> None:
    if not persistent:
        temp_sessions[session_id]["turns"].append({
            "user_message": user_message,
            "assistant_reply": assistant_reply,
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
            "created_at": now_iso(),
        })
        temp_sessions[session_id]["updated_at"] = now_iso()
        return

    conn = db_connect()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO turns (session_id, user_message, assistant_reply, meta_json, tool_json, subagent_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            session_id,
            user_message,
            assistant_reply,
            json.dumps(meta_json),
            json.dumps([]),
            json.dumps(subagent_outputs),
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

async def run_meta_agent(session_context: str) -> MetaOutput:
    meta_prompt = load_meta_agent_prompt()
    memory_context = load_memory_files(["identity.md", "soul.md", "user.md"])
    skills_list = list_available_skills()
    skills_text = ", ".join(skills_list) if skills_list else "none"
    system_content = f"{meta_prompt}\n\nAvailable skills: {skills_text}\n\n{memory_context}".strip()
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": f"Session Context:\n{session_context}"},
    ]
    response = await LLMProvider.call(
        config.meta,
        messages,
        response_format={"type": "json_object"},
    )
    content = response["choices"][0]["message"]["content"]
    meta_data = await parse_meta_json(config.meta, content)
    try:
        return MetaOutput(**meta_data)
    except Exception:
        return MetaOutput()


def build_main_messages(
    user_message: str,
    meta_output: MetaOutput,
    history_turns: List[Dict[str, Any]],
    subagent_outputs: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    main_prompt = load_main_agent_prompt()

    available_memory = list_available_memory_files()
    available_skills = list_available_skills()
    extra_instructions = meta_output.extra_instructions or ""

    memory_text = ", ".join(available_memory) if available_memory else "none"
    skills_text = ", ".join(available_skills) if available_skills else "none"

    system_parts = [
        main_prompt,
        f"Available memory files: {memory_text}. Use read_file to load if needed.",
        f"Available skills: {skills_text}. Use read_file to load if needed.",
    ]
    if extra_instructions:
        system_parts.append(f"Extra instructions:\n{extra_instructions}")

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": "\n\n".join(system_parts)},
        {"role": "system", "content": f"Meta analysis JSON:\n{meta_output.model_dump_json(indent=2)}"},
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
) -> Dict[str, Any]:
    tool_events: List[Dict[str, Any]] = []
    subagent_outputs: List[Dict[str, Any]] = []
    for _ in range(MAX_TOOL_ROUNDS):
        response = await LLMProvider.call(provider_cfg, messages, tools=tools)
        assistant_msg = response["choices"][0]["message"]
        tool_calls = assistant_msg.get("tool_calls")
        if tool_calls:
            for tool_call in tool_calls:
                if not tool_call.get("id"):
                    tool_call["id"] = f"call_{uuid.uuid4()}"

            def needs_approval(tc: Dict[str, Any]) -> bool:
                fn = tc.get("function", {})
                if fn.get("name") != "shell_command":
                    return False
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except Exception:
                    args = {}
                cmd = args.get("command", "")
                return should_allow_shell(cmd, user_message) and not allow_shell

            if approval_mode == "ask" and any(needs_approval(tc) for tc in tool_calls):
                pending_messages = messages + [{
                    "role": "assistant",
                    "content": assistant_msg.get("content", ""),
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
                "content": assistant_msg.get("content", ""),
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

                if name == "shell_command" and not should_allow_shell(args.get("command", ""), user_message):
                    result = "Tool rejected: not necessary. Respond directly without tools."
                    status = "rejected"
                elif name == "shell_command" and not allow_shell and approval_mode == "auto_deny":
                    result = "DENIED: shell_command is not allowed for this agent."
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
            "content": assistant_msg.get("content", ""),
            "tool_events": tool_events,
            "subagent_outputs": subagent_outputs,
        }

    return {
        "status": "ok",
        "content": "",
        "tool_events": tool_events,
        "subagent_outputs": subagent_outputs,
    }


async def run_subagents(subagents_spec: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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
    if not config:
        raise HTTPException(status_code=400, detail="Configuration not set")

    if request.session_id and session_exists(request.session_id):
        session_id = request.session_id
    else:
        session = create_persistent_session("New chat")
        session_id = session["id"]

    ensure_session_policy(session_id)
    persistent = session_id not in temp_sessions

    async def event_generator():
        yield sse_event("stage", {"name": "meta_start"})
        session_context = build_session_context(session_id)
        meta_output = await run_meta_agent(session_context)
        yield sse_event("meta", meta_output.model_dump())

        if meta_output.memory_actions:
            apply_memory_actions(meta_output.memory_actions)

        history_turns = get_recent_turns(session_id, MAX_HISTORY_TURNS)
        messages = build_main_messages(request.message, meta_output, history_turns, [])

        shell_allowed = session_policies[session_id].get("shell_command_allowed", False)
        tool_events: List[Dict[str, Any]] = []
        subagent_outputs: List[Dict[str, Any]] = []
        tools_started = False

        for _ in range(MAX_TOOL_ROUNDS):
            response = await LLMProvider.call(config.main, messages, tools=TOOLS_SPEC)
            assistant_msg = response["choices"][0]["message"]
            tool_calls = assistant_msg.get("tool_calls")

            if tool_calls:
                for tool_call in tool_calls:
                    if not tool_call.get("id"):
                        tool_call["id"] = f"call_{uuid.uuid4()}"

                def needs_approval(tc: Dict[str, Any]) -> bool:
                    fn = tc.get("function", {})
                    if fn.get("name") != "shell_command":
                        return False
                    try:
                        args = json.loads(fn.get("arguments") or "{}")
                    except Exception:
                        args = {}
                    cmd = args.get("command", "")
                    return should_allow_shell(cmd, request.message) and not shell_allowed

                if any(needs_approval(tc) for tc in tool_calls):
                    pending_messages = messages + [{
                        "role": "assistant",
                        "content": assistant_msg.get("content", ""),
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
                    }
                    pending_tools = [
                        {
                            "id": tc.get("id"),
                            "name": tc.get("function", {}).get("name"),
                            "arguments": tc.get("function", {}).get("arguments", "{}"),
                        }
                        for tc in tool_calls
                    ]
                    yield sse_event("needs_approval", {
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
                    })
                    return

                if not tools_started:
                    yield sse_event("stage", {"name": "tools_start"})
                    tools_started = True

                messages.append({
                    "role": "assistant",
                    "content": assistant_msg.get("content", ""),
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

                    yield sse_event("tool_call", {"name": name, "args": args})

                    if name == "shell_command" and not should_allow_shell(args.get("command", ""), request.message):
                        result = "Tool rejected: not necessary. Respond directly without tools."
                        status = "rejected"
                    elif name == "shell_command" and not shell_allowed:
                        result = "DENIED: shell_command is not allowed for this agent."
                        status = "denied"
                    elif name == "spawn_subagents":
                        yield sse_event("stage", {"name": "subagents_start"})
                        subagents_spec = args.get("subagents", [])
                        outputs = await run_subagents(subagents_spec)
                        subagent_outputs.extend(outputs)
                        for output in outputs:
                            yield sse_event("subagent", output)
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
                    yield sse_event("tool_result", {
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

            reply = assistant_msg.get("content", "")
            store_turn(
                session_id=session_id,
                persistent=persistent,
                user_message=request.message,
                assistant_reply=reply,
                meta_json=meta_output.model_dump(),
                tool_events=tool_events,
                subagent_outputs=subagent_outputs,
            )
            yield sse_event("assistant", {
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
            })
            yield sse_event("done", {"session_id": session_id})
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
        yield sse_event("assistant", {
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
        })
        yield sse_event("done", {"session_id": session_id})

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

    session_context = build_session_context(session_id)
    meta_output = await run_meta_agent(session_context)

    if meta_output.memory_actions:
        apply_memory_actions(meta_output.memory_actions)

    history_turns = get_recent_turns(session_id, MAX_HISTORY_TURNS)
    messages = build_main_messages(request.message, meta_output, history_turns, [])

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
    )

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
    )


@app.post("/api/tools/approve")
async def approve_tools(request: ToolApprovalRequest) -> ChatResponse:
    if request.session_id not in pending_approvals:
        raise HTTPException(status_code=404, detail="No pending tool approval")

    pending = pending_approvals.pop(request.session_id)
    messages = pending["messages"]
    tool_calls = pending["tool_calls"]
    tool_events = pending["tool_events"]
    original_user_message = pending.get("user_message", "(tool approval continuation)")
    meta_json = pending.get("meta_json", {})
    subagent_outputs = pending.get("subagent_outputs", [])
    persistent = pending.get("persistent", request.session_id not in temp_sessions)

    allow_shell = request.decision in ["run_once", "allow_session"]
    if request.decision == "allow_session":
        session_policies[request.session_id]["shell_command_allowed"] = True

    for tool_call in tool_calls:
        fn = tool_call.get("function", {})
        name = fn.get("name")
        args_raw = fn.get("arguments", "{}")
        try:
            args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
        except Exception:
            args = {}

        if name == "shell_command" and not should_allow_shell(args.get("command", ""), original_user_message):
            result = "Tool rejected: not necessary. Respond directly without tools."
            status = "rejected"
        elif name == "shell_command" and not allow_shell:
            result = "DENIED: user requested manual steps instead of running shell_command."
            status = "denied"
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
    )

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
    )


@app.get("/")
async def serve_frontend():
    return FileResponse("index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
