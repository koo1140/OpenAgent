"""
Agentic Gateway Backend
Multi-provider LLM orchestration with Meta, Main, and Sub agents
"""

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal
import httpx
import json
import os
from pathlib import Path
import asyncio

app = FastAPI()

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class ProviderConfig(BaseModel):
    provider: Literal["Groq", "OpenAI", "Mistral", "Anthropic", "Together", "OpenRouter", "DeepSeek", "Google"]
    model: str
    api_key: str
    base_url: Optional[str] = None  # For custom endpoints

class AgentConfig(BaseModel):
    meta: ProviderConfig
    main: ProviderConfig
    sub: ProviderConfig

class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str

class ToolCall(BaseModel):
    name: str
    arguments: Dict[str, Any]

class MetaOutput(BaseModel):
    """Meta Agent's structured output"""
    user_intent: str = Field(description="What the user wants to achieve")
    chat_subject: str = Field(description="The deeper subject being discussed")
    user_tone: str = Field(description="Current emotional tone")
    user_data_known: Dict[str, Any] = Field(default_factory=dict, description="Facts we know about the user")
    user_data_missing: List[str] = Field(default_factory=list, description="What we still don't know but should")
    knowledge_needed: List[str] = Field(default_factory=list, description="Skills/knowledge required")
    knowledge_gaps: List[str] = Field(default_factory=list, description="What the assistant lacks")
    learning_opportunities: List[str] = Field(default_factory=list, description="Chances to learn from user or context")
    aha_moments: List[str] = Field(default_factory=list, description="Insights to share with user")
    memory_actions: List[Dict[str, str]] = Field(default_factory=list, description="Memory file updates needed")
    keepMemoryFilesLoaded: bool = Field(default=False, description="Whether to inject memory files")
    recommended_plan: str = Field(description="High-level approach")
    subagents_needed: List[Dict[str, Any]] = Field(default_factory=list, description="Sub agents to spawn")
    tone_changes_noticed: Optional[str] = None
    mental_health_update: Optional[str] = None

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str

# ============================================================================
# GLOBAL STATE
# ============================================================================

config: Optional[AgentConfig] = None
conversation_history: List[ChatMessage] = []
tool_logs: List[Dict[str, Any]] = []
memory_dir = Path("memory")
memory_dir.mkdir(exist_ok=True)

# ============================================================================
# PROVIDER ABSTRACTION
# ============================================================================

class LLMProvider:
    """Unified interface for all LLM providers"""
    
    ENDPOINTS = {
        "Groq": "https://api.groq.com/openai/v1/chat/completions",
        "OpenAI": "https://api.openai.com/v1/chat/completions",
        "Mistral": "https://api.mistral.ai/v1/chat/completions",
        "Anthropic": "https://api.anthropic.com/v1/messages",
        "Together": "https://api.together.xyz/v1/chat/completions",
        "OpenRouter": "https://openrouter.ai/api/v1/chat/completions",
        "DeepSeek": "https://api.deepseek.com/v1/chat/completions",
        "Google": "https://generativelanguage.googleapis.com/v1beta/chat/completions",
    }
    
    @staticmethod
    async def call(provider_config: ProviderConfig, messages: List[Dict[str, str]], 
                   tools: Optional[List[Dict]] = None, response_format: Optional[Dict] = None) -> Dict[str, Any]:
        """Universal LLM API call"""
        
        endpoint = provider_config.base_url or LLMProvider.ENDPOINTS.get(provider_config.provider)
        if not endpoint:
            raise ValueError(f"Unknown provider: {provider_config.provider}")
        
        # Build request based on provider
        if provider_config.provider == "Anthropic":
            return await LLMProvider._call_anthropic(provider_config, messages, tools)
        else:
            return await LLMProvider._call_openai_compatible(provider_config, endpoint, messages, tools, response_format)
    
    @staticmethod
    async def _call_openai_compatible(config: ProviderConfig, endpoint: str, 
                                     messages: List[Dict], tools: Optional[List[Dict]], 
                                     response_format: Optional[Dict]) -> Dict[str, Any]:
        """OpenAI-compatible API call"""
        headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": config.model,
            "messages": messages,
        }
        
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        
        if response_format:
            payload["response_format"] = response_format
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(endpoint, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()
    
    @staticmethod
    async def _call_anthropic(config: ProviderConfig, messages: List[Dict], 
                             tools: Optional[List[Dict]]) -> Dict[str, Any]:
        """Anthropic-specific API call"""
        headers = {
            "x-api-key": config.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }
        
        # Separate system message
        system_msg = next((m["content"] for m in messages if m["role"] == "system"), None)
        conversation = [m for m in messages if m["role"] != "system"]
        
        payload = {
            "model": config.model,
            "max_tokens": 4096,
            "messages": conversation,
        }
        
        if system_msg:
            payload["system"] = system_msg
        
        if tools:
            payload["tools"] = tools
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            data = response.json()
            
            # Convert to OpenAI format
            return {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": data["content"][0]["text"] if data["content"] else ""
                    }
                }]
            }

# ============================================================================
# TOOLS
# ============================================================================

TOOLS_SPEC = [
    {
        "type": "function",
        "function": {
            "name": "shell_command",
            "description": "Execute a shell command",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The shell command to execute"}
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file. Returns full content if <200 lines, otherwise first 60 lines",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to read"}
                },
                "required": ["path"]
            }
        }
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
                    "mode": {"type": "string", "enum": ["write", "append"], "description": "Write mode"}
                },
                "required": ["path", "content"]
            }
        }
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
                    "recursive": {"type": "boolean", "description": "Search recursively in directories"}
                },
                "required": ["pattern", "path"]
            }
        }
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
                    "body": {"type": "string", "description": "Request body"}
                },
                "required": ["url", "method"]
            }
        }
    }
]

async def execute_tool(name: str, arguments: Dict[str, Any]) -> str:
    """Execute a tool and return result"""
    try:
        if name == "shell_command":
            proc = await asyncio.create_subprocess_shell(
                arguments["command"],
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()
            return f"STDOUT:\n{stdout.decode()}\nSTDERR:\n{stderr.decode()}"
        
        elif name == "read_file":
            path = Path(arguments["path"])
            if not path.exists():
                return f"Error: File {path} does not exist"
            
            lines = path.read_text().splitlines()
            if len(lines) <= 200:
                return "\n".join(lines)
            else:
                return "\n".join(lines[:60]) + f"\n... (truncated, {len(lines)} total lines)"
        
        elif name == "edit_file":
            path = Path(arguments["path"])
            path.parent.mkdir(parents=True, exist_ok=True)
            
            mode = arguments.get("mode", "write")
            content = arguments["content"]
            
            if mode == "append":
                with open(path, "a") as f:
                    f.write(content)
            else:
                path.write_text(content)
            
            return f"Successfully wrote to {path}"
        
        elif name == "regex_search":
            import re
            pattern = re.compile(arguments["pattern"])
            path = Path(arguments["path"])
            recursive = arguments.get("recursive", False)
            
            results = []
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
                        except:
                            pass
            
            return "\n".join(results) if results else "No matches found"
        
        elif name == "curl_request":
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.request(
                    method=arguments["method"],
                    url=arguments["url"],
                    headers=arguments.get("headers", {}),
                    content=arguments.get("body")
                )
                return f"Status: {response.status_code}\n{response.text}"
        
        return f"Unknown tool: {name}"
    
    except Exception as e:
        return f"Tool execution error: {str(e)}"

# ============================================================================
# AGENT ORCHESTRATION
# ============================================================================

async def run_meta_agent(session_context: str) -> MetaOutput:
    """Run the meta agent to analyze the session"""
    
    meta_prompt = load_meta_agent_prompt()
    
    messages = [
        {"role": "system", "content": meta_prompt},
        {"role": "user", "content": f"Session Context:\n{session_context}"}
    ]
    
    response = await LLMProvider.call(
        config.meta,
        messages,
        response_format={"type": "json_object"}
    )
    
    content = response["choices"][0]["message"]["content"]
    
    # Parse JSON output
    try:
        meta_data = json.loads(content)
        return MetaOutput(**meta_data)
    except Exception as e:
        print(f"Meta agent JSON parse error: {e}")
        # Return minimal output
        return MetaOutput(
            user_intent="unclear",
            chat_subject="general",
            user_tone="neutral",
            recommended_plan="respond conversationally"
        )

async def run_main_agent(user_message: str, meta_output: MetaOutput) -> str:
    """Run the main agent with meta guidance"""
    
    # Build context
    main_prompt = load_main_agent_prompt()
    
    # Load memory files if needed
    memory_context = ""
    if meta_output.keepMemoryFilesLoaded:
        memory_context = load_memory_files()
    
    # Build messages
    messages = [
        {"role": "system", "content": f"{main_prompt}\n\n{memory_context}"},
        {"role": "user", "content": f"Meta Analysis:\n{meta_output.model_dump_json(indent=2)}\n\nUser Message: {user_message}"}
    ]
    
    # Add conversation history
    for msg in conversation_history[-10:]:  # Last 10 messages
        messages.append({"role": msg.role, "content": msg.content})
    
    # Call main agent with tools
    response = await LLMProvider.call(
        config.main,
        messages,
        tools=TOOLS_SPEC
    )
    
    assistant_message = response["choices"][0]["message"]
    
    # Handle tool calls
    if "tool_calls" in assistant_message:
        tool_results = []
        for tool_call in assistant_message["tool_calls"]:
            func = tool_call["function"]
            result = await execute_tool(func["name"], json.loads(func["arguments"]))
            tool_results.append(result)
            
            # Log tool usage
            tool_logs.append({
                "tool": func["name"],
                "args": func["arguments"],
                "result": result
            })
        
        # Continue conversation with tool results
        messages.append({"role": "assistant", "content": assistant_message.get("content", "")})
        messages.append({"role": "user", "content": f"Tool Results:\n" + "\n\n".join(tool_results)})
        
        # Get final response
        final_response = await LLMProvider.call(config.main, messages)
        return final_response["choices"][0]["message"]["content"]
    
    return assistant_message.get("content", "")

def load_meta_agent_prompt() -> str:
    """Load the meta agent system prompt"""
    prompt_path = Path("prompts/meta_agent.txt")
    if prompt_path.exists():
        return prompt_path.read_text()
    return "You are a meta-cognitive agent analyzing user interactions."

def load_main_agent_prompt() -> str:
    """Load the main agent system prompt"""
    prompt_path = Path("prompts/main_agent.txt")
    if prompt_path.exists():
        return prompt_path.read_text()
    return "You are a helpful AI assistant with access to tools."

def load_memory_files() -> str:
    """Load identity, soul, and user memory files"""
    files = ["identity.md", "soul.md", "user.md"]
    content = []
    
    for filename in files:
        path = memory_dir / filename
        if path.exists():
            content.append(f"=== {filename.upper()} ===\n{path.read_text()}\n")
    
    return "\n".join(content)

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.post("/api/config")
async def save_config(agent_config: AgentConfig):
    """Save agent configuration"""
    global config
    config = agent_config
    
    # Save to disk
    config_path = Path("config.json")
    config_path.write_text(agent_config.model_dump_json(indent=2))
    
    return {"status": "success"}

@app.post("/api/chat")
async def chat(request: ChatRequest) -> ChatResponse:
    """Handle chat messages"""
    global conversation_history, tool_logs
    
    if not config:
        raise HTTPException(status_code=400, detail="Configuration not set")
    
    # Add user message to history
    conversation_history.append(ChatMessage(role="user", content=request.message))
    
    # Build session context
    session_context = {
        "conversation": [msg.model_dump() for msg in conversation_history],
        "tool_logs": tool_logs[-20:]  # Last 20 tool calls
    }
    
    # Run meta agent
    meta_output = await run_meta_agent(json.dumps(session_context, indent=2))
    
    # Update memory files based on meta output
    for action in meta_output.memory_actions:
        if action.get("type") == "update":
            file_path = memory_dir / action.get("file", "user.md")
            file_path.write_text(action.get("content", ""))
    
    # Run main agent
    reply = await run_main_agent(request.message, meta_output)
    
    # Add assistant message to history
    conversation_history.append(ChatMessage(role="assistant", content=reply))
    
    return ChatResponse(reply=reply)

@app.get("/")
async def serve_frontend():
    """Serve the HTML frontend"""
    return FileResponse("index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)