# OpenAgent
<img width="1918" height="913" alt="image" src="https://github.com/koo1140/OpenAgent/blob/main/547248235-0298b148-3327-4bee-9ecd-981ba20caeb7.png?raw=true" />


A sophisticated multi-agent LLM orchestration system with meta-cognitive capabilities, memory management, and adaptive learning.

## Overview

This system implements dual architecture modes:

- `legacy` (existing five-layer pipeline)
- `hierarchical_v2` (Main -> Summarizer optional -> Orchestrator -> Sub-Agent strict syntax)

`hierarchical_v2` includes loop guards:
- Terminal tool outcomes (`rejected`, `denied`, `unknown_tool`) stop repeated retries.
- Repeated failing actions are auto-finished with partial results and warnings.
- Main finalizes directly after orchestrator completion to avoid recursive orchestrator calls.

The legacy mode uses a sophisticated five-layer architecture:

1. **Layer 1: Meta-Analysis** - Observational consciousness analyzing intent, tone, and patterns across 6 dimensions.
2. **Layer 2: System Planner** - Strategic layer that processes meta-analysis to decide on memory loading, tool strategies, and sub-agent deployment.
3. **Layer 3: Main Agent** - The primary execution layer that interacts with the user and coordinates overall task completion.
4. **Layer 4: Sub-Agents** - Specialized, focused workers spawned for parallel execution of complex sub-tasks.
5. **Layer 5: Gatekeeper** - Final validation layer that summarizes the turn, updates working memory, and evolves persistent memory files.

## Key Features

### 🧠 Meta-Cognitive Intelligence
The Meta Agent doesn't just process requests - it:
- Understands deeper user intent beyond literal questions
- Notices tone changes and emotional patterns
- Identifies learning opportunities from the user
- Tracks what's known vs. unknown about the user
- Builds a living model of the user over time
- Creates "aha moments" showing genuine attention

### 📚 Adaptive Memory System
Three core memory files evolve through interactions:
- **identity.md** - Assistant's core values and principles
- **soul.md** - Deeper philosophical stance and purpose
- **user.md** - Growing profile of the user built from observations

Memory loading is intelligent - files are only loaded when context requires them, saving tokens and improving performance.

### 🔧 Universal LLM Provider Support
Supports OpenAI-compatible APIs plus Anthropic:
- Groq
- OpenAI
- Mistral
- Anthropic (Claude)
- Together AI
- OpenRouter
- DeepSeek
- Custom endpoints

### 🛠️ Powerful Tool System
Built-in tools for real work:
- **shell_command** - Execute system commands
- **read_file** - Read files (smart truncation for large files)
- **edit_file** - Create/modify files
- **regex_search** - Search text in files and directories
- **curl_request** - HTTP requests with full control

### 🗂️ Persistent Sessions
- Session list with resume/rename/delete
- SQLite-backed history for persistent chats
- Temporary chats that don’t persist

### 🔒 Tool Approval
- Shell commands require explicit approval
- Approve once or deny (`allow_session` is accepted for compatibility and treated as run-once)
- Tool output is truncated in UI with on-demand full view

### ⏳ Resilient Execution
- Automatic 0.5s delay between API calls to prevent rate limits.
- Built-in 429 (Rate Limit) handling with 30s automatic retry.
- Interactive user decision (Continue/Stop) if rate limits persist.
- Real-time progress visualization for every architectural stage.

### 🎯 Sub-Agent Orchestration
Complex tasks can be split across specialized sub-agents with:
- Limited context (just what they need)
- Specific tools for their role
- Clear deliverables
- Parallel execution

## Architecture

```
         User Input
             ↓
┌───────────────────────────┐
│  Layer 1: Meta-Analysis   │ (Intent, Tone, User, Subject, Needs, Patterns)
└─────────────┬─────────────┘
              ↓ (Structured Analysis)
┌─────────────┴─────────────┐
│  Layer 2: System Planner  │ (Strategy, Memory Actions, Sub-agent Tasks)
└─────────────┬─────────────┘
              ↓ (Strategic Plan)
┌─────────────┴─────────────┐
│    Layer 3: Main Agent    │ (Tool Use, Context Integration, User Response)
└─────────────┬─────────────┘
              ↓ (If Complex)
┌─────────────┴─────────────┐
│   Layer 4: Sub-Agents     │ (Parallel Focused Tasks)
└─────────────┬─────────────┘
              ↓ (Execution Results)
┌─────────────┴─────────────┐
│   Layer 5: Gatekeeper     │ (Turn Summary, Memory Evolution, Validation)
└─────────────┬─────────────┘
              ↓
        User Response
```
---
[![Star History Chart](https://api.star-history.com/svg?repos=koo1140/OpenAgent&type=date&legend=top-left)](https://www.star-history.com/#koo1140/OpenAgent&type=date&legend=top-left)
---
## Installation

```bash
# Open the folder
cd OpenAgent

# Install dependencies
pip install -r requirements.txt
```

## Configuration

1. Start the server:
```bash
python main.py
```

2. Open http://localhost:8000 in your browser

3. Click "Settings" and configure your LLM providers:

### Meta Agent
- **Purpose**: Observational analysis
- **Recommendation**: Fast, cheap model (e.g., Groq's Llama or Mixtral)
- **Why**: Runs on every message, needs speed

### Main Agent
- **Purpose**: User interaction and execution
- **Recommendation**: Balanced model (e.g., GPT-4, Claude Sonnet, Mixtral)
- **Why**: Needs intelligence + tool use capabilities

### Sub Agents
- **Purpose**: Specialized task execution
- **Recommendation**: Cheap, fast model (e.g., GPT-3.5, Groq Llama)
- **Why**: Simple, focused tasks with limited context

Example configuration:
```json
{
  "architecture_mode": "legacy",
  "meta": {
    "provider": "Groq",
    "provider_type": "openai_compatible",
    "model": "mixtral-8x7b-32768",
    "api_key": "your-groq-key",
    "base_url": null,
    "headers": null,
    "supports_tools": true,
    "supports_response_format": false
  },
  "main": {
    "provider": "Anthropic",
    "provider_type": "anthropic",
    "model": "claude-sonnet-4-20250514",
    "api_key": "your-anthropic-key",
    "base_url": null,
    "headers": null,
    "supports_tools": true,
    "supports_response_format": false
  },
  "sub": {
    "provider": "Groq",
    "provider_type": "openai_compatible",
    "model": "llama3-70b-8192",
    "api_key": "your-groq-key",
    "base_url": null,
    "headers": null,
    "supports_tools": true,
    "supports_response_format": false
  },
  "orchestrator": {
    "provider": "Groq",
    "provider_type": "openai_compatible",
    "model": "mixtral-8x7b-32768",
    "api_key": "your-groq-key",
    "base_url": null,
    "headers": null,
    "supports_tools": true,
    "supports_response_format": false
  },
  "summarizer": {
    "provider": "Groq",
    "provider_type": "openai_compatible",
    "model": "mixtral-8x7b-32768",
    "api_key": "your-groq-key",
    "base_url": null,
    "headers": null,
    "supports_tools": false,
    "supports_response_format": false
  }
}
```

## How It Works

### Meta Agent Analysis

Every user message goes through the Meta Agent first. It analyzes:

1. **User Intent** - What they really want (beyond literal request)
2. **Chat Subject** - The deeper topic being discussed
3. **User Tone** - Emotional state and changes
4. **Known Data** - Facts about the user
5. **Missing Data** - What we should learn
6. **Knowledge Needs** - Skills required for this task
7. **Knowledge Gaps** - What the assistant lacks
8. **Learning Opportunities** - Chances to learn from the user
9. **Aha Moments** - Insights to share ("I noticed...")
10. **Memory Actions** - Updates to user.md
11. **Plan** - Recommended approach
12. **Sub-Agents** - Whether to spawn specialized workers

The Meta Agent outputs structured JSON guiding the Main Agent.

### Main Agent Execution

The Main Agent receives:
- Meta Agent's JSON analysis
- Memory files (if Meta Agent deemed them relevant)
- Conversation history
- Tool access

It then:
1. Reads the meta analysis to understand context
2. Loads memory files if needed
3. Mentions "aha moments" to show attention
4. Asks questions from "learning opportunities"
5. Uses tools to accomplish tasks
6. Spawns sub-agents for complex work
7. Responds naturally to the user

### Memory Evolution

The system builds understanding over time:

**user.md grows with:**
- Discovered skills and expertise
- Observed preferences
- Emotional patterns
- Current projects
- Communication style
- Mental health trajectory

**identity.md and soul.md provide:**
- Consistent values across sessions
- Philosophical grounding
- Principled decision-making
- Authentic personality

### The Secret Sauce: Learning Opportunities

This is what makes the system exceptional. Instead of just answering questions, it actively looks for chances to learn:

```
User: "I always use composition over inheritance"

Meta Agent identifies:
{
  "learning_opportunities": [
    "user has strong opinions on OOP design - ask their reasoning",
    "could learn their approach to code architecture"
  ]
}

Main Agent asks:
"That's interesting! What led you to prefer composition? 
I'd love to understand your thinking."

[Genuinely learns from the answer and applies it in future interactions]
```

This creates a system that:
- Gets smarter about YOUR specific context
- Builds domain knowledge from YOU
- Adapts to YOUR preferences
- Grows WITH you

## Example Interactions

### Example 1: Simple Request

**User**: "How do I reverse a string in Python?"

**Meta Agent**:
```json
{
  "user_intent": "learn basic Python operation",
  "chat_subject": "Python programming basics",
  "user_tone": "curious, learning mode",
  "keepMemoryFilesLoaded": false,
  "recommended_plan": "teach with clear example, offer alternatives"
}
```

**Main Agent**:
"Here's the cleanest way:
```python
reversed_str = original[::-1]
```
This uses slice notation. Want to see other approaches?"

---

### Example 2: Complex Task

**User**: "My FastAPI server keeps crashing when I hit it with load"

**Meta Agent**:
```json
{
  "user_intent": "debug production issue",
  "chat_subject": "API performance under load",
  "user_tone": "frustrated, stressed",
  "knowledge_needed": ["FastAPI", "performance debugging", "system resources"],
  "aha_moments": ["user seems more stressed than usual - something urgent?"],
  "keepMemoryFilesLoaded": true,
  "recommended_plan": "diagnose with tools, show empathy for stress"
}
```

**Main Agent**:
"I can hear the frustration - let's figure this out. 

[Uses shell_command to check server logs]
[Uses read_file to check config]

Found it - you're hitting memory limits. Your workers are set to 100 but you only have 2GB RAM. Let me fix the config..."

---

### Example 3: Learning Opportunity

**User**: "I prefer TDD for all my API development"

**Meta Agent**:
```json
{
  "learning_opportunities": [
    "user practices TDD - learn their specific workflow",
    "could understand how they structure tests"
  ],
  "user_data_known": {
    "testing_approach": "TDD practitioner"
  }
}
```

**Main Agent**:
"That's great! I'm curious - what's your TDD workflow for APIs? Do you start with integration tests or unit tests?"

[User explains their approach]

**Main Agent** (in future):
"Since you do TDD, want me to write the tests first before implementing the endpoint?"

[Applied the learned knowledge]

## Advanced Features

### Tone Change Detection

The Meta Agent tracks emotional trajectory:

```json
{
  "tone_changes_noticed": "user was enthusiastic earlier but now seems tired - long session?",
  "aha_moments": ["You've been at this for a while - make sure to take a break!"]
}
```

### Mental Health Tracking

Observes patterns over time:

```json
{
  "mental_health_update": "user seems more stressed than last week - project deadline approaching?"
}
```

Updates user.md:
```markdown
## Mental Health
- [2024-02-08] More stressed than usual, project pressure
- [2024-02-01] Calm and focused, good work rhythm
```

### Adaptive Memory Loading

Meta Agent decides when to load expensive memory files:

**Load memory** when:
- Personal conversation
- Building on previous work
- User context matters
- Long-term project

**Don't load memory** when:
- Simple factual questions
- Self-contained tasks
- Speed matters
- First-time user

### Sub-Agent Spawning

For complex parallel work:

```json
{
  "subagents_needed": [
    {
      "role": "backend_dev",
      "task": "Implement the user authentication endpoints",
      "tools": ["read_file", "edit_file", "shell_command"]
    },
    {
      "role": "frontend_dev",
      "task": "Create the login UI component",
      "tools": ["read_file", "edit_file"]
    }
  ]
}
```

Main Agent coordinates their work and integrates results.

## File Structure

```
agentic-gateway/
├── main.py                 # FastAPI backend
├── index.html             # Frontend interface
├── requirements.txt       # Python dependencies
├── config.json           # Saved provider configuration
├── sessions.db           # SQLite session storage
├── prompts/
│   ├── meta_agent.txt    # Meta Agent instructions
│   └── main_agent.txt    # Main Agent instructions
├── skills/               # Optional skill files
└── memory/
    ├── identity.md       # Assistant's core identity
    ├── soul.md          # Philosophical purpose
    └── user.md          # User profile (grows over time)
```

## API Endpoints

### GET /api/providers
Returns provider registry (base URL + tool support).

### GET /api/config
Load saved provider configuration.

### POST /api/config
Save LLM provider configuration.

### GET /api/sessions
List persistent sessions.

### POST /api/sessions
Create a session.
```json
{ "title": "My chat", "persistent": true }
```

### GET /api/sessions/{id}
Load a session with turns.

### PATCH /api/sessions/{id}
Rename a session.

### DELETE /api/sessions/{id}
Delete a session.

### POST /api/chat
Send a message.
```json
{
  "message": "user message here",
  "session_id": "uuid"
}
```

Returns:
```json
{
  "reply": "assistant response",
  "meta": {...},
  "tool_events": [...],
  "subagent_outputs": [...],
  "status": "ok | needs_approval",
  "pending_tools": [...],
  "session_id": "uuid",
  "architecture_mode": "legacy | hierarchical_v2"
}
```

### POST /api/tools/approve
Approve shell_command tool calls.
```json
{
  "session_id": "uuid",
  "decision": "run_once | allow_session | deny"
}
```
`allow_session` is backward-compatible and behaves like `run_once`.

## Provider Configuration

Common fields:
- `provider`
- `provider_type` (`openai_compatible` or `anthropic`)
- `model`
- `api_key`
- `base_url` (optional override)
- `headers` (optional extra HTTP headers)
- `supports_tools` (optional override)
- `supports_response_format` (optional override)

### Groq
```json
{
  "provider": "Groq",
  "provider_type": "openai_compatible",
  "model": "mixtral-8x7b-32768",
  "api_key": "gsk_..."
}
```

### OpenAI
```json
{
  "provider": "OpenAI",
  "provider_type": "openai_compatible",
  "model": "gpt-4-turbo-preview",
  "api_key": "sk-..."
}
```

### Anthropic
```json
{
  "provider": "Anthropic",
  "provider_type": "anthropic",
  "model": "claude-sonnet-4-20250514",
  "api_key": "sk-ant-..."
}
```

### Custom Endpoint
```json
{
  "provider": "Custom",
  "provider_type": "openai_compatible",
  "model": "custom-model",
  "api_key": "your-key",
  "base_url": "https://your-endpoint.com/v1/chat/completions",
  "headers": {
    "X-Project": "my-app"
  }
}
```

## Best Practices

### Provider Selection

**Meta Agent**: Fast + Cheap
- Groq (Mixtral, Llama)
- OpenAI (GPT-3.5)
- Reason: Runs every message, analysis task

**Main Agent**: Smart + Capable
- Anthropic (Claude Sonnet/Opus)
- OpenAI (GPT-4)
- Mistral (Large)
- Reason: Complex reasoning, tool use, user interaction

**Sub Agents**: Fast + Focused
- Groq (Llama)
- OpenAI (GPT-3.5)
- Reason: Simple tasks, limited context

### Memory Management

- Review user.md periodically to ensure accuracy
- Keep identity.md and soul.md aligned with your values
- Let the Meta Agent update user.md automatically
- Manually edit for major user context changes

### Tool Usage

- Tools have limits (200 line threshold for files)
- Use regex_search for large codebases
- Chain tools thoughtfully (read → analyze → edit)
- Sub-agents can use tools independently

### Customization

Edit the prompt files to change behavior:
- `prompts/meta_agent.txt` - Analysis priorities
- `prompts/main_agent.txt` - Interaction style
- `memory/identity.md` - Assistant's values
- `memory/soul.md` - Philosophical stance

## Troubleshooting

### "Configuration not set" error
1. Go to Settings
2. Fill in all three agent configs
3. Click "Save Configuration"

### Meta Agent JSON parse errors
- Check Meta Agent model supports JSON output
- Some models need `response_format` parameter
- Use models known for structured output

### Tool execution failures
- Check file paths are correct
- Verify permissions for shell commands
- Large files may be truncated (>200 lines)

### Tool approval required
- When a tool call includes `shell_command`, the UI will ask for approval
- Choose Run once, Always allow this session, or Deny (manual steps)

### Memory files not loading
- Check `keepMemoryFilesLoaded` in Meta Agent output
- Ensure files exist in `memory/` directory
- Verify file format is valid markdown

## Why This System Works

### Traditional Chatbots
- Static knowledge from training
- No real user understanding
- Generic responses
- Forgettable interactions

### This System
- **Learns FROM the user** - builds personalized knowledge
- **Notices patterns** - tone changes, preferences, expertise
- **Shows attention** - "aha moments" prove we're paying attention
- **Grows together** - user.md evolves, assistant improves
- **Adapts approach** - meta-cognitive guidance for each interaction
- **Builds relationship** - shared history, inside knowledge, genuine partnership

The key insight: **True intelligence isn't about knowing everything - it's about noticing, learning, and adapting in context.**

## Contributing

This is a framework for intelligent, adaptive assistance. Customize it:
- Add new tools
- Modify agent prompts
- Extend memory structure
- Add new providers
- Enhance meta-analysis
Consider submitting a PR to help the project grow!

## License

MIT

## Credits

Built on the principle that AI assistance should feel like working with someone who truly knows you - not from surveillance, but from genuine attention and shared growth.

---

*"The best assistant isn't the one that knows everything - it's the one that really knows you."*
