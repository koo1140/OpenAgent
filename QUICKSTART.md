# Quick Start Guide

Get your Open Agent running in 3 minutes.

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Start the Server

```bash
./start.sh
# or
python main.py
```

## Step 3: Configure Providers

1. Open http://localhost:8000
2. Click **Settings**
3. Configure each agent:

### Meta Agent (Recommended: Fast & Cheap)
- **Provider**: Groq
- **Model**: `mixtral-8x7b-32768`
- **API Key**: Your Groq API key

### Main Agent (Recommended: Smart & Capable)
- **Provider**: Anthropic (or OpenAI)
- **Model**: `claude-sonnet-4-20250514` (or `gpt-4-turbo-preview`)
- **API Key**: Your Anthropic/OpenAI API key

### Sub Agents (Recommended: Fast & Focused)
- **Provider**: Groq
- **Model**: `llama3-70b-8192`
- **API Key**: Your Groq API key

### Architecture Mode
- `legacy` keeps the original meta/planner pipeline
- `hierarchical_v2` enables Main + Summarizer + Orchestrator + Sub-Agent strict `[TOOL]` syntax flow
- `hierarchical_v2` now includes loop guards to auto-stop repeated failing actions

4. Click **Save Configuration**
5. Click **Chat** and start chatting!

You can create, rename, and resume sessions from the Sessions panel.

## Getting API Keys

### Groq (Free Tier Available)
- Visit: https://console.groq.com
- Sign up and get your API key
- Recommended models: `mixtral-8x7b-32768`, `llama3-70b-8192`

### Anthropic
- Visit: https://console.anthropic.com
- Get your API key
- Recommended model: `claude-sonnet-4-20250514`

### OpenAI
- Visit: https://platform.openai.com
- Get your API key
- Recommended models: `gpt-4-turbo-preview`, `gpt-3.5-turbo`

### Mistral
- Visit: https://console.mistral.ai
- Get your API key
- Recommended models: `mistral-large-latest`, `mistral-medium-latest`

## Example Configuration

```json
{
  "architecture_mode": "legacy",
  "meta": {
    "provider": "Groq",
    "provider_type": "openai_compatible",
    "model": "mixtral-8x7b-32768",
    "api_key": "gsk_...",
    "base_url": null,
    "headers": null,
    "supports_tools": true,
    "supports_response_format": false
  },
  "main": {
    "provider": "Anthropic",
    "provider_type": "anthropic",
    "model": "claude-sonnet-4-20250514",
    "api_key": "sk-ant-...",
    "base_url": null,
    "headers": null,
    "supports_tools": true,
    "supports_response_format": false
  },
  "sub": {
    "provider": "Groq",
    "provider_type": "openai_compatible",
    "model": "llama3-70b-8192",
    "api_key": "gsk_...",
    "base_url": null,
    "headers": null,
    "supports_tools": true,
    "supports_response_format": false
  },
  "orchestrator": {
    "provider": "Groq",
    "provider_type": "openai_compatible",
    "model": "mixtral-8x7b-32768",
    "api_key": "gsk_...",
    "base_url": null,
    "headers": null,
    "supports_tools": true,
    "supports_response_format": false
  },
  "summarizer": {
    "provider": "Groq",
    "provider_type": "openai_compatible",
    "model": "mixtral-8x7b-32768",
    "api_key": "gsk_...",
    "base_url": null,
    "headers": null,
    "supports_tools": false,
    "supports_response_format": false
  }
}
```

## First Conversation

Try these to see the system in action:

### Simple Task
"Write me a Python function to calculate fibonacci numbers"

### Complex Task  
"Help me debug this FastAPI endpoint - it keeps timing out"
(paste some code)

### Learning Opportunity
"I always use composition over inheritance in my code"
(Watch the assistant ask follow-up questions to learn from you!)

### Personal Connection
Have a few conversations, then say:
"I'm feeling stuck on this project"
(Notice how it remembers your context and shows genuine attention)

## Tips

- The Meta Agent runs on EVERY message - use fast/cheap models
- The Main Agent needs to be smart and support tool use
- Sub Agents are for simple tasks - fast/cheap is fine
- Memory files evolve over time - check `memory/user.md` after a few chats
- Edit `prompts/meta_agent.txt` and `prompts/main_agent.txt` to customize behavior
- Shell commands require approval in the UI

## Troubleshooting

**"Configuration not set" error**
→ Go to Settings and save your configuration

**JSON parse errors**
→ Some models don't support structured output well. Try Groq's Mixtral or OpenAI's GPT-4

**Tool execution fails**
→ Check file paths and permissions

**hierarchical_v2 seems stuck**
→ Check tool result warnings in Details; repeated failing actions are auto-finished and surfaced there

**Tool approval required**
→ Approve once or deny and ask for manual steps (`allow_session` is treated as run-once)

**Meta agent not analyzing well**
→ Try a smarter model (but it will be slower/costlier)

## What Makes This Special?

Unlike traditional chatbots, this system:

1. **Actually notices you** - Tracks your tone, preferences, expertise
2. **Learns from you** - Asks questions to build personalized knowledge
3. **Shows attention** - "I noticed your tone changed" moments
4. **Grows over time** - user.md evolves with each conversation
5. **Adapts approach** - Meta agent guides each interaction differently

Read the full README.md for deep dive on architecture and advanced features.

---

**Ready to start?** → `./start.sh` then open http://localhost:8000
