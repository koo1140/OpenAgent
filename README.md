# 🚀 OpenAgent

**A sophisticated multi-agent LLM orchestration system featuring meta-cognitive capabilities, adaptive memory management, and deep learning-based evolution.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)

OpenAgent is not just a chatbot; it is an intelligent framework built on the principle that true AI assistance comes from **genuine attention and shared growth**. It learns *from* you, notices your patterns, and builds a living model of your needs over time.
<img width="1918" height="913" alt="image" src="https://github.com/koo1140/OpenAgent/blob/main/547248235-0298b148-3327-4bee-9ecd-981ba20caeb7.png?raw=true" />
---
## Guide
- [Meta-Cognitive Intelligence](#-the-secret-sauce-meta-cognitive-intelligence)
- [Architecture](#-architecture)
- [Adaptive Memory System](#-adaptive-memory-system)
- [Key Features](#-key-features)
- [Installation & Setup](#-installation--setup)
- [Repository Structure](#-repository-structure)
- [Contributing](#-contributing)
- [License](#-license)

## 🧠 The "Secret Sauce": Meta-Cognitive Intelligence

Unlike traditional chatbots that provide generic, forgettable interactions, OpenAgent uses a **Meta Agent** to analyze every message before the main response is even generated.

*   **Intent & Tone Detection:** Understands what you *really* want and notices emotional shifts (e.g., frustration or stress).
*   **"Aha Moments":** Identifies insights to share, proving the system is paying genuine attention.
*   **Learning Opportunities:** Actively asks questions to learn your specific workflows, like your preference for TDD or specific design patterns.
*   **Mental Health Tracking:** Observes stress patterns over time to provide empathetic context.

---
[![Star History Chart](https://api.star-history.com/svg?repos=koo1140/OpenAgent&type=date&legend=top-left)](https://www.star-history.com/#koo1140/OpenAgent&type=date&legend=top-left)
---

## 🏗️ Architecture

OpenAgent supports **dual architecture modes** to handle everything from simple chats to complex engineering tasks:

1.  **Hierarchical_v2 (Recommended):** A strict structure consisting of a Main Agent, optional Summarizer, Orchestrator, and Sub-Agents with loop guards to prevent recursive failures.
2.  **Legacy (5-Layer Pipeline):** A sophisticated sequential flow:

```text
User Input ↓
┌───────────────────────────┐
│  Layer 1: Meta-Analysis   │ (Intent, Tone, Patterns)
└─────────────┬─────────────┘
              ↓
┌─────────────┴─────────────┐
│  Layer 2: System Planner  │ (Strategy, Memory, Sub-agents)
└─────────────┬─────────────┘
              ↓
┌─────────────┴─────────────┐
│   Layer 3: Main Agent     │ (Tool Use, User Response)
└─────────────┬─────────────┘
              ↓ (If Complex)
┌─────────────┴─────────────┐
│   Layer 4: Sub-Agents     │ (Parallel Focused Tasks)
└─────────────┬─────────────┘
              ↓
┌─────────────┴─────────────┐
│   Layer 5: Gatekeeper     │ (Summary, Memory Evolution)
└─────────────┬─────────────┘
              ↓
        User Response
```

---

## 📚 Adaptive Memory System

The system manages three core Markdown-based memory files that evolve through interaction:
*   **`identity.md`**: The assistant’s core values and principles.
*   **`soul.md`**: Philosophical stance and deeper purpose.
*   **`user.md`**: A growing profile of you, including your expertise, preferences, and mental health trajectory.

**Smart Loading:** To save tokens and improve performance, memory files are only loaded when the Meta Agent determines they are relevant to the context.

---

## 🔧 Key Features

*   **Universal LLM Support:** Works with OpenAI, Anthropic (Claude), Groq, Mistral, Together AI, OpenRouter, DeepSeek, and custom endpoints.
*   **Powerful Toolset:** Built-in tools for `shell_command` execution (with approval), `read_file`, `edit_file`, `regex_search`, and `curl_request`.
*   **Resilient Execution:** Includes automatic 429 rate-limit handling (30s retry), loop guards for failing tools, and 0.5s delays between API calls.
*   **Persistent Sessions:** SQLite-backed history allows you to resume, rename, or delete chats.

---

## 🛠️ Installation & Setup

### 1. Clone and Install
```bash
# Open the folder
cd OpenAgent

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch the System
```bash
python main.py
```
Open **`http://localhost:8000`** in your browser to begin.

### 3. Recommended Configuration
Go to "Settings" to configure your providers. Best practice suggests:
*   **Meta Agent:** Fast & Cheap (e.g., Groq Llama 3, Mixtral, or GPT-3.5).
*   **Main Agent:** Smart & Capable (e.g., Claude 3.5 Sonnet, GPT-4, or Mistral Large).
*   **Sub-Agents:** Fast & Focused (e.g., GPT-3.5 or Llama 3).

---

## 📂 Repository Structure

*   `main.py`: FastAPI backend.
*   `index.html`: Frontend interface.
*   `prompts/`: Core instructions for Meta and Main agents.
*   `memory/`: Persistent memory files (`identity.md`, `soul.md`, `user.md`).
*   `sessions.db`: SQLite database for chat history.

---

## 🤝 Contributing

OpenAgent is a framework for intelligent, adaptive assistance. We welcome contributions:
*   Adding new tools or providers.
*   Enhancing meta-analysis logic.
*   Modifying agent prompts for different interaction styles.

---

## 📜 License

This project is licensed under the **MIT License**.

---

> *"The best assistant isn't the one that knows everything - it's the one that really knows you."*
