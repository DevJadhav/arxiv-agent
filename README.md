# ArXiv Agent

> **Your AI-Powered Research Assistant.**  
> Automatically discover, analyze, and implement research from arXiv papers using the power of advanced LLMs.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

---

**ArXiv Agent** is a terminal-based intelligent assistant designed for researchers and engineers who want to stay on top of the latest AI trends without drowning in PDFs. It combines **daily digests**, **RAG-powered chat**, and **automated implementation** into a single, cohesive workflow.

## 🚀 Key Features

| Feature | Description |
|---------|-------------|
| 📰 **Smart Daily Digests** | Wake up to a personalized summary of the latest papers in your field, delivered on a schedule you define |
| 💬 **Chat with Papers** | Use RAG to ask questions across your entire library or deep-dive into a specific paper |
| 🛠️ **Paper-to-Code** | Automatically generate implementation plans and code scaffolding from paper methodologies |
| 📚 **Personal Library** | Organize papers into collections with tags, notes, and full-text search |
| 📈 **Trend Analysis** | Discover trending topics and emerging research areas with topic modeling |
| 🧠 **Multi-Model Support** | Supports **Anthropic**, **OpenAI**, and **Google Gemini** with dynamic model selection |
| ⏰ **Scheduled Digests** | Background daemon for automated daily/weekly paper discovery |
| 🔒 **Privacy First** | Data stored locally (`~/.local/share/arxiv-agent`). API keys secured in system keyring |

---

## 📦 Installation

**Prerequisites:** Python 3.11+ and `pip` or `uv`

```bash
# Clone the repository
git clone https://github.com/DevJadhav/arxiv-agent.git
cd arxiv-agent

# Option 1: Install via pip
pip install .

# Option 2: Install via uv (Recommended)
uv tool install . --force
```

Verify installation:
```bash
arxiv-agent --help
arxiv-agent version
```

---

## ⚡ Quick Start

### 1. Setup Your LLM Provider

ArXiv Agent needs an LLM to function. Configure your preferred provider securely:

```bash
# Setup Anthropic (Recommended)
arxiv-agent config provider setup anthropic

# OR OpenAI
arxiv-agent config provider setup openai

# OR Google Gemini
arxiv-agent config provider setup gemini

# Check configured providers
arxiv-agent config provider list
```

### 2. Search & Discover Papers

```bash
# Search arXiv for papers
arxiv-agent search "transformer attention mechanism" --limit 5

# Get details for a specific paper
arxiv-agent paper 2401.12345

# View trending topics
arxiv-agent trends discover --category cs.AI
```

### 3. Build Your Library

```bash
# Add paper to library
arxiv-agent library add 2401.12345

# List papers in library
arxiv-agent library list

# Create collections
arxiv-agent library collections create "LLM Research"

# Add tags to papers
arxiv-agent library tag 2401.12345 --add "transformer,attention"

# Search your library
arxiv-agent library search "attention"

# Export library
arxiv-agent library export --format json
```

### 4. Analyze Papers

```bash
# Quick analysis
arxiv-agent analyze 2401.12345

# Full deep analysis
arxiv-agent analyze 2401.12345 --depth full

# Generate implementation plan (Paper-to-Code)
arxiv-agent paper2code 2401.12345
arxiv-agent paper2code 2401.12345 --output ./implementation

# View paper summary
arxiv-agent summary 2401.12345
```

### 5. Chat with Papers (RAG)

```bash
# Interactive chat with a paper
arxiv-agent chat 2401.12345

# Ask a single question
arxiv-agent chat 2401.12345 --ask "What is the main contribution?"

# Chat with your entire library
arxiv-agent chat --library

# View chat history
arxiv-agent history

# Export chat session
arxiv-agent export <session-id> --format markdown
```

### 6. Daily Digests

```bash
# Generate digest now
arxiv-agent digest run

# Configure digest topics
arxiv-agent digest config --keywords "llm,transformer,attention"

# View past digests
arxiv-agent digest list
arxiv-agent digest show 2026-01-15
```

---

## ⏰ Scheduling

ArXiv Agent includes a background scheduler for automated paper discovery:

### Digest Schedules

```bash
# Add a morning digest schedule
arxiv-agent digest schedules add morning --time "08:00" --keywords "llm,transformer"

# Add an evening ML digest  
arxiv-agent digest schedules add ml-papers --time "18:00" --categories "cs.LG,cs.AI"

# List all schedules
arxiv-agent digest schedules list

# Modify a schedule
arxiv-agent digest schedules modify morning --time "07:30"

# Remove a schedule
arxiv-agent digest schedules remove morning

# Enable/disable scheduling
arxiv-agent digest schedule enable
arxiv-agent digest schedule disable
arxiv-agent digest schedule status
```

### Background Daemon

```bash
# Start the scheduler daemon
arxiv-agent daemon start

# Check daemon status
arxiv-agent daemon status

# View scheduled jobs
arxiv-agent daemon jobs

# View execution logs
arxiv-agent daemon logs

# Pause/resume specific jobs
arxiv-agent daemon pause <job-id>
arxiv-agent daemon resume <job-id>

# Stop the daemon
arxiv-agent daemon stop
```

---

## ⚙️ Configuration

### Provider Management

```bash
# List providers and their status
arxiv-agent config provider list

# Set default provider
arxiv-agent config provider set anthropic

# Remove a provider's API key
arxiv-agent config provider remove openai
```

### Model Configuration

```bash
# Show current model config
arxiv-agent config models show

# Set model for specific tasks
arxiv-agent config models set

# Set model for a specific agent
arxiv-agent config models set code --provider openai
```

### Themes & Display

```bash
# List available themes
arxiv-agent config theme list

# Set theme
arxiv-agent config theme set monokai

# Show current theme
arxiv-agent config theme show
```

### General Settings

```bash
# Show all settings
arxiv-agent config show

# Show data paths
arxiv-agent config path

# Interactive setup wizard
arxiv-agent quickstart
```

---

## 📈 Trend Analysis

```bash
# Discover trending topics
arxiv-agent trends discover

# Filter by category
arxiv-agent trends discover --category cs.AI --days 7

# View topic details
arxiv-agent trends topic <topic-id>
```

---

## 📊 System Status

```bash
# Show system status and stats
arxiv-agent status

# Show version
arxiv-agent version
```

---

## 🛡️ Guardrails

ArXiv Agent includes built-in safety and reliability features:

| Feature | Description |
|---------|-------------|
| **Circuit Breakers** | Automatic failure detection and recovery for API calls |
| **Token Usage Tracking** | Daily usage tracked locally in `~/.local/share/arxiv-agent/` |
| **Rate Limiting** | Respects API rate limits with exponential backoff |
| **Context Window Management** | Checks request size before sending to avoid truncation |
| **JSON Repair** | Automatically fixes malformed JSON responses from LLMs |
| **Graceful Degradation** | Falls back to cached data when APIs are unavailable |

---

## 🗂️ Data Storage

All data is stored locally following XDG conventions:

| Path | Contents |
|------|----------|
| `~/.local/share/arxiv-agent/` | Database, PDFs, vector store |
| `~/.cache/arxiv-agent/` | Temporary files, API response cache |
| `~/.config/arxiv-agent/` | Configuration files |

---

## 📋 Command Reference

| Command | Description |
|---------|-------------|
| `arxiv-agent search <query>` | Search arXiv for papers |
| `arxiv-agent paper <id>` | Get paper details |
| `arxiv-agent analyze <id>` | Analyze a paper |
| `arxiv-agent paper2code <id>` | Generate implementation plan |
| `arxiv-agent chat <id>` | Chat with a paper |
| `arxiv-agent summary <id>` | Quick paper summary |
| `arxiv-agent library` | Manage your paper library |
| `arxiv-agent digest` | Daily digest management |
| `arxiv-agent trends` | Discover trending topics |
| `arxiv-agent daemon` | Background scheduler |
| `arxiv-agent config` | Settings management |
| `arxiv-agent status` | System status |
| `arxiv-agent version` | Version info |
| `arxiv-agent quickstart` | Interactive setup wizard |

Use `arxiv-agent <command> --help` for detailed options.

---

## 📚 Documentation

*   [**Architecture & Design**](docs/ARCHITECTURE.md): Deep dive into the system internals, database schema, and agent workflows.
*   [**Contribution Guide**](docs/CONTRIBUTING.md): How to set up a dev environment and contribute.

---

## 🧪 Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run linting
ruff check src/

# Run type checking
mypy src/
```

---

## 🤝 Contributing

We welcome contributions! Please check out our [Architecture Guide](docs/ARCHITECTURE.md) to understand the system before diving in.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
