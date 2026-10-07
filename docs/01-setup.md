# 01 — Setup Guide

> **Who this is for:** A senior JavaScript/TypeScript developer setting up Jarvis for the first time.
> Every Python concept is explained alongside its JS equivalent.

---

## Prerequisites

| Tool | Why you need it | JS equivalent |
|------|----------------|---------------|
| **Python 3.11+** | The runtime | Node.js |
| **uv** | Package manager + virtual env manager | npm / nvm combined |
| **Ollama** | Runs AI models locally (no API key needed) | A local Docker container serving a model API |
| **Git** | Version control | Git (same!) |

---

## Step 1 — Install Python 3.11+

Check your version:
```bash
python --version   # needs to be 3.11 or higher
```

If you need to install Python, download it from [python.org](https://www.python.org/downloads/).

On Windows you can also use `winget`:
```powershell
winget install Python.Python.3.11
```

---

## Step 2 — Install `uv`

`uv` is the modern Python package manager. Think of it as **npm + nvm combined** — it manages both packages and Python versions, and it's dramatically faster than `pip`.

```bash
# Install uv (works on Mac, Linux, Windows)
pip install uv

# Or on Windows with PowerShell:
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Verify:
```bash
uv --version
```

### uv vs npm — the mental model

| npm (JS) | uv (Python) | What it does |
|----------|-------------|-------------|
| `package.json` | `pyproject.toml` | Project manifest + dependency list |
| `package-lock.json` | `uv.lock` | Exact pinned versions |
| `node_modules/` | `.venv/` | Installed packages (local to project) |
| `npm install` | `uv pip install -e .` | Install all dependencies |
| `npm run dev` | `python ui/app.py` | Run a script |
| `.nvmrc` | `.python-version` | Pin a Python version |

---

## Step 3 — Install Ollama

Ollama runs AI models locally on your machine. No API key, no cloud, no cost.

1. Download from [ollama.com/download](https://ollama.com/download)
2. Install and run it — Ollama starts a local HTTP server at `http://localhost:11434`
3. Pull the two models Jarvis needs:

```bash
# The main chat model (~2 GB download)
# Used for reasoning, tool-calling, and generating responses
ollama pull llama3.2:3b

# The embedding model (~270 MB download)
# Used for long-term memory — converts text into vectors for semantic search
ollama pull nomic-embed-text
```

> **Why two models?** The chat model is like a brain — it thinks and talks.
> The embedding model is like a filing clerk — it converts text into numbers
> so we can find semantically similar content later. They do very different jobs.

Verify Ollama is running:
```bash
curl http://localhost:11434/api/tags
# Should return a JSON list of your pulled models
```

---

## Step 4 — Clone and Install Jarvis

```bash
git clone <your-repo-url>
cd jarvis

# Install all Python dependencies into a virtual environment
make install

# This runs: uv pip install -e ".[dev]"
# -e means "editable install" — like npm link, the source code IS the package
# .[dev] also installs dev dependencies (pytest etc.)
```

### What is a virtual environment?

In JS, when you run `npm install`, packages go into `./node_modules/` — local to the project.
Python's equivalent is a **virtual environment** (`.venv/` folder). Without it, packages
install globally and projects interfere with each other.

`uv` handles this automatically. You don't need to manually activate a venv when using `uv`.

### What is `pyproject.toml`?

`pyproject.toml` is Python's `package.json`. It defines:
- Project name, version, description
- Required dependencies (like `dependencies` in `package.json`)
- Dev dependencies (like `devDependencies`)
- Build system configuration

```toml
# pyproject.toml
[project]
name = "jarvis"
version = "0.1.0"
dependencies = [
    "langgraph>=0.2,<0.3",   # like "langgraph": "^0.2.0" in package.json
    "gradio>=5.0,<6.0",
    "aiosqlite>=0.20",
]
```

---

## Step 5 — Configure (optional)

Copy the example env file and edit if needed:
```bash
cp .env.example .env
```

The defaults work out of the box for local use. Only edit `.env` if you want to:
- Use a different model (e.g. `mistral:7b` for better quality on a GPU)
- Change where data is stored
- Connect to a remote Ollama instance

### How `config.py` works

```python
# config.py — reads from environment variables with defaults
import os
from dotenv import load_dotenv

load_dotenv()  # loads .env file if it exists

MODEL_NAME = os.getenv("MODEL_NAME", "llama3.2:3b")
#                       ^                ^
#                 env var name      default value
```

This is identical in concept to:
```javascript
// In Node.js
require('dotenv').config();
const MODEL_NAME = process.env.MODEL_NAME ?? 'llama3.2:3b';
```

---

## Step 6 — Run the Health Check

```bash
make health
```

Expected output:
```
🤖 Jarvis Health Check
────────────────────────────────────────
✅ Python 3.11 — OK (3.11+ required)
✅ All Python dependencies are installed
✅ Ollama is running at http://localhost:11434
✅ Model llama3.2:3b responded: "Hello! How can I help you today?"
✅ Embedding model nomic-embed-text is available

────────────────────────────────────────
🎉 Jarvis is ready to go!
   Run: make run   to start the web UI
   Run: make test  to run the test suite
```

If anything fails, the health check will tell you exactly what to fix.

---

## Step 7 — Run Jarvis

```bash
make run
```

This starts the Gradio web UI. Open your browser at **http://localhost:7860**.

To create safe demo records for checking the sidebar, run:

```bash
python tools/generate_mock_data.py
```

This writes to `data/mock_jarvis.db` and leaves the live `data/jarvis.db`
untouched by default.

---

## What just happened?

When you ran `make install`, Python:
1. Created a `.venv/` directory (your `node_modules` equivalent)
2. Installed all packages from `pyproject.toml` into that venv
3. Made the `jarvis` package itself importable in editable mode (`-e`)

When you ran `make health`:
1. `health_check.py` imported `config.py` which loaded your `.env`
2. It pinged `http://localhost:11434/api/tags` to confirm Ollama is running
3. It instantiated `ChatOllama` from `langchain-ollama` and sent a test message
4. It verified the embedding model is pulled

When you run `make run`:
1. `ui/app.py` starts a Gradio server on port 7860
2. Gradio serves a web UI backed by Python functions
3. Your browser connects to it via HTTP — just like any other web app

---

## Troubleshooting

**`make: command not found` on Windows**
- Install make via: `winget install GnuWin32.Make`
- Or run commands directly: `python health_check.py`

**`ollama: command not found`**
- Ollama isn't installed or not in PATH
- Download from [ollama.com/download](https://ollama.com/download) and restart your terminal

**`❌ Cannot reach Ollama at http://localhost:11434`**
- Ollama server isn't running. Run `ollama serve` in a separate terminal
- On Windows, Ollama usually runs in the system tray automatically after install

**`ModuleNotFoundError: No module named 'langchain'`**
- Dependencies aren't installed. Run `make install`

**Model responds slowly**
- Normal on CPU! `llama3.2:3b` runs at ~3-5 tokens/second on a typical CPU.
- For faster responses, use a GPU machine or try `phi3:mini` (smaller)
