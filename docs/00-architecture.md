# 00 — Architecture

> Full technical architecture of Jarvis — components, data flows, and design decisions.

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              JARVIS SYSTEM                                  │
│                                                                             │
│  Browser                                                                    │
│    │                                                                        │
│    ▼                                                                        │
│  ┌─────────────────────────────────────────────────────────────┐           │
│  │  Gradio Web UI  (ui/app.py)                                 │           │
│  │  gr.Blocks · streaming via astream_events · gr.State        │           │
│  └──────────────────────────┬──────────────────────────────────┘           │
│                             │                                               │
│                             ▼                                               │
│  ┌─────────────────────────────────────────────────────────────┐           │
│  │  Supervisor Graph  (agents/supervisor.py)                   │           │
│  │                                                             │           │
│  │  compress ──► supervisor ──► note_agent   ──┐              │           │
│  │                    │    └──► task_agent   ──┼──► supervisor │           │
│  │                    │    └──► memory_agent ──┘              │           │
│  │                    └──► finish ──► END                      │           │
│  └──────────────────────────┬──────────────────────────────────┘           │
│                             │                                               │
│           ┌─────────────────┼─────────────────┐                           │
│           ▼                 ▼                  ▼                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐            │
│  │  Note Tools  │  │  Task Tools  │  │    Memory Tools       │            │
│  │  (4 tools)   │  │  (5 tools)   │  │    (2 tools)          │            │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘            │
│         │                 │                       │                        │
│         ▼                 ▼                       ▼                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐            │
│  │  SQLite DB   │  │  SQLite DB   │  │  Chroma Vector DB    │            │
│  │  (notes)     │  │  (tasks)     │  │  (long-term memory)  │            │
│  └──────────────┘  └──────────────┘  └──────────────────────┘            │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────┐         │
│  │  Ollama (local)                                              │         │
│  │  llama3.2:3b (chat)  ·  nomic-embed-text (embeddings)        │         │
│  └──────────────────────────────────────────────────────────────┘         │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Layer Breakdown

### Layer 1 — Storage (`db/`, `data/`)

Two storage systems handle all persistence:

| Storage | Technology | Contents |
|---------|------------|----------|
| `data/jarvis.db` | SQLite (via `sqlite3`) | Notes table, tasks table, LangGraph checkpoints |
| `data/chroma_db/` | Chroma vector DB | Long-term memories, episodic summaries |

SQLite serves double duty: it stores application data (notes/tasks) AND LangGraph's `AsyncSqliteSaver` checkpoints (short-term conversation memory). The async checkpointer is opened lazily inside the running event loop so both the CLI and Gradio streaming paths work with current LangGraph versions.

### Layer 2 — Tools (`tools/`)

LangChain `@tool`-decorated functions that wrap the storage layer. These are the "actions" the agent can take:

```
note_tools.py → 4 tools: create_note, get_note, list_notes, search_notes_by_tag
task_tools.py → 5 tools: create_task, list_tasks, complete_task, get_task, delete_task
```

Tools are the boundary between the LLM world and the Python world. The LLM requests tool calls; Python executes them.

### Layer 3 — Memory (`memory/`)

```
long_term.py  → Chroma + OllamaEmbeddings → semantic search across stored facts
episodic.py   → LLM summarization → compress long conversations → store in Chroma
```

Memory sits alongside tools — the `memory_agent` calls memory tools just like the note/task agents call their respective tools.

### Layer 4 — Agents (`agents/`)

```
assistant.py  → Single ReAct agent (3-node graph) — used for simple interactions
supervisor.py → Multi-agent supervisor (7-node graph) — the production architecture
```

The supervisor graph is what the UI uses. The single agent in `assistant.py` demonstrates the foundational pattern that the supervisor builds on.

### Layer 5 — UI (`ui/`)

```
app.py → Gradio Blocks → streaming chat + sidebar panels
```

The UI is a thin layer — it initialises the supervisor lazily, calls `supervisor_graph.astream_events()`, and pipes tokens to the browser. All logic lives in the agent layer. The sidebar uses escaped HTML cards for readable pending-task and recent-note lists, and the assistant avatar is served from `ui/assets/jarvis-avatar.svg`.

---

## Data Flow: Single Message

```
1. User types "Create a task to write unit tests"

2. Gradio calls stream_response(message, history, thread_id)

3. supervisor_graph.astream_events() begins:

   a. compress_node: len(messages) <= 20 → no-op, pass through

   b. supervisor_node:
      LLM reads: "Create a task..."
      Decides: → "task_agent"
      Returns: {"next": "task_agent"}

   c. run_task_agent:
      task_agent (internal ReAct):
        call_model: LLM decides to call create_task
        tool_node:  create_task("Write unit tests", ...) → {"id": "abc", ...}
        call_model: LLM sees result, forms response
      Returns: messages with tool calls + result appended

   d. compress_node: still under threshold → no-op

   e. supervisor_node:
      LLM reads updated messages
      Decides: no more work needed → "FINISH"

   f. finish_node:
      LLM synthesises: "Done! Task 'Write unit tests' created as pending."

4. Gradio streams tokens to browser as they generate

5. sidebar panels refresh: new task appears in 📋 Pending Tasks
```

---

## Memory Architecture

```
                    ┌─────────────────────────────────┐
                    │         Three-Tier Memory        │
                    └─────────────────────────────────┘
                               │
           ┌───────────────────┼────────────────────┐
           ▼                   ▼                    ▼
   ┌───────────────┐  ┌────────────────┐  ┌─────────────────┐
   │  Short-term   │  │  Long-term     │  │  Episodic       │
   │               │  │                │  │                 │
   │ LangGraph     │  │ Chroma DB      │  │ Chroma DB       │
   │ state msgs    │  │ user facts     │  │ session         │
   │               │  │ (semantic      │  │ summaries       │
   │ Single        │  │  search)       │  │ (auto-stored    │
   │ session only  │  │                │  │  on compress)   │
   └───────────────┘  └────────────────┘  └─────────────────┘
   Filled by:         Filled by:          Filled by:
   every message      remember_fact tool  episodic.py
                      (user-explicit)     (automatic)
```

All three are queryable by the memory_agent via `recall_memories` — Chroma doesn't distinguish between long-term and episodic at query time, both return as relevant matches.

---

## LangGraph Graph Structure

### Single Agent (`assistant.py`)

```
START → call_model → tools → call_model → ... → END
                 ↑___________________|
                 (tool result feeds back in)
```

**Nodes:** `call_model`, `tools`
**Edges:** START→call_model, conditional (tools_condition), tools→call_model

### Supervisor (`supervisor.py`)

```
START → compress → supervisor → note_agent   → compress → supervisor → ...
                       │    └→ task_agent   →     (loop)
                       │    └→ memory_agent →
                       └→ finish → END
```

**Nodes:** `compress`, `supervisor`, `note_agent`, `task_agent`, `memory_agent`, `finish`
**Edges:** 3 unconditional, 1 conditional (route_to_worker), 3 back-edges to compress

---

## Technology Choices

| Decision | Choice | Why |
|----------|--------|-----|
| Agent framework | LangGraph | Stateful graphs, explicit control flow, 2025 standard |
| Local LLM runtime | Ollama | Zero API cost, privacy, CPU-friendly models available |
| Chat model | llama3.2:3b | Reliable tool-calling on CPU, ~3 GB RAM |
| Embedding model | nomic-embed-text | Purpose-built for embeddings, fast, high quality |
| Vector DB | Chroma | Local, file-based, zero config, Python-native |
| Relational DB | SQLite | Built into Python, zero setup, perfect for local tools |
| UI | Gradio | Python-native, streaming support, quick to build |
| Package manager | uv | Fast, modern, handles venvs automatically |

---

## File → Component Map

```
jarvis/
├── agents/
│   ├── assistant.py     ← Single ReAct agent (learning reference)
│   └── supervisor.py    ← Multi-agent supervisor (production)
├── db/
│   └── database.py      ← SQLite CRUD layer (notes + tasks)
├── memory/
│   ├── long_term.py     ← Chroma vector store + embedding tools
│   └── episodic.py      ← Conversation compression + summarization
├── tools/
│   ├── note_tools.py    ← @tool wrappers for note operations
│   └── task_tools.py    ← @tool wrappers for task operations
├── ui/
│   └── app.py           ← Gradio Blocks UI + streaming
├── tests/               ← pytest test suite (no LLM required)
├── config.py            ← Settings singleton (env vars + defaults)
└── health_check.py      ← Pre-flight checks for Ollama + models
```
