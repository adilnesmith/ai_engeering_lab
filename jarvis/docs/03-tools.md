# 03 — LangChain Tools

> How Python functions become callable actions for the AI agent.

---

## What is a LangChain Tool?

A **tool** is a Python function the LLM can choose to call when it decides it needs to take an action.

Without tools, an LLM can only generate text. With tools, it can:
- Read and write to a database
- Search the web
- Run code
- Call external APIs

The key insight: **the LLM doesn't execute code directly**. It outputs a structured message saying "I want to call this function with these arguments." Your Python code then executes the function and feeds the result back to the LLM.

For JS developers: think of tools as **API endpoints the LLM can call**. The `@tool` decorator is like `app.post('/create-note', handler)` — it registers the function with a name and schema so it's discoverable.

---

## The `@tool` Decorator

```python
from langchain_core.tools import tool

@tool
def create_note(title: str, content: str, tags: str = "") -> dict:
    """
    Create and save a new note with a title and content.

    Use this when the user wants to save information, jot something down,
    or record anything they want to remember later.

    Args:
        title:   A short, descriptive title for the note.
        content: The full body text of the note.
        tags:    Optional comma-separated tags (e.g. "work,meetings").
    """
    # ... implementation
```

The decorator does three things:
1. Wraps the function so it has a `.invoke()` method
2. Extracts the function signature as a **JSON schema** (from type hints)
3. Uses the **docstring** as the tool description

---

## The Docstring IS the Tool Description

This is the most important concept in tool design.

When you call `llm.bind_tools([create_note])`, LangChain sends the LLM something like this:

```json
{
  "name": "create_note",
  "description": "Create and save a new note with a title and content.\n\nUse this when the user wants to save information...",
  "parameters": {
    "type": "object",
    "properties": {
      "title": { "type": "string" },
      "content": { "type": "string" },
      "tags": { "type": "string", "default": "" }
    },
    "required": ["title", "content"]
  }
}
```

The LLM reads this and decides: *"The user said 'save a note', the `create_note` tool is for saving notes, I should call it."*

**Write docstrings for the LLM, not just for humans.** Be explicit about when to use the tool and what the arguments mean.

---

## Calling Tools — Two Ways

### 1. Direct invocation (no LLM — for testing)

```python
from tools.note_tools import create_note

# Pass a dict matching the function's parameters
result = create_note.invoke({"title": "My Note", "content": "Hello world"})
# → {"id": "abc-123", "title": "My Note", "content": "Hello world", "tags": [], ...}
```

This is how tests work — call the tool directly and assert on the result without needing an LLM running.

### 2. Via the LLM (in the agent)

```python
from langchain_ollama import ChatOllama
from tools.note_tools import NOTE_TOOLS

llm = ChatOllama(model="llama3.2:3b")
llm_with_tools = llm.bind_tools(NOTE_TOOLS)

# LLM generates a tool call message
response = llm_with_tools.invoke("Create a note called Meeting Notes")
# response.tool_calls → [{"name": "create_note", "args": {"title": "Meeting Notes", ...}}]
```

LangGraph's `ToolNode` handles the second part — it reads the `tool_calls` from the LLM response, executes the matching function, and returns the result as a `ToolMessage`.

---

## Tool Groups

Tools are exported as lists, grouped by domain:

```python
# tools/note_tools.py
NOTE_TOOLS = [create_note, get_note, list_notes, search_notes_by_tag]

# tools/task_tools.py
TASK_TOOLS = [create_task, list_tasks, complete_task, get_task, delete_task]
```

This grouping matters for the multi-agent architecture (Task 7). Each specialist agent gets only the tools it needs:

```python
# Note specialist only sees note tools
note_agent = create_react_agent(llm, NOTE_TOOLS)

# Task specialist only sees task tools
task_agent = create_react_agent(llm, TASK_TOOLS)
```

Giving an agent fewer tools reduces confusion and improves reliability.

---

## Module-level DB Singleton

```python
# At the top of note_tools.py
db = Database(settings.DB_PATH)   # created once when the module is imported
```

All tool functions share this single `db` instance — like a module-level client in a Node.js service:

```javascript
// JS equivalent
const db = new Database(config.DB_PATH);  // created at module load time
export function createNote(...) { return db.create(...)  }
```

In tests, this singleton is replaced with an in-memory instance using `patch.object` — see below.

---

## Testing Tools

```python
# tests/test_note_tools.py

@pytest.fixture(autouse=True)
def use_in_memory_db():
    mem_db = Database(":memory:")
    with patch.object(note_tools_module, "db", mem_db):  # swap the singleton
        yield mem_db
    mem_db.close()

def test_create_note_returns_dict_with_id():
    result = create_note.invoke({"title": "Test Note", "content": "Hello"})
    assert "id" in result
```

The `patch.object(note_tools_module, "db", mem_db)` line says:
> "For the duration of this test, replace `note_tools.db` with `mem_db`"

This is equivalent to Jest module mocking:
```javascript
jest.mock('../db', () => ({ db: new InMemoryDatabase() }));
```

---

## Running the Tests

```bash
# Test note tools only
pytest tests/test_note_tools.py -v

# Test task tools only
pytest tests/test_task_tools.py -v

# Test both
pytest tests/test_note_tools.py tests/test_task_tools.py -v
```

Expected output:
```
tests/test_note_tools.py::TestCreateNote::test_create_note_returns_dict_with_id PASSED
tests/test_note_tools.py::TestCreateNote::test_create_note_with_comma_separated_tags PASSED
...
tests/test_task_tools.py::test_full_task_lifecycle PASSED

21 passed in 0.34s
```

---

## Tools Reference

### Note Tools

| Tool | When the LLM uses it | Key args |
|------|---------------------|----------|
| `create_note` | User wants to save something | `title`, `content`, `tags` |
| `get_note` | Need to read a specific note | `note_id` |
| `list_notes` | User asks to see their notes | `limit` |
| `search_notes_by_tag` | User wants notes about a topic | `tag` |

### Task Tools

| Tool | When the LLM uses it | Key args |
|------|---------------------|----------|
| `create_task` | User has something to do | `title`, `description`, `due_date` |
| `list_tasks` | User asks what's pending/done | `status` |
| `complete_task` | User finished something | `task_id` |
| `get_task` | Need to check a specific task | `task_id` |
| `delete_task` | User explicitly wants to remove | `task_id` |
