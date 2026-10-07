# 04 — The ReAct Agent

> How the LLM and tools are wired together into a thinking, acting agent using LangGraph.

---

## What is a ReAct Agent?

**ReAct = Reason + Act**

A ReAct agent alternates between two steps in a loop:

```
User Input
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  REASON: LLM reads messages and decides what to do  │
└─────────────────────────────────────────────────────┘
    │
    ├──► "I have enough info to answer" ──► Final response to user
    │
    └──► "I need to call a tool" ──► call tool ──► result ──► back to REASON
```

Without ReAct, the LLM can only answer from its training data. With ReAct, it can take actions (call tools), observe results, and use those results in its next reasoning step — just like a human would.

**Example flow:**

```
You: "Create a task to review the README by Friday"

1. REASON: The user wants to create a task. I should call create_task.
2. ACT:    create_task(title="Review the README", due_date="2026-10-09")
3. OBSERVE: {"id": "abc-123", "title": "Review the README", "status": "pending", ...}
4. REASON: The task was created successfully. I can now tell the user.
5. RESPOND: "Done! I've created the task 'Review the README' due Friday."
```

---

## The LangGraph Graph

LangGraph models the agent as a **stateful directed graph**. Every concept maps cleanly:

| LangGraph concept | What it is | JS equivalent |
|------------------|-----------|---------------|
| `StateGraph` | The graph definition | A Redux store blueprint |
| `AgentState` | The shape of the state | A TypeScript interface |
| Node | A function that reads state and returns updates | A Redux action handler |
| Edge | A transition from one node to another | A state machine transition |
| Conditional edge | A transition decided at runtime | A switch statement on state |
| `compile()` | Build the runnable graph | `createStore()` in Redux |

### The 3-Node Graph

```
START ──► call_model ──────────────► END
               │   ▲
               │   │
               ▼   │
             tools ─┘
```

1. **`call_model`** — passes messages to the LLM. The LLM either responds (→ END) or requests a tool call (→ `tools`)
2. **`tools`** — executes the requested tool and appends the result as a `ToolMessage`
3. Back to **`call_model`** — LLM sees the tool result and continues

---

## AgentState — The Redux Reducer Pattern

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
```

The `Annotated[list[BaseMessage], add_messages]` declaration is saying:
> "The `messages` field is a list of messages. When a node returns new messages,
> **append** them to the existing list rather than replacing it."

`add_messages` is the **reducer** — it manages how state updates are merged.

Compare to Redux:
```javascript
// Redux equivalent
function messagesReducer(state = [], action) {
  switch (action.type) {
    case 'ADD_MESSAGE':
      return [...state, action.payload];  // append, not replace
    default:
      return state;
  }
}
```

The message list grows as the conversation progresses:
```
[HumanMessage("Create a task")]
  → + AIMessage(tool_calls=[create_task(...)])
    → + ToolMessage(result={"id": "..."})
      → + AIMessage("Done! Task created.")
```

---

## Message Types

LangChain has four message types that flow through the graph:

| Type | When it appears | Example |
|------|----------------|---------|
| `HumanMessage` | User input | "Create a task to review docs" |
| `AIMessage` | LLM response | "I'll create that task for you." |
| `AIMessage` (with tool_calls) | LLM requesting a tool | `tool_calls=[{"name": "create_task", "args": {...}}]` |
| `ToolMessage` | Result of a tool execution | `{"id": "abc", "status": "pending", ...}` |
| `SystemMessage` | Agent instructions | "You are Jarvis, a personal assistant..." |

---

## Conversation Persistence with AsyncSqliteSaver

```python
import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

connection = aiosqlite.connect("./data/jarvis.db")
await connection.__aenter__()
checkpointer = AsyncSqliteSaver(connection)
await checkpointer.setup()
graph = builder.compile(checkpointer=checkpointer)
```

Jarvis creates this connection lazily through `ensure_checkpoint_ready()` in
`agents/assistant.py` and `agents/supervisor.py`. This keeps graph imports safe
before an event loop exists and enables the async `ainvoke` and streaming APIs.

The checkpointer **saves and restores the full state** (all messages) after every step. When you invoke the graph with a `thread_id`, it:

1. Looks up that thread in the SQLite DB
2. Loads the existing messages (if any)
3. Appends the new message and runs the graph
4. Saves the updated state back to SQLite

```python
# Every invocation needs a config with thread_id
config = {"configurable": {"thread_id": "my-session"}}

# First message — starts a new conversation
result = await graph.ainvoke(
    {"messages": [{"role": "user", "content": "Hello"}]},
    config=config
)

# Second message — Jarvis remembers the first exchange
result = await graph.ainvoke(
    {"messages": [{"role": "user", "content": "What did I just say?"}]},
    config=config   # same thread_id → same conversation history
)
```

This is why the CLI flag `--thread-id` matters:
- Same ID = continue existing conversation
- New ID = start fresh

---

## bind_tools — What Actually Happens

```python
llm_with_tools = llm.bind_tools(NOTE_TOOLS + TASK_TOOLS)
```

`bind_tools` injects tool schemas into the LLM's context. Under the hood, the LLM receives something like:

```
You have access to these tools:

create_note(title: str, content: str, tags: str = "")
  → "Create and save a new note. Use this when the user wants to save information..."

create_task(title: str, description: str = "", due_date: str = "")
  → "Create a new task. Use this when the user wants to add to their to-do list..."

[... 7 more tools ...]

When you want to use a tool, respond with a tool_call instead of plain text.
```

The LLM then decides based on this context which tool (if any) to call.

---

## tools_condition — The Routing Logic

```python
builder.add_conditional_edges("call_model", tools_condition)
```

`tools_condition` is a pre-built function from `langgraph.prebuilt` that:
1. Reads the last message from state
2. If it has `tool_calls` → returns `"tools"` (route to the tools node)
3. If it doesn't → returns `END` (route to end, return response to user)

You could write this yourself:
```python
def route(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END
```

`tools_condition` is just the pre-packaged version.

---

## Running the CLI

```bash
python agents/assistant.py

# With a custom session name (to keep conversations separate)
python agents/assistant.py work-session
```

Sample interaction:
```
🤖 Jarvis CLI
   Model: llama3.2:3b
   Session: cli-session

──────────────────────────────────────────────────

You: Create a note called Project Ideas with content: Build a Jarvis AI assistant

Jarvis: Done! I've created a note called "Project Ideas" with your content. 
        Note ID: f47ac10b-58cc-4372-a567-0e02b2c3d479

You: Now create a task to review that note by tomorrow

Jarvis: Created! Task "Review Project Ideas note" is set as pending. 
        Task ID: 3e23e816-0c26-4815-8e44-8b02f7a8ed4e

You: What was the first note I created this session?

Jarvis: The first note you created this session was "Project Ideas" — 
        about building a Jarvis AI assistant.
```

Note how the last question is answered from conversation history — no tool call needed, the LLM just reads back from the `messages` in state.

---

## Exporting the Graph

`agents/assistant.py` exports a module-level `graph` instance:

```python
# agents/assistant.py
graph = build_graph(checkpointer=_checkpointer)
```

Other modules import this directly:

```python
# ui/app.py
from agents.assistant import graph

result = await graph.ainvoke(...)
```

This is the same pattern you'd use in Node.js — create the service once, export it, import it wherever needed.
