# 06 — Multi-Agent Architecture

> How Jarvis delegates work to specialist agents using a supervisor pattern.

---

## Why Multi-Agent?

A single agent with 9+ tools starts to break down:

| Problem | What happens |
|---------|-------------|
| Tool confusion | The LLM picks the wrong tool when too many are available |
| Context pollution | Every tool's schema adds tokens — hitting context limits faster |
| Hard to debug | When something fails, it's unclear which tool or reasoning step broke |
| Hard to extend | Adding new capabilities means one huge agent doing everything |

The supervisor pattern solves this by giving each specialist **only the tools it needs**.

---

## The Architecture

```
User message
     │
     ▼
┌─────────────────────────────────────────────────────┐
│  SUPERVISOR                                         │
│  Reads conversation → decides who should act next  │
└─────────────────────────────────────────────────────┘
     │               │               │
     ▼               ▼               ▼
┌──────────┐  ┌──────────┐  ┌──────────────┐
│note_agent│  │task_agent│  │memory_agent  │
│          │  │          │  │              │
│ 4 tools  │  │ 5 tools  │  │ 2 tools      │
└──────────┘  └──────────┘  └──────────────┘
     │               │               │
     └───────────────┴───────────────┘
                     │
                     ▼
              SUPERVISOR (again)
                     │
              ┌──────┴──────┐
              │             │
         more work?       FINISH
              │             │
           (loop)      ┌─────────┐
                       │ finish  │
                       │ node    │
                       └─────────┘
                            │
                            ▼
                     Final response
```

Each specialist runs its own internal ReAct loop — it can call tools multiple times before returning to the supervisor.

---

## The Supervisor Node

The supervisor is just an LLM with a routing prompt:

```python
SUPERVISOR_SYSTEM_PROMPT = """You are Jarvis supervisor.

Available specialists:
- note_agent    → notes (create, list, search, retrieve)
- task_agent    → tasks (create, list, complete, delete)
- memory_agent  → storing and recalling personal facts

Respond with ONLY one of: note_agent, task_agent, memory_agent, FINISH
"""

def supervisor_node(state: SupervisorState) -> dict:
    messages = [SystemMessage(SUPERVISOR_SYSTEM_PROMPT)] + state["messages"]
    response = llm.invoke(messages)
    decision = response.content.strip()   # e.g. "note_agent"
    return {"next": decision}
```

The routing decision is an LLM call — not hard-coded logic. The LLM reads the conversation and decides which specialist makes sense. This is why it can handle complex requests like "save a note AND create a task" — it routes to both specialists in sequence.

---

## SupervisorState

```python
class SupervisorState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    next: str   # "note_agent" | "task_agent" | "memory_agent" | "FINISH"
```

The `next` field carries the supervisor's routing decision between graph steps.

Compare to `AgentState` in `assistant.py`:
```python
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    # no `next` — single agent doesn't need routing
```

---

## `create_react_agent` — Pre-built Specialist Agents

```python
from langgraph.prebuilt import create_react_agent

note_agent = create_react_agent(
    llm,
    tools=NOTE_TOOLS,                   # only note tools, nothing else
    state_modifier=NOTE_AGENT_PROMPT,   # specialist system prompt
)
```

`create_react_agent` builds the same 3-node graph from `assistant.py` (call_model → tools → call_model → END) but in one line. Think of it as a factory function for ReAct agents.

The `state_modifier` injects the specialist's focused system prompt — "you are a note-taking specialist, your only job is notes" — which keeps each agent on-task.

---

## The Routing Flow — Step by Step

**User asks:** "Create a note about today's standup and mark my review task as done"

```
Step 1:  Supervisor reads message
         → LLM decides: "This needs both note and task work"
         → {"next": "note_agent"}

Step 2:  note_agent runs
         → Internal ReAct: call create_note("Today's Standup", "...")
         → Appends AIMessage + ToolMessage + AIMessage to state

Step 3:  Supervisor reads updated messages (now includes note_agent's work)
         → LLM decides: "Notes are done, still need tasks"
         → {"next": "task_agent"}

Step 4:  task_agent runs
         → Internal ReAct: call complete_task("review-task-id")
         → Appends its messages to state

Step 5:  Supervisor reads all messages again
         → LLM decides: "Both done, nothing more needed"
         → {"next": "FINISH"}

Step 6:  finish_node runs
         → LLM synthesises a clean response from all the work
         → "Done! I saved your standup note and marked the review task as completed."
```

**CLI output:**
```
You: Create a note about today's standup and mark my review task as done

[Supervisor] → routing to: note_agent
[Supervisor] → routing to: task_agent
[Supervisor] → FINISH — composing final response

Jarvis: Done! I've saved a note titled "Today's Standup" and marked your 
        review task as completed.
```

---

## Conditional Edges — The Routing Mechanism

```python
builder.add_conditional_edges(
    "supervisor",           # source node
    route_to_worker,        # function that reads state and returns next node name
    {
        "note_agent":   "note_agent",
        "task_agent":   "task_agent",
        "memory_agent": "memory_agent",
        "finish":       "finish",
    },
)
```

`route_to_worker` reads `state["next"]` and returns the node name. The dict maps return values to actual node names. LangGraph then jumps to that node.

After each specialist, edges go back to supervisor:
```python
builder.add_edge("note_agent", "supervisor")
builder.add_edge("task_agent", "supervisor")
builder.add_edge("memory_agent", "supervisor")
```

This creates the loop: supervisor → specialist → supervisor → specialist → ... → finish.

---

## The Finish Node

The specialist agents append technical messages with tool calls, tool results, and internal reasoning. The finish node runs one final LLM call to synthesise everything into a clean, user-facing response:

```python
def finish_node(state: SupervisorState) -> dict:
    synthesis_prompt = """Based on the work done by specialist agents,
    provide a clear, friendly summary to the user."""
    messages = [SystemMessage(synthesis_prompt)] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}
```

This separation keeps the UX clean — the user sees a polished response, not the internal specialist chatter.

---

## Contrast: One Agent vs Supervisor Pattern

| Scenario | Single Agent | Supervisor Pattern |
|----------|-------------|-------------------|
| "Create a note" | Works fine | Works fine |
| "Create a note AND a task" | Sometimes confuses tools | Routes cleanly to each specialist |
| "What do you remember about me?" | May ignore memory tools | Routes to memory_agent specifically |
| Adding a new capability (e.g. calendar) | Add all tools to one agent | Add new specialist, update supervisor prompt |
| Debugging a failed tool call | Hard to trace | Clear log shows which specialist failed |
| 15+ tools total | Context length / confusion issues | Each agent only sees 2-5 tools |

---

## Running the Multi-Agent CLI

```bash
python agents/supervisor.py

# With a custom session
python agents/supervisor.py my-session
```

The supervisor module exposes `supervisor_graph` for use by the Gradio UI. It is
created lazily because it uses the async SQLite checkpointer:

```python
import agents.supervisor as supervisor_module

await supervisor_module.ensure_checkpoint_ready()
result = await supervisor_module.supervisor_graph.ainvoke(...)
```
