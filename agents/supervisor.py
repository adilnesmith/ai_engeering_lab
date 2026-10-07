"""
agents/supervisor.py — Multi-agent supervisor architecture using LangGraph.

This replaces the single monolithic agent from assistant.py with a system
where a supervisor LLM routes tasks to specialist sub-agents, each with
a focused toolset.

Architecture:
                        ┌─────────────────────────────────────────┐
                        │                                         │
    START ──► [supervisor] ──► [note_agent]   ──► [supervisor] ──┘
                    │    │──► [task_agent]   ──►       │
                    │    └──► [memory_agent] ──►       │
                    │                                  │
                 FINISH ◄─────────────────────────────┘

How it works:
  1. Supervisor reads the user's message and decides which specialist to call
  2. The chosen specialist runs its own internal ReAct loop (tool calls etc.)
  3. Specialist returns its result to the supervisor
  4. Supervisor decides: call another specialist, or FINISH and respond

Why this is better than one big agent:
  - Each specialist has only relevant tools → less confusion, better accuracy
  - Separation of concerns → easier to debug, extend, and replace specialists
  - Demonstrates a real production pattern used in agentic AI systems

For JS developers: think of the supervisor as an API Gateway that routes
requests to microservices. Each microservice (specialist) handles one domain
and returns results to the gateway.

Usage:
    from agents.supervisor import supervisor_graph

    result = await supervisor_graph.ainvoke(
        {"messages": [{"role": "user", "content": "Create a note and a task"}]},
        config={"configurable": {"thread_id": "session-1"}}
    )
"""

import asyncio
import sys
from typing import Annotated, Literal

import aiosqlite
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
try:
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
except ImportError:
    from langgraph_checkpoint_sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from typing_extensions import TypedDict

from config import settings
from memory.episodic import maybe_compress_messages, needs_compression
from memory.long_term import MEMORY_TOOLS, format_memories_for_context
from tools.note_tools import NOTE_TOOLS
from tools.task_tools import TASK_TOOLS

# ── LLM ──────────────────────────────────────────────────────────────────────
llm = ChatOllama(
    model=settings.MODEL_NAME,
    base_url=settings.OLLAMA_BASE_URL,
    temperature=0,
)

# ── Available workers ─────────────────────────────────────────────────────────
# This list is the supervisor's menu — it uses it in the routing prompt.
WORKERS = ["note_agent", "task_agent", "memory_agent"]

# ── Supervisor state ──────────────────────────────────────────────────────────
# Extends the basic message state with a `next` field.
# `next` tells the graph which node to route to after the supervisor runs.
#
# Compare to assistant.py's AgentState:
#   AgentState has only `messages`
#   SupervisorState adds `next` to carry the routing decision between nodes

class SupervisorState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    next: str   # one of WORKERS or "FINISH"


# ── Specialist system prompts ─────────────────────────────────────────────────
# Each specialist gets a focused system prompt that scopes its behaviour.
# This is how the supervisor "programs" each sub-agent.

NOTE_AGENT_PROMPT = """You are a note-taking specialist called by the Jarvis supervisor.

Your ONLY job is to handle note-related requests using the available note tools.
Be efficient: call the appropriate tool immediately without over-explaining.

After completing the task, summarise what you did in 1-2 sentences.
Do not ask follow-up questions — complete the task and return.
"""

TASK_AGENT_PROMPT = """You are a task management specialist called by the Jarvis supervisor.

Your ONLY job is to handle task-related requests using the available task tools.
Be efficient: call the appropriate tool immediately without over-explaining.

After completing the task, summarise what you did in 1-2 sentences.
Do not ask follow-up questions — complete the task and return.
"""

MEMORY_AGENT_PROMPT = """You are a memory specialist called by the Jarvis supervisor.

Your ONLY job is to store and retrieve information about the user using memory tools.
- Use remember_fact to store new information the user shares about themselves
- Use recall_memories to retrieve relevant past memories

Be efficient: call the appropriate tool immediately.
After completing the task, summarise what you found or stored in 1-2 sentences.
"""

# ── Specialist agents ─────────────────────────────────────────────────────────
# create_react_agent is a LangGraph convenience function that builds the same
# 3-node ReAct graph from assistant.py (call_model → tools → call_model → END)
# but pre-packaged in one line.
#
# state_modifier injects the specialist's system prompt before every LLM call.

note_agent = create_react_agent(
    llm,
    tools=NOTE_TOOLS,
    state_modifier=NOTE_AGENT_PROMPT,
)

task_agent = create_react_agent(
    llm,
    tools=TASK_TOOLS,
    state_modifier=TASK_AGENT_PROMPT,
)

memory_agent = create_react_agent(
    llm,
    tools=MEMORY_TOOLS,
    state_modifier=MEMORY_AGENT_PROMPT,
)


# ── Supervisor node ───────────────────────────────────────────────────────────

SUPERVISOR_SYSTEM_PROMPT = """You are Jarvis, a personal AI assistant supervisor.

Your job is to read the conversation and decide which specialist should handle
the next step, or whether you have enough information to give a final response.

Available specialists:
- note_agent    → handles everything related to notes (create, list, search, retrieve)
- task_agent    → handles everything related to tasks (create, list, complete, delete)
- memory_agent  → handles storing and recalling personal facts about the user

Rules:
1. If the request involves notes → route to note_agent
2. If the request involves tasks → route to task_agent
3. If the user shares personal info OR asks what you remember → route to memory_agent
4. If multiple specialists are needed → route to them one at a time
5. If all required work is done → respond with FINISH

Respond with ONLY one of: note_agent, task_agent, memory_agent, FINISH
No explanation. Just the routing decision.
"""

def supervisor_node(state: SupervisorState) -> dict:
    """
    Supervisor node: decides which specialist to call next, or FINISH.

    Reads all messages in the conversation, applies the routing prompt,
    and returns {"next": "<worker_name>"} or {"next": "FINISH"}.

    Args:
        state: Current graph state with messages and next field.

    Returns:
        Dict with "next" key containing the routing decision.
    """
    # Pull relevant memories to help the supervisor understand context
    last_human_message = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            last_human_message = msg.content
            break

    memory_context = ""
    if last_human_message:
        memory_context = format_memories_for_context(last_human_message, k=3)

    # Build the supervisor prompt with optional memory context
    system_content = SUPERVISOR_SYSTEM_PROMPT
    if memory_context:
        system_content += f"\n\nContext from memory:\n{memory_context}"

    messages = [SystemMessage(content=system_content)] + state["messages"]

    response = llm.invoke(messages)
    decision = response.content.strip()

    # Sanitise — only accept known workers or FINISH
    if decision not in WORKERS + ["FINISH"]:
        # If the LLM returns something unexpected, try to extract a known word
        for worker in WORKERS:
            if worker in decision:
                decision = worker
                break
        else:
            decision = "FINISH"

    return {"next": decision}


# ── Specialist wrapper nodes ──────────────────────────────────────────────────
# Each wrapper runs the specialist agent and routes back to the supervisor.
# The specialist's response is automatically appended to messages via
# the add_messages reducer, so the supervisor sees what each specialist did.

def run_note_agent(state: SupervisorState) -> dict:
    """Run the note specialist and return its messages."""
    print(f"\033[94m[Supervisor]\033[0m → routing to: \033[92mnote_agent\033[0m")
    result = note_agent.invoke({"messages": state["messages"]})
    return {"messages": result["messages"]}


def run_task_agent(state: SupervisorState) -> dict:
    """Run the task specialist and return its messages."""
    print(f"\033[94m[Supervisor]\033[0m → routing to: \033[92mtask_agent\033[0m")
    result = task_agent.invoke({"messages": state["messages"]})
    return {"messages": result["messages"]}


def run_memory_agent(state: SupervisorState) -> dict:
    """Run the memory specialist and return its messages."""
    print(f"\033[94m[Supervisor]\033[0m → routing to: \033[92mmemory_agent\033[0m")
    result = memory_agent.invoke({"messages": state["messages"]})
    return {"messages": result["messages"]}


async def compress_node(state: SupervisorState) -> dict:
    """
    Episodic memory node: compress messages when the conversation gets too long.

    Fires before the supervisor runs when message count exceeds the threshold.
    Replaces the full message list with [summary SystemMessage] + [last 4 messages].
    The summary is also stored in Chroma for cross-session retrieval.

    Args:
        state: Current graph state.

    Returns:
        Dict with compressed "messages" list, or unchanged state if under threshold.
    """
    if needs_compression(state["messages"]):
        compressed = await maybe_compress_messages(state["messages"])
        print(
            f"\033[93m[Episodic]\033[0m Compressed "
            f"{len(state['messages'])} → {len(compressed)} messages"
        )
        # Return the full compressed list as a replacement (not append)
        # We need to bypass add_messages reducer here — return special format
        return {"messages": compressed}
    return state


def finish_node(state: SupervisorState) -> dict:
    """
    Final node: supervisor composes a clean response to the user.

    The specialist agents append technical messages to the state.
    This node runs one final LLM call to synthesise a clean, friendly
    response from all the work that was done.
    """
    print(f"\033[94m[Supervisor]\033[0m → \033[93mFINISH\033[0m — composing final response")

    synthesis_prompt = """You are Jarvis, a personal AI assistant.

Based on the conversation history and the work done by the specialist agents,
provide a clear, friendly, concise summary response to the user.

Focus on:
- What was accomplished (tasks created, notes saved, memories retrieved)
- Any relevant results the user should know
- Be conversational and natural, not technical
"""
    messages = [SystemMessage(content=synthesis_prompt)] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


# ── Routing function ──────────────────────────────────────────────────────────

def route_to_worker(
    state: SupervisorState,
) -> Literal["note_agent", "task_agent", "memory_agent", "finish"]:
    """
    Read state["next"] and return the name of the next node to visit.

    This is used as the conditional edge function from the supervisor node.
    LangGraph calls this after each supervisor_node execution to decide
    where to go next.
    """
    next_step = state.get("next", "FINISH")
    if next_step == "FINISH":
        return "finish"
    return next_step  # type: ignore[return-value]


# ── Graph construction ────────────────────────────────────────────────────────

def build_supervisor_graph(checkpointer=None):
    """
    Build and compile the supervisor multi-agent graph.

    Args:
        checkpointer: Optional LangGraph checkpointer for persistence.

    Returns:
        Compiled StateGraph.
    """
    builder = StateGraph(SupervisorState)

    # Register all nodes
    builder.add_node("compress", compress_node)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("note_agent", run_note_agent)
    builder.add_node("task_agent", run_task_agent)
    builder.add_node("memory_agent", run_memory_agent)
    builder.add_node("finish", finish_node)

    # Entry: compress first (no-op if under threshold), then supervisor
    builder.add_edge(START, "compress")
    builder.add_edge("compress", "supervisor")

    # Conditional edge from supervisor: route based on state["next"]
    builder.add_conditional_edges(
        "supervisor",
        route_to_worker,
        {
            "note_agent": "note_agent",
            "task_agent": "task_agent",
            "memory_agent": "memory_agent",
            "finish": "finish",
        },
    )

    # After each specialist runs, compress if needed then return to supervisor
    builder.add_edge("note_agent", "compress")
    builder.add_edge("task_agent", "compress")
    builder.add_edge("memory_agent", "compress")

    # Finish node is the terminal state
    builder.add_edge("finish", END)

    return builder.compile(checkpointer=checkpointer)


# ── Module-level graph ────────────────────────────────────────────────────────
# This is what ui/app.py imports.

# `from_conn_string()` is a context manager in current langgraph versions.
# Keep the underlying connection alive for the module-level graph instead of
# deferring a missing `get_next_version` failure until the first UI request.
_checkpoint_connection = None
_checkpointer = None
_checkpoint_ready = False
_checkpoint_lock = asyncio.Lock()
supervisor_graph = None


async def ensure_checkpoint_ready() -> None:
    """Open and initialise the async checkpoint database on first use."""
    global _checkpoint_connection, _checkpointer, _checkpoint_ready, supervisor_graph
    if _checkpoint_ready:
        return

    async with _checkpoint_lock:
        if _checkpoint_ready:
            return
        _checkpoint_connection = aiosqlite.connect(settings.DB_PATH)
        # Compatibility shim: the installed aiosqlite release exposes the
        # worker thread directly, while this LangGraph saver asks for the
        # older `is_alive()` helper.
        if not hasattr(_checkpoint_connection, "is_alive"):
            _checkpoint_connection.is_alive = _checkpoint_connection._thread.is_alive
        _checkpointer = AsyncSqliteSaver(_checkpoint_connection)
        await _checkpoint_connection.__aenter__()
        await _checkpointer.setup()
        supervisor_graph = build_supervisor_graph(checkpointer=_checkpointer)
        _checkpoint_ready = True


# ── CLI runner ────────────────────────────────────────────────────────────────

async def run_cli(thread_id: str = "supervisor-session") -> None:
    """Interactive CLI for testing the supervisor graph."""
    await ensure_checkpoint_ready()
    config = {"configurable": {"thread_id": thread_id}}

    BLUE  = "\033[94m"
    GREEN = "\033[92m"
    RESET = "\033[0m"
    BOLD  = "\033[1m"

    print(f"\n{BOLD}🤖 Jarvis (Multi-Agent){RESET}")
    print(f"   Mode: Supervisor + Specialists")
    print(f"   Model: {settings.MODEL_NAME}")
    print(f"   Session: {thread_id}")
    print(f"   Type 'exit' to stop\n")
    print("─" * 50)

    while True:
        try:
            user_input = input(f"\n{BLUE}You:{RESET} ").strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n\n{GREEN}Jarvis:{RESET} Goodbye! 👋")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit", "bye"):
            print(f"\n{GREEN}Jarvis:{RESET} Goodbye! 👋")
            break

        try:
            result = await supervisor_graph.ainvoke(
                {"messages": [{"role": "user", "content": user_input}]},
                config=config,
            )
            response = result["messages"][-1].content
            print(f"\n{GREEN}Jarvis:{RESET} {response}")

        except Exception as e:
            if "connection refused" in str(e).lower() or "cannot connect" in str(e).lower():
                print(f"\n❌ Cannot connect to Ollama. Run: ollama serve")
            else:
                print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    thread = sys.argv[1] if len(sys.argv) > 1 else "supervisor-session"
    asyncio.run(run_cli(thread_id=thread))
