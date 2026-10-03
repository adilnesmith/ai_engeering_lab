"""
agents/assistant.py — Single ReAct agent built with LangGraph.

This is the core of Jarvis: a stateful AI agent that can reason about
what to do, call tools, observe results, and respond — all in a loop.

Architecture (3-node graph):
                    ┌─────────────────────────────┐
                    │                             │
    START ──► [call_model] ──► [tool_node] ──────┘
                    │
                  (no tool call)
                    │
                   END

The loop:
  1. call_model  — LLM reads messages + decides: respond OR call a tool
  2. tool_node   — executes the tool, appends result as a ToolMessage
  3. back to (1) — LLM sees the tool result, decides next action

This is the ReAct pattern (Reason + Act). The LLM alternates between
"thinking" (generating reasoning) and "acting" (calling tools) until
it has enough information to give a final response.

Conversation persistence:
  SqliteSaver stores all messages to a SQLite DB keyed by thread_id.
  Starting a new session with the same thread_id restores the full
  conversation history — the agent remembers previous turns.

For JS developers: think of this as a state machine where:
  - AgentState  = Redux store (typed dict of messages)
  - add_messages = Redux reducer (appends new messages to the list)
  - Nodes        = reducer action handlers
  - Edges        = transitions between states
  - thread_id    = session ID (like a cookie/session token)

Usage:
    # Run the CLI:
    python agents/assistant.py

    # Import the graph for use in other modules:
    from agents.assistant import graph
"""

import asyncio
import sys
from typing import Annotated

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict

from config import settings
from tools.note_tools import NOTE_TOOLS
from tools.task_tools import TASK_TOOLS

# ── System prompt ─────────────────────────────────────────────────────────────
# This is the LLM's "personality" and capability description.
# It's prepended to every conversation.

SYSTEM_PROMPT = """You are Jarvis, a helpful personal AI assistant.

You help the user manage their notes and tasks. You have access to tools
to create, list, search, complete, and delete notes and tasks.

Guidelines:
- Be concise and direct in your responses
- When creating notes or tasks, confirm what you created
- When listing items, format them clearly and readably
- If the user's request is ambiguous, ask for clarification before acting
- Always use tools when the user asks to save, create, list, or manage notes/tasks
- Do not make up task or note IDs — always get them from tool results

Current capabilities:
- Notes: create, retrieve, list, search by tag
- Tasks: create, list (by status), complete, retrieve, delete
"""

# ── Agent state ───────────────────────────────────────────────────────────────
# TypedDict defines the shape of the graph's state — like a TypeScript interface.
#
# The `add_messages` reducer is the key:
#   - When a node returns {"messages": [new_message]}, it APPENDS to the list
#   - It does NOT replace the whole list (that's what the Annotated[] wrapper means)
#   - This gives us automatic conversation history without manual management
#
# Compare to Redux:
#   type State = { messages: Message[] }
#   function reducer(state, action) { return [...state.messages, action.payload] }

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ── LLM setup ─────────────────────────────────────────────────────────────────
ALL_TOOLS = NOTE_TOOLS + TASK_TOOLS

llm = ChatOllama(
    model=settings.MODEL_NAME,
    base_url=settings.OLLAMA_BASE_URL,
    temperature=0,          # deterministic responses — better for tool use
)

# bind_tools tells the LLM about available tools by injecting their
# JSON schemas into the system context. The LLM can then output
# structured tool_call messages instead of plain text.
llm_with_tools = llm.bind_tools(ALL_TOOLS)


# ── Graph nodes ───────────────────────────────────────────────────────────────

def call_model(state: AgentState) -> dict:
    """
    Node 1: Call the LLM with the current conversation history.

    Prepends the system prompt, then invokes the LLM.
    The LLM either:
      a) Returns a plain text AIMessage  → the graph routes to END
      b) Returns an AIMessage with tool_calls → the graph routes to tool_node

    Args:
        state: Current graph state containing the messages list.

    Returns:
        Dict with "messages" key containing the new AIMessage to append.
    """
    # Prepend the system prompt to every LLM call
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    # Returning {"messages": [response]} triggers the add_messages reducer
    # which appends this AIMessage to state["messages"]
    return {"messages": [response]}


# ToolNode is a pre-built LangGraph node that:
#   1. Reads tool_calls from the last AIMessage
#   2. Executes the matching tool function
#   3. Returns the results as ToolMessage objects
tool_node = ToolNode(ALL_TOOLS)


# ── Graph construction ────────────────────────────────────────────────────────

def build_graph(checkpointer=None):
    """
    Construct and compile the ReAct agent graph.

    Args:
        checkpointer: Optional LangGraph checkpointer for conversation persistence.
                      If None, the graph has no memory between separate invocations.

    Returns:
        A compiled LangGraph graph ready to invoke.
    """
    builder = StateGraph(AgentState)

    # Register nodes — each node is a function that takes state and returns a dict
    builder.add_node("call_model", call_model)
    builder.add_node("tools", tool_node)

    # Entry point — where the graph starts
    builder.add_edge(START, "call_model")

    # Conditional edge from call_model:
    #   tools_condition is a pre-built function that checks if the last message
    #   has tool_calls. If yes → route to "tools". If no → route to END.
    builder.add_conditional_edges("call_model", tools_condition)

    # After tools execute, always go back to call_model to process results
    builder.add_edge("tools", "call_model")

    return builder.compile(checkpointer=checkpointer)


# ── Persistence setup ─────────────────────────────────────────────────────────
# SqliteSaver stores the full conversation history in the same SQLite file
# as notes and tasks. Each "thread" is an independent conversation.
#
# thread_id is the session identifier — like a cookie or session token.
# Use the same thread_id to continue a conversation, a new one to start fresh.

_checkpointer = SqliteSaver.from_conn_string(settings.DB_PATH)

# Module-level graph — import this in other files (ui/app.py, supervisor.py)
graph = build_graph(checkpointer=_checkpointer)


# ── CLI runner ────────────────────────────────────────────────────────────────

async def run_cli(thread_id: str = "cli-session") -> None:
    """
    Interactive CLI loop for testing the agent in the terminal.

    Uses a fixed thread_id so conversation persists across restarts.
    Change thread_id to start a fresh conversation.

    Args:
        thread_id: Session identifier for conversation persistence.
    """
    config = {"configurable": {"thread_id": thread_id}}

    # ── ANSI colours for terminal output ──
    BLUE  = "\033[94m"
    GREEN = "\033[92m"
    RESET = "\033[0m"
    BOLD  = "\033[1m"

    print(f"\n{BOLD}🤖 Jarvis CLI{RESET}")
    print(f"   Model: {settings.MODEL_NAME}")
    print(f"   Session: {thread_id}")
    print(f"   Type 'exit' or 'quit' to stop\n")
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
            result = await graph.ainvoke(
                {"messages": [{"role": "user", "content": user_input}]},
                config=config,
            )
            # The last message is always the agent's final response
            response = result["messages"][-1].content
            print(f"\n{GREEN}Jarvis:{RESET} {response}")

        except Exception as e:
            # Surface errors clearly in CLI mode — helps with debugging
            if "connection refused" in str(e).lower() or "cannot connect" in str(e).lower():
                print(f"\n❌ Cannot connect to Ollama. Is it running?")
                print(f"   Run: ollama serve")
            else:
                print(f"\n❌ Error: {e}")
                print(f"   Run 'make health' to diagnose.")


if __name__ == "__main__":
    # Allow passing a custom thread_id as a CLI argument:
    #   python agents/assistant.py my-session-name
    thread = sys.argv[1] if len(sys.argv) > 1 else "cli-session"
    asyncio.run(run_cli(thread_id=thread))
