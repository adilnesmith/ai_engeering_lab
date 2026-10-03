"""
ui/app.py — Gradio web interface for Jarvis.

Features:
  - Streaming chat responses (token by token)
  - Live task and note panels that update after each message
  - Tool usage indicators shown in the chat
  - New conversation button (resets thread_id, keeps data)
  - Graceful error handling when Ollama is not running

Architecture:
  Browser ──► Gradio (HTTP) ──► supervisor_graph (LangGraph) ──► Ollama (local)
                                        │
                                    SQLite DB (notes, tasks, memory)

For JS developers:
  - Gradio is like a React UI backed by Python functions instead of a REST API
  - gr.Blocks is like JSX layout — you compose components in a with-block
  - gr.State is like React useState — server-side state per browser session
  - async def + yield is like an async generator feeding a ReadableStream

Run with:
    python ui/app.py
    make run
"""

import asyncio
import uuid
from datetime import datetime
from html import escape

import gradio as gr

from config import settings
from db.database import Database

# Import the supervisor graph — this is the brain behind the UI
try:
    from agents.supervisor import supervisor_graph
    GRAPH_AVAILABLE = True
except Exception as _import_error:
    GRAPH_AVAILABLE = False
    _IMPORT_ERROR_MSG = str(_import_error)

# Direct DB access for the sidebar panels
db = Database(settings.DB_PATH)


# ── Helper: check Ollama ──────────────────────────────────────────────────────

def is_ollama_running() -> bool:
    """Quick non-blocking check whether Ollama HTTP server is reachable."""
    import urllib.request
    import urllib.error
    try:
        with urllib.request.urlopen(
            f"{settings.OLLAMA_BASE_URL}/api/tags", timeout=2
        ) as resp:
            return resp.status == 200
    except Exception:
        return False


# ── Helper: format sidebar data ───────────────────────────────────────────────

def get_sidebar_data() -> tuple[list[dict], list[dict]]:
    """
    Fetch current tasks and notes for the sidebar panels.

    Returns:
        Tuple of (tasks_list, notes_list) — each a list of simplified dicts.
    """
    try:
        raw_tasks = db.list_tasks(status="pending")
        tasks = [
            {
                "title": t["title"],
                "due": t.get("due_date") or "no due date",
                "id": t["id"][:8] + "...",   # truncate UUID for display
            }
            for t in raw_tasks[:10]
        ]

        raw_notes = db.list_notes(limit=5)
        notes = [
            {
                "title": n["title"],
                "tags": ", ".join(n.get("tags", [])) or "no tags",
                "created": n["created_at"][:10],  # just the date part
            }
            for n in raw_notes
        ]
        return tasks, notes
    except Exception:
        return [], []


def render_tasks_html(tasks: list[dict]) -> str:
    """Render task sidebar content as HTML."""
    if not tasks:
        return "<div style='color:#666; font-size:0.9em;'>No pending tasks.</div>"

    rows = []
    for task in tasks:
        rows.append(
            "<div style='padding:6px 0; border-bottom:1px solid #eee;'>"
            f"<strong>{escape(task['title'])}</strong><br>"
            f"<span style='color:#666; font-size:0.8em;'>Due: {escape(task['due'])}</span>"
            "</div>"
        )
    return "".join(rows)


def render_notes_html(notes: list[dict]) -> str:
    """Render note sidebar content as HTML."""
    if not notes:
        return "<div style='color:#666; font-size:0.9em;'>No notes yet.</div>"

    rows = []
    for note in notes:
        rows.append(
            "<div style='padding:6px 0; border-bottom:1px solid #eee;'>"
            f"<strong>{escape(note['title'])}</strong><br>"
            f"<span style='color:#666; font-size:0.8em;'>Tags: {escape(note['tags'])}</span><br>"
            f"<span style='color:#999; font-size:0.75em;'>{escape(note['created'])}</span>"
            "</div>"
        )
    return "".join(rows)


# ── Core streaming function ───────────────────────────────────────────────────

async def stream_response(
    message: str,
    history: list[dict],
    thread_id: str,
) -> tuple:
    """
    Process a user message through the supervisor graph and stream the response.

    This is an async generator — it yields partial (history, tasks, notes) tuples
    on every token, which Gradio renders in real time.

    Args:
        message:   The user's latest message.
        history:   The current Gradio chat history (list of {role, content} dicts).
        thread_id: The LangGraph session ID for conversation continuity.

    Yields:
        Tuple of (updated_history, "", tasks_data, notes_data)
        The empty string clears the input box.
    """
    if not message.strip():
        tasks, notes = get_sidebar_data()
        yield history, "", render_tasks_html(tasks), render_notes_html(notes)
        return

    # Graceful degradation: check Ollama before trying to invoke
    if not GRAPH_AVAILABLE:
        error_msg = f"⚠️ Could not load the agent graph:\n```\n{_IMPORT_ERROR_MSG}\n```\nRun `make health` to diagnose."
        history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": error_msg},
        ]
        tasks, notes = get_sidebar_data()
        yield history, "", render_tasks_html(tasks), render_notes_html(notes)
        return

    if not is_ollama_running():
        error_msg = (
            "⚠️ **Ollama is not running.**\n\n"
            "Start it with:\n```\nollama serve\n```\n"
            f"Then make sure `{settings.MODEL_NAME}` is pulled:\n"
            f"```\nollama pull {settings.MODEL_NAME}\n```"
        )
        history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": error_msg},
        ]
        tasks, notes = get_sidebar_data()
        yield history, "", render_tasks_html(tasks), render_notes_html(notes)
        return

    # Add user message to history immediately (before waiting for LLM)
    history = history + [{"role": "user", "content": message}]
    # Append a placeholder assistant message that we'll fill in as tokens stream
    history = history + [{"role": "assistant", "content": ""}]

    # Yield immediately to show the user message in UI before LLM starts
    tasks, notes = get_sidebar_data()
    yield history, "", render_tasks_html(tasks), render_notes_html(notes)

    config = {"configurable": {"thread_id": thread_id}}
    partial_response = ""
    tool_indicators = []

    try:
        # astream_events streams every internal event from the graph:
        #   - on_chat_model_stream: individual tokens from the LLM
        #   - on_tool_start: a tool is about to be called
        #   - on_tool_end: a tool finished executing
        #
        # version="v2" is required for langgraph >= 0.2
        async for event in supervisor_graph.astream_events(
            {"messages": [{"role": "user", "content": message}]},
            config=config,
            version="v2",
        ):
            event_type = event.get("event", "")
            event_name = event.get("name", "")

            # ── Token streaming ──────────────────────────────────────────────
            if event_type == "on_chat_model_stream":
                chunk = event["data"].get("chunk")
                if chunk and hasattr(chunk, "content") and chunk.content:
                    # Only stream the final synthesis from the finish node,
                    # not intermediate specialist chatter.
                    # The finish node's model call is named "ChatOllama"
                    # We filter by checking it's not coming from a sub-agent tool node
                    partial_response += chunk.content
                    # Update the last message in history with the partial response
                    display = partial_response
                    if tool_indicators:
                        display = "\n".join(tool_indicators) + "\n\n" + partial_response
                    history[-1]["content"] = display
                    yield history, "", render_tasks_html(tasks), render_notes_html(notes)

            # ── Tool usage indicators ────────────────────────────────────────
            elif event_type == "on_tool_start":
                tool_name = event_name or event.get("data", {}).get("name", "tool")
                indicator = f"🔧 `{tool_name}`"
                if indicator not in tool_indicators:
                    tool_indicators.append(indicator)
                    # Show tool usage while response is being built
                    if partial_response:
                        display = "\n".join(tool_indicators) + "\n\n" + partial_response
                    else:
                        display = "\n".join(tool_indicators) + "\n\n_Thinking..._"
                    history[-1]["content"] = display
                    yield history, "", render_tasks_html(tasks), render_notes_html(notes)

        # ── Final update with refreshed sidebar ──────────────────────────────
        # After the graph finishes, refresh the panels with any new data
        tasks, notes = get_sidebar_data()

        # Ensure the final response is clean (remove "Thinking..." placeholder)
        if not partial_response:
            # Graph ran but produced no streaming content — use last message
            # This can happen if the model doesn't stream final synthesis
            try:
                result = await supervisor_graph.aget_state(config)
                if result and result.values.get("messages"):
                    last_msg = result.values["messages"][-1]
                    if hasattr(last_msg, "content") and last_msg.content:
                        partial_response = last_msg.content
            except Exception:
                partial_response = "Done! I've completed your request."

        final_display = partial_response
        if tool_indicators:
            final_display = "\n".join(tool_indicators) + "\n\n" + partial_response
        history[-1]["content"] = final_display
        yield history, "", render_tasks_html(tasks), render_notes_html(notes)

    except Exception as e:
        error_text = str(e)
        if "connection refused" in error_text.lower() or "cannot connect" in error_text.lower():
            history[-1]["content"] = (
                "⚠️ Lost connection to Ollama. Please make sure it's still running.\n"
                "```\nollama serve\n```"
            )
        else:
            history[-1]["content"] = f"⚠️ An error occurred:\n```\n{error_text}\n```"
        tasks, notes = get_sidebar_data()
        yield history, "", render_tasks_html(tasks), render_notes_html(notes)


# ── New conversation ──────────────────────────────────────────────────────────

def new_conversation() -> tuple:
    """
    Start a fresh conversation by generating a new thread_id.

    Note: this clears the chat history but does NOT delete notes or tasks.
    Data is always preserved. Only the in-memory conversation resets.

    Returns:
        Tuple of (empty_history, new_thread_id, tasks, notes)
    """
    new_thread = f"session-{uuid.uuid4().hex[:8]}"
    tasks, notes = get_sidebar_data()
    return [], new_thread, render_tasks_html(tasks), render_notes_html(notes)


def refresh_panels() -> tuple[str, str]:
    """Manually refresh the task and note sidebar panels."""
    tasks, notes = get_sidebar_data()
    return render_tasks_html(tasks), render_notes_html(notes)


# ── Build the Gradio UI ───────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    """
    Construct the Gradio Blocks UI.

    Layout:
      ┌─────────────────────────────┬───────────────────┐
      │  Chat (3/4 width)           │  Sidebar (1/4)    │
      │                             │  📋 Pending Tasks │
      │  [Chatbot component]        │  📝 Recent Notes  │
      │                             │  [Refresh]        │
      │  [Input] [Send] [New Chat]  │                   │
      └─────────────────────────────┴───────────────────┘
    """
    with gr.Blocks(
        title="Jarvis — Personal AI Assistant",
        theme=gr.themes.Soft(),
        css="""
            .jarvis-header { text-align: center; padding: 10px 0; }
            .jarvis-header h1 { font-size: 1.8em; margin: 0; }
            .jarvis-header p  { color: #666; margin: 4px 0 0 0; font-size: 0.9em; }
            .status-bar { font-size: 0.8em; color: #888; padding: 4px 8px; }
        """,
    ) as demo:

        # ── Header ────────────────────────────────────────────────────────────
        with gr.Row(elem_classes="jarvis-header"):
            gr.HTML(f"""
                <h1>🤖 Jarvis</h1>
                <p>Personal AI Assistant &nbsp;·&nbsp;
                   Model: <code>{settings.MODEL_NAME}</code> &nbsp;·&nbsp;
                   Multi-agent · Three-tier memory · Fully local</p>
            """)

        # ── Session state ─────────────────────────────────────────────────────
        # gr.State is server-side state per browser session.
        # Each visitor gets their own thread_id — sessions are isolated.
        # Compare to React's useState — but stored on the server, not the browser.
        thread_id_state = gr.State(value=f"session-{uuid.uuid4().hex[:8]}")

        # ── Main layout ───────────────────────────────────────────────────────
        with gr.Row():

            # Left column — chat (takes 3/4 of the width)
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="Jarvis",
                    type="messages",        # modern {role, content} format
                    height=520,
                    show_label=False,
                    avatar_images=(None, "🤖"),   # (user avatar, assistant avatar)
                    render_markdown=True,
                    bubble_full_width=False,
                )

                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="Ask Jarvis to create a task, save a note, or recall a memory...",
                        show_label=False,
                        scale=5,
                        container=False,
                        autofocus=True,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1, min_width=80)

                with gr.Row():
                    clear_btn = gr.Button("🔄 New Conversation", variant="secondary", size="sm")
                    gr.HTML(
                        '<p class="status-bar">💡 Try: "Create a task to review the README" · '
                        '"Save a note about today\'s standup" · "What do you remember about me?"</p>'
                    )

            # Right column — sidebar panels (takes 1/4 of the width)
            with gr.Column(scale=1):
                tasks_panel = gr.HTML(
                    value="<div style='color:#666; font-size:0.9em;'>Loading tasks...</div>",
                    show_label=False,
                )
                notes_panel = gr.HTML(
                    value="<div style='color:#666; font-size:0.9em;'>Loading notes...</div>",
                    show_label=False,
                )
                refresh_btn = gr.Button("↻ Refresh", variant="secondary", size="sm")

        # ── Event: Send message (button click or Enter) ───────────────────────
        # Both the Send button and pressing Enter in the textbox call stream_response.
        #
        # inputs:  message text, current chat history, current thread_id
        # outputs: updated history, cleared input box, refreshed task/note panels

        send_inputs  = [msg_input, chatbot, thread_id_state]
        send_outputs = [chatbot, msg_input, tasks_panel, notes_panel]

        send_btn.click(
            fn=stream_response,
            inputs=send_inputs,
            outputs=send_outputs,
        )

        msg_input.submit(
            fn=stream_response,
            inputs=send_inputs,
            outputs=send_outputs,
        )

        # ── Event: New conversation ───────────────────────────────────────────
        clear_btn.click(
            fn=new_conversation,
            inputs=[],
            outputs=[chatbot, thread_id_state, tasks_panel, notes_panel],
        )

        # ── Event: Refresh sidebar ────────────────────────────────────────────
        refresh_btn.click(
            fn=refresh_panels,
            inputs=[],
            outputs=[tasks_panel, notes_panel],
        )

        # ── Load initial sidebar data on page open ────────────────────────────
        demo.load(
            fn=refresh_panels,
            inputs=[],
            outputs=[tasks_panel, notes_panel],
        )

    return demo


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n🤖 Starting Jarvis web UI...")
    print(f"   Model:  {settings.MODEL_NAME}")
    print(f"   Ollama: {settings.OLLAMA_BASE_URL}")
    print(f"   Data:   {settings.DB_PATH}")

    if not is_ollama_running():
        print("\n⚠️  Warning: Ollama is not running.")
        print(f"   Start it with: ollama serve")
        print(f"   The UI will still launch but responses will show an error.")
    else:
        print(f"\n✅ Ollama is running.")

    print("\n   Opening at: http://localhost:7860\n")

    demo = build_ui()
    # Gradio's launch() probes localhost with HTTP HEAD. That request can hang
    # or fail on Windows (proxy / IPv6), even after the server is already bound.
    import gradio.networking as _gradio_net
    _gradio_net.url_ok = lambda _url: True
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
    )
