"""
tools/note_tools.py — LangChain tools wrapping the notes database layer.

This is where the database layer becomes "agentic" — each function is
decorated with @tool, which registers it as a callable the LLM can choose
to invoke when it decides the tool is needed.

Key concept: the docstring IS the tool description.
The LLM reads the docstring to understand what the tool does and when to use it.
Write them like you're writing documentation for a teammate, not for a machine.

For JS developers: think of these as API route handlers that the LLM can call.
The @tool decorator is like Express's app.post() — it registers the function
with a name and description so it's discoverable.

Usage:
    from tools.note_tools import NOTE_TOOLS
    llm_with_tools = llm.bind_tools(NOTE_TOOLS)

    # Direct invocation (no LLM, useful for testing):
    from tools.note_tools import create_note
    result = create_note.invoke({"title": "Test", "content": "Hello"})
"""

from langchain_core.tools import tool

from db.database import Database
from config import settings

# Module-level singleton — one DB connection shared across all tool calls.
# In tests, this gets patched with an in-memory instance.
db = Database(settings.DB_PATH)


# ─────────────────────────────────────────────────────────────────────────────
# Tool definitions
# Each function:
#   1. Is decorated with @tool
#   2. Has a clear, descriptive docstring (the LLM reads this)
#   3. Has typed parameters (the LLM uses these to build the call)
#   4. Returns a JSON-serialisable value (dict or list)
# ─────────────────────────────────────────────────────────────────────────────

@tool
def create_note(title: str, content: str, tags: str = "") -> dict:
    """
    Create and save a new note with a title and content.

    Use this when the user wants to save information, jot something down,
    record meeting notes, capture ideas, or write anything they want to
    remember later.

    Args:
        title:   A short, descriptive title for the note (e.g. "Meeting Notes - Oct 3").
        content: The full body text of the note.
        tags:    Optional comma-separated tags for categorisation (e.g. "work,meetings").
                 Leave empty if no tags are needed.

    Returns:
        A dict with the created note including its generated id.
    """
    # Convert comma-separated tag string to a list, stripping whitespace
    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
    return db.create_note(title, content, tag_list)


@tool
def get_note(note_id: str) -> dict:
    """
    Retrieve a specific note by its ID.

    Use this when you have a note ID and need to read its full content.
    Note IDs are returned by create_note and list_notes.

    Args:
        note_id: The UUID string of the note to retrieve.

    Returns:
        A dict with the note's fields, or an error dict if not found.
    """
    note = db.get_note(note_id)
    if note is None:
        return {"error": f"Note with id '{note_id}' not found."}
    return note


@tool
def list_notes(limit: int = 10) -> list:
    """
    List the most recently created notes.

    Use this when the user asks to see their notes, wants to browse what
    they've saved, or asks "what notes do I have?".

    Args:
        limit: Maximum number of notes to return. Defaults to 10.
               Use a higher number if the user wants to see more.

    Returns:
        A list of note dicts, ordered newest first.
        Each dict contains: id, title, content, tags, created_at.
    """
    return db.list_notes(limit=limit)


@tool
def search_notes_by_tag(tag: str) -> list:
    """
    Find all notes that have a specific tag.

    Use this when the user wants to find notes about a topic, or asks
    something like "show me my work notes" or "find notes tagged with meetings".

    Args:
        tag: A single tag to search for (e.g. "work", "meetings", "ideas").
             Only one tag at a time is supported.

    Returns:
        A list of matching note dicts, or an empty list if none found.
    """
    return db.search_notes_by_tag(tag)


# ── Exports ───────────────────────────────────────────────────────────────────
# This list is what gets passed to llm.bind_tools() and ToolNode.
# Keeping tools grouped by domain makes it easy to assign them to
# the right specialist agent later.

NOTE_TOOLS = [create_note, get_note, list_notes, search_notes_by_tag]
