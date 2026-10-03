"""
tools/task_tools.py — LangChain tools wrapping the tasks database layer.

Follows the exact same pattern as note_tools.py.
See that file for the conceptual explanation of @tool and docstrings.

Usage:
    from tools.task_tools import TASK_TOOLS
    llm_with_tools = llm.bind_tools(TASK_TOOLS)

    # Direct invocation (no LLM):
    from tools.task_tools import create_task
    result = create_task.invoke({"title": "Buy groceries"})
"""

from langchain_core.tools import tool

from db.database import Database
from config import settings

# Module-level singleton — shared DB connection.
# Patched in tests with an in-memory instance.
db = Database(settings.DB_PATH)


@tool
def create_task(title: str, description: str = "", due_date: str = "") -> dict:
    """
    Create a new task and save it with 'pending' status.

    Use this when the user wants to add something to their to-do list,
    create a reminder, or track something that needs to be done.

    Args:
        title:       A short, clear title for the task (e.g. "Review PR #42").
        description: Optional longer description of what needs to be done.
        due_date:    Optional due date in YYYY-MM-DD format (e.g. "2026-10-10").
                     Leave empty if no deadline is specified.

    Returns:
        A dict with the created task including its generated id and status 'pending'.
    """
    return db.create_task(title, description, due_date or None)


@tool
def list_tasks(status: str = "pending") -> list:
    """
    List tasks filtered by their status.

    Use this when the user asks to see their tasks, wants to know what's
    pending, or asks "what do I need to do?".

    Args:
        status: Filter tasks by status. Use:
                - "pending"   to see tasks not yet done (default)
                - "completed" to see finished tasks
                - "all"       to see every task regardless of status

    Returns:
        A list of task dicts, ordered newest first.
        Each dict contains: id, title, description, status, due_date, created_at.
    """
    if status == "all":
        return db.list_tasks(status=None)
    return db.list_tasks(status=status)


@tool
def complete_task(task_id: str) -> dict:
    """
    Mark a task as completed.

    Use this when the user says they've finished a task, wants to check
    something off, or says something like "done with X" or "mark X as done".

    Args:
        task_id: The UUID string of the task to mark as completed.
                 Get this from list_tasks or create_task output.

    Returns:
        The updated task dict with status 'completed', or an error dict if not found.
    """
    updated = db.complete_task(task_id)
    if updated is None:
        return {"error": f"Task with id '{task_id}' not found."}
    return updated


@tool
def get_task(task_id: str) -> dict:
    """
    Retrieve a specific task by its ID.

    Use this when you have a task ID and need to check its current details or status.

    Args:
        task_id: The UUID string of the task to retrieve.

    Returns:
        A dict with the task's fields, or an error dict if not found.
    """
    task = db.get_task(task_id)
    if task is None:
        return {"error": f"Task with id '{task_id}' not found."}
    return task


@tool
def delete_task(task_id: str) -> dict:
    """
    Permanently delete a task.

    Use this when the user explicitly asks to remove or delete a task.
    This cannot be undone. For finished tasks, prefer complete_task instead.

    Args:
        task_id: The UUID string of the task to delete.

    Returns:
        A dict with success status and a message.
    """
    deleted = db.delete_task(task_id)
    if not deleted:
        return {"success": False, "message": f"Task with id '{task_id}' not found."}
    return {"success": True, "message": f"Task '{task_id}' has been deleted."}


# ── Exports ───────────────────────────────────────────────────────────────────
TASK_TOOLS = [create_task, list_tasks, complete_task, get_task, delete_task]
