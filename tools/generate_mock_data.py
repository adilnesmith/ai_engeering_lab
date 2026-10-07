"""Create realistic local data for exercising Jarvis without an LLM.

The generator writes to ``data/mock_jarvis.db`` by default so it never
overwrites a user's working database. Pass ``--db-path`` to target another
database explicitly, for example when manually checking the Gradio sidebar.

Usage:
    python tools/generate_mock_data.py
    python tools/generate_mock_data.py --reset
    python tools/generate_mock_data.py --db-path data/manual-test.db --reset
"""

from __future__ import annotations

import argparse
from pathlib import Path

from db.database import Database


MOCK_TASKS = [
    {
        "title": "Review agent routing logs",
        "description": "Check note, task, and memory specialist hand-offs.",
        "due_date": "2026-10-08",
    },
    {
        "title": "Add golden-path regression test",
        "description": "Cover create task → complete task → refresh sidebar.",
        "due_date": "2026-10-10",
    },
    {
        "title": "Document Ollama troubleshooting",
        "description": "Keep setup guidance friendly for Windows users.",
        "due_date": None,
    },
    {
        "title": "Archive old sprint notes",
        "description": "Move completed planning notes out of the active view.",
        "due_date": "2026-10-14",
    },
]

MOCK_NOTES = [
    {
        "title": "Sprint goals",
        "content": "Improve the local assistant's reliability, UI clarity, and test coverage.",
        "tags": ["work", "planning"],
    },
    {
        "title": "UI review findings",
        "content": "Keep the chat primary, make sidebar state scannable, and show useful empty states.",
        "tags": ["ui", "accessibility"],
    },
    {
        "title": "Ollama setup",
        "content": "Chat uses llama3.2:3b; semantic memory uses nomic-embed-text.",
        "tags": ["setup", "ai"],
    },
    {
        "title": "Demo prompt ideas",
        "content": "Create a task, save a note, complete a task, then ask what Jarvis remembers.",
        "tags": ["demo", "testing"],
    },
    {
        "title": "Privacy promise",
        "content": "All notes, tasks, and memory remain on the local machine.",
        "tags": ["security", "product"],
    },
]


def populate(db_path: str, reset: bool = False) -> tuple[int, int]:
    """Populate ``db_path`` and return ``(task_count, note_count)``."""
    path = Path(db_path)
    if reset and path.exists() and db_path != ":memory:":
        path.unlink()

    db = Database(db_path)
    try:
        if reset:
            with db.conn:
                db.conn.execute("DELETE FROM tasks")
                db.conn.execute("DELETE FROM notes")

        existing_tasks = {task["title"] for task in db.list_tasks(status="all")}
        existing_notes = {note["title"] for note in db.list_notes(limit=1000)}

        task_count = 0
        for task in MOCK_TASKS:
            if task["title"] not in existing_tasks:
                db.create_task(**task)
                task_count += 1

        note_count = 0
        for note in MOCK_NOTES:
            if note["title"] not in existing_notes:
                db.create_note(**note)
                note_count += 1

        return task_count, note_count
    finally:
        db.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db-path",
        default="data/mock_jarvis.db",
        help="SQLite path (defaults to a safe mock database)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="clear the selected database before inserting mock records",
    )
    args = parser.parse_args()
    tasks, notes = populate(args.db_path, reset=args.reset)
    print(f"Created {tasks} tasks and {notes} notes in {args.db_path}")


if __name__ == "__main__":
    main()
