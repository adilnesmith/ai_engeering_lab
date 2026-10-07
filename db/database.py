"""
db/database.py — SQLite persistence layer for notes and tasks.

This module provides the durable storage used by the agent tools and the
Gradio UI. It exposes a small, thread-safe-enough wrapper for the local
single-user app while keeping all SQL parameterised.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class Database:
    """Thin SQLite wrapper for notes and tasks data."""

    def __init__(self, db_path: str = "data/jarvis.db") -> None:
        self.db_path = db_path
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        """Create the notes and tasks schema if it does not already exist."""
        with self.conn:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS notes (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    tags TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL
                )
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'pending',
                    due_date TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )

    def close(self) -> None:
        """Close the SQLite connection."""
        self.conn.close()

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _normalize_tags(tags: Any) -> list[str]:
        """Return clean tags from either a list or the stored JSON string."""
        if tags is None:
            return []
        if isinstance(tags, str):
            try:
                parsed = json.loads(tags)
            except json.JSONDecodeError:
                return []
            if not isinstance(parsed, list):
                return []
            tags = parsed
        if isinstance(tags, (list, tuple)):
            return [str(item).strip() for item in tags if str(item).strip()]
        return []

    @staticmethod
    def _note_from_row(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "title": row["title"],
            "content": row["content"],
            "tags": Database._normalize_tags(row["tags"]),
            "created_at": row["created_at"],
        }

    @staticmethod
    def _task_from_row(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "title": row["title"],
            "description": row["description"],
            "status": row["status"],
            "due_date": row["due_date"],
            "created_at": row["created_at"],
        }

    # ── Notes ───────────────────────────────────────────────────────────────

    def create_note(
        self, title: str, content: str, tags: list[str] | None = None
    ) -> dict[str, Any]:
        """Insert a note and return the saved representation."""
        note_id = str(uuid.uuid4())
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO notes (id, title, content, tags, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    note_id,
                    title.strip(),
                    content.strip(),
                    json.dumps(self._normalize_tags(tags)),
                    self._utc_now(),
                ),
            )
        return self.get_note(note_id)  # type: ignore[return-value]

    def get_note(self, note_id: str) -> dict[str, Any] | None:
        """Fetch a note by id, or return ``None`` when it is missing."""
        row = self.conn.execute(
            "SELECT * FROM notes WHERE id = ?", (note_id,)
        ).fetchone()
        return self._note_from_row(row) if row is not None else None

    def list_notes(self, limit: int = 10) -> list[dict[str, Any]]:
        """List recent notes newest-first."""
        safe_limit = max(0, int(limit))
        rows = self.conn.execute(
            "SELECT * FROM notes ORDER BY rowid DESC LIMIT ?", (safe_limit,)
        ).fetchall()
        return [self._note_from_row(row) for row in rows]

    def search_notes_by_tag(self, tag: str) -> list[dict[str, Any]]:
        """Return notes whose tags contain the given tag, case-insensitively."""
        tag_name = tag.strip().lower()
        rows = self.conn.execute(
            "SELECT * FROM notes ORDER BY created_at DESC"
        ).fetchall()
        return [
            self._note_from_row(row)
            for row in rows
            if tag_name in {item.lower() for item in self._normalize_tags(row["tags"])}
        ]

    # ── Tasks ────────────────────────────────────────────────────────────────

    def create_task(
        self,
        title: str,
        description: str = "",
        due_date: str | None = None,
    ) -> dict[str, Any]:
        """Insert a task with pending status and return the saved record."""
        task_id = str(uuid.uuid4())
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO tasks (id, title, description, status, due_date, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    task_id,
                    title.strip(),
                    description.strip(),
                    "pending",
                    (due_date or "").strip(),
                    self._utc_now(),
                ),
            )
        return self.get_task(task_id)  # type: ignore[return-value]

    def get_task(self, task_id: str) -> dict[str, Any] | None:
        """Fetch a task by id, or ``None`` when it is missing."""
        row = self.conn.execute(
            "SELECT * FROM tasks WHERE id = ?", (task_id,)
        ).fetchone()
        return self._task_from_row(row) if row is not None else None

    def list_tasks(
        self, status: str | None = "pending", limit: int | None = None
    ) -> list[dict[str, Any]]:
        """List tasks newest-first, with ``None``/``all`` meaning every task."""
        if status is None or status == "all":
            sql = "SELECT * FROM tasks ORDER BY rowid DESC"
            params: tuple[Any, ...] = ()
        else:
            sql = "SELECT * FROM tasks WHERE status = ? ORDER BY rowid DESC"
            params = (status,)

        if limit is not None:
            sql += " LIMIT ?"
            params += (max(0, int(limit)),)

        rows = self.conn.execute(sql, params).fetchall()
        return [self._task_from_row(row) for row in rows]

    def complete_task(self, task_id: str) -> dict[str, Any] | None:
        """Set task status to completed and return the updated task."""
        with self.conn:
            cursor = self.conn.execute(
                "UPDATE tasks SET status = 'completed' WHERE id = ?", (task_id,)
            )
        return self.get_task(task_id) if cursor.rowcount else None

    def delete_task(self, task_id: str) -> bool:
        """Delete a task permanently and return whether it existed."""
        with self.conn:
            cursor = self.conn.execute(
                "DELETE FROM tasks WHERE id = ?", (task_id,)
            )
        return cursor.rowcount > 0
