"""
tests/test_database.py — Tests for the SQLite persistence layer.

All tests use an in-memory database (:memory:) so they:
  - Run fast (no disk I/O)
  - Are fully isolated (each test gets a fresh DB)
  - Leave no files behind

For JS developers: this is the equivalent of using an in-memory SQLite
or a mock database in Jest — same concept, built into Python's sqlite3.

Run with: pytest tests/test_database.py -v
"""

import pytest
from db.database import Database


# ── Fixtures ──────────────────────────────────────────────────────────────────
# A pytest fixture is like a beforeEach() setup in Jest.
# The function runs before each test that declares it as a parameter.

@pytest.fixture
def db():
    """Create a fresh in-memory database for each test."""
    database = Database(":memory:")
    yield database          # yield = "here's the value for the test"
    database.close()        # runs after the test (like afterEach / cleanup)


# ── Note tests ────────────────────────────────────────────────────────────────

class TestNotes:
    """Tests for the notes CRUD operations."""

    def test_create_note_returns_dict_with_all_fields(self, db):
        """Creating a note should return a dict with id, title, content, tags, created_at."""
        note = db.create_note("My Note", "Some content")

        assert note["title"] == "My Note"
        assert note["content"] == "Some content"
        assert note["tags"] == []
        assert "id" in note
        assert len(note["id"]) == 36      # UUID format: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
        assert "created_at" in note

    def test_create_note_with_tags(self, db):
        """Tags should be stored and returned as a Python list."""
        note = db.create_note("Tagged Note", "Content", tags=["work", "ideas"])

        assert note["tags"] == ["work", "ideas"]

    def test_get_note_returns_correct_note(self, db):
        """get_note should retrieve a note by its ID."""
        created = db.create_note("Retrievable Note", "Hello world")

        fetched = db.get_note(created["id"])

        assert fetched is not None
        assert fetched["id"] == created["id"]
        assert fetched["title"] == "Retrievable Note"
        assert fetched["content"] == "Hello world"

    def test_get_note_returns_none_for_missing_id(self, db):
        """get_note should return None when the ID doesn't exist."""
        result = db.get_note("non-existent-id")

        assert result is None

    def test_list_notes_returns_all_notes(self, db):
        """list_notes should return all created notes."""
        db.create_note("Note A", "Content A")
        db.create_note("Note B", "Content B")
        db.create_note("Note C", "Content C")

        notes = db.list_notes()

        assert len(notes) == 3

    def test_list_notes_respects_limit(self, db):
        """list_notes should respect the limit parameter."""
        for i in range(5):
            db.create_note(f"Note {i}", f"Content {i}")

        notes = db.list_notes(limit=3)

        assert len(notes) == 3

    def test_list_notes_newest_first(self, db):
        """list_notes should return newest notes first."""
        db.create_note("First Note", "Created first")
        db.create_note("Second Note", "Created second")

        notes = db.list_notes()

        # Newest is first — "Second Note" should come before "First Note"
        assert notes[0]["title"] == "Second Note"
        assert notes[1]["title"] == "First Note"

    def test_search_notes_by_tag_finds_matching_notes(self, db):
        """search_notes_by_tag should return notes with the matching tag."""
        db.create_note("Work Note", "Meeting notes", tags=["work", "meetings"])
        db.create_note("Personal Note", "Grocery list", tags=["personal"])
        db.create_note("Another Work Note", "Project update", tags=["work", "projects"])

        results = db.search_notes_by_tag("work")

        assert len(results) == 2
        titles = [n["title"] for n in results]
        assert "Work Note" in titles
        assert "Another Work Note" in titles

    def test_search_notes_by_tag_returns_empty_for_no_match(self, db):
        """search_notes_by_tag should return empty list when no notes match."""
        db.create_note("Untagged Note", "No matching tags", tags=["other"])

        results = db.search_notes_by_tag("nonexistent-tag")

        assert results == []

    def test_tags_survive_roundtrip(self, db):
        """Tags should be stored as JSON and come back as a Python list."""
        original_tags = ["python", "ai", "langchain"]
        note = db.create_note("Tech Note", "About AI", tags=original_tags)

        # Fetch fresh from DB to confirm the roundtrip
        fetched = db.get_note(note["id"])

        assert fetched["tags"] == original_tags


# ── Task tests ────────────────────────────────────────────────────────────────

class TestTasks:
    """Tests for the tasks CRUD operations."""

    def test_create_task_returns_dict_with_all_fields(self, db):
        """Creating a task should return a dict with all expected fields."""
        task = db.create_task("Write tests", "Write unit tests for the DB layer")

        assert task["title"] == "Write tests"
        assert task["description"] == "Write unit tests for the DB layer"
        assert task["status"] == "pending"
        assert task["due_date"] == ""
        assert "id" in task
        assert "created_at" in task

    def test_create_task_with_due_date(self, db):
        """due_date should be stored and returned correctly."""
        task = db.create_task("Deploy app", due_date="2026-12-31")

        assert task["due_date"] == "2026-12-31"

    def test_get_task_returns_correct_task(self, db):
        """get_task should retrieve a task by its ID."""
        created = db.create_task("Fetch me", "I should be fetchable")

        fetched = db.get_task(created["id"])

        assert fetched is not None
        assert fetched["id"] == created["id"]
        assert fetched["title"] == "Fetch me"

    def test_get_task_returns_none_for_missing_id(self, db):
        """get_task should return None for non-existent IDs."""
        result = db.get_task("does-not-exist")

        assert result is None

    def test_list_tasks_returns_all_tasks(self, db):
        """list_tasks with no filter should return all tasks."""
        db.create_task("Task 1")
        db.create_task("Task 2")
        db.create_task("Task 3")

        tasks = db.list_tasks()

        assert len(tasks) == 3

    def test_list_tasks_filters_by_pending_status(self, db):
        """list_tasks(status='pending') should only return pending tasks."""
        task = db.create_task("Pending task")
        db.complete_task(task["id"])     # complete it
        db.create_task("Still pending")  # this one stays pending

        pending = db.list_tasks(status="pending")

        assert len(pending) == 1
        assert pending[0]["title"] == "Still pending"

    def test_list_tasks_filters_by_completed_status(self, db):
        """list_tasks(status='completed') should only return completed tasks."""
        task = db.create_task("Task to complete")
        db.create_task("Another pending task")
        db.complete_task(task["id"])

        completed = db.list_tasks(status="completed")

        assert len(completed) == 1
        assert completed[0]["title"] == "Task to complete"
        assert completed[0]["status"] == "completed"

    def test_complete_task_changes_status(self, db):
        """complete_task should update the task status to 'completed'."""
        task = db.create_task("Complete me")
        assert task["status"] == "pending"

        updated = db.complete_task(task["id"])

        assert updated is not None
        assert updated["status"] == "completed"

    def test_complete_task_returns_none_for_missing_id(self, db):
        """complete_task on a non-existent ID should return None."""
        result = db.complete_task("ghost-id")

        assert result is None

    def test_delete_task_removes_it_from_db(self, db):
        """delete_task should remove the task permanently."""
        task = db.create_task("Delete me")

        deleted = db.delete_task(task["id"])
        fetched = db.get_task(task["id"])

        assert deleted is True
        assert fetched is None

    def test_delete_task_returns_false_for_missing_id(self, db):
        """delete_task on a non-existent ID should return False."""
        result = db.delete_task("non-existent-id")

        assert result is False

    def test_full_task_lifecycle(self, db):
        """Integration test: create → list pending → complete → list completed → delete."""
        # Create
        task = db.create_task("Lifecycle task", "Full roundtrip test")
        assert task["status"] == "pending"

        # List pending
        pending = db.list_tasks(status="pending")
        ids = [t["id"] for t in pending]
        assert task["id"] in ids

        # Complete
        db.complete_task(task["id"])
        pending_after = db.list_tasks(status="pending")
        completed_after = db.list_tasks(status="completed")
        assert task["id"] not in [t["id"] for t in pending_after]
        assert task["id"] in [t["id"] for t in completed_after]

        # Delete
        db.delete_task(task["id"])
        assert db.get_task(task["id"]) is None
