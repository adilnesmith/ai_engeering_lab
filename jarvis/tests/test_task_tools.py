"""
tests/test_task_tools.py — Tests for the task LangChain tools.

Same pattern as test_note_tools.py — tools are called directly via .invoke()
and the module-level db is patched with an in-memory instance.

Run with: pytest tests/test_task_tools.py -v
"""

import pytest
from unittest.mock import patch

from db.database import Database
import tools.task_tools as task_tools_module
from tools.task_tools import create_task, list_tasks, complete_task, get_task, delete_task


# ── Fixture ───────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def use_in_memory_db():
    """Replace the module-level db with a fresh in-memory instance per test."""
    mem_db = Database(":memory:")
    with patch.object(task_tools_module, "db", mem_db):
        yield mem_db
    mem_db.close()


# ── create_task tests ─────────────────────────────────────────────────────────

class TestCreateTask:

    def test_create_task_returns_dict_with_required_fields(self):
        """create_task.invoke() should return a complete task dict."""
        result = create_task.invoke({"title": "Write tests"})

        assert isinstance(result, dict)
        assert "id" in result
        assert result["title"] == "Write tests"
        assert result["status"] == "pending"

    def test_create_task_with_description(self):
        """Description should be stored and returned."""
        result = create_task.invoke({
            "title": "Deploy app",
            "description": "Push to production server"
        })

        assert result["description"] == "Push to production server"

    def test_create_task_with_due_date(self):
        """Due date in YYYY-MM-DD format should be stored correctly."""
        result = create_task.invoke({
            "title": "Deadline task",
            "due_date": "2026-12-31"
        })

        assert result["due_date"] == "2026-12-31"

    def test_create_task_defaults_to_pending_status(self):
        """Newly created tasks should always have status 'pending'."""
        result = create_task.invoke({"title": "New task"})

        assert result["status"] == "pending"

    def test_create_task_empty_due_date_stored_as_none(self):
        """Empty due_date string should be treated as no due date."""
        result = create_task.invoke({"title": "No deadline", "due_date": ""})

        # The DB stores None/empty, both are acceptable
        assert result["due_date"] in (None, "")


# ── list_tasks tests ──────────────────────────────────────────────────────────

class TestListTasks:

    def test_list_tasks_returns_list(self):
        """list_tasks.invoke() should always return a list."""
        result = list_tasks.invoke({})

        assert isinstance(result, list)

    def test_list_tasks_shows_pending_by_default(self):
        """Default status filter should be 'pending'."""
        task = create_task.invoke({"title": "Pending task"})
        complete_task.invoke({"task_id": task["id"]})
        create_task.invoke({"title": "Still pending"})

        result = list_tasks.invoke({})

        assert len(result) == 1
        assert result[0]["title"] == "Still pending"

    def test_list_tasks_filter_pending(self):
        """status='pending' should return only pending tasks."""
        create_task.invoke({"title": "Task A"})
        task_b = create_task.invoke({"title": "Task B"})
        complete_task.invoke({"task_id": task_b["id"]})

        result = list_tasks.invoke({"status": "pending"})

        assert len(result) == 1
        assert result[0]["title"] == "Task A"

    def test_list_tasks_filter_completed(self):
        """status='completed' should return only completed tasks."""
        task_a = create_task.invoke({"title": "Task A"})
        create_task.invoke({"title": "Task B"})  # stays pending
        complete_task.invoke({"task_id": task_a["id"]})

        result = list_tasks.invoke({"status": "completed"})

        assert len(result) == 1
        assert result[0]["title"] == "Task A"
        assert result[0]["status"] == "completed"

    def test_list_tasks_all_returns_everything(self):
        """status='all' should return tasks regardless of status."""
        task_a = create_task.invoke({"title": "Task A"})
        create_task.invoke({"title": "Task B"})
        complete_task.invoke({"task_id": task_a["id"]})

        result = list_tasks.invoke({"status": "all"})

        assert len(result) == 2


# ── complete_task tests ───────────────────────────────────────────────────────

class TestCompleteTask:

    def test_complete_task_changes_status_to_completed(self):
        """complete_task should update status from 'pending' to 'completed'."""
        task = create_task.invoke({"title": "Complete me"})
        assert task["status"] == "pending"

        result = complete_task.invoke({"task_id": task["id"]})

        assert result["status"] == "completed"

    def test_complete_task_returns_error_for_missing_id(self):
        """complete_task should return an error dict for unknown task ids."""
        result = complete_task.invoke({"task_id": "ghost-uuid"})

        assert "error" in result
        assert "ghost-uuid" in result["error"]

    def test_complete_task_does_not_affect_other_tasks(self):
        """Completing one task should not change other tasks."""
        task_a = create_task.invoke({"title": "Task A"})
        task_b = create_task.invoke({"title": "Task B"})

        complete_task.invoke({"task_id": task_a["id"]})

        task_b_fresh = get_task.invoke({"task_id": task_b["id"]})
        assert task_b_fresh["status"] == "pending"


# ── get_task tests ────────────────────────────────────────────────────────────

class TestGetTask:

    def test_get_task_returns_correct_task(self):
        """get_task.invoke() should return the task matching the given id."""
        created = create_task.invoke({"title": "Find me"})

        fetched = get_task.invoke({"task_id": created["id"]})

        assert fetched["id"] == created["id"]
        assert fetched["title"] == "Find me"

    def test_get_task_returns_error_for_missing_id(self):
        """get_task should return an error dict for unknown ids."""
        result = get_task.invoke({"task_id": "no-such-id"})

        assert "error" in result


# ── delete_task tests ─────────────────────────────────────────────────────────

class TestDeleteTask:

    def test_delete_task_removes_it(self):
        """After deleting, get_task should return an error for that id."""
        task = create_task.invoke({"title": "Delete me"})

        result = delete_task.invoke({"task_id": task["id"]})
        fetched = get_task.invoke({"task_id": task["id"]})

        assert result["success"] is True
        assert "error" in fetched  # task is gone

    def test_delete_task_returns_failure_for_missing_id(self):
        """Deleting a non-existent task should return success=False."""
        result = delete_task.invoke({"task_id": "nonexistent"})

        assert result["success"] is False

    def test_delete_does_not_affect_other_tasks(self):
        """Deleting one task should leave other tasks intact."""
        task_a = create_task.invoke({"title": "Task A"})
        task_b = create_task.invoke({"title": "Task B"})

        delete_task.invoke({"task_id": task_a["id"]})

        task_b_fresh = get_task.invoke({"task_id": task_b["id"]})
        assert "error" not in task_b_fresh
        assert task_b_fresh["title"] == "Task B"


# ── Full lifecycle integration test ───────────────────────────────────────────

def test_full_task_lifecycle():
    """
    Integration test: create → list pending → complete → list completed → delete.
    This mirrors the demo described in the plan.
    """
    # Create
    task = create_task.invoke({
        "title": "Full lifecycle task",
        "description": "Tests the whole flow",
        "due_date": "2026-12-01"
    })
    assert task["status"] == "pending"

    # Appears in pending list
    pending = list_tasks.invoke({"status": "pending"})
    assert any(t["id"] == task["id"] for t in pending)

    # Complete it
    completed_task = complete_task.invoke({"task_id": task["id"]})
    assert completed_task["status"] == "completed"

    # No longer in pending, appears in completed
    pending_after = list_tasks.invoke({"status": "pending"})
    completed_after = list_tasks.invoke({"status": "completed"})
    assert not any(t["id"] == task["id"] for t in pending_after)
    assert any(t["id"] == task["id"] for t in completed_after)

    # Delete it
    deleted = delete_task.invoke({"task_id": task["id"]})
    assert deleted["success"] is True

    # Gone from DB
    fetched = get_task.invoke({"task_id": task["id"]})
    assert "error" in fetched


# ── TASK_TOOLS export tests ───────────────────────────────────────────────────

def test_task_tools_list_has_five_tools():
    """TASK_TOOLS should export exactly 5 tools."""
    from tools.task_tools import TASK_TOOLS
    assert len(TASK_TOOLS) == 5


def test_task_tools_are_callable():
    """Each tool in TASK_TOOLS should have .invoke, .name, and .description."""
    from tools.task_tools import TASK_TOOLS
    for tool in TASK_TOOLS:
        assert hasattr(tool, "invoke"), f"{tool} missing .invoke()"
        assert hasattr(tool, "name"), f"{tool} missing .name"
        assert hasattr(tool, "description"), f"{tool} missing .description"
