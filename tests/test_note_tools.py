"""
tests/test_note_tools.py — Tests for the note LangChain tools.

These tests call the tools directly (without an LLM) using tool.invoke({}).
This verifies the tool logic is correct before wiring it to an agent.

The key testing trick here is patching the module-level `db` singleton
in note_tools.py with an in-memory Database instance.

For JS developers: this is equivalent to Jest module mocking —
  jest.mock('./db', () => new InMemoryDatabase())

Run with: pytest tests/test_note_tools.py -v
"""

import pytest
from unittest.mock import patch

from db.database import Database
import tools.note_tools as note_tools_module
from tools.note_tools import create_note, get_note, list_notes, search_notes_by_tag


# ── Fixture ───────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def use_in_memory_db():
    """
    Replace the module-level db singleton with a fresh in-memory instance.

    autouse=True means this fixture runs automatically for every test in
    this file — you don't need to declare it as a parameter.

    The patch() context manager is like jest.spyOn() — it replaces the
    attribute for the duration of the test and restores it afterwards.
    """
    mem_db = Database(":memory:")
    # Patch the `db` name inside the note_tools module
    with patch.object(note_tools_module, "db", mem_db):
        yield mem_db
    mem_db.close()


# ── create_note tests ─────────────────────────────────────────────────────────

class TestCreateNote:

    def test_create_note_returns_dict_with_id(self):
        """create_note.invoke() should return a dict containing an id."""
        result = create_note.invoke({"title": "Test Note", "content": "Hello world"})

        assert isinstance(result, dict)
        assert "id" in result
        assert result["title"] == "Test Note"
        assert result["content"] == "Hello world"

    def test_create_note_without_tags_defaults_to_empty(self):
        """Tags should default to an empty list when not provided."""
        result = create_note.invoke({"title": "No Tags", "content": "Content"})

        assert result["tags"] == []

    def test_create_note_with_comma_separated_tags(self):
        """Tags string 'work,ideas' should be split into ['work', 'ideas']."""
        result = create_note.invoke({
            "title": "Tagged Note",
            "content": "Some content",
            "tags": "work, ideas, python"
        })

        assert result["tags"] == ["work", "ideas", "python"]

    def test_create_note_with_single_tag(self):
        """A single tag with no commas should produce a one-element list."""
        result = create_note.invoke({
            "title": "Single Tag",
            "content": "Content",
            "tags": "work"
        })

        assert result["tags"] == ["work"]

    def test_create_note_strips_whitespace_from_tags(self):
        """Tag whitespace should be stripped: ' work , ideas ' → ['work', 'ideas']."""
        result = create_note.invoke({
            "title": "Whitespace Tags",
            "content": "Content",
            "tags": "  work  ,  ideas  "
        })

        assert result["tags"] == ["work", "ideas"]


# ── get_note tests ────────────────────────────────────────────────────────────

class TestGetNote:

    def test_get_note_returns_correct_note(self):
        """get_note.invoke() should return the note matching the given id."""
        created = create_note.invoke({"title": "Fetchable", "content": "Content"})

        fetched = get_note.invoke({"note_id": created["id"]})

        assert fetched["id"] == created["id"]
        assert fetched["title"] == "Fetchable"

    def test_get_note_returns_error_for_missing_id(self):
        """get_note should return an error dict (not raise) for unknown ids."""
        result = get_note.invoke({"note_id": "non-existent-uuid"})

        assert "error" in result
        assert "non-existent-uuid" in result["error"]


# ── list_notes tests ──────────────────────────────────────────────────────────

class TestListNotes:

    def test_list_notes_returns_list(self):
        """list_notes.invoke() should always return a list."""
        result = list_notes.invoke({})

        assert isinstance(result, list)

    def test_list_notes_returns_empty_when_no_notes(self):
        """With no notes in the DB, list_notes should return an empty list."""
        result = list_notes.invoke({})

        assert result == []

    def test_list_notes_returns_created_notes(self):
        """Created notes should appear in list_notes output."""
        create_note.invoke({"title": "Note A", "content": "Content A"})
        create_note.invoke({"title": "Note B", "content": "Content B"})

        result = list_notes.invoke({})

        assert len(result) == 2
        titles = [n["title"] for n in result]
        assert "Note A" in titles
        assert "Note B" in titles

    def test_list_notes_respects_limit(self):
        """list_notes should not return more than the requested limit."""
        for i in range(5):
            create_note.invoke({"title": f"Note {i}", "content": f"Content {i}"})

        result = list_notes.invoke({"limit": 3})

        assert len(result) == 3


# ── search_notes_by_tag tests ─────────────────────────────────────────────────

class TestSearchNotesByTag:

    def test_search_finds_notes_with_matching_tag(self):
        """search_notes_by_tag should return notes that have the given tag."""
        create_note.invoke({"title": "Work Note", "content": "...", "tags": "work"})
        create_note.invoke({"title": "Personal Note", "content": "...", "tags": "personal"})

        results = search_notes_by_tag.invoke({"tag": "work"})

        assert len(results) == 1
        assert results[0]["title"] == "Work Note"

    def test_search_returns_empty_for_no_match(self):
        """search_notes_by_tag should return empty list when no notes match."""
        create_note.invoke({"title": "A Note", "content": "...", "tags": "other"})

        results = search_notes_by_tag.invoke({"tag": "nonexistent"})

        assert results == []

    def test_search_returns_multiple_matches(self):
        """Multiple notes with the same tag should all be returned."""
        create_note.invoke({"title": "Note 1", "content": "...", "tags": "ai,work"})
        create_note.invoke({"title": "Note 2", "content": "...", "tags": "ai,ideas"})
        create_note.invoke({"title": "Note 3", "content": "...", "tags": "personal"})

        results = search_notes_by_tag.invoke({"tag": "ai"})

        assert len(results) == 2


# ── NOTE_TOOLS export test ────────────────────────────────────────────────────

def test_note_tools_list_has_four_tools():
    """NOTE_TOOLS should export exactly 4 tools."""
    from tools.note_tools import NOTE_TOOLS
    assert len(NOTE_TOOLS) == 4

def test_note_tools_are_callable():
    """Each tool in NOTE_TOOLS should be a LangChain tool (has .invoke)."""
    from tools.note_tools import NOTE_TOOLS
    for tool in NOTE_TOOLS:
        assert hasattr(tool, "invoke"), f"{tool} does not have .invoke()"
        assert hasattr(tool, "name"), f"{tool} does not have .name"
        assert hasattr(tool, "description"), f"{tool} does not have .description"
