"""
tests/test_memory.py — Tests for the long-term memory module.

These tests use a temporary directory (tmp_path pytest fixture) for the
Chroma DB so they don't pollute the real data directory and are cleaned
up automatically after each test.

Note: These tests require Ollama + nomic-embed-text to be running.
They are marked with pytest.mark.integration so you can skip them
in CI environments without Ollama:

    pytest tests/test_memory.py                    # run all
    pytest tests/test_memory.py -m "not integration"  # skip integration tests

Run with: pytest tests/test_memory.py -v
"""

import pytest
from unittest.mock import patch, MagicMock

import memory.long_term as long_term_module
from memory.long_term import store_memory, retrieve_memories, format_memories_for_context


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_vectorstore():
    """
    Mock the Chroma vectorstore for unit tests that don't need a real embedding model.

    This patches the module-level vectorstore with a mock object, letting us
    test the logic of store_memory and retrieve_memories without needing
    Ollama running.
    """
    mock_vs = MagicMock()
    with patch.object(long_term_module, "vectorstore", mock_vs):
        yield mock_vs


# ── store_memory tests ────────────────────────────────────────────────────────

class TestStoreMemory:

    def test_store_memory_calls_add_documents(self, mock_vectorstore):
        """store_memory should call vectorstore.add_documents with a Document."""
        mock_vectorstore.add_documents.return_value = ["doc-id-1"]

        result = store_memory("The user's name is Alex")

        mock_vectorstore.add_documents.assert_called_once()
        assert result["success"] is True
        assert result["fact"] == "The user's name is Alex"

    def test_store_memory_returns_id(self, mock_vectorstore):
        """store_memory should return the document id from add_documents."""
        mock_vectorstore.add_documents.return_value = ["test-id-123"]

        result = store_memory("User prefers Python over JavaScript")

        assert result["id"] == "test-id-123"

    def test_store_memory_strips_whitespace(self, mock_vectorstore):
        """Leading/trailing whitespace should be stripped before storing."""
        mock_vectorstore.add_documents.return_value = ["id-1"]

        result = store_memory("  Some fact with spaces  ")

        assert result["fact"] == "Some fact with spaces"

    def test_store_memory_rejects_empty_string(self, mock_vectorstore):
        """Empty strings should not be stored."""
        result = store_memory("")

        mock_vectorstore.add_documents.assert_not_called()
        assert result["success"] is False
        assert "error" in result

    def test_store_memory_rejects_whitespace_only(self, mock_vectorstore):
        """Whitespace-only strings should not be stored."""
        result = store_memory("   ")

        mock_vectorstore.add_documents.assert_not_called()
        assert result["success"] is False

    def test_store_memory_includes_timestamp_in_metadata(self, mock_vectorstore):
        """Stored documents should have a stored_at timestamp in metadata."""
        mock_vectorstore.add_documents.return_value = ["id-1"]
        captured_docs = []

        def capture_docs(docs):
            captured_docs.extend(docs)
            return ["id-1"]

        mock_vectorstore.add_documents.side_effect = capture_docs

        store_memory("Some fact")

        assert len(captured_docs) == 1
        assert "stored_at" in captured_docs[0].metadata

    def test_store_memory_accepts_custom_metadata(self, mock_vectorstore):
        """Custom metadata should be merged with the base metadata."""
        mock_vectorstore.add_documents.return_value = ["id-1"]
        captured_docs = []

        def capture_docs(docs):
            captured_docs.extend(docs)
            return ["id-1"]

        mock_vectorstore.add_documents.side_effect = capture_docs

        store_memory("A fact", metadata={"source": "conversation", "session": "abc"})

        assert captured_docs[0].metadata["source"] == "conversation"
        assert captured_docs[0].metadata["session"] == "abc"


# ── retrieve_memories tests ───────────────────────────────────────────────────

class TestRetrieveMemories:

    def test_retrieve_memories_returns_list_of_strings(self, mock_vectorstore):
        """retrieve_memories should return a list of strings."""
        from langchain_core.documents import Document
        mock_vectorstore.similarity_search.return_value = [
            Document(page_content="The user's name is Alex"),
            Document(page_content="User is a software engineer"),
        ]

        results = retrieve_memories("what do you know about me?")

        assert isinstance(results, list)
        assert all(isinstance(r, str) for r in results)

    def test_retrieve_memories_returns_correct_content(self, mock_vectorstore):
        """retrieve_memories should return the page_content of matching documents."""
        from langchain_core.documents import Document
        mock_vectorstore.similarity_search.return_value = [
            Document(page_content="User prefers morning meetings"),
        ]

        results = retrieve_memories("meetings preferences")

        assert "User prefers morning meetings" in results

    def test_retrieve_memories_passes_k_to_similarity_search(self, mock_vectorstore):
        """The k parameter should be forwarded to similarity_search."""
        mock_vectorstore.similarity_search.return_value = []

        retrieve_memories("some query", k=3)

        mock_vectorstore.similarity_search.assert_called_once_with("some query", k=3)

    def test_retrieve_memories_returns_empty_on_exception(self, mock_vectorstore):
        """Should return empty list if Chroma raises (e.g. empty collection)."""
        mock_vectorstore.similarity_search.side_effect = Exception("Collection is empty")

        results = retrieve_memories("some query")

        assert results == []

    def test_retrieve_memories_returns_empty_when_no_docs(self, mock_vectorstore):
        """Should return empty list when similarity_search returns no results."""
        mock_vectorstore.similarity_search.return_value = []

        results = retrieve_memories("obscure query")

        assert results == []


# ── format_memories_for_context tests ────────────────────────────────────────

class TestFormatMemoriesForContext:

    def test_format_returns_empty_string_when_no_memories(self, mock_vectorstore):
        """Should return empty string when no memories exist."""
        mock_vectorstore.similarity_search.return_value = []

        result = format_memories_for_context("some query")

        assert result == ""

    def test_format_includes_numbered_memories(self, mock_vectorstore):
        """Formatted output should number each memory."""
        from langchain_core.documents import Document
        mock_vectorstore.similarity_search.return_value = [
            Document(page_content="Memory one"),
            Document(page_content="Memory two"),
        ]

        result = format_memories_for_context("some query")

        assert "1." in result
        assert "2." in result
        assert "Memory one" in result
        assert "Memory two" in result

    def test_format_includes_header(self, mock_vectorstore):
        """Formatted output should have a header line."""
        from langchain_core.documents import Document
        mock_vectorstore.similarity_search.return_value = [
            Document(page_content="Some fact"),
        ]

        result = format_memories_for_context("query")

        assert "remember" in result.lower()


# ── Tool wrapper tests ────────────────────────────────────────────────────────

class TestMemoryTools:

    def test_remember_fact_tool_exists(self):
        """remember_fact should be a LangChain tool with name and description."""
        from memory.long_term import remember_fact
        assert hasattr(remember_fact, "invoke")
        assert hasattr(remember_fact, "name")
        assert hasattr(remember_fact, "description")

    def test_recall_memories_tool_exists(self):
        """recall_memories should be a LangChain tool with name and description."""
        from memory.long_term import recall_memories
        assert hasattr(recall_memories, "invoke")
        assert hasattr(recall_memories, "name")
        assert hasattr(recall_memories, "description")

    def test_memory_tools_list_has_two_tools(self):
        """MEMORY_TOOLS should export exactly 2 tools."""
        from memory.long_term import MEMORY_TOOLS
        assert len(MEMORY_TOOLS) == 2

    def test_recall_memories_returns_no_memories_message(self, mock_vectorstore):
        """recall_memories tool should return a friendly message when empty."""
        from memory.long_term import recall_memories
        mock_vectorstore.similarity_search.return_value = []

        result = recall_memories.invoke({"query": "what do you know about me?"})

        assert "don't have" in result.lower() or "no" in result.lower()

    def test_recall_memories_formats_results(self, mock_vectorstore):
        """recall_memories tool should format found memories into a readable string."""
        from langchain_core.documents import Document
        from memory.long_term import recall_memories
        mock_vectorstore.similarity_search.return_value = [
            Document(page_content="User is named Alex"),
        ]

        result = recall_memories.invoke({"query": "user name"})

        assert "Alex" in result
        assert isinstance(result, str)
