"""
memory/long_term.py — Long-term semantic memory using Chroma + embeddings.

This module gives Jarvis the ability to remember facts about the user across
completely separate sessions. Unlike short-term memory (conversation messages
stored in SqliteSaver), long-term memory persists indefinitely and is
retrieved by semantic similarity — meaning you find related memories even
when the exact words don't match.

How it works:
  1. A fact string is converted to a vector (list of ~768 numbers) by the
     embedding model. Similar sentences produce numerically close vectors.
  2. The vector + original text is stored in Chroma (a local vector database).
  3. On retrieval, the query is also embedded and Chroma finds the closest
     stored vectors — returning the most semantically relevant memories.

Example:
  Store: "The user's name is Alex and they work as a software engineer"
  Query: "what do I know about the user?"
  → Returns the stored fact (semantically similar, even though the words differ)

For JS developers: this is like a smarter version of localStorage.
  - Regular localStorage: exact key lookup → getItem("username")
  - Vector memory: semantic similarity → "find anything related to the user's job"

Usage:
    from memory.long_term import store_memory, retrieve_memories, MEMORY_TOOLS
"""

from datetime import datetime, timezone

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_ollama import OllamaEmbeddings

from config import settings

# ── Embedding model setup ─────────────────────────────────────────────────────
# OllamaEmbeddings converts text → vectors using the nomic-embed-text model.
# This is a DIFFERENT model from the chat model — it's optimised specifically
# for creating embeddings, not for conversation.
#
# Run: ollama pull nomic-embed-text   (one-time setup)

embeddings = OllamaEmbeddings(
    model=settings.EMBED_MODEL,
    base_url=settings.OLLAMA_BASE_URL,
)

# ── Chroma vector store ───────────────────────────────────────────────────────
# Chroma is a local vector database that persists to disk.
# persist_directory tells it where to store the index files.
# collection_name namespaces the data (like a table name in SQL).

vectorstore = Chroma(
    collection_name="long_term_memory",
    embedding_function=embeddings,
    persist_directory=settings.CHROMA_PATH,
)


# ── Core functions ────────────────────────────────────────────────────────────

def store_memory(fact: str, metadata: dict | None = None) -> dict:
    """
    Embed a fact string and store it in the Chroma vector store.

    Args:
        fact:     The text to store (e.g. "User prefers morning meetings").
        metadata: Optional extra data to attach (timestamps, source, etc.)

    Returns:
        A dict confirming storage with the document id.
    """
    if not fact or not fact.strip():
        return {"success": False, "error": "Cannot store an empty memory."}

    base_metadata = {
        "stored_at": datetime.now(timezone.utc).isoformat(),
        "source": metadata.get("source", "direct") if metadata else "direct",
    }
    if metadata:
        base_metadata.update(metadata)

    # Document is LangChain's wrapper for a piece of text + metadata
    doc = Document(page_content=fact.strip(), metadata=base_metadata)

    # add_documents embeds + stores in one call
    ids = vectorstore.add_documents([doc])

    return {
        "success": True,
        "id": ids[0] if ids else None,
        "fact": fact.strip(),
    }


def retrieve_memories(query: str, k: int = 5) -> list[str]:
    """
    Retrieve the most semantically relevant memories for a query.

    Args:
        query: A natural language query (e.g. "what does the user do for work?").
        k:     Maximum number of memories to return (default 5).

    Returns:
        A list of fact strings, most relevant first.
        Returns empty list if no memories are stored yet.
    """
    try:
        # similarity_search embeds the query and finds the k closest vectors
        docs = vectorstore.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]
    except Exception:
        # Chroma raises if the collection is empty — handle gracefully
        return []


def format_memories_for_context(query: str, k: int = 5) -> str:
    """
    Retrieve memories and format them as a context string for the system prompt.

    Args:
        query: The user's current message (used to find relevant memories).
        k:     Number of memories to retrieve.

    Returns:
        A formatted string ready to inject into the system prompt,
        or empty string if no relevant memories exist.
    """
    memories = retrieve_memories(query, k=k)
    if not memories:
        return ""

    lines = ["Relevant things I remember about you:"]
    for i, memory in enumerate(memories, 1):
        lines.append(f"  {i}. {memory}")

    return "\n".join(lines)


# ── LangChain tool wrappers ───────────────────────────────────────────────────
# These wrap the core functions as @tool so the memory agent can call them.

@tool
def remember_fact(fact: str) -> dict:
    """
    Store an important fact about the user for future reference.

    Use this when the user shares personal information, preferences, or context
    about themselves that would be useful to remember in future conversations.

    Examples of things worth remembering:
    - "My name is Alex"
    - "I prefer concise bullet-point responses"
    - "I'm building a personal AI assistant project"
    - "I work as a software engineer at Acme Corp"

    Args:
        fact: A clear, self-contained sentence describing what to remember.
              Write it in a way that makes sense read back in future sessions.

    Returns:
        A dict confirming the memory was stored.
    """
    return store_memory(fact)


@tool
def recall_memories(query: str) -> str:
    """
    Recall relevant facts and memories related to a query.

    Use this when the user asks about something you should remember,
    or when context from past sessions would help give a better response.

    Examples of when to use:
    - "Do you remember my name?"
    - "What do you know about me?"
    - "What are my preferences?"

    Args:
        query: A natural language question or topic to search memories for.

    Returns:
        A formatted string of relevant memories, or a message indicating
        no relevant memories were found.
    """
    memories = retrieve_memories(query, k=5)
    if not memories:
        return "I don't have any relevant memories stored yet."

    lines = ["Here's what I remember:"]
    for i, memory in enumerate(memories, 1):
        lines.append(f"  {i}. {memory}")
    return "\n".join(lines)


# ── Exports ───────────────────────────────────────────────────────────────────
MEMORY_TOOLS = [remember_fact, recall_memories]
