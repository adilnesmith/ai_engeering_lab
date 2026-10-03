# 05 — Memory System

> How Jarvis remembers things — across a conversation, and across sessions.

---

## Three Types of Memory

Jarvis has three distinct memory systems, each serving a different purpose:

| Type | What it stores | Lifespan | Technology | Analogy |
|------|---------------|----------|------------|---------|
| **Short-term** | All messages in the current conversation | Single session | LangGraph state + SqliteSaver | RAM — fast, temporary |
| **Long-term** | Facts about the user and their preferences | Forever | Chroma vector DB + embeddings | A notebook the assistant keeps |
| **Episodic** | Summaries of past conversations | Forever | Chroma vector DB + embeddings | Meeting minutes — compressed records |

---

## Short-term Memory

This is just the `messages` list in `AgentState`. Every `HumanMessage`, `AIMessage`, and `ToolMessage` is appended to this list as the conversation progresses.

`SqliteSaver` persists this list to SQLite after every step, keyed by `thread_id`. Starting a new session with the same `thread_id` restores the full history.

```python
# The whole conversation is in state["messages"]
state["messages"] = [
    HumanMessage("Create a task to review the docs"),
    AIMessage(tool_calls=[{"name": "create_task", ...}]),
    ToolMessage(content='{"id": "abc", "status": "pending"}'),
    AIMessage("Done! Task created."),
    HumanMessage("What was the last task I created?"),
    # LLM reads all of the above to answer this question
]
```

No special retrieval needed — the full history is always in context. This is why it's "short-term": it's bounded by the model's context window. When conversations get too long, episodic memory (Task 9) compresses them.

---

## Long-term Memory

Long-term memory answers the question: *"What do you remember about me from previous sessions?"*

### How Vector Embeddings Work

An **embedding model** converts any piece of text into a list of numbers (a vector). The key property: **similar texts produce numerically close vectors**.

```
"The user is a software engineer"  →  [0.23, -0.87, 0.41, ...]  (768 numbers)
"Alex works in software development" →  [0.21, -0.85, 0.44, ...]  (numerically close!)
"I like pizza"                      →  [-0.92, 0.31, -0.67, ...]  (far away)
```

When you query "what does the user do for work?", its vector is also computed and the database finds the stored vectors that are closest — returning semantically related memories even when the exact words differ.

For JS developers: this is like full-text search, but semantic rather than keyword-based. Imagine a `localStorage.search("work")` that returns "software engineer", "developer", and "programmer" entries because they're all related concepts — not because they contain the literal word "work".

### The Two Models

Jarvis uses **two separate Ollama models**:

```
Chat model (llama3.2:3b)    → Generates text responses, decides which tools to call
Embedding model (nomic-embed-text) → Converts text to vectors for semantic search
```

They do fundamentally different jobs. The embedding model is smaller, faster, and purpose-built for this task.

### Storing a Memory

```python
from memory.long_term import store_memory

store_memory("The user's name is Alex and they are transitioning from JS to Python AI development")
# → Embeds the text and saves to Chroma with a timestamp
# → {"success": True, "id": "chroma-doc-id", "fact": "..."}
```

Under the hood:
1. `embeddings.embed_query(fact)` → turns text into a `[0.23, -0.87, ...]` vector
2. Chroma stores: `{ vector: [...], content: fact, metadata: {stored_at: ...} }`

### Retrieving Memories

```python
from memory.long_term import retrieve_memories

memories = retrieve_memories("what do you know about the user?", k=5)
# → Chroma finds the 5 stored vectors closest to the query vector
# → ["The user's name is Alex...", "Alex prefers concise responses", ...]
```

### As Agent Tools

The memory functions are also wrapped as `@tool` so the memory agent can call them:

```python
remember_fact.invoke({"fact": "User prefers morning standups"})
# → Stores the fact permanently in Chroma

recall_memories.invoke({"query": "user preferences"})
# → Returns a formatted string of relevant memories
```

---

## Why Chroma?

Chroma is a local, file-based vector database. Like SQLite for relational data, it needs no server and stores everything in a local directory (`data/chroma_db/`).

Alternatives considered:

| Option | Pros | Cons |
|--------|------|------|
| **Chroma** ✅ | Local, zero-config, file-based | Not for high scale |
| Pinecone | Managed cloud service | Requires API key, costs money |
| pgvector | SQL + vectors together | Requires PostgreSQL running |
| FAISS | Fast, mature | No persistence without extra work |

Chroma is the right choice for a local personal assistant.

---

## Memory Data Flow

```
User says: "My name is Alex"
                │
                ▼
    [Memory Agent]  decides to call: remember_fact("User's name is Alex")
                │
                ▼
    OllamaEmbeddings → [0.23, -0.87, ...]
                │
                ▼
    Chroma stores: { vector, content: "User's name is Alex", stored_at: ... }


New session — User asks: "Do you remember my name?"
                │
                ▼
    [Memory Agent]  decides to call: recall_memories("user name")
                │
                ▼
    OllamaEmbeddings → query vector for "user name"
                │
                ▼
    Chroma finds closest stored vectors → ["User's name is Alex", ...]
                │
                ▼
    Agent responds: "Yes! Your name is Alex."
```

---

## Testing Without Ollama

The memory tests use `MagicMock` to replace the Chroma vectorstore, so they run without needing Ollama:

```python
@pytest.fixture
def mock_vectorstore():
    mock_vs = MagicMock()
    with patch.object(long_term_module, "vectorstore", mock_vs):
        yield mock_vs

def test_store_memory(mock_vectorstore):
    mock_vectorstore.add_documents.return_value = ["doc-id-1"]
    result = store_memory("The user's name is Alex")
    assert result["success"] is True
```

`MagicMock` is Python's equivalent of `jest.fn()` — a fake object that records calls and lets you control return values.

---

## Running the Tests

```bash
pytest tests/test_memory.py -v
```

All tests use mocks so they run without Ollama. Expected output:
```
tests/test_memory.py::TestStoreMemory::test_store_memory_calls_add_documents PASSED
tests/test_memory.py::TestStoreMemory::test_store_memory_rejects_empty_string PASSED
...
tests/test_memory.py::TestMemoryTools::test_memory_tools_list_has_two_tools PASSED

18 passed in 0.21s
```
