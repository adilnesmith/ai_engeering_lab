# 08 — Episodic Memory

> How Jarvis handles long conversations without forgetting context.

---

## The Problem: Context Windows

Every language model has a **context window** — a maximum number of tokens it can process at once. With `llama3.2:3b`, this is around 8,000-128,000 tokens depending on the version.

A long conversation in Jarvis easily generates thousands of tokens:
- Each user message: ~20-100 tokens
- Each LLM response: ~50-200 tokens  
- Each tool call + result: ~50-300 tokens

After 20+ exchanges, you're approaching the limit. When the context overflows:
- The model silently drops the oldest messages
- It "forgets" tasks it created earlier
- Responses become inconsistent

---

## The Solution: Compress and Store

When the message count exceeds a threshold (default: 20), Jarvis:

1. **Summarizes** everything except the last 4 messages into 3-5 sentences
2. **Stores** the summary in Chroma as an episodic memory
3. **Replaces** the long message list with: `[summary] + [last 4 messages]`

```
Before compression (22 messages):
  HumanMessage("Hello")
  AIMessage("Hi! How can I help?")
  HumanMessage("Create a task...")
  AIMessage(tool_calls=[create_task(...)])
  ToolMessage({"id": "abc", ...})
  AIMessage("Done! Task created.")
  ... 16 more messages ...
  HumanMessage("What tasks did I create today?")   ← recent
  AIMessage("You created...")                       ← recent
  HumanMessage("Mark the review task as done")     ← recent
  AIMessage(tool_calls=[complete_task(...)])        ← recent

After compression (5 messages):
  SystemMessage("Previous summary: The user created a task 'Review README'
                 and a note 'Today's Standup'. They asked about their pending
                 tasks and completed the review task.")
  HumanMessage("What tasks did I create today?")
  AIMessage("You created...")
  HumanMessage("Mark the review task as done")
  AIMessage(tool_calls=[complete_task(...)])
```

The conversation shrinks from 22 messages to 5 — well within the context window — but the key information is preserved in the summary.

---

## Memory Timeline

```
Session 1
    │
    ├── Messages 1-20: normal conversation
    │
    ├── Message 21: COMPRESSION FIRES
    │       │
    │       ├── Summarize messages 1-17
    │       ├── Store summary in Chroma → "Conversation summary: ..."
    │       └── Keep messages 18-21
    │
    └── Messages 22+: continue from compressed state

Session 2 (new session, fresh thread_id)
    │
    ├── User: "What did we work on last time?"
    │
    └── memory_agent calls recall_memories("previous session")
            │
            └── Chroma returns: "Conversation summary: The user created..."
```

The episodic memory (session summary) feeds into long-term memory — they share the same Chroma vector store. The `source` metadata field distinguishes them:

```python
store_memory(
    f"Conversation summary: {summary_text}",
    metadata={"source": "episodic_compression"}  # vs "direct" for user-told facts
)
```

---

## The Code

### `maybe_compress_messages()`

The main entry point — call it with any message list, get back a (possibly compressed) list:

```python
from memory.episodic import maybe_compress_messages

# If len(messages) > 20 → summarize + compress + store in Chroma
# If len(messages) <= 20 → return unchanged
compressed = await maybe_compress_messages(messages)
```

### Where It Fires in the Graph

The `compress_node` is wired into the supervisor graph as the entry point and after every specialist returns:

```
START → compress → supervisor → specialist → compress → supervisor → ... → finish
```

This means compression is checked:
- At the start of each user message
- After each specialist completes its work

The `compress_node` is a no-op (returns unchanged state) when under threshold — so there's no performance cost on short conversations.

### The Summarization Prompt

```
Create a concise summary (3-5 sentences) that captures:
- Tasks that were created, completed, or deleted
- Notes that were saved or retrieved
- Personal information the user shared
- Any preferences or important context mentioned
- Key decisions or conclusions reached
```

The prompt is designed to extract only **actionable information** — not pleasantries or filler. A good summary means the agent can continue as if it had read the full history.

---

## Seeing It in Action

In the CLI or Gradio UI, compression is logged:

```
You: [message 21]

[Episodic] Compressed 22 → 5 messages
[Supervisor] → routing to: task_agent
[task_agent] ...
[Supervisor] → FINISH

Jarvis: Done! ...
```

After the session, the summary is queryable:

```
You: Do you remember anything from our previous conversations?

[Supervisor] → routing to: memory_agent
[memory_agent] calls recall_memories("previous conversations")

Jarvis: Yes! From our last session: you created a "Review README" task,
        saved a "Today's Standup" note, and completed the review task.
```

---

## Configuration

Control compression via `.env`:

```bash
# Number of messages before compression kicks in
COMPRESSION_THRESHOLD=20    # default

# Set higher for more context retention (uses more memory)
COMPRESSION_THRESHOLD=40

# Set lower for very aggressive compression
COMPRESSION_THRESHOLD=10
```

---

## Connection to the Three-Tier Memory System

| Memory type | Technology | Filled by |
|------------|------------|-----------|
| Short-term | LangGraph state | Every message |
| Long-term | Chroma (source: "direct") | `remember_fact` tool |
| **Episodic** | **Chroma (source: "episodic_compression")** | **Automatic compression** |

All three are retrievable by the `memory_agent` via the `recall_memories` tool — it searches Chroma regardless of source. The three types complement each other:

- Short-term: what we're talking about right now
- Episodic: compressed record of what we discussed in the past
- Long-term: explicit facts the user told Jarvis directly
