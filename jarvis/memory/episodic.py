"""
memory/episodic.py — Episodic memory via conversation summarization.

The Problem:
  Language models have a fixed context window — they can only "see" a limited
  number of tokens at once. A long conversation eventually overflows this window,
  causing the model to forget early messages or fail entirely.

The Solution:
  When a conversation exceeds a threshold (default: 20 messages), compress it:
    1. Summarize everything except the last 4 messages into 3-5 sentences
    2. Store the summary in Chroma as an episodic memory (retrievable in future sessions)
    3. Replace the old messages with: [SystemMessage(summary)] + [last 4 messages]

The result: the conversation stays short but the key context is preserved.
Summaries stored in Chroma are also retrievable in future sessions via semantic
search — connecting episodic memory to long-term memory.

Timeline:
  Session 1: 25 messages → compress → 1 summary + 4 recent messages
                                        └─► stored in Chroma
  Session 2: user asks "what did we do last time?"
             → recall_memories("previous session") → finds the stored summary

For JS developers: think of this like pagination meets caching.
  You keep the "current page" (last 4 messages) and cache the "previous pages"
  (summary) to disk so they're retrievable later.

Usage:
    from memory.episodic import maybe_compress_messages, COMPRESSION_THRESHOLD
"""

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from config import settings

# Compression fires when message count exceeds this threshold
COMPRESSION_THRESHOLD = settings.COMPRESSION_THRESHOLD

# Number of recent messages to keep uncompressed (always in context)
MESSAGES_TO_KEEP = 4

# ── LLM for summarization ─────────────────────────────────────────────────────
# Uses the same model as the agent — no extra setup needed.
_summarization_llm = ChatOllama(
    model=settings.MODEL_NAME,
    base_url=settings.OLLAMA_BASE_URL,
    temperature=0,
)

# ── Summarization prompt ──────────────────────────────────────────────────────
SUMMARIZATION_PROMPT = """You are summarizing a conversation between a user and their AI assistant Jarvis.

Create a concise summary (3-5 sentences) that captures:
- Tasks that were created, completed, or deleted
- Notes that were saved or retrieved
- Personal information the user shared about themselves
- Any preferences or important context mentioned
- Key decisions or conclusions reached

Write in past tense. Be specific about names, IDs, and details.
Do not include pleasantries or filler. Focus only on actionable information."""


async def summarize_messages(messages: list[BaseMessage]) -> str:
    """
    Use the LLM to summarize a list of messages into a few sentences.

    Args:
        messages: The messages to summarize (typically everything except the last few).

    Returns:
        A summary string, or a fallback message if summarization fails.
    """
    if not messages:
        return ""

    # Filter out system messages for the summary input — they're instructions,
    # not conversation content worth summarizing
    content_messages = [
        m for m in messages
        if not isinstance(m, SystemMessage)
    ]

    if not content_messages:
        return ""

    try:
        response = await _summarization_llm.ainvoke(
            [
                SystemMessage(content=SUMMARIZATION_PROMPT),
                *content_messages,
            ]
        )
        return response.content.strip()
    except Exception as e:
        # Return a minimal fallback rather than crashing the whole agent
        return f"[Previous conversation — {len(content_messages)} messages]"


async def maybe_compress_messages(
    messages: list[BaseMessage],
    threshold: int = COMPRESSION_THRESHOLD,
) -> list[BaseMessage]:
    """
    Compress the message list if it exceeds the threshold.

    This is the main entry point. Call it at the start of each graph step
    to keep message lists manageable.

    Args:
        messages:  The current list of messages in the conversation.
        threshold: Message count that triggers compression (default from config).

    Returns:
        Either the original list (if under threshold) or a compressed list:
        [SystemMessage(summary), ...last MESSAGES_TO_KEEP messages]
    """
    if len(messages) <= threshold:
        return messages

    # Split: everything to summarize vs the recent messages to keep
    messages_to_summarize = messages[:-MESSAGES_TO_KEEP]
    recent_messages = messages[-MESSAGES_TO_KEEP:]

    summary_text = await summarize_messages(messages_to_summarize)

    if not summary_text:
        # Summarization failed — return recent messages only to avoid crashing
        return recent_messages

    # Store the summary as an episodic memory in Chroma so it's retrievable
    # in future sessions (connects episodic → long-term memory)
    try:
        from memory.long_term import store_memory
        store_memory(
            f"Conversation summary: {summary_text}",
            metadata={"source": "episodic_compression"},
        )
    except Exception:
        # Memory storage failing should never crash the agent
        pass

    # Return the compressed message list:
    # A single SystemMessage containing the summary + the last few messages
    compressed = [
        SystemMessage(
            content=f"[Previous conversation summary]\n{summary_text}"
        ),
        *recent_messages,
    ]

    return compressed


def needs_compression(
    messages: list[BaseMessage],
    threshold: int = COMPRESSION_THRESHOLD,
) -> bool:
    """
    Check if a message list needs compression without compressing it.

    Useful for conditional logic in graph nodes.

    Args:
        messages:  The message list to check.
        threshold: The compression threshold.

    Returns:
        True if compression should be applied.
    """
    return len(messages) > threshold


def get_compression_stats(messages: list[BaseMessage]) -> dict:
    """
    Return statistics about the current message list.

    Useful for debugging and for displaying compression status in the UI.

    Args:
        messages: The current message list.

    Returns:
        Dict with count, threshold, needs_compression, and estimated token count.
    """
    # Rough token estimate: average 4 chars per token
    total_chars = sum(
        len(m.content) if hasattr(m, "content") and m.content else 0
        for m in messages
    )
    estimated_tokens = total_chars // 4

    return {
        "message_count": len(messages),
        "threshold": COMPRESSION_THRESHOLD,
        "needs_compression": needs_compression(messages),
        "estimated_tokens": estimated_tokens,
        "messages_to_keep": MESSAGES_TO_KEEP,
    }
