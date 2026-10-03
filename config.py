"""
config.py — Centralised configuration for Jarvis.

All settings are read from environment variables with sensible defaults.
This is the single source of truth for paths, model names, and thresholds.

For JS developers: this is the equivalent of a .env + dotenv setup you'd
use in a Node project. Python's os.getenv() maps directly to process.env
in Node — same concept, different syntax.

Usage:
    from config import settings
    print(settings.MODEL_NAME)
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file if it exists (won't overwrite real env vars already set)
load_dotenv()

# ── Base directory ──────────────────────────────────────────────────────────
# Resolve to the directory this file lives in, so relative paths work
# regardless of where you run the script from.
BASE_DIR = Path(__file__).parent


class Settings:
    """
    Jarvis configuration settings.

    All values can be overridden via environment variables (see .env.example).
    Defaults are designed for local CPU-only use with Ollama.
    """

    # ── Ollama / LLM ────────────────────────────────────────────────────────
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # The main chat model. llama3.2:3b is the recommended starting point:
    # fast on CPU, supports tool-calling reliably, ~3 GB RAM.
    MODEL_NAME: str = os.getenv("MODEL_NAME", "llama3.2:3b")

    # Dedicated embedding model — separate from the chat model.
    # Used for converting text to vectors for semantic search.
    EMBED_MODEL: str = os.getenv("EMBED_MODEL", "nomic-embed-text")

    # ── Storage paths ────────────────────────────────────────────────────────
    # SQLite database for notes and tasks
    DB_PATH: str = os.getenv("DB_PATH", str(BASE_DIR / "data" / "jarvis.db"))

    # Chroma vector store for long-term and episodic memory
    CHROMA_PATH: str = os.getenv("CHROMA_PATH", str(BASE_DIR / "data" / "chroma_db"))

    # ── Agent behaviour ──────────────────────────────────────────────────────
    # How many messages in a conversation before automatic summarization fires
    COMPRESSION_THRESHOLD: int = int(os.getenv("COMPRESSION_THRESHOLD", "20"))

    def ensure_data_dirs(self) -> None:
        """Create data directories if they don't exist yet."""
        Path(self.DB_PATH).parent.mkdir(parents=True, exist_ok=True)
        Path(self.CHROMA_PATH).mkdir(parents=True, exist_ok=True)


# Singleton — import this everywhere: `from config import settings`
settings = Settings()

# Make sure data/ exists as soon as config is imported
settings.ensure_data_dirs()
