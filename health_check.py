"""
health_check.py — Verify Jarvis dependencies are ready before running.

This script checks:
  1. Ollama HTTP server is reachable
  2. The configured chat model (llama3.2:3b) responds to a test prompt
  3. The configured embedding model (nomic-embed-text) is available

Run this before starting Jarvis for the first time:
    python health_check.py

Or via Makefile:
    make health
"""

import sys
import asyncio
import urllib.request
import urllib.error
import json

from config import settings


# ── ANSI colour helpers (work on Windows PowerShell ≥ 5.1) ──────────────────
def green(text: str) -> str:
    return f"\033[92m{text}\033[0m"


def red(text: str) -> str:
    return f"\033[91m{text}\033[0m"


def yellow(text: str) -> str:
    return f"\033[93m{text}\033[0m"


def bold(text: str) -> str:
    return f"\033[1m{text}\033[0m"


# ── Individual checks ────────────────────────────────────────────────────────

def check_ollama_running() -> bool:
    """Ping the Ollama health endpoint."""
    url = f"{settings.OLLAMA_BASE_URL}/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            if resp.status == 200:
                print(green(f"✅ Ollama is running at {settings.OLLAMA_BASE_URL}"))
                return True
    except urllib.error.URLError as e:
        print(red(f"❌ Cannot reach Ollama at {settings.OLLAMA_BASE_URL}"))
        print(yellow(f"   Error: {e}"))
        print(yellow("   Make sure Ollama is installed and running:"))
        print(yellow("   → https://ollama.com/download"))
        print(yellow("   → Run: ollama serve"))
        return False


def check_model_available(model_name: str) -> bool:
    """Check if a model is already pulled in Ollama."""
    url = f"{settings.OLLAMA_BASE_URL}/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read())
            pulled_models = [m["name"].split(":")[0] for m in data.get("models", [])]
            model_base = model_name.split(":")[0]
            if model_base in pulled_models or model_name in [m["name"] for m in data.get("models", [])]:
                return True
            return False
    except Exception:
        return False


async def check_chat_model() -> bool:
    """Send a test prompt to the chat model and verify it responds."""
    try:
        from langchain_ollama import ChatOllama
        from langchain_core.messages import HumanMessage

        print(f"\n   Testing chat model: {bold(settings.MODEL_NAME)}")

        # Check if model is pulled
        if not check_model_available(settings.MODEL_NAME):
            print(yellow(f"   ⚠️  Model '{settings.MODEL_NAME}' not found locally."))
            print(yellow(f"   Run: ollama pull {settings.MODEL_NAME}"))
            print(yellow("   (This may take a few minutes on first run)"))
            return False

        llm = ChatOllama(
            model=settings.MODEL_NAME,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=0,
        )
        response = await llm.ainvoke([HumanMessage(content="Say only: Hello! in your response.")])
        reply = response.content.strip()[:60]  # truncate for display
        print(green(f"✅ Model {settings.MODEL_NAME} responded: \"{reply}\""))
        return True

    except ImportError:
        print(red("❌ langchain-ollama not installed. Run: make install"))
        return False
    except Exception as e:
        print(red(f"❌ Chat model check failed: {e}"))
        return False


async def check_embed_model() -> bool:
    """Verify the embedding model is available."""
    try:
        if not check_model_available(settings.EMBED_MODEL):
            print(yellow(f"\n   ⚠️  Embedding model '{settings.EMBED_MODEL}' not found locally."))
            print(yellow(f"   Run: ollama pull {settings.EMBED_MODEL}"))
            print(yellow("   Long-term memory features will not work without this model."))
            return False

        print(green(f"✅ Embedding model {settings.EMBED_MODEL} is available"))
        return True

    except Exception as e:
        print(red(f"❌ Embedding model check failed: {e}"))
        return False


def check_python_version() -> bool:
    """Ensure Python 3.11+ is being used."""
    major, minor = sys.version_info[:2]
    if major >= 3 and minor >= 11:
        print(green(f"✅ Python {major}.{minor} — OK (3.11+ required)"))
        return True
    else:
        print(red(f"❌ Python {major}.{minor} — too old. Jarvis requires Python 3.11+"))
        return False


def check_dependencies() -> bool:
    """Check that all required packages are importable."""
    packages = {
        "langchain": "langchain",
        "langchain_ollama": "langchain-ollama",
        "langchain_chroma": "langchain-chroma",
        "langgraph": "langgraph",
        "chromadb": "chromadb",
        "gradio": "gradio",
        "pydantic": "pydantic",
    }
    all_ok = True
    missing = []
    for module, pkg_name in packages.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(pkg_name)
            all_ok = False

    if all_ok:
        print(green("✅ All Python dependencies are installed"))
    else:
        print(red(f"❌ Missing packages: {', '.join(missing)}"))
        print(yellow("   Run: make install   (or: pip install -e .[dev])"))
    return all_ok


# ── Main ─────────────────────────────────────────────────────────────────────

async def main() -> None:
    print(bold("\n🤖 Jarvis Health Check\n") + "─" * 40)

    results = []

    # 1. Python version
    results.append(check_python_version())

    # 2. Dependencies
    results.append(check_dependencies())

    # 3. Ollama server
    ollama_ok = check_ollama_running()
    results.append(ollama_ok)

    if ollama_ok:
        # 4. Chat model
        results.append(await check_chat_model())

        # 5. Embedding model
        results.append(await check_embed_model())

    print("\n" + "─" * 40)

    if all(results):
        print(green(bold("🎉 Jarvis is ready to go!")))
        print(f"   Run: {bold('make run')}   to start the web UI")
        print(f"   Run: {bold('make test')}  to run the test suite\n")
    else:
        failed = results.count(False)
        print(red(bold(f"⚠️  {failed} check(s) failed. See messages above.")))
        print(yellow("   Fix the issues above, then re-run: make health\n"))
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
