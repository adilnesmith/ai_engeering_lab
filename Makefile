# ─────────────────────────────────────────────────────────────────────────────
# Jarvis — Makefile
#
# Usage: make <target>
#
# Targets:
#   make install   Install all dependencies
#   make health    Verify Ollama + models are ready
#   make run       Start the Gradio web UI
#   make cli       Run the multi-agent CLI (no browser)
#   make test      Run the full test suite
#   make test-fast Run tests that don't need Ollama (unit tests only)
#   make clean     Remove generated data and cache
#   make freeze    Export requirements.txt from current environment
#   make help      Show this help
# ─────────────────────────────────────────────────────────────────────────────

.PHONY: install health run cli test test-fast clean freeze help

# ── Install ───────────────────────────────────────────────────────────────────
install:
	@echo "📦 Installing Jarvis dependencies..."
	pip install -e ".[dev]"
	@echo ""
	@echo "✅ Done. Next steps:"
	@echo "   1. Install Ollama: https://ollama.com/download"
	@echo "   2. ollama pull llama3.2:3b"
	@echo "   3. ollama pull nomic-embed-text"
	@echo "   4. make health"

# ── Health check ──────────────────────────────────────────────────────────────
health:
	@echo "🔍 Running Jarvis health check..."
	python health_check.py

# ── Run web UI ────────────────────────────────────────────────────────────────
run:
	@echo "🚀 Starting Jarvis web UI..."
	@echo "   Open your browser at http://localhost:7860"
	@echo "   Press Ctrl+C to stop"
	python ui/app.py

# ── Run CLI ───────────────────────────────────────────────────────────────────
cli:
	@echo "🤖 Starting Jarvis multi-agent CLI..."
	python agents/supervisor.py

# ── Tests ─────────────────────────────────────────────────────────────────────
test:
	@echo "🧪 Running full test suite..."
	pytest tests/ -v

test-fast:
	@echo "🧪 Running unit tests (no Ollama required)..."
	pytest tests/test_database.py tests/test_note_tools.py tests/test_task_tools.py tests/test_memory.py -v

# ── Freeze requirements ───────────────────────────────────────────────────────
freeze:
	@echo "📄 Exporting requirements.txt..."
	pip freeze > requirements.txt
	@echo "✅ requirements.txt updated"

# ── Clean ─────────────────────────────────────────────────────────────────────
clean:
	@echo "🧹 Cleaning generated files..."
	@echo "   ⚠️  This will delete your local notes, tasks, and memories."
	@if exist data rmdir /s /q data
	@if exist .pytest_cache rmdir /s /q .pytest_cache
	@for /r . %%d in (__pycache__) do @if exist "%%d" rmdir /s /q "%%d"
	@for /r . %%f in (*.pyc) do @if exist "%%f" del /q "%%f"
	@echo "✅ Clean complete."

# ── Help ──────────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  🤖 Jarvis — Personal AI Assistant"
	@echo "  ════════════════════════════════"
	@echo ""
	@echo "  make install     Install all Python dependencies"
	@echo "  make health      Verify Ollama + models are ready"
	@echo "  make run         Start the Gradio web UI (http://localhost:7860)"
	@echo "  make cli         Run the multi-agent CLI in terminal"
	@echo "  make test        Run all tests with pytest"
	@echo "  make test-fast   Run unit tests only (no Ollama needed)"
	@echo "  make freeze      Export requirements.txt"
	@echo "  make clean       Remove data/ and cache (⚠️ deletes your data)"
	@echo "  make help        Show this help"
	@echo ""
	@echo "  First time? Run:  make install  then  make health  then  make run"
	@echo ""
