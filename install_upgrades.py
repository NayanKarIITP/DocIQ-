"""
install_upgrades.py
────────────────────
Installs all new packages for DocIQ v2 upgrades.
Run from project root with venv activated.

Usage:
    python install_upgrades.py
"""
import subprocess
import sys

def install(package, label=None):
    label = label or package
    print(f"  Installing {label}...", end=" ", flush=True)
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", package, "--quiet"],
        capture_output=True
    )
    if result.returncode == 0:
        print("✓")
    else:
        print(f"⚠ (may already be installed)")

print("=" * 55)
print("  DocIQ v2 — Installing upgrade packages")
print("=" * 55)

print("\n[1/4] LangChain stack...")
install("langchain",              "langchain")
install("langchain-core",         "langchain-core")
install("langchain-google-genai", "langchain-google-genai")
install("langchain-qdrant",       "langchain-qdrant")
install("langchain-community",    "langchain-community")

print("\n[2/4] LangGraph (agentic AI)...")
install("langgraph", "langgraph")
install("duckduckgo-search", "duckduckgo-search (web tool)")

print("\n[3/4] HuggingFace local embedder...")
install("sentence-transformers", "sentence-transformers")
# CPU-only torch (smaller download)
install(
    "torch --index-url https://download.pytorch.org/whl/cpu",
    "torch (CPU-only)"
)

print("\n[4/4] W&B experiment tracking...")
install("wandb", "wandb")

print("""
═══════════════════════════════════════════════
  All packages installed!

  Next steps:
  1. Copy new files to your project:
     - retrieval/langchain_retriever.py
     - retrieval/hf_embedder.py
     - agents/dociq_agent.py  (new folder)
     - agents/__init__.py     (empty file)
     - tracking/wandb_tracker.py (new folder)
     - tracking/__init__.py   (empty file)
     - generation/prompt_engineer.py
     - notebooks/feedback_analysis.py
     - Replace api/main.py
     - Replace ui/app.py

  2. Set up W&B (optional but recommended):
     wandb login
     (paste your key from https://wandb.ai/authorize)
     Add to .env: WANDB_API_KEY=your_key

  3. Restart the API:
     python -m uvicorn api.main:app --reload

  4. Open http://localhost:8501 — you'll see
     the new mode selector in the sidebar!
═══════════════════════════════════════════════
""")
