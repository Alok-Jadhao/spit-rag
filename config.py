"""Central configuration. Override anything with environment variables."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ---- knowledge-base layout (Task 1) ----
KB_DIR = ROOT / "knowledge_base"
WEB_DIR = KB_DIR / "website_pages"
PDF_DIR = KB_DIR / "institutional_pdfs"
ERP_DIR = KB_DIR / "erp_export"          # put ma'am's anonymized ERP export here
META_CSV = KB_DIR / "metadata.csv"       # source name + source URL for every file

INDEX_DIR = ROOT / "index"
RESULTS_DIR = ROOT / "results"
DISTRACTOR_DIR = ROOT / "distractors"    # for Experiment C
for _d in (WEB_DIR, PDF_DIR, ERP_DIR, INDEX_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---- chunking / embeddings ----
# bge-small accepts 512 tokens, so 400-600 token chunks are NOT truncated.
# (all-MiniLM-L6-v2 was trained on 256 tokens - good as the "second embedding model" for the advanced challenge.)
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text:latest")
# Accept a plain Ollama tag or the old invalid HF-style alias from earlier setup attempts.
if EMBED_MODEL.lower().startswith("sentence-transformers/nomic-embed-text"):
    EMBED_MODEL = "ollama:" + EMBED_MODEL.rsplit("/", 1)[-1]
elif "/" not in EMBED_MODEL and (":" in EMBED_MODEL or EMBED_MODEL.lower().startswith("nomic-embed-text")):
    EMBED_MODEL = "ollama:" + EMBED_MODEL
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
ALT_EMBED_MODEL = os.getenv("ALT_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_TOKENS = int(os.getenv("CHUNK_TOKENS", 450))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 60))
CHUNK_STRATEGY = os.getenv("CHUNK_STRATEGY", "sentence")   # "sentence" | "fixed"
DEFAULT_K = 3

# ---- generator LLM ----
# LLM_PROVIDER = "anthropic"  -> needs ANTHROPIC_API_KEY
# LLM_PROVIDER = "openai"     -> any OpenAI-compatible server:
#     OpenAI:  OPENAI_API_KEY (+ LLM_MODEL=gpt-4o-mini or similar)
#     Ollama:  OPENAI_BASE_URL=http://localhost:11434/v1  LLM_MODEL=gemma4:e2b
#     Use whichever your instructor has approved.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
LLM_MODEL = os.getenv("LLM_MODEL", "gemma4:e2b")
LLM_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
# Ollama does not require a real API key; rag.py uses the placeholder "ollama".
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", 500))
