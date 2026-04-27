"""
All settings are loaded from .env (never hardcoded here).
Copy .env.example → .env and fill in your values.
"""

import os
from dotenv import load_dotenv

load_dotenv()


def _get(key: str, default: str) -> str:
    return os.getenv(key) or default


def _get_int(key: str, default: int) -> int:
    return int(os.getenv(key) or default)


# ── Claude ────────────────────────────────────────────────────────────────────
CLAUDE_MODEL = _get("CLAUDE_MODEL", "claude-3-5-sonnet-20241022")

# ── Paths ─────────────────────────────────────────────────────────────────────
DOCS_DIR = _get("DOCS_DIR", "docs")
DB_DIR = _get("DB_DIR", "chroma_db")

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE = _get_int("CHUNK_SIZE", 800)
CHUNK_OVERLAP = _get_int("CHUNK_OVERLAP", 100)

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K = _get_int("TOP_K", 5)

# ── Embedding model (runs locally, no API needed) ─────────────────────────────
EMBEDDING_MODEL = _get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
