# ── LLM backend ──────────────────────────────────────────────────────────────
# "ollama"  → fully local, no API key needed (install Ollama first)
# "openai"  → OpenAI cloud API (set OPENAI_API_KEY env var or paste key below)
LLM_BACKEND = "ollama"

# Ollama settings (https://ollama.com — run: ollama pull llama3.2)
OLLAMA_MODEL = "llama3.2"
OLLAMA_BASE_URL = "http://localhost:11434"

# OpenAI settings (only used when LLM_BACKEND = "openai")
OPENAI_API_KEY = ""   # leave blank to read from OPENAI_API_KEY env var instead
OPENAI_MODEL = "gpt-4o-mini"

# ── Paths ─────────────────────────────────────────────────────────────────────
DOCS_DIR = "docs"       # drop your PDFs / DOCX / Markdown files here
DB_DIR = "chroma_db"    # ChromaDB persists here (auto-created)

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE = 800        # characters per chunk
CHUNK_OVERLAP = 100     # characters of overlap between consecutive chunks

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K = 5               # number of chunks to retrieve per query

# ── Embedding model (runs locally on CPU, no API needed) ──────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # ~90 MB, fast, good quality
