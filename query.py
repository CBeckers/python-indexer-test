"""
query.py — Interactive Q&A over your indexed documentation.

Usage:
    python query.py               # interactive loop
    python query.py "your question here"   # single question, then exit
"""

import os
import sys

import chromadb
from sentence_transformers import SentenceTransformer

import config


# ── Retrieval ──────────────────────────────────────────────────────────────────

def retrieve(collection, model, question: str) -> tuple[list[str], list[str]]:
    """Return the top-k most relevant chunks and their source filenames."""
    embedding = model.encode([question])[0].tolist()
    results = collection.query(
        query_embeddings=[embedding],
        n_results=config.TOP_K,
        include=["documents", "metadatas"],
    )
    chunks: list[str] = results["documents"][0]
    sources: list[str] = [m["source"] for m in results["metadatas"][0]]
    return chunks, sources


# ── LLM call ──────────────────────────────────────────────────────────────────

def ask_llm(context_chunks: list[str], question: str) -> str:
    context = "\n\n---\n\n".join(context_chunks)
    system_prompt = (
        "You are a helpful assistant. Answer questions using only the provided documentation context. "
        "If the answer is not contained in the context, say so clearly rather than guessing. "
        "Be concise, accurate, and cite relevant detail from the context."
    )
    user_message = f"Context from documentation:\n\n{context}\n\nQuestion: {question}"

    return _ask_claude(system_prompt, user_message)


def _ask_claude(system_prompt: str, user_message: str) -> str:
    import anthropic
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key or api_key.startswith("your-"):
        raise ValueError(
            "No Anthropic API key found. Set ANTHROPIC_API_KEY in your .env file."
        )
    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=config.CLAUDE_MODEL,
        max_tokens=2048,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )
    return response.content[0].text


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Loading embedding model '{config.EMBEDDING_MODEL}'...")
    model = SentenceTransformer(config.EMBEDDING_MODEL)

    client = chromadb.PersistentClient(path=config.DB_DIR)
    try:
        collection = client.get_collection("docs")
    except Exception:
        print("No index found. Run 'python ingest.py' first.")
        sys.exit(1)

    count = collection.count()
    if count == 0:
        print("Index is empty. Run 'python ingest.py' to index your documents first.")
        sys.exit(1)

    print(f"Index loaded ({count} chunks across your docs).")
    print("Type 'quit' or press Ctrl+C to exit.\n")

    # Allow a single question passed as a CLI argument
    one_shot = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else None

    while True:
        if one_shot:
            question = one_shot
        else:
            try:
                question = input("Question: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye.")
                break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break

        try:
            chunks, sources = retrieve(collection, model, question)
            answer = ask_llm(chunks, question)
        except Exception as exc:
            print(f"[error] {exc}\n")
            if one_shot:
                sys.exit(1)
            continue

        unique_sources = list(dict.fromkeys(sources))
        print(f"\nAnswer:\n{answer}")
        print(f"\nSources: {', '.join(unique_sources)}\n")
        print("-" * 60 + "\n")

        if one_shot:
            break


if __name__ == "__main__":
    main()
