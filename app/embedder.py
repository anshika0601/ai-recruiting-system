
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from app.config import settings

MODEL_NAME   = "all-MiniLM-L6-v2"
COLLECTION   = "resumes"

# Weights for combining section embeddings into one resume representation.
# Experience and skills matter most for JD matching.
SECTION_WEIGHTS: Dict[str, float] = {
    "summary":    0.15,
    "experience": 0.40,
    "skills":     0.30,
    "education":  0.10,
    "projects":   0.05,
}


# Singleton helpers — model and Chroma client load once per process
# ---------------------------------------------------------------------------

_model:  SentenceTransformer | None = None
_client: chromadb.ClientAPI   | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"[embedder] Loading model '{MODEL_NAME}' (first run downloads ~80 MB)…")
        _model = SentenceTransformer(MODEL_NAME)
        print("[embedder] Model ready.")
    return _model


def _get_client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        persist_dir = Path(settings.chroma_persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(persist_dir))
    return _client


def _get_collection() -> chromadb.Collection:
    return _get_client().get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},   # cosine similarity, not L2
    )


# Core: build a single embedding vector for a parsed resume
# ---------------------------------------------------------------------------

def _resume_to_text(parsed: Dict[str, Any]) -> str:
    """
    Combine resume sections into one weighted text string for embedding.

    Sections with higher weight are repeated proportionally so the
    embedding model "sees" them more — a simple but effective trick
    when you can't control the model's attention directly.
    """
    parts = []
    sections: Dict[str, str] = parsed.get("sections", {})

    for section, weight in SECTION_WEIGHTS.items():
        content = sections.get(section, "").strip()
        if not content:
            continue
        # Repeat high-weight sections (floor so weight 0.15 → 0 repeats, 0.40 → 1)
        repeats = max(1, round(weight * 4))
        parts.extend([content] * repeats)

    # Always prepend name + email so the vector DB doc is self-contained
    header = f"{parsed.get('name', '')} {parsed.get('email', '')}".strip()
    if header:
        parts.insert(0, header)

    return "\n\n".join(parts)


# Public API
# ---------------------------------------------------------------------------

def embed_resume(parsed: Dict[str, Any], resume_id: str) -> None:
    """
    Embed a parsed resume and upsert it into Chroma.

    Args:
        parsed:    Output of parse_resume() from app/parser.py.
        resume_id: Unique string ID (e.g. filename stem or UUID).
    """
    text  = _resume_to_text(parsed)
    model = _get_model()
    vec   = model.encode(text, normalize_embeddings=True).tolist()

    # Store lightweight metadata alongside the vector for display
    metadata = {
        "name":       parsed.get("name", "unknown"),
        "email":      parsed.get("email", "") or "",
        "confidence": parsed.get("parse_confidence", ""),
        "layout":     parsed.get("layout", ""),
        "mixed":      json.dumps(parsed.get("mixed_sections", [])),
    }

    _get_collection().upsert(
        ids        =[resume_id],
        embeddings =[vec],
        documents  =[text],
        metadatas  =[metadata],
    )
    print(f"[embedder] Stored '{resume_id}' ({parsed.get('name', '?')})")


def embed_all_resumes(parsed_list: List[Dict[str, Any]]) -> None:
    """
    Embed and store a batch of parsed resumes.

    Args:
        parsed_list: List of (resume_id, parsed_dict) tuples.
    """
    for resume_id, parsed in parsed_list:
        embed_resume(parsed, resume_id)


def search_resumes(jd_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Given a job description, return the top-k most semantically similar resumes.

    Args:
        jd_text: Raw job description text.
        top_k:   Number of results to return (default 5).

    Returns:
        List of dicts, each with keys:
            id         resume ID
            name       candidate name
            email      candidate email
            score      cosine similarity 0-1 (higher = better match)
            mixed      sections flagged as mixed content
            snippet    first 300 chars of stored resume text
    """
    model = _get_model()
    vec   = model.encode(jd_text, normalize_embeddings=True).tolist()

    results = _get_collection().query(
        query_embeddings=[vec],
        n_results=min(top_k, _get_collection().count()),
        include=["documents", "metadatas", "distances"],
    )

    if not results["ids"][0]:
        return []

    output = []
    for rid, doc, meta, dist in zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        # Chroma cosine distance = 1 - similarity, so flip it
        similarity = round(1 - dist, 4)
        output.append({
            "id":      rid,
            "name":    meta.get("name", "unknown"),
            "email":   meta.get("email", ""),
            "score":   similarity,
            "mixed":   json.loads(meta.get("mixed", "[]")),
            "snippet": doc[:300],
        })

    # Sort highest similarity first (Chroma usually returns sorted)
    return sorted(output, key=lambda x: x["score"], reverse=True)


def list_stored_resumes() -> List[Dict[str, str]]:
    """Return all resumes currently stored in Chroma (id + name + email)."""
    col = _get_collection()
    if col.count() == 0:
        return []
    all_items = col.get(include=["metadatas"])
    return [
        {"id": rid, "name": m.get("name", ""), "email": m.get("email", "")}
        for rid, m in zip(all_items["ids"], all_items["metadatas"])
    ]


def clear_all_resumes() -> None:
    """Delete all stored resumes. Useful for resetting during testing."""
    _get_client().delete_collection(COLLECTION)
    print("[embedder] Collection cleared.")


# CLI test — run:  python -m app.embedder
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path
    from app.parser import parse_resume

    # Step 1: embed all PDFs passed as args
    pdf_paths = [p for p in sys.argv[1:] if p.endswith(".pdf")]

    if not pdf_paths:
        print("Usage: python -m app.embedder resume1.pdf resume2.pdf ... --jd 'your JD text'")
        sys.exit(1)

    for pdf_path in pdf_paths:
        parsed = parse_resume(pdf_path)
        resume_id = Path(pdf_path).stem
        embed_resume(parsed, resume_id)

    # Step 2: if --jd flag provided, run a search
    if "--jd" in sys.argv:
        jd_idx = sys.argv.index("--jd")
        jd_text = sys.argv[jd_idx + 1] if jd_idx + 1 < len(sys.argv) else ""

        if jd_text:
            print("\n--- Ranked results ---")
            results = search_resumes(jd_text, top_k=5)
            for i, r in enumerate(results, 1):
                print(f"\n#{i}  {r['name']}  (score: {r['score']})")
                print(f"Email : {r['email']}")
                print(f"Mixed : {r['mixed']}")
                print(f"Snippet: {r['snippet'][:150]}…")
        else:
            print("Provide JD text after --jd flag.")

    print(f"\n[embedder] Total resumes in store: {_get_collection().count()}")