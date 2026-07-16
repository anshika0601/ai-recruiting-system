"""
app/routes.py

Day 18: FastAPI endpoints wiring the full pipeline into the API.

Endpoints:
  POST /resumes/upload       — parse + embed a resume PDF
  POST /screen               — run full pipeline on a resume vs JD
  GET  /resumes              — list all stored resumes
  GET  /resumes/{resume_id}  — get metadata for one resume
  DELETE /resumes/{resume_id}— remove a resume from Chroma
  POST /search               — semantic search (embedding only, fast)
  GET  /health               — health check with component status
"""
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.parser   import parse_resume
from app.embedder import (
    embed_resume,
    search_resumes,
    list_stored_resumes,
    _get_collection,
)
from graphs.pipeline import run_pipeline

router = APIRouter()


# ── request / response models ─────────────────────────────────────────────────

class ScreenRequest(BaseModel):
    resume_id: str
    jd_text:   str


class SearchRequest(BaseModel):
    jd_text: str
    top_k:   int = 5


class ScreenResponse(BaseModel):
    resume_id:       str
    candidate_name:  str
    final_score:     Optional[float]
    needs_review:    Optional[bool]
    score_breakdown: Optional[Dict[str, Any]]
    guard_flags:     Optional[List[Dict]]
    extracted_facts: Optional[Dict[str, Any]]
    error:           Optional[str]


# ── POST /resumes/upload ──────────────────────────────────────────────────────

@router.post("/resumes/upload", summary="Upload and embed a resume PDF")
async def upload_resume(file: UploadFile = File(...)):
    """
    Upload a resume PDF. Steps:
      1. Save to temp file
      2. Parse with column-aware parser
      3. Embed and store in Chroma
    Returns parsed metadata + resume_id for use in /screen.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files accepted")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        parsed    = parse_resume(tmp_path)
        resume_id = Path(file.filename).stem
        embed_resume(parsed, resume_id)

        return {
            "resume_id":        resume_id,
            "candidate_name":   parsed["name"],
            "email":            parsed["email"],
            "parse_confidence": parsed["parse_confidence"],
            "layout":           parsed.get("layout", "unknown"),
            "mixed_sections":   parsed.get("mixed_sections", []),
            "sections_found":   [k for k, v in parsed["sections"].items() if v.strip()],
            "message":          f"Resume '{resume_id}' uploaded and embedded successfully",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

    finally:
        os.unlink(tmp_path)


# ── POST /screen ──────────────────────────────────────────────────────────────

@router.post("/screen", response_model=ScreenResponse,
             summary="Run full agent pipeline on a stored resume vs JD")
async def screen_resume(request: ScreenRequest):
    """
    Run the 4-agent pipeline (extractor → scorer → guard → aggregator)
    on a stored resume against a job description.

    resume_id must exist in Chroma (uploaded via /resumes/upload).
    Returns full score breakdown with evidence per dimension.
    Takes 30-60 seconds — 15+ LLM calls per run.
    """
    try:
        col   = _get_collection()
        items = col.get(ids=[request.resume_id], include=["metadatas", "documents"])

        if not items["ids"]:
            raise HTTPException(
                status_code=404,
                detail=f"Resume '{request.resume_id}' not found. Upload first via /resumes/upload"
            )

        doc_text       = items["documents"][0]
        candidate_name = items["metadatas"][0].get("name", "unknown")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chroma fetch failed: {str(e)}")

    try:
        result = run_pipeline(
            resume_text    = doc_text,
            jd_text        = request.jd_text,
            resume_id      = request.resume_id,
            candidate_name = candidate_name,
        )

        return ScreenResponse(
            resume_id       = request.resume_id,
            candidate_name  = candidate_name,
            final_score     = result.get("final_score"),
            needs_review    = result.get("needs_review"),
            score_breakdown = result.get("score_breakdown"),
            guard_flags     = result.get("guard_flags"),
            extracted_facts = result.get("extracted_facts"),
            error           = result.get("error"),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {str(e)}")


# ── GET /resumes ──────────────────────────────────────────────────────────────

@router.get("/resumes", summary="List all stored resumes")
def list_resumes():
    """Return all resumes currently in Chroma with basic metadata."""
    resumes = list_stored_resumes()
    return {"total": len(resumes), "resumes": resumes}


# ── GET /resumes/{resume_id} ──────────────────────────────────────────────────

@router.get("/resumes/{resume_id}", summary="Get metadata for one resume")
def get_resume(resume_id: str):
    """Return stored metadata for a specific resume."""
    try:
        col   = _get_collection()
        items = col.get(ids=[resume_id], include=["metadatas"])

        if not items["ids"]:
            raise HTTPException(status_code=404, detail=f"Resume '{resume_id}' not found")

        return {"resume_id": resume_id, "metadata": items["metadatas"][0]}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── DELETE /resumes/{resume_id} ───────────────────────────────────────────────

@router.delete("/resumes/{resume_id}", summary="Remove a resume from storage")
def delete_resume(resume_id: str):
    """Delete a resume from Chroma by ID."""
    try:
        col = _get_collection()
        col.delete(ids=[resume_id])
        return {"message": f"Resume '{resume_id}' deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── POST /search ──────────────────────────────────────────────────────────────

@router.post("/search", summary="Fast semantic search — no agents, embedding only")
def search(request: SearchRequest):
    """
    Semantic search using embeddings only (~100ms vs 60s for full pipeline).
    Use this to shortlist candidates before running /screen on top results.
    """
    results = search_resumes(request.jd_text, top_k=request.top_k)
    return {
        "jd_preview": request.jd_text[:100] + "...",
        "results":    results,
    }


# ── GET /health ───────────────────────────────────────────────────────────────

@router.get("/health", summary="Health check with component status")
def health():
    """Check status of all system components."""
    status = {"api": "ok", "components": {}}

    try:
        col = _get_collection()
        status["components"]["chroma"] = {
            "status":       "ok",
            "resume_count": col.count(),
        }
    except Exception as e:
        status["components"]["chroma"] = {"status": "error", "detail": str(e)}

    groq_key = os.getenv("GROQ_API_KEY", "")
    status["components"]["groq"] = {
        "status": "ok" if groq_key.startswith("gsk_") else "missing",
    }

    ls_key = os.getenv("LANGCHAIN_API_KEY", "")
    status["components"]["langsmith"] = {
        "status":  "ok" if ls_key.startswith("ls__") else "disabled",
        "tracing": os.getenv("LANGCHAIN_TRACING_V2", "false"),
    }

    return status