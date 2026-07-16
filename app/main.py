"""
app/main.py

FastAPI application — updated Day 18 to include all pipeline routes.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routes import router

app = FastAPI(
    title="AI Recruiting Platform",
    description="Multi-agent resume screening pipeline with LangGraph + RAG",
    version="0.2.0",
)

# CORS — needed for Next.js frontend (Week 3)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount all routes
app.include_router(router)


@app.get("/", include_in_schema=False)
def root():
    return {
        "name":    "AI Recruiting Platform",
        "version": "0.2.0",
        "docs":    "/docs",
        "health":  "/health",
    }