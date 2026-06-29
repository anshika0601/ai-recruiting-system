"""
FastAPI skeleton. Day 1 goal: app boots, health check works, and we can
trigger the toy LangGraph from an endpoint so you see the two pieces wired
together end to end (even before any real resume logic exists).
"""
from fastapi import FastAPI

from app.config import settings
from graphs.toy_graph import run_toy_graph

app = FastAPI(title="AI Recruiter", version="0.1.0")


@app.get("/health")
def health():
    return {"status": "ok", "env": settings.app_env}


@app.get("/toy-graph")
def toy_graph_demo():
    """
    Hits the 2-node LangGraph toy example. Replace this route in Week 2
    with the real extractor -> scorer pipeline.
    """
    result = run_toy_graph("candidate Jane Doe")
    return result
