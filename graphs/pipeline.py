"""
graphs/pipeline.py

Full LangGraph pipeline — all agents wired together.

Graph structure:
  START → extractor → scorer → guard → aggregator → END

Conditional edge after extractor:
  if error → END early (don't waste LLM calls scoring a broken extract)
  else     → scorer

This is the file that makes everything a real multi-agent pipeline
rather than four separate scripts. State flows through automatically.
"""
import json
from typing import Any, Dict

from langgraph.graph import StateGraph, END

from graphs.pipeline_state import PipelineState
from graphs.extractor_agent import extractor_node
from graphs.scorer_agent    import scorer_node
from graphs.guard_agent     import guard_node
from graphs.aggregator      import aggregator_node


# Conditional edge — skip scoring if extraction failed
# ---------------------------------------------------------------------------

def should_continue(state: PipelineState) -> str:
    """
    After extractor runs: if error or no extracted_facts, end early.
    Otherwise proceed to scorer.
    """
    if state.get("error") or not state.get("extracted_facts"):
        print("[pipeline] ✗ Extraction failed — ending pipeline early")
        return "end"
    return "scorer"


# Build and compile the graph
# ---------------------------------------------------------------------------

def build_pipeline() -> Any:
    """
    Construct the LangGraph StateGraph with all 4 nodes and edges.
    Returns a compiled, runnable graph.
    """
    graph = StateGraph(PipelineState)

    # Add all nodes
    graph.add_node("extractor",  extractor_node)
    graph.add_node("scorer",     scorer_node)
    graph.add_node("guard",      guard_node)
    graph.add_node("aggregator", aggregator_node)

    # Entry point
    graph.set_entry_point("extractor")

    # Conditional edge: extractor → scorer OR end
    graph.add_conditional_edges(
        "extractor",
        should_continue,
        {"scorer": "scorer", "end": END},
    )

    # Linear edges for the rest
    graph.add_edge("scorer",     "guard")
    graph.add_edge("guard",      "aggregator")
    graph.add_edge("aggregator", END)

    return graph.compile()


# Compile once at module load — reused across all requests
_pipeline = build_pipeline()


# Public API
# ---------------------------------------------------------------------------

def run_pipeline(
    resume_text:    str,
    jd_text:        str,
    resume_id:      str,
    candidate_name: str,
) -> Dict[str, Any]:
    """
    Run the full screening pipeline for one candidate.

    Args:
        resume_text:    Raw text from parser
        jd_text:        Full job description
        resume_id:      Unique ID (filename stem or UUID)
        candidate_name: Candidate's name from parser

    Returns:
        Complete PipelineState dict after all agents have run.
    """
    initial_state: PipelineState = {
        "resume_text":     resume_text,
        "jd_text":         jd_text,
        "resume_id":       resume_id,
        "candidate_name":  candidate_name,
        "extracted_facts":  None,
        "dimension_scores": None,
        "guard_flags":      None,
        "guard_penalty":    None,
        "final_score":      None,
        "score_breakdown":  None,
        "needs_review":     None,
        "error":            None,
    }

    return _pipeline.invoke(initial_state)


# CLI — python -m graphs.pipeline resume.pdf
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from app.parser import parse_resume

    if len(sys.argv) < 2:
        print("Usage: python -m graphs.pipeline resume.pdf")
        sys.exit(1)

    test_jd = """
    Senior Software Engineer — Python
    Requirements:
    - 3+ years Python development experience
    - REST API design and development
    - AWS cloud services (S3, Lambda, EC2)
    - React or similar frontend framework
    - Strong problem-solving and communication skills
    """

    parsed = parse_resume(sys.argv[1])

    final_state = run_pipeline(
        resume_text    = parsed["raw_text"],
        jd_text        = test_jd,
        resume_id      = sys.argv[1].split("/")[-1].replace(".pdf", ""),
        candidate_name = parsed["name"],
    )

    print("\n── Full Score Breakdown ──")
    if final_state.get("score_breakdown"):
        breakdown = final_state["score_breakdown"]
        for dim, b in breakdown["dimensions"].items():
            print(f"\n{b['label']} ({b['median_score']}/5):")
            print(f"  Evidence  : {b['evidence']}")
            print(f"  Reasoning : {b['reasoning']}")
            print(f"  Contribution: {b['contribution']:+.2f} pts")

        print(f"\nRaw score    : {breakdown['raw_score']}")
        print(f"Guard penalty: -{breakdown['guard_penalty']}")
        print(f"FINAL SCORE  : {breakdown['final_score']} / 10")
        print(f"Needs review : {final_state.get('needs_review')}")

    if final_state.get("guard_flags"):
        print(f"\n── Guard Flags ──")
        for f in final_state["guard_flags"]:
            print(f"  [{f['severity'].upper()}] {f['type']}: {f['evidence']}")