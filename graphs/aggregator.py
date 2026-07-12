"""
graphs/aggregator.py

The Aggregator — fourth and final node in the pipeline.

Job: combine dimension scores + guard penalty into one final weighted
score with a fully transparent breakdown. Every number is traceable
back to evidence from the resume.

Input  (from PipelineState): dimension_scores, guard_flags, guard_penalty
Output (to PipelineState):   final_score, score_breakdown, needs_review
"""
from typing import Any, Dict

from graphs.pipeline_state import PipelineState
from graphs.rubric import RUBRIC, DIMENSION_ORDER
from langsmith import traceable

@traceable(name="Aggregator", tags=["aggregator"])
def aggregator_node(state: PipelineState) -> Dict[str, Any]:
    """
    LangGraph node: compute weighted final score from dimension medians.

    Scoring formula:
      weighted_sum = Σ (dimension_median/5 × weight) for non-penalty dims
      penalty_dims reduce the weighted sum directly
      final_score  = max(0, weighted_sum × 10 − guard_penalty)

    Scaled to 0–10 so it reads naturally ("7.2 out of 10").
    """
    candidate = state.get("candidate_name", "unknown")
    print(f"[aggregator] Computing final score for: {candidate}")

    if not state.get("dimension_scores"):
        print("[aggregator] ✗ No dimension_scores in state")
        return {
            "final_score":     None,
            "score_breakdown": None,
            "needs_review":    None,
            "error": "Aggregator requires dimension_scores from scorer agent",
        }

    dim_scores   = state["dimension_scores"]
    guard_penalty = state.get("guard_penalty", 0.0) or 0.0
    guard_flags   = state.get("guard_flags", []) or []

    # ── compute weighted sum ────────────────────────────────────────────────
    breakdown = {}
    weighted_sum = 0.0

    for dim in DIMENSION_ORDER:
        rubric_def = RUBRIC[dim]
        agent_score = dim_scores.get(dim)

        if not agent_score:
            continue

        median  = agent_score["median"]
        weight  = rubric_def["weight"]
        label   = rubric_def["label"]
        is_pen  = rubric_def["is_penalty"]

        # Penalty dimension: a score of 5 = no penalty, 1 = max penalty.
        # Normalise so 5→0.0 deduction, 1→weight×10 deduction
        if is_pen:
            deduction = ((5 - median) / 4) * weight * 10
            contribution = -round(deduction, 3)
        else:
            contribution = round((median / 5) * weight * 10, 3)

        weighted_sum += contribution

        breakdown[dim] = {
            "label":        label,
            "median_score": median,
            "weight":       weight,
            "contribution": contribution,
            "is_penalty":   is_pen,
            "disagreement": agent_score["disagreement"],
            "evidence":     agent_score["run_1"]["evidence"],
            "reasoning":    agent_score["run_1"]["reasoning"],
        }

    # ── apply guard penalty ─────────────────────────────────────────────────
    raw_score   = max(0.0, weighted_sum)
    final_score = round(max(0.0, raw_score - guard_penalty), 2)

    # ── needs_review flag ───────────────────────────────────────────────────
    any_disagreement = any(
        dim_scores[d]["disagreement"]
        for d in DIMENSION_ORDER
        if d in dim_scores
    )
    needs_review = any_disagreement or len(guard_flags) > 0

    # ── console output ──────────────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print(f"  CANDIDATE : {candidate}")
    print(f"{'─'*50}")
    for dim, b in breakdown.items():
        flag = "⚠" if b["disagreement"] else " "
        sign = "" if b["contribution"] >= 0 else ""
        print(f"  {flag} {b['label']:26s} {b['median_score']}/5  "
              f"{sign}{abs(b['contribution']):.2f} pts")
    if guard_penalty > 0:
        print(f"    {'Guard penalty':26s}        -{guard_penalty:.2f} pts")
    print(f"{'─'*50}")
    print(f"  FINAL SCORE : {final_score:.2f} / 10.00")
    if needs_review:
        print(f"  ⚠ FLAGGED FOR HUMAN REVIEW")
    print(f"{'─'*50}\n")

    return {
        "final_score": final_score,
        "score_breakdown": {
            "dimensions":     breakdown,
            "raw_score":      round(raw_score, 2),
            "guard_penalty":  guard_penalty,
            "guard_flags":    guard_flags,
            "final_score":    final_score,
        },
        "needs_review": needs_review,
        "error": None,
    }