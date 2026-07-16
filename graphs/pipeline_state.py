"""
graphs/pipeline_state.py

The shared state that flows through every agent in the pipeline.

Every agent reads from this state and writes partial updates back to it.
LangGraph merges those updates automatically — no manual passing of return values.
Updated added domain_verdict, domain_penalty, transferable_skills.

"""
from typing import Any, Dict, List, Optional, TypedDict


class DimensionScore(TypedDict):
    """One agent's score for a single rubric dimension."""
    evidence:  List[str]   # direct quotes/facts from resume supporting this score
    reasoning: str         # 1-2 sentence explanation of how evidence maps to anchor
    score:     int         # 1-5 integer matching the rubric anchor


class AgentScore(TypedDict):
    """Full scoring result for one rubric dimension across 3 self-consistency runs."""
    run_1:       DimensionScore
    run_2:       DimensionScore
    run_3:       DimensionScore
    median:      int            # median score across 3 runs
    disagreement: bool          # True if any run differs by >1 from median


class PipelineState(TypedDict):
    """
    Shared state flowing through the full agent pipeline.

    Population order:
      Step 0  (input)        → resume_text, jd_text, resume_id, candidate_name
      Step 1  (extractor)    → extracted_facts
      Step 2  (scorer)       → dimension_scores
      Step 3  (guard)        → guard_flags, guard_penalty
      Step 4  (aggregator)   → final_score, score_breakdown, needs_review
      Step 5  (output)       → error (if anything went wrong)
    """

    # ── inputs ──────────────────────────────────────────────────────────────
    resume_text:      str              # raw text from parser
    jd_text:          str              # full job description text
    resume_id:        str              # filename stem or UUID
    candidate_name:   str              # from parser output

    # ── extractor output ────────────────────────────────────────────────────
    extracted_facts: Optional[Dict[str, Any]]
    # Shape:
    # {
    #   "skills_found":      ["Python", "REST APIs", "AWS"],
    #   "years_experience":  5,
    #   "companies":         ["Google", "Startup X"],
    #   "achievements":      ["Reduced latency by 40%", "Led team of 6"],
    #   "education":         ["BSc Computer Science, MIT 2018"],
    #   "red_flags":         ["6-month gap 2021", "4 jobs in 3 years"],
    #   "jd_requirements":   ["Python", "3+ years", "AWS"],   ← parsed from JD
    # }
    
    
    # ── domain check output (Day 22) ─────────────────────────────────────────
    domain_verdict:      Optional[str]    # MATCH | ADJACENT | MISMATCH
    domain_penalty:      Optional[float]  # score deduction applied in aggregator
    transferable_skills: Optional[List[str]]  # passed to scorer for ADJACENT

    # ── scorer output ────────────────────────────────────────────────────────
    dimension_scores: Optional[Dict[str, AgentScore]]
    # Keys: "core_skill_match", "experience_relevance",
    #       "achievement_evidence", "trajectory", "red_flags"

    # ── guard output ─────────────────────────────────────────────────────────
    guard_flags:   Optional[List[str]]   # list of detected issues
    guard_penalty: Optional[float]       # 0.0–1.0 deduction from final score

    # ── aggregator output ────────────────────────────────────────────────────
    final_score:     Optional[float]     # 0–10 weighted final score
    score_breakdown: Optional[Dict[str, Any]]  # per-dimension scores + weights
    needs_review:    Optional[bool]      # True if any dimension had disagreement

    # ── error handling ───────────────────────────────────────────────────────
    error: Optional[str]