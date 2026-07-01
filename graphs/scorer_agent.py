"""
graphs/scorer_agent.py

The Scorer Agent — second node in the pipeline.

The score each rubric dimension independently, with evidence cited
BEFORE the number is assigned (forces grounded reasoning, reduces
hallucinated justification). Each dimension is scored 3 times
(self-consistency voting) — median is taken, disagreement is flagged.

Input (from PipelineState): extracted_facts, jd_text
Output (to PipelineState):   dimension_scores
"""
import json
import os
import statistics
from typing import Any, Dict, List
from dotenv import load_dotenv
from groq import Groq
from graphs.pipeline_state import PipelineState, DimensionScore, AgentScore
from graphs.rubric import RUBRIC, DIMENSION_ORDER, format_anchors

load_dotenv()

_client: Groq | None = None

def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return _client

MODEL = "llama-3.3-70b-versatile"

# Self-consistency: run each dimension this many times, take the median
VOTES_PER_DIMENSION = 3

# If any single run differs from the median by more than this, flag disagreement
DISAGREEMENT_THRESHOLD = 1

# Scoring prompt — evidence first, reasoning second, score last.
# Order matters: forces the model to ground its number in evidence
# rather than picking a number and rationalizing it after.
# ---------------------------------------------------------------------------

SCORER_SYSTEM = """You are a calibrated, evidence-based resume scorer.
You score ONE dimension at a time using only the structured facts provided.
You must respond with valid JSON only. No preamble, no markdown fences.
Be conservative: if evidence is weak or absent, score low. Never guess."""

SCORER_PROMPT = """Score the candidate on this single dimension: {label}

ANCHOR SCALE (you must pick the closest match):
{anchors}

JOB DESCRIPTION (for context):
{jd_text}

CANDIDATE FACTS (extracted from resume):
{facts_json}

Steps:
1. List up to 3 short pieces of evidence from the candidate facts relevant
   to "{label}". If none exist, evidence should be an empty list.
2. In 1-2 sentences, explain how that evidence maps to the anchor scale.
3. Output a single integer 1-5 matching the closest anchor.

Return ONLY this JSON:
{{
  "evidence": ["short evidence fragment 1", "short evidence fragment 2"],
  "reasoning": "1-2 sentence explanation",
  "score": <integer 1-5>
}}"""

# Single scoring run for one dimension
# ---------------------------------------------------------------------------

def _score_dimension_once(
    dimension: str,
    facts: Dict[str, Any],
    jd_text: str,
) -> DimensionScore:
    """
    Run the scoring prompt once for a single dimension.
    Returns a DimensionScore dict. On failure, returns a safe low-score default
    rather than crashing the whole pipeline.
    """
    dim_def = RUBRIC[dimension]

    prompt = SCORER_PROMPT.format(
        label=dim_def["label"],
        anchors=format_anchors(dimension),
        jd_text=jd_text[:2000],
        facts_json=json.dumps(facts, indent=2)[:2500],
    )

    try:
        response = _get_client().chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SCORER_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            max_tokens=400,
            temperature=0.2,  # low but not zero — slight variation needed for
                              # self-consistency voting to be meaningful
        )

        raw = response.choices[0].message.content.strip()

        # Strip markdown fences
        if raw.startswith("```"):
         raw = raw.split("```")[1]
        if raw.startswith("json"):
         raw = raw[4:]
        raw = raw.strip()

        # Find the JSON object boundaries — model sometimes adds text before/after
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start != -1 and end > start:
         raw = raw[start:end]

        # Replace smart quotes that some models output
        raw = raw.replace("\u2018", "'").replace("\u2019", "'")
        raw = raw.replace("\u201c", '"').replace("\u201d", '"')
        try:
           parsed = json.loads(raw)
        except json.JSONDecodeError:
        # Last resort: extract fields individually via regex
          import re
          evidence_match=re.search(r'"evidence"\s*:\s*\[(.*?)\]', raw, re.DOTALL)
          reasoning_match=re.search(r'"reasoning"\s*:\s*"(.*?)"', raw, re.DOTALL)
          score_match=re.search(r'"score"\s*:\s*(\d)', raw)

          parsed = {
        "evidence":  json.loads(f"[{evidence_match.group(1)}]") if evidence_match else [],
        "reasoning": reasoning_match.group(1) if reasoning_match else "",
        "score":     int(score_match.group(1)) if score_match else 1,
    }
          print(f"[scorer]   ⚠ Used regex fallback for JSON extraction")
  

        # Clamp score to valid 1-5 range in case model drifts
        score = int(parsed.get("score", 1))
        score = max(1, min(5, score))

        return {
            "evidence": parsed.get("evidence", []),
            "reasoning": parsed.get("reasoning", ""),
            "score": score,
        }

    except Exception as e:
        print(f"[scorer]   ✗ run failed for {dimension}: {e}")
        # Safe fallback: lowest score, flagged in reasoning so it's visible
        # downstream rather than silently treated as a real low score.
        return {
            "evidence": [],
            "reasoning": f"SCORING_FAILED: {str(e)}",
            "score": 1,
        }


# Self-consistency voting wrapper for one dimension
# ---------------------------------------------------------------------------

def _score_dimension_with_voting(
    dimension: str,
    facts: Dict[str, Any],
    jd_text: str,
) -> AgentScore:
    """
    Run _score_dimension_once 3 times, take the median score, flag
    disagreement if any individual run deviates from the median by
    more than DISAGREEMENT_THRESHOLD.
    """
    runs: List[DimensionScore] = []
    for i in range(VOTES_PER_DIMENSION):
        result = _score_dimension_once(dimension, facts, jd_text)
        runs.append(result)

    scores = [r["score"] for r in runs]
    median_score = int(statistics.median(scores))

    disagreement = any(
        abs(s - median_score) > DISAGREEMENT_THRESHOLD for s in scores
    )

    print(
        f"[scorer]   {RUBRIC[dimension]['label']:24s} "
        f"runs={scores} median={median_score} "
        f"{'⚠ DISAGREEMENT' if disagreement else '✓'}"
    )

    return {
        "run_1": runs[0],
        "run_2": runs[1],
        "run_3": runs[2],
        "median": median_score,
        "disagreement": disagreement,
    }


# Scorer node — entry point called by LangGraph
# ---------------------------------------------------------------------------

def scorer_node(state: PipelineState) -> Dict[str, Any]:
    """
    LangGraph node: score all 5 rubric dimensions with self-consistency voting.
    Returns partial state update with dimension_scores populated.
    """
    candidate = state.get("candidate_name", "unknown")
    print(f"[scorer] Scoring: {candidate}")

    if not state.get("extracted_facts"):
        print("[scorer] ✗ No extracted_facts in state — extractor must run first")
        return {
            "dimension_scores": None,
            "error": "Scorer requires extracted_facts from extractor agent",
        }

    facts = state["extracted_facts"]
    jd_text = state["jd_text"]

    dimension_scores: Dict[str, AgentScore] = {}
    for dimension in DIMENSION_ORDER:
        dimension_scores[dimension] = _score_dimension_with_voting(
            dimension, facts, jd_text
        )

    flagged = [d for d, s in dimension_scores.items() if s["disagreement"]]
    if flagged:
        print(f"[scorer] ⚠ {len(flagged)} dimension(s) flagged for disagreement: {flagged}")
    else:
        print(f"[scorer] ✓ All dimensions scored with no disagreement")

    return {"dimension_scores": dimension_scores, "error": None}


# CLI test — python -m graphs.scorer_agent resume.pdf
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from app.parser import parse_resume
    from graphs.extractor_agent import extractor_node

    if len(sys.argv) < 2:
        print("Usage: python -m graphs.scorer_agent resume.pdf")
        sys.exit(1)

    test_jd = """
    Senior Software Engineer — Python
    Requirements:
    - 3+ years Python development
    - REST API design and development
    - AWS cloud services (S3, Lambda, EC2)
    - React or similar frontend framework
    - Strong problem-solving and communication skills
    """

    parsed = parse_resume(sys.argv[1])

    state: PipelineState = {
        "resume_text": parsed["raw_text"],
        "jd_text": test_jd,
        "resume_id": "test",
        "candidate_name": parsed["name"],
        "extracted_facts": None,
        "dimension_scores": None,
        "guard_flags": None,
        "guard_penalty": None,
        "final_score": None,
        "score_breakdown": None,
        "needs_review": None,
        "error": None,
    }

    # Step 1: extractor must run first
    extract_result = extractor_node(state)
    state.update(extract_result)

    if not state.get("extracted_facts"):
        print("Extraction failed, cannot proceed to scoring.")
        sys.exit(1)

    # Step 2: scorer
    score_result = scorer_node(state)
    state.update(score_result)

    print("\n── Dimension Scores (median + disagreement) ──")
    for dim, s in state["dimension_scores"].items():
        print(f"\n{RUBRIC[dim]['label']}: {s['median']}/5  "
              f"{'⚠ DISAGREEMENT' if s['disagreement'] else ''}")
        print(f"  Evidence: {s['run_1']['evidence']}")
        print(f"  Reasoning: {s['run_1']['reasoning']}")