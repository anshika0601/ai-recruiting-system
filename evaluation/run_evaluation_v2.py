"""
evaluation/run_evaluation_v2.py

Re-run evaluation after domain check improvement.
Compares v1 (no domain check) vs v2 (with domain check) using
multiple rank-quality metrics — not just strict position matching,
which is noisy on small candidate pools.

Usage:
    python -m evaluation.run_evaluation_v2
"""
import json
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import chromadb
from graphs.pipeline import run_pipeline

# ── NEW IMPORT ──────────────────────────────────────────────────────────
from evaluation.metrics import evaluate_ranking, format_metrics
# ────────────────────────────────────────────────────────────────────────

JD_SE = """
Software Engineer
Requirements:
- Backend development experience (Python, Java, or similar)
- REST API design and development
- Database knowledge (MySQL, PostgreSQL, or similar)
- Version control with Git
- Problem-solving skills
- Personal or professional projects demonstrating coding ability
- Fresh graduates and experienced candidates both welcome
"""

HUMAN_RANKINGS = {
    "Anshika_Bijalwan_Resume_Final":     2,
    "Akhilesh-Rawat-Resume":             1,
    "Ux-designer-resume-example-5":      3,
    "Sydney-Resume-Template-Modern":     4,
    "Account-Manager-Example-PDF":       5,
    "Dublin-Resume-Template-Modern":     7,
    "Personal-trainer-resume-example-3": 8,
    "Amsterdam-Modern-Resume-Template":  9,
    "Moscow-Creative-Resume-Template":   6,
}

# v1 results from Day 15 (before domain check)
V1_SCORES = {
    "Akhilesh-Rawat-Resume":             4.58,
    "Anshika_Bijalwan_Resume_Final":     3.47,
    "Ux-designer-resume-example-5":      3.03,
    "Moscow-Creative-Resume-Template":   1.90,
    "Amsterdam-Modern-Resume-Template":  1.65,
    "Dublin-Resume-Template-Modern":     1.52,
    "Sydney-Resume-Template-Modern":     1.02,
    "Personal-trainer-resume-example-3": 0.75,
    "Account-Manager-Example-PDF":       0.20,
}


def rank_by_score(scores):
    """Convert {id: score} into {id: rank} where rank 1 = highest score."""
    return {
        rid: rank + 1
        for rank, (rid, _) in enumerate(
            sorted(scores.items(), key=lambda x: x[1], reverse=True)
        )
    }


def aligned_rank_lists(human_map, pipeline_ranks):
    """
    Convert two dicts keyed by resume_id into two index-aligned lists,
    which is what evaluate_ranking() expects.

    Skips any resume that appears in human_map but not in pipeline_ranks
    (e.g. a resume that failed to run) so the two lists stay the same length.
    """
    human_list    = []
    pipeline_list = []
    for rid, human_rank in human_map.items():
        if rid in pipeline_ranks:
            human_list.append(human_rank)
            pipeline_list.append(pipeline_ranks[rid])
    return human_list, pipeline_list


def run_evaluation_v2():
    print("\n" + "=" * 62)
    print("  DAY 23 — Evaluation v2 (with domain check)")
    print("=" * 62)

    client  = chromadb.PersistentClient(path="./chroma_data")
    col     = client.get_collection("resumes")
    items   = col.get(include=["metadatas", "documents"])
    resumes = list(zip(items["ids"], items["metadatas"], items["documents"]))

    print(f"\n  Running pipeline on {len(resumes)} resumes (SE JD)...\n")

    v2_scores   = {}
    v2_verdicts = {}

    for resume_id, meta, doc_text in resumes:
        candidate = meta.get("name", resume_id)
        print(f"  [v2] {candidate}")

        result = run_pipeline(
            resume_text    = doc_text,
            jd_text        = JD_SE,
            resume_id      = resume_id,
            candidate_name = candidate,
        )

        score   = result.get("final_score") or 0.0
        verdict = result.get("domain_verdict") or "MATCH"
        v2_scores[resume_id]   = score
        v2_verdicts[resume_id] = verdict
        print(f"  [v2] Score: {score:.2f}/10  Domain: {verdict}")
        time.sleep(1)

    # ── Ranking derived from scores ─────────────────────────────────────
    v1_ranks = rank_by_score(V1_SCORES)
    v2_ranks = rank_by_score(v2_scores)

    # ── Per-candidate breakdown table (unchanged) ───────────────────────
    print(f"\n  {'─' * 62}")
    print(f"  {'Candidate':<26} {'Human':>6} {'v1':>5} {'v2':>5} "
          f"{'v1M':>5} {'v2M':>5} {'Domain':>10}")
    print(f"  {'─'*26} {'─'*6} {'─'*5} {'─'*5} {'─'*5} {'─'*5} {'─'*10}")

    for rid, hr in sorted(HUMAN_RANKINGS.items(), key=lambda x: x[1]):
        meta = next((m for r, m, _ in resumes if r == rid), {})
        name = meta.get("name", rid)[:24]
        v1r  = v1_ranks.get(rid, "?")
        v2r  = v2_ranks.get(rid, "?")
        v1m  = "OK" if isinstance(v1r, int) and abs(v1r - hr) <= 1 else "X"
        v2m  = "OK" if isinstance(v2r, int) and abs(v2r - hr) <= 1 else "X"
        dom  = v2_verdicts.get(rid, "?")
        print(f"  {name:<26} {hr:>6} {str(v1r):>5} {str(v2r):>5} "
              f"{v1m:>5} {v2m:>5} {dom:>10}")

    print(f"  {'─' * 62}")

    # ── NEW: multi-metric evaluation ────────────────────────────────────
    v1_human, v1_pipe = aligned_rank_lists(HUMAN_RANKINGS, v1_ranks)
    v2_human, v2_pipe = aligned_rank_lists(HUMAN_RANKINGS, v2_ranks)

    v1_metrics = evaluate_ranking(v1_human, v1_pipe, top_k=3)
    v2_metrics = evaluate_ranking(v2_human, v2_pipe, top_k=3)

    print("\n  ── v1 Metrics (before domain check) ──")
    print(format_metrics(v1_metrics))

    print("\n  ── v2 Metrics (after domain check) ──")
    print(format_metrics(v2_metrics))

    # Diff summary
    print(f"\n  ── Improvement (v2 vs v1) ──")

    mae_delta      = v2_metrics["mae"]           - v1_metrics["mae"]
    spearman_delta = v2_metrics["spearman"]      - v1_metrics["spearman"]
    top_k_delta    = v2_metrics["top_k_overlap"] - v1_metrics["top_k_overlap"]
    exact_delta    = v2_metrics["exact_match"]   - v1_metrics["exact_match"]

    def arrow(delta, higher_is_better=True):
        if delta == 0:
            return "  =  "
        good = (delta > 0) if higher_is_better else (delta < 0)
        return "  ▲  " if good else "  ▼  "

    print(f"  MAE (rank error)     : {v1_metrics['mae']:.2f} "
          f"→ {v2_metrics['mae']:.2f}"
          f"  {arrow(mae_delta, higher_is_better=False)} "
          f"({mae_delta:+.2f})   lower is better")
    print(f"  Top-3 overlap        : {v1_metrics['top_k_overlap']:.1%} "
          f"→ {v2_metrics['top_k_overlap']:.1%}"
          f"  {arrow(top_k_delta)} "
          f"({top_k_delta:+.1%})")
    print(f"  Spearman correlation : {v1_metrics['spearman']:+.2f} "
          f"→ {v2_metrics['spearman']:+.2f}"
          f"  {arrow(spearman_delta)} "
          f"({spearman_delta:+.2f})")
    print(f"  Exact-position match : {v1_metrics['exact_match']:.1%} "
          f"→ {v2_metrics['exact_match']:.1%}"
          f"  {arrow(exact_delta)} "
          f"({exact_delta:+.1%})   noisy on small samples")
    print(f"  {'─' * 62}\n")

    # ── Save richer results ─────────────────────────────────────────────
    results = {
        "v1": {
            "scores":  V1_SCORES,
            "ranks":   v1_ranks,
            "metrics": v1_metrics,
        },
        "v2": {
            "scores":   v2_scores,
            "ranks":    v2_ranks,
            "verdicts": v2_verdicts,
            "metrics":  v2_metrics,
        },
        "human_baseline": HUMAN_RANKINGS,
        "improvement": {
            "mae_delta":      round(mae_delta, 3),
            "spearman_delta": round(spearman_delta, 3),
            "top_k_delta":    round(top_k_delta, 3),
            "exact_delta":    round(exact_delta, 3),
        },
    }

    os.makedirs("evaluation", exist_ok=True)
    with open("evaluation/day23_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved -> evaluation/day23_results.json\n")

    return results


if __name__ == "__main__":
    run_evaluation_v2()