"""
evaluation/run_evaluation.py

Full pipeline evaluation against human rankings.

Runs all resumes in Chroma through the 4-agent pipeline for both JDs,
compares agent rankings vs human rankings from Week 1 baseline,
and computes before-vs-after accuracy improvement.

"""
import json
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import chromadb
from graphs.pipeline import run_pipeline

# ── JD definitions ────────────────────────────────────────────────────────────

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


JD_FD = """
Senior Fashion Designer
Requirements:
- 5+ years fashion design experience
- Adobe Suite (Illustrator, Photoshop)
- Trend forecasting and seasonal collections
- Experience with luxury or high-end brands
- Strong portfolio of apparel or accessories design
"""

JDS = {
    "Software Engineer": JD_SE,
    "Fashion Designer": JD_FD,
    
}

# ── Human rankings ────────────────────────────────────
# Keys must partially match your Chroma resume IDs (case-insensitive)

HUMAN_RANKINGS = {
    "anshika":     {"Software Engineer": 1, "Fashion Designer": 6},
    "akhilesh":    {"Software Engineer": 2, "Fashion Designer": 5},
    "john":        {"Software Engineer": 3, "Fashion Designer": 2},
    "kristen":     {"Software Engineer": 4, "Fashion Designer": 3},
    "michelle-s":  {"Software Engineer": 5, "Fashion Designer": 7},
    "esther":      {"Software Engineer": 6, "Fashion Designer": 4},
    "charly":      {"Software Engineer": 7, "Fashion Designer": 8},
    "michelle-l":  {"Software Engineer": 9, "Fashion Designer": 1},
    "julie":       {"Software Engineer": 8, "Fashion Designer": 9},
}

BASELINE_ACCURACY = {
    "Software Engineer": 0.0,
    "Fashion Designer":  0.80,
}


# ── helpers ───────────────────────────────────────────────────────────────────

def get_all_resumes():
    client = chromadb.PersistentClient(path="./chroma_data")
    col    = client.get_collection("resumes")
    items  = col.get(include=["metadatas", "documents"])
    return list(zip(items["ids"], items["metadatas"], items["documents"]))


def rank_by_score(results):
    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
    return {rid: rank + 1 for rank, (rid, _) in enumerate(sorted_results)}


def get_human_rank(resume_id, jd_name):
    rid_lower = resume_id.lower()
    for key, ranks in HUMAN_RANKINGS.items():
        if key.lower() in rid_lower:
            return ranks.get(jd_name)
    return None


# ── main evaluation ───────────────────────────────────────────────────────────

def run_evaluation():
    print("\n" + "=" * 62)
    print("  FULL EVALUATION  —  Multi-Agent Pipeline vs Baseline")
    print("=" * 62)

    resumes = get_all_resumes()
    print(f"\n  Resumes in Chroma: {len(resumes)}")
    for rid, meta, _ in resumes:
        print(f"    {rid:35s} -> {meta.get('name', '?')}")

    eval_summary = {}

    for jd_name, jd_text in JDS.items():
        print(f"\n\n{'=' * 62}")
        print(f"  JD: {jd_name}")
        print(f"{'=' * 62}")

        jd_scores   = {}
        jd_details  = {}

        for resume_id, meta, doc_text in resumes:
            candidate = meta.get("name", resume_id)
            print(f"\n  [pipeline] {candidate}")

            result = run_pipeline(
                resume_text    = doc_text,
                jd_text        = jd_text,
                resume_id      = resume_id,
                candidate_name = candidate,
            )

            score = result.get("final_score") or 0.0
            jd_scores[resume_id]  = score
            jd_details[resume_id] = result
            print(f"  [pipeline] Score: {score:.2f}/10")
            time.sleep(3)

        agent_ranks = rank_by_score(list(jd_scores.items()))

        print(f"\n  {'─' * 62}")
        print(f"  {'Candidate':<26} {'Agent':>6} {'Human':>6} {'Score':>7} {'Match':>6}")
        print(f"  {'─' * 26} {'─' * 6} {'─' * 6} {'─' * 7} {'─' * 6}")

        matches = 0
        total   = 0
        details = []

        for resume_id, agent_rank in sorted(agent_ranks.items(), key=lambda x: x[1]):
            meta       = next(m for rid, m, _ in resumes if rid == resume_id)
            candidate  = meta.get("name", resume_id)
            score      = jd_scores[resume_id]
            human_rank = get_human_rank(resume_id, jd_name)

            if human_rank is not None:
                matched = abs(agent_rank - human_rank) <= 1
                matches += int(matched)
                total   += 1
                symbol  = "OK" if matched else "X"
            else:
                matched = None
                symbol  = "?"

            print(f"  {candidate:<26} {agent_rank:>6} {str(human_rank or '?'):>6} "
                  f"{score:>6.2f}  {symbol:>6}")

            details.append({
                "resume_id":    resume_id,
                "candidate":    candidate,
                "agent_rank":   agent_rank,
                "human_rank":   human_rank,
                "score":        score,
                "match":        matched,
                "needs_review": jd_details[resume_id].get("needs_review"),
                "guard_flags":  len(jd_details[resume_id].get("guard_flags") or []),
            })

        accuracy = matches / total if total > 0 else 0
        baseline = BASELINE_ACCURACY.get(jd_name, 0)
        delta    = accuracy - baseline
        delta_str = f"+{delta:.0%}" if delta >= 0 else f"{delta:.0%}"

        print(f"  {'─' * 62}")
        print(f"  Agent accuracy   : {accuracy:.0%}  ({matches}/{total} within 1 rank)")
        print(f"  Baseline Week 1  : {baseline:.0%}")
        print(f"  Improvement      : {delta_str}")
        print(f"  {'─' * 62}")

        eval_summary[jd_name] = {
            "accuracy": accuracy,
            "baseline": baseline,
            "delta":    delta,
            "matches":  matches,
            "total":    total,
            "details":  details,
        }

    # ── final summary ─────────────────────────────────────────────────────────
    print(f"\n\n{'=' * 62}")
    print(f"  FINAL SUMMARY")
    print(f"{'=' * 62}")
    print(f"  {'JD':<22} {'Baseline':>9} {'Agent':>8} {'Delta':>7}")
    print(f"  {'─' * 22} {'─' * 9} {'─' * 8} {'─' * 7}")

    for jd_name, s in eval_summary.items():
        delta_str = f"+{s['delta']:.0%}" if s['delta'] >= 0 else f"{s['delta']:.0%}"
        print(f"  {jd_name:<22} {s['baseline']:>9.0%} {s['accuracy']:>8.0%} {delta_str:>7}")

    print(f"{'=' * 62}")

    out_path = "evaluation/day15_results.json"
    os.makedirs("evaluation", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(eval_summary, f, indent=2, default=str)
    print(f"\n  Results saved -> {out_path}")
    print("  Paste results into your evaluation spreadsheet.\n")

    return eval_summary


if __name__ == "__main__":
    run_evaluation()