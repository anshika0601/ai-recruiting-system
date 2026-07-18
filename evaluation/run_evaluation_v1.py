"""
evaluation/run_evaluation.py

Full pipeline evaluation against human rankings.

Runs all resumes in Chroma through the 4-agent pipeline for each enabled JD,
compares agent rankings with the Week 1 human-ranking baseline, and computes
the before-vs-after accuracy improvement.
"""

import json
import os
import sys
import time
from pathlib import Path

# Make imports such as `from graphs.pipeline import run_pipeline` work even
# when this file is run from inside the evaluation directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

import chromadb

from graphs.pipeline import run_pipeline


# ── Configuration ─────────────────────────────────────────────────────────────

CHROMA_PATH = PROJECT_ROOT / "chroma_data"
CHROMA_COLLECTION = "resumes"
OUTPUT_PATH = PROJECT_ROOT / "evaluation" / "day15_results_1.json"
PIPELINE_DELAY_SECONDS = 3


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
#
#JD_FD = """
#Senior Fashion Designer
#Requirements:
#- 5+ years fashion design experience
#- Adobe Suite (Illustrator, Photoshop)
#- Trend forecasting and seasonal collections
#- Experience with luxury or high-end brands
#- Strong portfolio of apparel or accessories design
#"""

JDS = {
    "Software Engineer": JD_SE,
    # "Fashion Designer": JD_FD,
}


# ── Human rankings ────────────────────────────────────────────────────────────
# Every key below exactly matches a Chroma resume ID. Do not replace these keys
# with candidate names: most stored IDs are based on resume template filenames.

HUMAN_RANKINGS = {
    "Account-Manager-Example-PDF": {
        "Software Engineer": 5,
    },  # Michelle Smith
    "Amsterdam-Modern-Resume-Template": {
        "Software Engineer": 9,
    },  # Julie Monroe
    "Dublin-Resume-Template-Modern": {
        "Software Engineer": 7,
    },  # Esther Scott
    "Moscow-Creative-Resume-Template": {
        "Software Engineer": 6,
    },  # Michelle Lopez
    "Personal-trainer-resume-example-3": {
        "Software Engineer": 8,
    },  # Charly Dolman
    "Ux-designer-resume-example-5": {
        "Software Engineer": 3,
    },  # John Huber
    "Sydney-Resume-Template-Modern": {
        "Software Engineer": 4,
    },  # Kristen Connelly
    "Anshika_Bijalwan_Resume_Final": {
        "Software Engineer": 2,
    },  # Anshika Bijalwan
    "Akhilesh-Rawat-Resume": {
        "Software Engineer": 1,
    },  # Akhilesh Rawat
}

BASELINE_ACCURACY = {
    "Software Engineer": 0.0,
}


# ── Helpers ───────────────────────────────────────────────────────────────────


def get_all_resumes():
    """Return all Chroma resumes as (resume_id, metadata, document) tuples."""
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_collection(CHROMA_COLLECTION)
    items = collection.get(include=["metadatas", "documents"])

    ids = items.get("ids") or []
    metadatas = items.get("metadatas") or [{} for _ in ids]
    documents = items.get("documents") or ["" for _ in ids]

    return [
        (resume_id, metadata or {}, document or "")
        for resume_id, metadata, document in zip(ids, metadatas, documents)
    ]


def rank_by_score(results):
    """Convert (resume_id, score) pairs into 1-based descending ranks."""
    sorted_results = sorted(results, key=lambda item: item[1], reverse=True)
    return {
        resume_id: rank
        for rank, (resume_id, _) in enumerate(sorted_results, start=1)
    }


def get_human_rank(resume_id, jd_name):
    """Return the human rank using an exact Chroma-ID lookup."""
    rankings = HUMAN_RANKINGS.get(resume_id)
    if rankings is None:
        return None
    return rankings.get(jd_name)


def validate_human_rankings(resumes):
    """Fail before evaluation if Chroma IDs and ranking keys do not match."""
    chroma_ids = {resume_id for resume_id, _, _ in resumes}
    ranking_ids = set(HUMAN_RANKINGS)

    missing_ranking_ids = chroma_ids - ranking_ids
    unknown_ranking_ids = ranking_ids - chroma_ids

    if missing_ranking_ids:
        print("\nERROR: Chroma IDs missing from HUMAN_RANKINGS:")
        for resume_id in sorted(missing_ranking_ids):
            print(f"  - {resume_id!r}")

    if unknown_ranking_ids:
        print("\nWARNING: HUMAN_RANKINGS keys not found in Chroma:")
        for resume_id in sorted(unknown_ranking_ids):
            print(f"  - {resume_id!r}")

    missing_jd_ranks = []
    for resume_id in sorted(chroma_ids & ranking_ids):
        for jd_name in JDS:
            if HUMAN_RANKINGS[resume_id].get(jd_name) is None:
                missing_jd_ranks.append((resume_id, jd_name))

    if missing_jd_ranks:
        print("\nERROR: Human ranks missing for enabled JDs:")
        for resume_id, jd_name in missing_jd_ranks:
            print(f"  - {resume_id!r}: {jd_name!r}")

    if missing_ranking_ids or missing_jd_ranks:
        raise ValueError(
            "Human ranking validation failed. Fix the IDs/ranks shown above "
            "before running the pipeline."
        )


def normalize_score(result, resume_id):
    """Read final_score from a pipeline result and return it as a float."""
    raw_score = result.get("final_score")
    if raw_score is None:
        return 0.0

    try:
        return float(raw_score)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Pipeline returned a non-numeric final_score for {resume_id!r}: "
            f"{raw_score!r}"
        ) from exc


# ── Evaluation ────────────────────────────────────────────────────────────────


def run_evaluation():
    print("\n" + "=" * 62)
    print("  FULL EVALUATION  —  Multi-Agent Pipeline vs Baseline")
    print("=" * 62)

    resumes = get_all_resumes()
    if not resumes:
        raise RuntimeError(
            f"No resumes were found in Chroma collection "
            f"{CHROMA_COLLECTION!r} at {str(CHROMA_PATH)!r}."
        )

    print(f"\n  Resumes in Chroma: {len(resumes)}")
    for resume_id, metadata, _ in resumes:
        print(f"    {resume_id:35s} -> {metadata.get('name', '?')}")

    # Validate IDs before making any potentially expensive pipeline calls.
    validate_human_rankings(resumes)

    metadata_by_id = {
        resume_id: metadata for resume_id, metadata, _ in resumes
    }
    eval_summary = {}

    for jd_name, jd_text in JDS.items():
        print(f"\n\n{'=' * 62}")
        print(f"  JD: {jd_name}")
        print("=" * 62)

        jd_scores = {}
        jd_details = {}

        for index, (resume_id, metadata, document_text) in enumerate(resumes):
            candidate = metadata.get("name") or resume_id
            print(f"\n  [pipeline] {candidate}")

            result = run_pipeline(
                resume_text=document_text,
                jd_text=jd_text,
                resume_id=resume_id,
                candidate_name=candidate,
            )

            if not isinstance(result, dict):
                raise TypeError(
                    f"run_pipeline returned {type(result).__name__} for "
                    f"{resume_id!r}; expected a dictionary."
                )

            score = normalize_score(result, resume_id)
            jd_scores[resume_id] = score
            jd_details[resume_id] = result
            print(f"  [pipeline] Score: {score:.2f}/10")

            # Avoid an unnecessary delay after the final resume.
            if index < len(resumes) - 1 and PIPELINE_DELAY_SECONDS > 0:
                time.sleep(PIPELINE_DELAY_SECONDS)

        agent_ranks = rank_by_score(jd_scores.items())

        print(f"\n  {'─' * 62}")
        print(
            f"  {'Candidate':<26} {'Agent':>6} {'Human':>6} "
            f"{'Score':>7} {'Match':>6}"
        )
        print(f"  {'─' * 26} {'─' * 6} {'─' * 6} {'─' * 7} {'─' * 6}")

        matches = 0
        total = 0
        details = []

        for resume_id, agent_rank in sorted(
            agent_ranks.items(), key=lambda item: item[1]
        ):
            metadata = metadata_by_id[resume_id]
            candidate = metadata.get("name") or resume_id
            score = jd_scores[resume_id]
            human_rank = get_human_rank(resume_id, jd_name)

            if human_rank is not None:
                matched = abs(agent_rank - human_rank) <= 1
                matches += int(matched)
                total += 1
                symbol = "OK" if matched else "X"
                human_rank_display = str(human_rank)
            else:
                # Validation should prevent this branch, but retain it as a
                # safeguard if the code is changed later.
                matched = None
                symbol = "?"
                human_rank_display = "?"

            print(
                f"  {candidate:<26} {agent_rank:>6} "
                f"{human_rank_display:>6} {score:>6.2f}  {symbol:>6}"
            )

            details.append(
                {
                    "resume_id": resume_id,
                    "candidate": candidate,
                    "agent_rank": agent_rank,
                    "human_rank": human_rank,
                    "score": score,
                    "match": matched,
                    "needs_review": jd_details[resume_id].get("needs_review"),
                    "guard_flags": len(
                        jd_details[resume_id].get("guard_flags") or []
                    ),
                }
            )

        accuracy = matches / total if total > 0 else 0.0
        baseline = BASELINE_ACCURACY.get(jd_name, 0.0)
        delta = accuracy - baseline
        delta_str = f"{delta:+.0%}"

        print(f"  {'─' * 62}")
        print(
            f"  Agent accuracy   : {accuracy:.0%}  "
            f"({matches}/{total} within 1 rank)"
        )
        print(f"  Baseline Week 1  : {baseline:.0%}")
        print(f"  Improvement      : {delta_str}")
        print(f"  {'─' * 62}")

        eval_summary[jd_name] = {
            "accuracy": accuracy,
            "baseline": baseline,
            "delta": delta,
            "matches": matches,
            "total": total,
            "details": details,
        }

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n\n{'=' * 62}")
    print("  FINAL SUMMARY")
    print("=" * 62)
    print(f"  {'JD':<22} {'Baseline':>9} {'Agent':>8} {'Delta':>7}")
    print(f"  {'─' * 22} {'─' * 9} {'─' * 8} {'─' * 7}")

    for jd_name, summary in eval_summary.items():
        delta_str = f"{summary['delta']:+.0%}"
        print(
            f"  {jd_name:<22} {summary['baseline']:>9.0%} "
            f"{summary['accuracy']:>8.0%} {delta_str:>7}"
        )

    print("=" * 62)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as output_file:
        json.dump(eval_summary, output_file, indent=2, default=str)

    try:
        displayed_output_path = OUTPUT_PATH.relative_to(PROJECT_ROOT)
    except ValueError:
        displayed_output_path = OUTPUT_PATH

    print(f"\n  Results saved -> {displayed_output_path}")
    print("  Paste results into your evaluation spreadsheet.\n")

    return eval_summary


if __name__ == "__main__":
    run_evaluation()
