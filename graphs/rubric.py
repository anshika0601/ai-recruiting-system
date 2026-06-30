"""
graphs/rubric.py

The scoring rubric — single source of truth for dimension definitions,
weights, and anchor scales. Both the scorer agent and the aggregator
import from this file so they can never drift out of sync.
"""
from typing import Dict, List, TypedDict


class DimensionDef(TypedDict):
    name:        str          # canonical key, e.g. "core_skill_match"
    label:       str          # human-readable, e.g. "Core Skill Match"
    weight:      float        # contribution to final score (sums to 1.0 across non-penalty dims)
    is_penalty:  bool         # True only for red_flags
    anchors:     Dict[int, str]  # 1-5 scale anchor descriptions


RUBRIC: Dict[str, DimensionDef] = {

    "core_skill_match": {
        "name": "core_skill_match",
        "label": "Core Skill Match",
        "weight": 0.30,
        "is_penalty": False,
        "anchors": {
            1: "No required skills present",
            2: "1-2 required skills, no evidence of depth",
            3: "Most required skills present, surface-level evidence",
            4: "All required skills present with project-level evidence",
            5: "All required skills present with measurable, repeated production usage",
        },
    },

    "experience_relevance": {
        "name": "experience_relevance",
        "label": "Experience Relevance",
        "weight": 0.25,
        "is_penalty": False,
        "anchors": {
            1: "No relevant experience to the role",
            2: "Tangentially related experience only",
            3: "Some directly relevant experience, but below required years/seniority",
            4: "Experience matches required years and domain closely",
            5: "Experience exceeds requirements in years and domain depth",
        },
    },

    "achievement_evidence": {
        "name": "achievement_evidence",
        "label": "Achievement Evidence",
        "weight": 0.20,
        "is_penalty": False,
        "anchors": {
            1: "No achievements listed, only vague responsibilities",
            2: "Achievements listed but not quantified",
            3: "Some quantified achievements, limited scope",
            4: "Multiple quantified achievements with clear business impact",
            5: "Consistently quantified, high-impact achievements across roles/projects",
        },
    },

    "career_trajectory": {
        "name": "career_trajectory",
        "label": "Career Trajectory",
        "weight": 0.15,
        "is_penalty": False,
        "anchors": {
            1: "Declining scope/seniority over time, or no career history to assess",
            2: "Stagnant — no growth in scope or seniority",
            3: "Some growth, unclear pattern (e.g. student/early career, hard to assess)",
            4: "Clear upward trajectory in scope or seniority",
            5: "Strong, consistent upward trajectory with increasing responsibility",
        },
    },

    "red_flags": {
        "name": "red_flags",
        "label": "Red Flags",
        "weight": 0.10,
        "is_penalty": True,
        "anchors": {
            1: "Severe concerns — major unexplained gaps, frequent job-hopping, inconsistencies",
            2: "Notable concerns — one significant gap or pattern of short tenures",
            3: "Minor concerns — small gap or one short tenure, explainable",
            4: "No concerns — clean history with minor non-issues",
            5: "No concerns whatsoever — fully consistent, clean history",
        },
    },
}

DIMENSION_ORDER: List[str] = list(RUBRIC.keys())


def format_anchors(dimension: str) -> str:
    """Render a dimension's anchor scale as readable text for prompts."""
    anchors = RUBRIC[dimension]["anchors"]
    lines = [f"{score} = {desc}" for score, desc in anchors.items()]
    return "\n".join(lines)