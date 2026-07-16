"""
graphs/domain_check.py

Domain relevance pre-check — runs between extractor and scorer.

Problem it solves:
  The scorer treats all resumes equally regardless of domain.
  A fashion designer and a software engineer both go through
  the same 15 LLM scoring calls, producing misleadingly similar
  scores when the domain mismatch should be obvious immediately.

Solution:
  A lightweight pre-check that classifies candidate domain vs JD domain.
  Three outcomes:
    MATCH    -> proceed normally, no adjustment
    ADJACENT -> proceed with transferable_skills flag passed to scorer
    MISMATCH -> apply domain penalty, skip expensive scorer entirely

Fixes both Day 15 failure modes:
  Michelle Lopez (Fashion -> SE JD):   MISMATCH -> penalty -3.0 pts
  Kristen Connelly (Video -> SE JD):   ADJACENT -> small penalty + credit
"""
import json
import os
from typing import Any, Dict

from dotenv import load_dotenv
from groq import Groq
from graphs.pipeline_state import PipelineState

load_dotenv()

_client: Groq | None = None

def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return _client

MODEL = "llama-3.1-8b-instant"

DOMAIN_PENALTIES = {
    "MISMATCH": 3.0,
    "ADJACENT": 0.5,
    "MATCH":    0.0,
}

DOMAIN_CHECK_SYSTEM = """You are a domain relevance classifier for a technical hiring pipeline.
Analyze the candidate's background against the job description and classify domain relevance.
Respond with valid JSON only. No preamble, no markdown, no explanations."""

DOMAIN_CHECK_PROMPT = """Classify the domain relevance between this candidate and the target role.

TARGET ROLE: {target_role}
JOB DESCRIPTION:
{jd_text}

CANDIDATE BACKGROUND:
Titles: {job_titles}
Skills: {skills}
Companies: {companies}
Key Achievements/Metrics: {achievements}
Resume Context (first 1200 chars):
{resume_context}

CLASSIFICATION RULES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MATCH: Direct overlap with target domain (e.g., Software Engineer for SE role).

ADJACENT: Different primary field BUT shows clear transferable signals.
Classify as ADJACENT if ANY TWO of these are present:
  • Technical tooling proficiency (Adobe Suite, analytics platforms, certified software)
  • Quantified metrics showing analytical/data-driven decisions (%, $, time saved, scale)
  • Large-scale project/team coordination (50+ people, multi-stage workflows, pipelines)
  • Professional certifications or structured methodology adoption
  • Systems thinking or process optimization

MISMATCH: Zero technical overlap. No tooling, no metrics, no scale coordination,
no certifications. Purely non-technical role with no transferable signals.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CRITICAL RULES:
1. Default toward ADJACENT over MISMATCH when uncertain.
2. Do NOT require programming experience for ADJACENT.
3. Certifications or quantified project metrics alone are sufficient for ADJACENT.
4. Only use MISMATCH when there is genuinely zero overlap.

Return ONLY this JSON:
{{
  "verdict": "MATCH" | "ADJACENT" | "MISMATCH",
  "candidate_domain": "one phrase describing their field",
  "jd_domain": "one phrase from the JD",
  "reasoning": "one sentence explaining the classification based on the rules above",
  "transferable_skills": ["list specific transferable signals found, e.g., 'Adobe CS5 certification', 'data-driven editing (20% metric)', '100-person production coordination']
}}

For MISMATCH, transferable_skills must be [].
For ADJACENT, transferable_skills must contain at least 2 items.
"""


def domain_check_node(state: PipelineState) -> Dict[str, Any]:
    """
    LangGraph node: classify domain relevance before scoring.
    Returns domain_verdict, domain_penalty, transferable_skills.
    """
    candidate = state.get("candidate_name", "unknown")
    print(f"[domain] Checking: {candidate}")

    facts = state.get("extracted_facts")
    if not facts:
        return {
            "domain_verdict": "MATCH",
            "domain_penalty": 0.0,
            "transferable_skills": [],
            "error": None,
        }

    # Safely extract context for the LLM
    target_role = state.get("jd_text", "").split("\n")[0].strip() or "Software Engineering"
    job_titles = ", ".join(facts.get("job_titles", []) or ["unknown"])
    skills = ", ".join(facts.get("skills_found", [])[:20])
    companies = ", ".join(facts.get("companies", []) or ["unknown"])
    achievements = "\n".join(facts.get("achievements", [])[:5]) if facts.get("achievements") else "Not explicitly extracted"
    resume_context = state.get("resume_text", "")[:1200]

    prompt = DOMAIN_CHECK_PROMPT.format(
        target_role    = target_role,
        jd_text        = state["jd_text"][:1000],
        job_titles     = job_titles,
        skills         = skills,
        companies      = companies,
        achievements   = achievements,
        resume_context = resume_context,
    )

    try:
        response = _get_client().chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": DOMAIN_CHECK_SYSTEM},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=300,
            temperature=0.0,
        )

        raw = response.choices[0].message.content.strip()
        
        # Strip markdown code blocks if present
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start != -1 and end > start:
            raw = raw[start:end]

        result   = json.loads(raw)
        verdict  = result.get("verdict", "MATCH")
        penalty  = DOMAIN_PENALTIES.get(verdict, 0.0)
        transfer = result.get("transferable_skills", [])

        # Enforce ADJACENT contract
        if verdict == "ADJACENT" and len(transfer) < 2:
            transfer = [result.get("reasoning", "transferable signals present")]

        print(f"[domain] {verdict} — {result.get('candidate_domain')} "
              f"vs {result.get('jd_domain')}")
        if verdict == "MISMATCH":
            print(f"[domain] ⚠ Applying domain penalty: -{penalty} pts")
        elif verdict == "ADJACENT":
            print(f"[domain] ℹ Transferable skills: {transfer}")

        return {
            "domain_verdict":      verdict,
            "domain_penalty":      penalty,
            "transferable_skills": transfer,
            "error": None,
        }

    except Exception as e:
        print(f"[domain] ✗ Error: {e} — defaulting to MATCH (safe fallback)")
        return {
            "domain_verdict":      "MATCH",
            "domain_penalty":      0.0,
            "transferable_skills": [],
            "error": None,
        }