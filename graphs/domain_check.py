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

DOMAIN_CHECK_PROMPT = """You must classify a candidate's domain relevance to a Software Engineering role.

━━━ CLASSIFICATION EXAMPLES (learn from these) ━━━

EXAMPLE 1 → MISMATCH:
Titles: Hair Stylist, Salon Manager
Skills: Cutting, Colouring, Customer Service
Achievements: None quantified
Certifications: None technical
Result: {{"verdict": "MISMATCH", "reasoning": "Zero technical overlap"}}

EXAMPLE 2 → ADJACENT:
Titles: Video Producer, Production Coordinator
Skills: Adobe CS5, Hootsuite, Final Cut Pro
Achievements: Increased engagement by 20%, coordinated 100-person team
Certifications: Hootsuite Certified, Adobe CS5 Certified
Result: {{"verdict": "ADJACENT", "reasoning": "Technical certifications + quantified metrics + large-scale pipeline coordination"}}

EXAMPLE 3 → ADJACENT:
Titles: Marketing Analyst, Campaign Manager
Skills: Google Analytics, Salesforce, Excel, A/B testing
Achievements: Reduced CPA by 35%, managed $2M budget
Certifications: Google Analytics Certified
Result: {{"verdict": "ADJACENT", "reasoning": "Data analytics tools + quantified outcomes + systems thinking"}}

EXAMPLE 4 → MATCH:
Titles: Backend Developer, Software Engineer Intern
Skills: Python, REST APIs, PostgreSQL, Docker
Achievements: Reduced latency by 40%
Result: {{"verdict": "MATCH", "reasoning": "Direct software engineering background"}}

EXAMPLE 5 → MISMATCH:
Titles: Fashion Designer, Textile Artist
Skills: Sketching, Fabric selection, Hand sewing
Achievements: None quantified, no tools
Result: {{"verdict": "MISMATCH", "reasoning": "No technical tools, no metrics, no engineering overlap"}}

━━━ CLASSIFICATION RULE SUMMARY ━━━
ADJACENT requires ANY TWO of:
  [A] Technical software certification (Adobe, Hootsuite, AWS, Google, Salesforce)
  [B] Quantified metric (%, $, time, people count, scale)
  [C] Coordination of 20+ people or multi-stage technical workflow
  [D] Data-driven decision making (analytics, reporting, optimisation)

MISMATCH = zero of the above signals present.

━━━ NOW CLASSIFY THIS CANDIDATE ━━━

JOB DESCRIPTION (target role):
{jd_text}

CANDIDATE:
Titles: {job_titles}
Skills: {skills}
Companies: {companies}
Achievements: {achievements}
Resume excerpt: {resume_context}

Return ONLY this JSON (no other text):
{{
  "verdict": "MATCH" or "ADJACENT" or "MISMATCH",
  "candidate_domain": "short phrase",
  "jd_domain": "short phrase",
  "reasoning": "cite which signals A/B/C/D triggered ADJACENT, or why none exist for MISMATCH",
  "transferable_skills": ["signal 1", "signal 2"]
}}"""

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
            "domain_verdict":      "MATCH",
            "domain_penalty":      0.0,
            "transferable_skills": [],
            "error":               None,
        }

    # Build context for the LLM
    target_role    = state.get("jd_text", "").split("\n")[0].strip() or "Software Engineering"
    job_titles     = ", ".join(facts.get("job_titles",   []) or ["unknown"])
    skills         = ", ".join(facts.get("skills_found", [])[:20])
    companies      = ", ".join(facts.get("companies",    []) or ["unknown"])
    achievements   = (
        "\n".join(facts.get("achievements", [])[:5])
        if facts.get("achievements")
        else "See resume excerpt below"
    )
    # Pass raw resume text so certifications and metrics are visible
    resume_context = state.get("resume_text", "")[:1500]

    prompt = DOMAIN_CHECK_PROMPT.format(
        target_role    = target_role,
        jd_text        = state["jd_text"][:800],
        job_titles     = job_titles,
        skills         = skills,
        companies      = companies,
        achievements   = achievements,
        resume_context = resume_context,
    )

    try:
        response = _get_client().chat.completions.create(
            model       = MODEL,
            messages    = [
                {"role": "system", "content": DOMAIN_CHECK_SYSTEM},
                {"role": "user",   "content": prompt},
            ],
            max_tokens  = 300,
            temperature = 0.0,
        )

        raw = response.choices[0].message.content.strip()

        # Strip markdown code fences if model wraps output
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        # Extract JSON object robustly
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start != -1 and end > start:
            raw = raw[start:end]

        result  = json.loads(raw)
        verdict = result.get("verdict", "MATCH").upper().strip()

        # Normalise any unexpected model output
        if verdict not in ("MATCH", "ADJACENT", "MISMATCH"):
            verdict = "MATCH"

        penalty  = DOMAIN_PENALTIES.get(verdict, 0.0)
        transfer = result.get("transferable_skills", [])

        # Enforce ADJACENT contract — must have at least 2 signals
        if verdict == "ADJACENT" and len(transfer) < 2:
            transfer = [result.get("reasoning", "transferable signals detected")]

        # Logging
        print(f"[domain] {verdict} — {result.get('candidate_domain')} "
              f"vs {result.get('jd_domain')}")

        if verdict == "MISMATCH":
            print(f"[domain] ⚠ Applying domain penalty: -{penalty} pts")
        elif verdict == "ADJACENT":
            print(f"[domain] ℹ Transferable: {transfer}")
            print(f"[domain] ℹ Penalty: -{penalty} pts")

        return {
            "domain_verdict":      verdict,
            "domain_penalty":      penalty,
            "transferable_skills": transfer,
            "error":               None,
        }

    except json.JSONDecodeError as e:
        print(f"[domain] ✗ JSON parse error: {e} — defaulting to MATCH")
        return {
            "domain_verdict":      "MATCH",
            "domain_penalty":      0.0,
            "transferable_skills": [],
            "error":               str(e),
        }

    except Exception as e:
        print(f"[domain] ✗ Unexpected error: {e} — defaulting to MATCH")
        return {
            "domain_verdict":      "MATCH",
            "domain_penalty":      0.0,
            "transferable_skills": [],
            "error":               str(e),
        }
