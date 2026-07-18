"""
graphs/guard_agent.py

The Guard Agent — third node in the pipeline.

Job: detect keyword stuffing, skill inflation, and resume gaming.
Why it exists: embedding similarity and rubric scoring can both be
gamed by stuffing a resume with JD keywords. The guard agent's only
job is to look for signs that the resume is optimised to beat automated
screening rather than reflect genuine skills.

Input  (from PipelineState): extracted_facts, jd_text, resume_text
Output (to PipelineState):   guard_flags, guard_penalty

Key rules for fair assessment:
  - Student/personal projects COUNT as skill demonstration
  - 0 years paid experience is EXPECTED for new grads — not a flag
  - Only flag skills with ZERO evidence anywhere in resume text
  - EXPERIENCE_MISMATCH is only valid when years claimed > evidence shows
"""
import json
import os
from typing import Any, Dict, List
from langsmith import traceable  
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

# Upgraded model — 8b cannot follow nuanced multi-rule guard prompts
MODEL = "llama-3.3-70b-versatile"

# Penalty per confirmed flag
SEVERITY_WEIGHTS = {"low": 0.25, "medium": 0.5, "high": 1.0}
PENALTY_PER_FLAG = 0.5
MAX_PENALTY      = 2.0


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

GUARD_SYSTEM = """You are a fair and precise resume authenticity checker.
Your job is to detect signs of resume gaming — but you must NOT penalise
candidates for legitimate experience gained through projects, internships,
or self-directed work.

ABSOLUTE RULES you must follow:
  1. Student projects, personal projects, and open-source work COUNT as
     valid skill demonstration. If a skill appears in ANY project 
     description, it is NOT keyword stuffing.
  2. 0 years of paid work experience is NORMAL and EXPECTED for new
     graduates and interns. Never flag this as EXPERIENCE_MISMATCH.
  3. EXPERIENCE_MISMATCH only applies when someone CLAIMS seniority
     (e.g., "10 years", "Senior Engineer") that contradicts their evidence.
  4. Only flag KEYWORD_STUFFING when a skill appears in the skills list
     AND has zero mention anywhere else in the entire resume text.
  5. Be conservative — when uncertain, do NOT flag.

Respond with valid JSON only. No preamble, no markdown."""

GUARD_PROMPT = """Check this resume for authenticity issues.
Read the FULL resume text carefully before flagging anything.

━━━ JOB DESCRIPTION ━━━
{jd_text}

━━━ EXTRACTED FACTS ━━━
{facts_json}

━━━ FULL RESUME TEXT ━━━
{resume_text}

━━━ WHAT TO CHECK ━━━

1. KEYWORD_STUFFING
   ONLY flag if: skill is in the skills list AND the word does not appear
   anywhere in projects, achievements, or experience descriptions.
   DO NOT flag if: the skill appears in any project bullet point.
   Evidence required: name the skill AND confirm it has zero project usage.

2. SKILL_INFLATION  
   ONLY flag if: skill is claimed at expert/senior level but the only
   evidence is a beginner tutorial or single trivial usage.
   DO NOT flag project-level usage as inflation.

3. VAGUE_ACHIEVEMENTS
   ONLY flag if: an achievement claims a number (%, $, time) but the
   number is literally missing or replaced with "X" or "N".
   DO NOT flag achievements that have real numbers with missing baselines —
   those are LOW severity at most.

4. JD_MIRRORING
   ONLY flag if: 5+ consecutive words from the JD appear verbatim in the
   resume in a context that suggests copy-paste.
   DO NOT flag common industry terminology that naturally overlaps.

5. EXPERIENCE_MISMATCH
   ONLY flag if: candidate CLAIMS a specific number of years that is
   contradicted by their actual timeline.
   NEVER flag: 0 years experience for a student or new grad.
   NEVER flag: "shipped production systems" if those are student projects
               — student production projects are legitimate.

━━━ SEVERITY GUIDE ━━━
  high   — Clear, unambiguous gaming that materially inflates candidacy
  medium — Probable issue but some legitimate explanation possible
  low    — Minor concern, worth noting but not penalising heavily

━━━ OUTPUT ━━━
Return ONLY this JSON:
{{
  "flags": [
    {{
      "type": "KEYWORD_STUFFING | SKILL_INFLATION | VAGUE_ACHIEVEMENTS | JD_MIRRORING | EXPERIENCE_MISMATCH",
      "evidence": "quote the specific text that triggered this flag",
      "severity": "low | medium | high",
      "reasoning": "why this is a flag given the rules above"
    }}
  ],
  "overall_assessment": "clean | suspicious | likely_gamed",
  "summary": "1-2 sentence overall assessment"
}}

If no clear issues found → return flags as [] and overall_assessment as "clean".
When in doubt → do NOT flag.
"""


# Penalty calculation
# ---------------------------------------------------------------------------
@traceable(name="GuardAgent", tags=["guard"])  
def _calculate_penalty(flags: List[Dict]) -> float:
    """
    Convert flagged issues into a score penalty (0.0 to MAX_PENALTY).
    High severity flags cost more than low severity ones.
    """
    if not flags:
        return 0.0
    total = sum(
        SEVERITY_WEIGHTS.get(f.get("severity", "low"), 0.25)
        for f in flags
    )
    return round(min(total * PENALTY_PER_FLAG, MAX_PENALTY), 3)


# ---------------------------------------------------------------------------
# Guard node
# ---------------------------------------------------------------------------
@traceable(name="GuardAgent", tags=["guard"])  

def guard_node(state: PipelineState) -> Dict[str, Any]:
    """
    LangGraph node: check for resume gaming and keyword stuffing.
    Returns partial state update with guard_flags and guard_penalty.
    """
    candidate = state.get("candidate_name", "unknown")
    print(f"[guard] Checking: {candidate}")

    if not state.get("extracted_facts"):
        print("[guard] ✗ No extracted_facts — skipping guard check")
        return {"guard_flags": [], "guard_penalty": 0.0}

    facts       = state["extracted_facts"]
    jd_text     = state["jd_text"]
    resume_text = state["resume_text"]

    prompt = GUARD_PROMPT.format(
        jd_text     = jd_text[:2000],
        facts_json  = json.dumps(facts, indent=2)[:2000],
        # Pass full resume text so guard sees ALL project mentions
        resume_text = resume_text[:3000],
    )

    try:
        response = _get_client().chat.completions.create(
            model    = MODEL,
            messages = [
                {"role": "system", "content": GUARD_SYSTEM},
                {"role": "user",   "content": prompt},
            ],
            max_tokens  = 800,
            temperature = 0.0,   # deterministic — guard must be consistent
        )

        raw = response.choices[0].message.content.strip()

        # Strip markdown fences if model adds them
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start != -1 and end > start:
            raw = raw[start:end]

        result     = json.loads(raw)
        flags      = result.get("flags", [])
        penalty    = _calculate_penalty(flags)
        assessment = result.get("overall_assessment", "clean")

        if flags:
            print(f"[guard] ⚠ {len(flags)} flag(s) detected — "
                  f"assessment: {assessment} — penalty: -{penalty}")
            for f in flags:
                print(f"[guard]   [{f.get('severity','?').upper()}] "
                      f"{f.get('type','?')}: {f.get('evidence','')[:80]}")
        else:
            print(f"[guard] ✓ Clean — no issues detected")

        return {
            "guard_flags":   flags,
            "guard_penalty": penalty,
            "error":         None,
        }

    except json.JSONDecodeError as e:
        print(f"[guard] ✗ JSON parse error: {e} — applying zero penalty")
        return {
            "guard_flags":   [],
            "guard_penalty": 0.0,
            "error":         f"Guard JSON error: {str(e)}",
        }

    except Exception as e:
        print(f"[guard] ✗ Unexpected error: {e} — applying zero penalty")
        return {
            "guard_flags":   [],
            "guard_penalty": 0.0,
            "error":         f"Guard failed: {str(e)}",
        }