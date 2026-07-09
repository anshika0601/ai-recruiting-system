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
"""
import json
import os
from typing import Any, Dict, List

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

# Penalty per confirmed flag (applied as deduction from final score 0-10)
PENALTY_PER_FLAG = 0.5
MAX_PENALTY      = 2.0   # cap so one bad resume can't go below 0


GUARD_SYSTEM = """You are a resume authenticity checker.
Your job is to detect signs that a resume has been artificially optimised
to pass automated screening rather than reflecting genuine skills.
Respond with valid JSON only. No preamble, no markdown fences."""

GUARD_PROMPT = """Check this candidate's resume for the following red flags.
Be fair — only flag clear evidence, not suspicion.

JOB DESCRIPTION:
{jd_text}

EXTRACTED FACTS:
{facts_json}

RESUME RAW TEXT (for pattern checking):
{resume_text}

Check for these specific issues:

1. KEYWORD_STUFFING: Does the skills list contain keywords that appear
   nowhere in actual experience or projects? (Skills listed but never
   demonstrated in any role or project)

2. SKILL_INFLATION: Are skills listed at an implied expert level but
   only demonstrated in trivial contexts? (e.g. "AWS" listed but only
   used one S3 bucket in a tutorial)

3. VAGUE_ACHIEVEMENTS: Are achievements worded to sound quantified but
   actually contain no real numbers? (e.g. "improved performance by X%"
   where X is not stated)

4. JD_MIRRORING: Does the resume seem to copy exact phrases from the JD
   verbatim? (sign of last-minute tailoring beyond normal customisation)

5. EXPERIENCE_MISMATCH: Does the stated experience level contradict the
   evidence? (e.g. claims "senior" but has 1 year experience)

Return ONLY this JSON:
{{
  "flags": [
    {{
      "type": "KEYWORD_STUFFING | SKILL_INFLATION | VAGUE_ACHIEVEMENTS | JD_MIRRORING | EXPERIENCE_MISMATCH",
      "evidence": "specific example from resume that triggered this flag",
      "severity": "low | medium | high"
    }}
  ],
  "overall_assessment": "clean | suspicious | likely_gamed",
  "summary": "1-2 sentence overall assessment"
}}

If no issues found, return flags as empty list [] and overall_assessment as "clean"."""


# Penalty calculation
# ---------------------------------------------------------------------------

SEVERITY_WEIGHTS = {"low": 0.25, "medium": 0.5, "high": 1.0}

def _calculate_penalty(flags: List[Dict]) -> float:
    """
    Convert flagged issues into a score penalty (0.0 to MAX_PENALTY).
    High severity flags cost more than low severity ones.
    """
    if not flags:
        return 0.0
    total = sum(SEVERITY_WEIGHTS.get(f.get("severity", "low"), 0.25) for f in flags)
    return min(total * PENALTY_PER_FLAG, MAX_PENALTY)


# Guard node
# ---------------------------------------------------------------------------

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

    facts     = state["extracted_facts"]
    jd_text   = state["jd_text"]
    resume_text = state["resume_text"]

    prompt = GUARD_PROMPT.format(
        jd_text=jd_text[:2000],
        facts_json=json.dumps(facts, indent=2)[:2000],
        resume_text=resume_text[:2000],
    )

    try:
        response = _get_client().chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": GUARD_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            max_tokens=600,
            temperature=0.0,  # guard must be deterministic
        )

        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start != -1 and end > start:
            raw = raw[start:end]

        result  = json.loads(raw)
        flags   = result.get("flags", [])
        penalty = _calculate_penalty(flags)
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
            "error": None,
        }

    except Exception as e:
        print(f"[guard] ✗ Error: {e}")
        # On failure, apply no penalty — don't punish candidate for our bug
        return {
            "guard_flags":   [],
            "guard_penalty": 0.0,
            "error": f"Guard failed: {str(e)}",
        }