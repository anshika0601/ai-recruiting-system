"""
graphs/extractor_agent.py

The Extractor Agent — first node in the pipeline.

Job: read raw resume text + JD, output clean structured facts.
Why it exists: every downstream agent (scorer, guard) works from
structured facts, not raw text. This keeps scorer prompts short,
focused, and consistent across all resumes regardless of format.

Input  (from PipelineState): resume_text, jd_text
Output (to PipelineState):   extracted_facts
"""
import json
import os
from typing import Any, Dict

from dotenv import load_dotenv
from groq import Groq

from graphs.pipeline_state import PipelineState

load_dotenv()

# Groq client — loaded once
# ---------------------------------------------------------------------------

_client: Groq | None = None

def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return _client

MODEL = "llama-3.1-8b-instant"


# Extraction prompt
# ---------------------------------------------------------------------------

EXTRACTOR_SYSTEM = """You are a precise resume data extractor. 
Your only job is to extract structured facts from a resume against a job description.
You must respond with valid JSON only. No preamble, no explanation, no markdown fences.
Every field is required. Use empty lists [] if no data found, never null."""

EXTRACTOR_PROMPT = """Extract the following from the resume below.

JOB DESCRIPTION:
{jd_text}

RESUME:
{resume_text}

Return ONLY this JSON structure:
{{
  "skills_found": ["list of technical and soft skills explicitly mentioned in resume"],
  "jd_requirements": ["list of skills/requirements extracted from the JD"],
  "skills_matched": ["skills present in BOTH resume and JD requirements"],
  "skills_missing": ["JD requirements NOT found anywhere in resume"],
  "years_experience": <integer total years of work experience, 0 if unclear>,
  "companies": ["list of company names worked at"],
  "job_titles": ["list of job titles held, most recent first"],
  "achievements": ["list of quantified achievements e.g. 'Increased revenue by 30%'"],
  "education": ["list of degrees e.g. 'BSc Computer Science, MIT, 2018'"],
  "red_flags": ["list of concerns e.g. '8-month employment gap in 2021', 'changed jobs 4 times in 2 years'"],
  "career_trajectory": "improving | stable | declining | unclear"
}}"""


# Extractor node
# ---------------------------------------------------------------------------

def extractor_node(state: PipelineState) -> Dict[str, Any]:
    """
    LangGraph node: extract structured facts from resume + JD.
    Returns partial state update — LangGraph merges it automatically.
    """
    print(f"[extractor] Processing: {state.get('candidate_name', 'unknown')}")

    prompt = EXTRACTOR_PROMPT.format(
        jd_text=state["jd_text"][:3000],       # cap to avoid token limits
        resume_text=state["resume_text"][:4000]
    )

    try:
        response = _get_client().chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": EXTRACTOR_SYSTEM},
                {"role": "user",   "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.0,   # extraction must be deterministic
        )

        raw = response.choices[0].message.content.strip()

        # Strip markdown fences if model adds them despite instructions
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        facts = json.loads(raw)

        # Validate required keys exist
        required = [
            "skills_found", "jd_requirements", "skills_matched",
            "skills_missing", "years_experience", "companies",
            "job_titles", "achievements", "education",
            "red_flags", "career_trajectory"
        ]
        for key in required:
            if key not in facts:
                facts[key] = [] if key != "years_experience" else 0
                if key == "career_trajectory":
                    facts[key] = "unclear"

        print(f"[extractor] ✓ Found {len(facts['skills_found'])} skills, "
              f"{facts['years_experience']} yrs exp, "
              f"{len(facts['skills_matched'])} JD matches")

        return {"extracted_facts": facts, "error": None}

    except json.JSONDecodeError as e:
        print(f"[extractor] ✗ JSON parse failed: {e}")
        print(f"[extractor] Raw response: {raw[:200]}")
        return {
            "extracted_facts": None,
            "error": f"Extractor JSON parse failed: {str(e)}"
        }
    except Exception as e:
        print(f"[extractor] ✗ Error: {e}")
        return {
            "extracted_facts": None,
            "error": f"Extractor failed: {str(e)}"
        }



# CLI test 
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from app.parser import parse_resume

    if len(sys.argv) < 2:
        print("Usage: python -m graphs.extractor_agent resume.pdf")
        sys.exit(1)

    # Sample job description for testing
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

    # Simulate pipeline state
    state: PipelineState = {
        "resume_text":    parsed["raw_text"],
        "jd_text":        test_jd,
        "resume_id":      "test",
        "candidate_name": parsed["name"],
        "extracted_facts":   None,
        "dimension_scores":  None,
        "guard_flags":       None,
        "guard_penalty":     None,
        "final_score":       None,
        "score_breakdown":   None,
        "needs_review":      None,
        "error":             None,
    }

    result = extractor_node(state)

    print("\n── Extracted Facts ──")
    print(json.dumps(result["extracted_facts"], indent=2))