
import re
from typing import Dict, List, Optional, Any


import pdfplumber 

SECTION_HEADERS: Dict[str, List[str]] = {
    "experience": [
        "experience", "work experience", "employment history",
        "work history", "professional experience", "career history",
        "relevant experience",
    ],
    "education": [
        "education", "educational background", "education history",
        "academic background", "qualifications", "academic qualifications",
        "academic history",
    ],
    "skills": [
        "skills", "technical skills", "core competencies",
        "key skills", "expertise", "technologies", "tools",
        "skills & tools", "skills and tools",
    ],
    "projects": [
        "projects", "personal projects", "key projects",
        "portfolio", "notable projects", "selected projects",
    ],
    "summary": [
        "summary", "profile", "about", "objective",
        "professional summary", "career objective",
        "personal statement", "about me",
    ],
}
 
# Lines containing these strings are almost certainly not a person's name
_NAME_SKIP_PATTERNS = [
    "profile", "curriculum vitae", "cv", "resume", "@",
    "http", "www.", "phone", "address", "linkedin", "github",
    "tel:", "mob", "mobile",
]
 
# If a section supposedly contains these keywords it probably has mixed content
_SKILL_KEYWORDS = [
    "python", "java", "sql", "excel", "javascript", "typescript",
    "react", "node", "aws", "docker", "git", "html", "css",
    "machine learning", "data analysis", "ms office", "communication",
    "leadership", "teamwork",
]
 


def parse_resume(pdf_path: str) -> Dict[str, Any]:
    """
    Extract raw text, name, email, and sections from a resume PDF.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        dict: Contains keys:
            - raw_text (str): Full extracted text.
            - name (str): First non-empty line from page 1.
            - email (str): First email address found via regex.
            - sections (dict): Mapping of section headers (EXPERIENCE, EDUCATION,
                                SKILLS, PROJECTS) to their corresponding text.
            - mixed_sections (list): List of section names that are likely to have mixed content.
            - parse_confidence (float): Confidence score for the parsing accuracy.
    """
    # 1. Extract raw text from PDF using pdfplumber
    raw_text = _extract_pdf_text(pdf_path)

    # 2. Parse name: first non-empty line of page 1
    name = _extract_name(raw_text)

    # 3. Parse email using provided regex
    email = _extract_email(raw_text)

    # 4. Split into sections by common headers
    sections = _split_into_sections(raw_text)
    #5&6. Determine if sections are likely
    mixed = _flag_mixed_sections(sections)
    confidence = _compute_confidence(name, sections)

    return {
        "raw_text": raw_text,
        "name": name,
        "email": email,
        "sections": sections,
        "mixed_sections": mixed,
        "parse_confidence": confidence,
    }


def _extract_pdf_text(pdf_path: str) -> str:
    """
    Extract text from all pages using pdfplumber.
    Preserves page breaks with newlines for better separation.
    """
    text_parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    return "\n".join(text_parts)


def _extract_name(raw_text: str) -> Optional[str]:
    """Return the name from the beginning of the text."""
    lines = [l.strip() for l in raw_text.split("\n") if l.strip()]
 
    for line in lines[:8]:
        line_lower = line.lower()
 
        # Skip lines that contain known non-name content
        if any(skip in line_lower for skip in _NAME_SKIP_PATTERNS):
            continue
 
        word_count = len(line.split())
 
        # Names are typically 2–5 words
        if not (2 <= word_count <= 5):
            continue
 
        # Long all-caps line is probably a section header, not a name
        if line.isupper() and len(line) > 20:
            continue
 
        return line
 
    return "unknown"


def _extract_email(raw_text: str) -> Optional[str]:
    """Return the first email match using the given regex."""
    pattern = r"[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}"
    matches = re.findall(pattern, raw_text)
    return matches[0] if matches else None

def _detect_section_header(line: str) -> Optional[str]:
    """
    If `line` matches a known section header synonym, return the canonical
    section name (e.g. 'experience'). Otherwise return None.
    """
    cleaned = line.strip().lower().rstrip(":").strip()
    for canonical, variants in SECTION_HEADERS.items():
        if cleaned in variants:
            return canonical
    return None
 

def _split_into_sections(raw_text: str) -> Dict[str, str]:
    
    """
    Walk through lines, detect section headers via the synonym map, and
    bucket content into canonical sections.
 
    Lines before the first detected header go into 'summary'.
    Unrecognised content after a header stays in that header's bucket.
    """
    # Initialise all canonical sections as empty
    buckets: Dict[str, List[str]] = {k: [] for k in SECTION_HEADERS}
    current_section = "summary"
 
    for line in raw_text.split("\n"):
        detected = _detect_section_header(line)
        if detected:
            current_section = detected
        else:
            buckets[current_section].append(line)
 
    return {k: "\n".join(v).strip() for k, v in buckets.items()}

def _flag_mixed_sections(sections: Dict[str, str]) -> List[str]:
    """
    Returns section names that appear to contain content from other sections.
 
    Example: an 'education' block that also lists Python, SQL, etc. is mixed.
    The scoring agent will receive this list and adjust accordingly.
    """
    flagged = []
 
    # Education containing skill-like keywords
    edu = sections.get("education", "").lower()
    if any(kw in edu for kw in _SKILL_KEYWORDS):
        flagged.append("education")
 
    # Skills section containing degree/university keywords
    skills = sections.get("skills", "").lower()
    if any(kw in skills for kw in ["bachelor", "master", "university", "college", "degree"]):
        flagged.append("skills")
 
    return flagged
 

def _compute_confidence(name: str, sections: Dict[str, str]) -> str:
    """
    "low" if name extraction failed or 3+ sections came back empty.
    "low" confidence will trigger OCR fallback on Day 3.
    """
    empty_sections = sum(1 for v in sections.values() if not v.strip())
 
    if name == "unknown" or empty_sections >= 3:
        return "low"
    return "high"
 
if __name__ == "__main__":
    import sys
    import json
 
    path = sys.argv[1] if len(sys.argv) > 1 else "resume.pdf"
 
    result = parse_resume(path)

 
    # Print everything except raw_text (too verbose)
    summary = {k: v for k, v in result.items() if k != "raw_text"}
    print(json.dumps(summary, indent=2))
    print(f"\nraw_text length: {len(result['raw_text'])} chars")
 