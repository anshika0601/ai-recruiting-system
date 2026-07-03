
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
        "personal statement", "about me","hobbies",                # ← add this so hobbies don't bleed into skills
    "accomplishments",
    "languages",
    "links",
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
    raw_text, _layout = _extract_pdf_text(pdf_path)


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

def _find_column_split(page) -> Optional[float]:
    """
    Find the x-coordinate that divides a two-column page.
 
    Method: cluster all word x0 values into left-side and right-side groups.
    If two clear clusters exist with a gap between them, it's two-column.
    Returns the split x value, or None if single-column.
 
    This is more robust than a fixed midpoint threshold because it works
    for asymmetric layouts (e.g. 40/60 column splits) and sidebars.
    """
    words = page.extract_words()
    if len(words) < 10:           # too few words to decide
        return None
 
    # Collect unique x0 starts, rounded to nearest 5pt to reduce noise
    x_starts = sorted(set(round(float(w["x0"]) / 5) * 5 for w in words))
 
    if not x_starts:
        return None
 
    page_width = page.width
    left_margin  = page_width * 0.15   # ignore page margins
    right_margin = page_width * 0.85
 
    # Look for the largest gap between consecutive x-start clusters
    # that falls in the middle 30-70% of the page
    centre_min = page_width * 0.30
    centre_max = page_width * 0.70
 
    best_gap   = 0.0
    split_x    = None
 
    for i in range(len(x_starts) - 1):
        gap = x_starts[i + 1] - x_starts[i]
        mid = (x_starts[i] + x_starts[i + 1]) / 2
        if centre_min <= mid <= centre_max and gap > best_gap:
            best_gap = gap
            split_x  = mid
 
    # Only call it two-column if the gap is meaningful (>= 15pt ~ 0.5cm)
    if best_gap >= 15:
        return split_x
 
    return None
 
 
def _extract_column_text(page, x_min: float, x_max: float) -> str:
    """Extract text from a horizontal slice of the page."""
    cropped = page.crop((x_min, 0, x_max, page.height))
    return cropped.extract_text() or ""
 
 
def _extract_page_text(page) -> tuple[str, str]:
    """
    Extract text from one page, handling single vs two-column layouts.
    Returns (text, layout_type) where layout_type is 'single' or 'two_column'.
    """
    split_x = _find_column_split(page)
 
    if split_x:
        left  = _extract_column_text(page, 0, split_x)
        right = _extract_column_text(page, split_x, page.width)
        # Newline between columns so the right column's first header is
        # on its own line and gets detected by _detect_section_header
        return left.rstrip() + "\n\n" + right.lstrip(), "two_column"
 
    return page.extract_text() or "", "single_column"
 
 
def _extract_pdf_text(pdf_path: str) -> tuple[str, str]:
    """
    Extract text — tries column-aware pdfplumber first.
    If yield is too low (image-based PDF), calls the OCR fallback hook.
    Returns (full_text, layout_label).
    """
    parts   = []
    layouts = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text, layout = _extract_page_text(page)   # ← unpack tuple
            if text.strip():
                parts.append(text)
            layouts.append(layout)

    full_text = "\n".join(parts)

    overall_layout = (
        "two_column"
        if layouts.count("two_column") > layouts.count("single_column")
        else "single_column"
    )

    # ── OCR fallback hook ────────────────────────────────────────────────
    if len(full_text.strip()) < 100:
        print("[parser] ⚠ Low text yield — OCR fallback needed")
        print("[parser] ℹ Image-based PDF detected. "
              "Connect a cloud vision API here for production.")
        full_text = _ocr_fallback(pdf_path)

    return full_text, overall_layout


def _ocr_fallback(pdf_path: str) -> str:
    """
    OCR fallback stub — returns empty string with a clear message.

    Production swap: replace body with a call to:
      - Mistral Pixtral API  (free tier, recommended)
      - AWS Textract         (pay per page)
      - Google Document AI   (pay per page)

    This function just needs to return extracted text as a plain string.
    """
    print("[parser] ✗ OCR fallback not implemented — returning empty text")
    print("[parser] ✗ To fix: implement _ocr_fallback() with a cloud vision API")
    return ""
 



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


def _detect_section_header(line: str) -> tuple[Optional[str], str]:
    """
    Check if a line starts with a section header.
    Returns (canonical_name, remainder) where remainder is any content
    on the same line after the header, or (None, line) if no header found.

    Handles both:
      "SKILLS"                     → ("skills", "")
      "SKILLS HTML5, CSS, React"   → ("skills", "HTML5, CSS, React")
      "Employment History Senior…" → ("experience", "Senior…")
    """
    stripped = line.strip()
    lower    = stripped.lower()

    for canonical, variants in SECTION_HEADERS.items():
        for variant in variants:
            # Match header at start of line (case-insensitive)
            if lower == variant:
                return canonical, ""
            if lower.startswith(variant + " ") or lower.startswith(variant + ":"):
                remainder = stripped[len(variant):].lstrip(": ").strip()
                return canonical, remainder

    return None, line


def _split_into_sections(raw_text: str) -> Dict[str, str]:
    """
    Walk lines, detect section headers (even inline ones), bucket content.
    If a header has content on the same line, that content goes into the
    new section's bucket immediately.
    """
    buckets: Dict[str, List[str]] = {k: [] for k in SECTION_HEADERS}
    current = "summary"

    for line in raw_text.split("\n"):
        canonical, remainder = _detect_section_header(line)
        if canonical:
            current = canonical
            if remainder:                    # content was on the same line as header
                buckets[current].append(remainder)
        else:
            buckets[current].append(line)

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
 