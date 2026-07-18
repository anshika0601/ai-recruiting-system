# AI Recruiting Platform

> Multi-agent resume screening pipeline with evidence-cited scores, self-consistency voting, domain relevance gating, and adversarial keyword-stuffing detection.

**Python · LangGraph · Groq · sentence-transformers · Chroma · FastAPI**

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2-purple)](https://langchain-ai.github.io/langgraph/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)](https://fastapi.tiangolo.com)
[![Tests](https://img.shields.io/badge/Tests-39%20passing-brightgreen)]()

---

## The Problem

Most resume screening tools are black boxes — they rank candidates but can't explain why. A recruiter who receives *"Candidate A: 87/100"* with no evidence cannot defend that decision legally, cannot trust it, and cannot improve it.

This platform solves that with a multi-agent pipeline where **every score cites specific evidence from the resume**, every dimension is scored three times and voted on for consistency, and adversarial patterns (keyword stuffing, JD mirroring, experience inflation) are flagged explicitly rather than hidden.

---

## Demo

Upload a resume PDF → paste a job description → get a fully explained, evidence-backed score in 30–60 seconds.
[extractor] ✓ Found 36 skills, 7 JD matches
[domain] MATCH — Machine Learning Engineering vs ML Engineering
[scorer] Core Skill Match runs=[3,3,4] median=3 ✓
[scorer] Experience Relevance runs=[4,4,4] median=4 ✓
[scorer] Achievement Evidence runs=[4,4,4] median=4 ✓
[scorer] Career Trajectory runs=[4,4,4] median=4 ✓
[scorer] Red Flags runs=[3,3,3] median=3 ✓
[guard] ✓ Clean — no issues detected

──────────────────────────────────────────────────
CANDIDATE : Anshika Bijalwan
──────────────────────────────────────────────────
Core Skill Match 3/5 1.80 pts
Experience Relevance 4/5 2.00 pts
Achievement Evidence 4/5 1.60 pts
Career Trajectory 4/5 1.20 pts
Red Flags 3/5 -0.50 pts
──────────────────────────────────────────────────
FINAL SCORE : 6.10 / 10.00
──────────────────────────────────────────────────
Every score is backed by specific evidence quoted from the resume — no black-box numbers.

---

## Architecture
┌─────────────────────────────────────────────────────────────────┐
│ INPUT LAYER │
│ │
│ Resume PDFs ──────────────────────► Job Description (text) │
│ │ │ │
│ ▼ │ │
│ ┌──────────┐ │ │
│ │ Parser │ column-aware pdfplumber │ │
│ │ │ x-coord clustering │ │
│ └────┬─────┘ synonym section detection │ │
│ │ │ │
│ ▼ │ │
│ ┌──────────┐ │ │
│ │ Embedder │ all-MiniLM-L6-v2 (local) │ │
│ │ │ section-weighted text │ │
│ │ │ Chroma vector store │ │
│ └────┬─────┘ │ │
│ │ semantic search (top-K) │ │
└────────┼───────────────────────────────────────┼───────────────┘
│ │
└───────────────┬───────────────────────┘
│
┌────────────────────────▼────────────────────────────────────────┐
│ LANGGRAPH PIPELINE │
│ │
│ ┌─────────────┐ │
│ │ Extractor │ resume → structured facts │
│ │ Agent │ skills, years exp, achievements, red flags │
│ │ │ temperature=0.0 (deterministic) │
│ └──────┬──────┘ │
│ │ │
│ ▼ │
│ ┌─────────────┐ MATCH ─────────────────────────────────┐ │
│ │ Domain │ │ │
│ │ Check │ ADJACENT ─────────────────────────┐ │ │
│ │ │ │ │ │
│ └──────┬──────┘ MISMATCH → skip scorer (−3.0 pts) │ │ │
│ │ (saves 15 LLM calls on wrong-domain) │ │ │
│ ▼ │ │ │
│ ┌─────────────┐ ◄───────────────────────────────────── │ │
│ │ Scorer │ 5 dimensions × 3 runs = 15 LLM calls │ │
│ │ Agent │ evidence → reasoning → score (order!) │ │
│ │ │ median vote + disagreement flag │ │
│ └──────┬──────┘ ◄─────────────────────────────────────── │
│ │ │
│ ▼ │
│ ┌─────────────┐ │
│ │ Guard │ keyword stuffing detection │
│ │ Agent │ skill inflation, JD mirroring │
│ │ │ experience mismatch │
│ └──────┬──────┘ │
│ │ │
│ ▼ │
│ ┌─────────────┐ │
│ │ Aggregator │ weighted sum → 0-10 final score │
│ │ │ domain + guard penalties applied │
│ └─────────────┘ │
│ │
└─────────────────────────────────────────────────────────────────┘
│
▼
┌────────────────────────────────────────────────────────────────┐
│ OUTPUT LAYER │
│ │
│ FastAPI REST API ──────────────────► Recruiter UI │
│ /resumes/upload drag-drop PDF upload │
│ /screen score breakdown │
│ /search (fast) evidence per dim │
│ /health guard flag display │
└────────────────────────────────────────────────────────────────┘
---

## Evaluation

The pipeline was evaluated against a human-ranked baseline of **9 resumes** spanning MATCH (real SE candidates), ADJACENT (transferable-skill candidates), and MISMATCH (wrong-domain) buckets, all scored against the same Software Engineer JD.

### Multi-metric comparison — v1 (no domain check) vs v2 (with domain check)

| Metric | v1 | v2 | Change |
|---|---|---|---|
| **Top-3 overlap with human** | **100%** | **100%** | unchanged — pipeline consistently surfaces the correct shortlist |
| **Mean absolute rank error** | 1.56 | **1.33** | −14% (improved) |
| **Spearman rank correlation** | +0.62 | **+0.75** | crossed 0.70 "strong correlation" threshold |
| Exact-position match | 44.4% | 33.3% | noisy on n=9; kept for continuity, not decisions |

**Why Top-3 overlap matters most.** In a real screening workflow, a recruiter interviews the top few candidates. If the pipeline surfaces the same top-3 as a human reviewer, downstream ranking noise has no operational impact. Both versions achieve 100% on this metric.

**Why Spearman improved.** Adding the domain-check node (Day 22) let the pipeline distinguish domain relevance (MATCH / ADJACENT / MISMATCH) *before* scoring. This prevented the scorer from producing false-positive high scores on wrong-domain resumes and let the overall rank order align more closely with human judgment.

### Ablation — single-prompt LLM vs multi-agent pipeline

| Approach | Score std dev (lower is better) | Notes |
|---|---|---|
| Single-prompt LLM | 3.30 | One call, no rubric, no voting — erratic scores |
| Multi-agent pipeline | 1.20 | Rubric + 3× voting + guard + domain — consistent |

The multi-agent pipeline reduces score variance by **~64%**. The pipeline's primary value is **consistency and explainability**, not just raw ranking accuracy.

### Known limitation

The ADJACENT domain classification is currently too permissive on candidates with non-technical certifications (fashion, personal training). Full analysis and proposed fix in [`evaluation/evaluation.md`](evaluation/evaluation.md).

---

## Architecture Decisions

### 1. Column-aware PDF parsing
Most parsers fail on two-column layouts — Education (left) and Skills (right) get interleaved into one blob. Fix: detect columns via word x-coordinate clustering. Find the largest gap between consecutive x-positions in the middle 30–70% of the page. If gap ≥ 15pt, extract each column separately via `page.crop()`.

### 2. Section header synonym map
Resumes use 20+ variants for 5 canonical sections. `"Employment History"`, `"Career History"`, and `"Work Experience"` all map to `experience`. Inline headers (`"SKILLS HTML5 CSS"` on one line) are handled by startswith matching plus remainder extraction.

### 3. Evidence-before-score prompt order
The scorer prompt forces: **evidence → reasoning → score**. This order is deliberate — models that assign a number first and then rationalize it produce inconsistent, hallucinated justifications. Evidence-first grounds the score in what's actually in the resume.

### 4. Self-consistency voting
Each dimension is scored 3× independently at temperature=0.2, and the median is taken. If any run differs from the median by more than 1 point, the candidate is flagged for human review. Disagreement rate is a reported metric — high disagreement signals ambiguous candidates that deserve a human second look.

### 5. Domain relevance pre-check
A lightweight classifier runs between the extractor and the scorer. MISMATCH candidates (fashion designer → SE role) receive a −3.0pt penalty and skip 15 scorer LLM calls entirely. ADJACENT candidates (video producer → SE role) receive −0.5pt and proceed with their transferable skills surfaced. This saves LLM tokens *and* prevents misleadingly-detailed scores for wrong-domain resumes.

### 6. Fair guard prompt design
The guard agent went through explicit iteration. An early version flagged every new-grad resume as `EXPERIENCE_MISMATCH` (interpreting "0 paid years" as gaming), and flagged skills demonstrated in project bullets as `KEYWORD_STUFFING` (only checking job titles, not projects). The corrected prompt treats student/project experience as legitimate and requires zero-mention verification before flagging keyword stuffing. Impact: a clean legitimate resume went from **4 false-positive flags (−1.375 penalty)** to **0 flags**.

### 7. Ablation-driven design
Every major architectural choice was validated against a single-prompt baseline on the same test set. The multi-agent pipeline was retained not because it was more complex, but because measurement showed it produced substantially more consistent scores (−64% variance).

---

## Stack

| Component | Technology | Why |
|---|---|---|
| Agent orchestration | LangGraph | Explicit state machine, conditional edges for domain/disagreement branching |
| LLM | Groq (llama-3.3-70b-versatile) | Free tier, fast inference, strong structured output |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) | Local, no API key, no cost, sufficient quality for section-level similarity |
| Vector DB | Chroma | Persistent cosine similarity search, zero infrastructure setup |
| PDF parsing | pdfplumber | Word-level x/y coordinates enable column detection |
| Backend | FastAPI | Async, Pydantic validation, auto-generated Swagger docs |
| Tracing | LangSmith | Per-agent latency, token cost, input/output logging |
| Tests | pytest | 39 unit tests covering rubric math, voting, parsing, penalties, JSON robustness |
| Config | pydantic-settings | Type-safe environment variable loading |

---

## Project Structure
ai-recruiting-system/
├── app/
│ ├── config.py # pydantic-settings env loading
│ ├── main.py # FastAPI app + CORS
│ ├── routes.py # 7 REST endpoints
│ ├── parser.py # column-aware PDF parser
│ └── embedder.py # sentence-transformers + Chroma
├── graphs/
│ ├── pipeline_state.py # shared LangGraph TypedDict state
│ ├── pipeline.py # full graph: 5 nodes + conditional edges
│ ├── extractor_agent.py # resume → structured facts
│ ├── domain_check.py # MATCH | ADJACENT | MISMATCH classifier
│ ├── scorer_agent.py # rubric scoring × 3 runs per dimension
│ ├── guard_agent.py # keyword stuffing + gaming detection
│ ├── aggregator.py # weighted final score
│ └── rubric.py # single source of truth for dimensions
├── evaluation/
│ ├── metrics.py # MAE, Top-K, Spearman, exact-match
│ ├── run_evaluation_v2.py # baseline evaluation harness
│ ├── ablation.py # single-prompt vs pipeline comparison
│ ├── run_evaluation_v2.py # post-domain-check comparison
│ ├── v1_results.json # measured metrics per version
  ├── v2_results.json # measured metrics per version
│ └── EVALUATION.md # detailed v1 vs v2 writeup
├── frontend/
│ └── index.html # recruiter dashboard (drag-drop, score UI)
├── tests/
│ └── test_pipeline.py # 39 pytest unit tests
├── pdf_parder.py
└── requirements.txt
---

## Setup

```bash
git clone https://github.com/anshika0601/ai-recruiting-system
cd ai-recruiting-system

python -m venv venv
venv\Scripts\activate            # Windows
# source venv/bin/activate       # macOS/Linux

pip install -r requirements.txt
cp .env.example .env             # add your GROQ_API_KEY
Embed resumes:

Bash

python -m app.embedder resume1.pdf resume2.pdf
Screen a single candidate from the CLI:

Bash

python -m graphs.pipeline resume.pdf
Run the API server + frontend:

Bash

uvicorn app.main:app --reload
# then open frontend/index.html in a browser
Run the test suite:

Bash

pytest tests/ -v
Reproduce the evaluation:

Bash

python -m evaluation.run_evaluation_v2
API Reference
Method	Endpoint	Description
POST	/resumes/upload	Upload a PDF → parse → embed → store
POST	/screen	Run the full 5-agent pipeline on a stored resume
GET	/resumes	List all stored resumes
GET	/resumes/{id}	Get resume metadata by ID
DELETE	/resumes/{id}	Remove a resume from storage
POST	/search	Fast semantic search (embeddings only, no agents)
GET	/health	Component status check
Full interactive OpenAPI docs at http://localhost:8000/docs.

Known Limitations
Image-based PDFs. pdfplumber requires a text layer. Scanned resumes return no text. The parser detects this via a character-yield threshold and calls _ocr_fallback() — a documented stub designed to swap in Mistral Pixtral or AWS Textract for production.

ADJACENT domain classification is too permissive. Candidates with non-technical certifications (fashion, personal training) are currently classified ADJACENT rather than MISMATCH, inflating their rank. Documented in docs/EVALUATION.md with a proposed one-line prompt fix.

Groq free-tier token limits. Capped at 100K tokens/day. Running the full 9-resume evaluation consumes roughly half this budget. Production fix: overnight batch processing or an upgrade to Groq's dev tier.

JSON parse failures (~5% of runs). LLM output occasionally contains smart quotes or unescaped apostrophes that break json.loads. A regex fallback handles most cases but defaults to score=1 when it fails, which can bias median calculations for that dimension.

Evaluation sample size (n=9). Large enough to demonstrate direction of improvement (Spearman up, MAE down, guard false-positives eliminated) but too small for statistically-tight accuracy claims. A production version would need 50+ labelled resumes across multiple JDs.

What I'd Do Differently
Parallelize scorer dimensions. Five sequential LLM calls could run concurrently with asyncio.gather(), reducing latency from ~45s to ~15s per candidate.
Tighten ADJACENT classification. Require specifically technical certifications (cloud, analytics, developer tools, project management platforms) rather than any certification. This would flip Michelle Lopez and Julie Monroe from ADJACENT to MISMATCH and lift Spearman further.
Fine-tune section detection. Replace the rule-based synonym map with a small NER model trained on real resume headers.
Persist pipeline results. Store screening results in SQLite so recruiters can compare runs over time and audit past decisions.
Report confidence intervals. Surface scores as "6.1 ± 0.8" using the disagreement spread across the 3 scorer runs, rather than a false-precision single number.
Expand the evaluation set. n=9 is enough for direction; n=50+ is needed for magnitude claims.
What This Project Demonstrates
Multi-agent orchestration with real conditional routing (LangGraph)
Prompt engineering with measured impact — every prompt change validated against ground-truth ranking
Self-consistency techniques — 3× voting, median aggregation, disagreement flagging
Adversarial-input defense — a dedicated guard agent that resists keyword stuffing and JD mirroring, with an explicit fairness contract for legitimate candidates
Explainability by construction — every score carries evidence, every penalty is itemized
Evaluation discipline — labeled test set, multiple metrics, ablation study, honest documentation of known limitations
Production-oriented engineering — 39 pytest unit tests, structured logging, LangSmith tracing, FastAPI REST layer with OpenAPI docs

Built by Anshika Bijalwan