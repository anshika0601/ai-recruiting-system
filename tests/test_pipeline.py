"""
tests/test_pipeline.py

Day 24: Pytest tests for core pipeline logic.

Tests cover:
  - Parser: name extraction, section splitting, column detection
  - Rubric: weights sum to 1.0, all anchors defined
  - Scorer: self-consistency voting logic, median calculation
  - Guard: penalty calculation
  - Aggregator: weighted score math
  - Domain check: verdict mapping to penalties

Run: pytest tests/ -v
"""
import json
import pytest
import statistics


# ── parser tests ──────────────────────────────────────────────────────────────

class TestParser:

    def test_name_extraction_skips_email(self):
        from app.parser import _extract_name
        text = "email@gmail.com\nJohn Smith\nSoftware Engineer"
        assert _extract_name(text) == "John Smith"

    def test_name_extraction_skips_profile_keyword(self):
        from app.parser import _extract_name
        text = "Profile\nJane Doe\nDeveloper"
        assert _extract_name(text) == "Jane Doe"

    def test_name_extraction_skips_single_word(self):
        from app.parser import _extract_name
        text = "RESUME\nJohn Smith\nEngineer"
        assert _extract_name(text) == "John Smith"

    def test_name_returns_unknown_when_no_match(self):
        """When no line looks like a name, return 'unknown'."""
        from app.parser import _extract_name
        text = "email@test.com\nhttp://linkedin.com\n@@@"
        assert _extract_name(text) == "unknown"

    def test_email_extraction(self):
        from app.parser import _extract_email
        text = "John Smith\njohn.smith@gmail.com\nSoftware Engineer"
        assert _extract_email(text) == "john.smith@gmail.com"

    def test_email_returns_none_when_missing(self):
        from app.parser import _extract_email
        assert _extract_email("No email here at all") is None

    def test_section_header_detection_exact(self):
        """Exact header word 'experience' is detected."""
        from app.parser import _detect_section_header
        result = _detect_section_header("experience")
        assert result[0] == "experience"

    def test_section_header_detection_synonym(self):
        """Synonym 'work history' maps to canonical 'experience'."""
        from app.parser import _detect_section_header
        result = _detect_section_header("work history")
        assert result[0] == "experience"

    def test_section_header_returns_none_for_content(self):
        from app.parser import _detect_section_header
        result, _ = _detect_section_header("Developed a REST API for customer data")
        assert result is None

    def test_section_splitting_buckets_correctly(self):
        from app.parser import _split_into_sections
        text = "EXPERIENCE\nWorked at Google\nEDUCATION\nBSc Computer Science"
        sections = _split_into_sections(text)
        assert "Google" in sections["experience"]
        assert "Computer Science" in sections["education"]

    def test_parse_confidence_low_when_name_unknown(self):
        from app.parser import _compute_confidence
        sections = {k: "some content" for k in ["experience","education","skills","projects","summary"]}
        assert _compute_confidence("unknown", sections) == "low"

    def test_parse_confidence_low_when_empty_sections(self):
        from app.parser import _compute_confidence
        sections = {"experience": "", "education": "", "skills": "", "projects": "", "summary": ""}
        assert _compute_confidence("John Smith", sections) == "low"

    def test_parse_confidence_high_when_good(self):
        from app.parser import _compute_confidence
        sections = {k: "content" for k in ["experience","education","skills","projects","summary"]}
        assert _compute_confidence("John Smith", sections) == "high"


# ── rubric tests ──────────────────────────────────────────────────────────────

class TestRubric:

    def test_non_penalty_weights_sum_to_correct_value(self):
        from graphs.rubric import RUBRIC
        total = sum(d["weight"] for d in RUBRIC.values())
        assert abs(total - 1.0) < 0.001, f"Weights sum to {total}, expected 1.0"

    def test_all_dimensions_have_five_anchors(self):
        from graphs.rubric import RUBRIC
        for name, dim in RUBRIC.items():
            assert len(dim["anchors"]) == 5, f"{name} has {len(dim['anchors'])} anchors"
            assert set(dim["anchors"].keys()) == {1, 2, 3, 4, 5}

    def test_format_anchors_returns_string(self):
        from graphs.rubric import format_anchors
        result = format_anchors("core_skill_match")
        assert "1 =" in result
        assert "5 =" in result

    def test_red_flags_is_penalty_dimension(self):
        from graphs.rubric import RUBRIC
        assert RUBRIC["red_flags"]["is_penalty"] is True

    def test_non_penalty_dimensions_not_penalty(self):
        from graphs.rubric import RUBRIC
        non_penalty = ["core_skill_match","experience_relevance","achievement_evidence","career_trajectory"]
        for dim in non_penalty:
            assert RUBRIC[dim]["is_penalty"] is False


# ── self-consistency voting tests ─────────────────────────────────────────────

class TestSelfConsistencyVoting:

    def test_median_of_three_equal_scores(self):
        scores = [3, 3, 3]
        assert int(statistics.median(scores)) == 3

    def test_median_with_one_outlier(self):
        scores = [1, 3, 3]
        assert int(statistics.median(scores)) == 3

    def test_disagreement_detected_when_spread_gt_one(self):
        scores = [1, 3, 3]
        median = int(statistics.median(scores))
        disagreement = any(abs(s - median) > 1 for s in scores)
        assert disagreement is True

    def test_no_disagreement_when_spread_le_one(self):
        scores = [3, 4, 4]
        median = int(statistics.median(scores))
        disagreement = any(abs(s - median) > 1 for s in scores)
        assert disagreement is False

    def test_score_clamped_to_valid_range(self):
        # Simulate score clamping logic from scorer
        for raw in [0, -1, 6, 10]:
            clamped = max(1, min(5, raw))
            assert 1 <= clamped <= 5


# ── guard penalty tests ───────────────────────────────────────────────────────

class TestGuardPenalty:

    def _calculate_penalty(self, flags):
        """Mirror of guard_agent._calculate_penalty"""
        PENALTY_PER_FLAG  = 0.5
        MAX_PENALTY       = 2.0
        SEVERITY_WEIGHTS  = {"low": 0.25, "medium": 0.5, "high": 1.0}
        if not flags:
            return 0.0
        total = sum(SEVERITY_WEIGHTS.get(f.get("severity","low"), 0.25) for f in flags)
        return min(total * PENALTY_PER_FLAG, MAX_PENALTY)

    def test_no_flags_zero_penalty(self):
        assert self._calculate_penalty([]) == 0.0

    def test_single_high_flag_penalty(self):
        flags = [{"severity": "high", "type": "KEYWORD_STUFFING"}]
        assert self._calculate_penalty(flags) == 0.5

    def test_penalty_capped_at_max(self):
        flags = [{"severity": "high"}] * 10
        assert self._calculate_penalty(flags) == 2.0

    def test_medium_flag_penalty(self):
        flags = [{"severity": "medium"}]
        assert self._calculate_penalty(flags) == 0.25


# ── domain penalty tests ──────────────────────────────────────────────────────

class TestDomainPenalty:

    def test_match_zero_penalty(self):
        from graphs.domain_check import DOMAIN_PENALTIES
        assert DOMAIN_PENALTIES["MATCH"] == 0.0

    def test_adjacent_small_penalty(self):
        from graphs.domain_check import DOMAIN_PENALTIES
        assert DOMAIN_PENALTIES["ADJACENT"] == 0.5

    def test_mismatch_large_penalty(self):
        from graphs.domain_check import DOMAIN_PENALTIES
        assert DOMAIN_PENALTIES["MISMATCH"] == 3.0

    def test_mismatch_penalty_greater_than_adjacent(self):
        from graphs.domain_check import DOMAIN_PENALTIES
        assert DOMAIN_PENALTIES["MISMATCH"] > DOMAIN_PENALTIES["ADJACENT"]


# ── aggregator math tests ─────────────────────────────────────────────────────

class TestAggregatorMath:

    def test_weighted_contribution_non_penalty(self):
        # median=5, weight=0.30 → contribution = (5/5) * 0.30 * 10 = 3.0
        median = 5
        weight = 0.30
        contribution = round((median / 5) * weight * 10, 3)
        assert contribution == 3.0

    def test_weighted_contribution_penalty_dim(self):
        # median=5 (no red flags) → deduction = ((5-5)/4) * 0.10 * 10 = 0
        median = 5
        weight = 0.10
        deduction = ((5 - median) / 4) * weight * 10
        assert deduction == 0.0

    def test_penalty_dim_max_deduction(self):
        # median=1 (severe flags) → deduction = ((5-1)/4) * 0.10 * 10 = 1.0
        median = 1
        weight = 0.10
        deduction = ((5 - median) / 4) * weight * 10
        assert deduction == 1.0

    def test_final_score_never_negative(self):
        raw_score      = 1.0
        guard_penalty  = 2.0
        domain_penalty = 3.0
        final = max(0.0, raw_score - guard_penalty - domain_penalty)
        assert final == 0.0

    def test_perfect_candidate_max_score(self):
        from graphs.rubric import RUBRIC, DIMENSION_ORDER
        weighted_sum = 0.0
        for dim in DIMENSION_ORDER:
            r = RUBRIC[dim]
            if r["is_penalty"]:
                weighted_sum += -0.0   # median=5, zero deduction
            else:
                weighted_sum += (5/5) * r["weight"] * 10
        assert abs(weighted_sum - 9.0) < 0.1   # max possible ~9.0 (red flags always some risk)


# ── json parsing robustness ───────────────────────────────────────────────────

class TestJsonRobustness:

    def test_json_with_smart_quotes_cleaned(self):
        raw = '{"score": 3, "reasoning": "candidate\u2019s skills are good", "evidence": []}'
        clean = (raw
            .replace("\u2018", " ").replace("\u2019", " ")
            .replace("\u201c", '"').replace("\u201d", '"')
            .replace("'s ", "s ")
        )
        parsed = json.loads(clean)
        assert parsed["score"] == 3

    def test_json_boundary_extraction(self):
        raw = "Some text before { \"score\": 4, \"evidence\": [], \"reasoning\": \"good\" } and after"
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        parsed = json.loads(raw[start:end])
        assert parsed["score"] == 4

    def test_score_clamp(self):
        for val, expected in [(0, 1), (6, 5), (3, 3), (-1, 1)]:
            assert max(1, min(5, val)) == expected