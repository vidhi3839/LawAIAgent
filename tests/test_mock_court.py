"""
Tests for tasks/mock_court.py

Covers pure scoring logic directly. LLM-driven functions
(analyze_counter_arguments, evaluate_proof_strength, identify_judicial_gaps,
run_full_analysis) are tested with `llm.invoke` mocked so no real Groq call
happens.
"""
import pytest
from unittest.mock import patch, MagicMock

from tasks import mock_court


# ── _stem_or_exact_match ─────────────────────────────────────────────────

class TestStemOrExactMatch:
    def test_multiword_phrase_exact_match(self):
        words = "we filed this in good faith today".split()
        assert mock_court._stem_or_exact_match(
            "good faith", "we filed this in good faith today", words
        ) is True

    def test_multiword_phrase_no_match(self):
        words = "we filed this within days".split()
        assert mock_court._stem_or_exact_match(
            "good faith", "we filed this within days", words
        ) is False

    def test_short_word_exact_boundary_match(self):
        # "act" is < 8 chars — must be a whole-word match, not a substring
        # of "action"/"actual"/"activity"
        words = ["action", "under", "the", "act"]
        assert mock_court._stem_or_exact_match("act", "action under the act", words) is True

    def test_short_word_does_not_falsely_match_inside_longer_word(self):
        words = ["action", "actual", "activity"]
        arg_lower = "the action was actual and part of routine activity"
        # "act" should NOT appear as its own word here
        assert mock_court._stem_or_exact_match("act", arg_lower, words) is False

    def test_long_word_stem_match_catches_morphological_variant(self):
        # "retaliation" (11 chars) stems to "retaliat" (len-3=8 chars) —
        # should match "retaliated" via startswith
        words = ["the", "employee", "retaliated", "against", "him"]
        assert mock_court._stem_or_exact_match(
            "retaliation", "the employee retaliated against him", words
        ) is True


# ── compute_statutory_grounding ──────────────────────────────────────────

class TestComputeStatutoryGrounding:
    def test_usc_citation_scores_full(self):
        assert mock_court.compute_statutory_grounding(
            "This is governed by 42 U.S.C. § 1983"
        ) == 1.0

    def test_rule_citation_scores_full(self):
        assert mock_court.compute_statutory_grounding("See FRE Rule 401") == 1.0

    def test_legal_terminology_without_citation_scores_half(self):
        assert mock_court.compute_statutory_grounding(
            "This raises a due process concern under common law"
        ) == 0.5

    def test_no_legal_terms_scores_zero(self):
        assert mock_court.compute_statutory_grounding(
            "The defendant did a bad thing and should be punished"
        ) == 0.0


# ── compute_precedent_support ─────────────────────────────────────────────

class TestComputePrecedentSupport:
    def test_empty_cases_returns_floor_score(self):
        assert mock_court.compute_precedent_support([]) == 0.1

    @pytest.mark.parametrize("similarities,expected", [
        ([0.60, 0.58], 1.0),
        ([0.45, 0.42], 0.7),
        ([0.35, 0.32], 0.4),
        ([0.20, 0.15], 0.1),
    ])
    def test_similarity_bands(self, similarities, expected):
        cases = [{"similarity_score": s} for s in similarities]
        assert mock_court.compute_precedent_support(cases) == expected

    def test_malformed_cases_returns_zero_not_exception(self):
        assert mock_court.compute_precedent_support([{"no_score_key": True}]) == 0.0


# ── compute_vulnerability_exposure ────────────────────────────────────────

class TestComputeVulnerabilityExposure:
    def test_short_argument_addressing_all_expected_defenses_scores_full(self):
        # word_count < 50 → expected = 2
        arg = "We acted in good faith and there is no standing to sue here."
        score = mock_court.compute_vulnerability_exposure(arg)
        assert score == 1.0

    def test_argument_addressing_no_defenses_scores_zero(self):
        arg = "The defendant caused harm and must be held accountable for it."
        assert mock_court.compute_vulnerability_exposure(arg) == 0.0

    def test_long_argument_scales_expected_count(self, sample_argument_strong):
        # sample_argument_strong hits good_faith, statute_of_limitations,
        # standing, safe_harbour explicitly (4 of 5) at word_count >= 50
        score = mock_court.compute_vulnerability_exposure(sample_argument_strong)
        assert 0.0 < score <= 1.0

    def test_false_positive_words_dont_count(self):
        # Regression check: this exact sentence used to falsely count as
        # addressing statute-of-limitations because of bare "within"/"days"
        arg = "We filed this within days of the incident, well before affected parties suffered loss."
        score = mock_court.compute_vulnerability_exposure(arg)
        assert score == 0.0


# ── compute_strategic_strength ────────────────────────────────────────────

class TestComputeStrategicStrength:
    def test_weighted_sum_and_rating_bands(self, sample_retrieved_cases_high_similarity):
        result = mock_court.compute_strategic_strength(
            "Pursuant to 42 U.S.C. § 1983, we acted in good faith.",
            sample_retrieved_cases_high_similarity,
        )
        s, p, v = result["s_score"], result["p_score"], result["v_score"]
        expected = round((s * 0.4) + (p * 0.4) + (v * 0.2), 3)
        assert result["final_score"] == expected
        assert result["task"] == "mock_court"

    @pytest.mark.parametrize("final_score,expected_flag", [
        (0.85, "STRONG"),
        (0.6, "MODERATE"),
        (0.3, "HIGH RISK"),
    ])
    def test_rating_bands(self, final_score, expected_flag, monkeypatch):
        # Force the three sub-scores so final_score lands exactly where we want
        monkeypatch.setattr(mock_court, "compute_statutory_grounding", lambda a: final_score)
        monkeypatch.setattr(mock_court, "compute_precedent_support", lambda c: final_score)
        monkeypatch.setattr(mock_court, "compute_vulnerability_exposure", lambda a: final_score)
        result = mock_court.compute_strategic_strength("irrelevant text", [])
        assert result["flag"] == expected_flag


# ── _deduplicate_cases ────────────────────────────────────────────────────

class TestDeduplicateCases:
    def test_keeps_only_highest_similarity_chunk_per_case(self, sample_retrieved_cases_duplicated):
        result = mock_court._deduplicate_cases(sample_retrieved_cases_duplicated)
        citations = [c["citation"] for c in result]
        assert citations.count("248 N.Y. 339") == 1
        palsgraf = next(c for c in result if c["citation"] == "248 N.Y. 339")
        assert palsgraf["similarity_score"] == 0.55  # the highest of the three chunks

    def test_sorted_descending_by_similarity(self, sample_retrieved_cases_duplicated):
        result = mock_court._deduplicate_cases(sample_retrieved_cases_duplicated)
        scores = [c["similarity_score"] for c in result]
        assert scores == sorted(scores, reverse=True)

    def test_empty_input_returns_empty_list(self):
        assert mock_court._deduplicate_cases([]) == []


# ── LLM-driven functions (mocked) ─────────────────────────────────────────

class TestAnalyzeCounterArguments:
    def test_returns_analysis_and_strips_think_tags(self, monkeypatch, fake_llm):
        fake_llm.invoke.return_value.content = "<think>reasoning here</think>Final counter-argument text."
        monkeypatch.setattr(mock_court, "llm", fake_llm)
        result = mock_court.analyze_counter_arguments("My argument is X.", [])
        assert result["analysis"] == "Final counter-argument text."

    def test_llm_failure_is_caught_not_raised(self, monkeypatch):
        fake_llm = MagicMock()
        fake_llm.invoke.side_effect = RuntimeError("Groq API down")
        monkeypatch.setattr(mock_court, "llm", fake_llm)
        result = mock_court.analyze_counter_arguments("My argument is X.", [])
        assert "Could not generate analysis" in result["analysis"]


class TestIdentifyJudicialGaps:
    def test_unaddressed_defences_lists_missing_categories(self, monkeypatch, fake_llm):
        monkeypatch.setattr(mock_court, "llm", fake_llm)
        # Addresses only good_faith — the other 4 categories should show as unaddressed
        result = mock_court.identify_judicial_gaps("We acted in good faith throughout.")
        assert "Good faith defence" not in result["unaddressed_defences"]
        assert "Standing / injury" in result["unaddressed_defences"]


class TestRunFullAnalysis:
    def test_shares_one_case_retrieval_across_all_subfunctions(self, monkeypatch, fake_llm):
        """Regression guard: run_full_analysis must call search_past_cases
        exactly ONCE, not once per sub-function (the bug this was fixed
        to prevent)."""
        monkeypatch.setattr(mock_court, "llm", fake_llm)
        fake_search = MagicMock(return_value={"results": [
            {"case_name": "A", "citation": "1", "similarity_score": 0.5, "text": "t"}
        ]})
        with patch.dict("sys.modules", {"tasks.past_cases": MagicMock(search_past_cases=fake_search)}):
            result = mock_court.run_full_analysis("My argument is that we acted properly.")
        assert fake_search.call_count == 1
        assert "score" in result
        assert "counter_arguments" in result