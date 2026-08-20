"""
Tests for tasks/summarize.py

compute_semantic_grounding loads a real sentence-transformers model, so it
is mocked (via monkeypatching summarize._get_embedding_model) wherever it's
a dependency of the function under test. It gets its own smoke test with a
fake embedder so the *logic* (weakest-best-match aggregation) is still
checked without a real model load.
"""
import pytest
from unittest.mock import patch, MagicMock

from tasks import summarize


# ── _strip_think_tags ─────────────────────────────────────────────────────

class TestStripThinkTags:
    def test_strips_think_block_when_present(self):
        text = "<think>internal reasoning</think>Final answer text."
        assert summarize._strip_think_tags(text) == "Final answer text."

    def test_leaves_text_unchanged_when_no_think_tags(self):
        text = "Just a normal answer."
        assert summarize._strip_think_tags(text) == text


# ── _extract_first_date ───────────────────────────────────────────────────

class TestExtractFirstDate:
    def test_finds_month_day_year_format(self):
        assert summarize._extract_first_date("Filed on January 5, 2021 in court") == "january 5, 2021"

    def test_finds_slash_date_format(self):
        assert summarize._extract_first_date("Due 03/15/2022 per order") == "03/15/2022"

    def test_bare_year_accepted_in_plausible_range(self):
        assert summarize._extract_first_date("The case was decided in 1998") == "1998"

    def test_bare_year_rejected_outside_plausible_range(self):
        # 1899 is outside 1900-2099 — must not be treated as a date
        assert summarize._extract_first_date("Docket number 1899 filed") is None

    def test_no_date_returns_none(self):
        assert summarize._extract_first_date("no temporal reference here at all") is None


# ── _looks_like_decline ───────────────────────────────────────────────────

class TestLooksLikeDecline:
    @pytest.mark.parametrize("answer", [
        "This is not stated in the document.",
        "The document does not specify a date.",
        "Cannot determine the holding from this text.",
    ])
    def test_decline_phrases_detected(self, answer):
        assert summarize._looks_like_decline(answer) is True

    def test_normal_answer_not_flagged_as_decline(self):
        assert summarize._looks_like_decline("The court held that the motion was granted.") is False


# ── _extract_section ──────────────────────────────────────────────────────

class TestExtractSection:
    def test_extracts_text_between_headers(self):
        extraction = (
            "TIMELINE:\n- 2021 — filing\nCAST OF CHARACTERS:\n- Jane Doe — Plaintiff\n"
            "CAUSES OF ACTION:\n- Negligence — Tort\nKEY FACTS:\n- Something happened."
        )
        result = summarize._extract_section(extraction, "CAST OF CHARACTERS:")
        assert "Jane Doe" in result
        assert "Negligence" not in result

    def test_extracts_to_end_when_no_next_header(self):
        extraction = "KEY FACTS:\n- Fact one.\n- Fact two."
        result = summarize._extract_section(extraction, "KEY FACTS:")
        assert "Fact one" in result and "Fact two" in result

    def test_missing_header_returns_empty_string(self):
        assert summarize._extract_section("no headers here", "TIMELINE:") == ""


# ── compute_chronological_integrity ───────────────────────────────────────

class TestComputeChronologicalIntegrity:
    def test_no_dates_found_line_scores_full(self):
        extraction = "TIMELINE:\nNo explicit dates found.\nCAST OF CHARACTERS:\n- x"
        assert summarize.compute_chronological_integrity(extraction, "raw text") == 1.0

    def test_date_present_and_near_matching_event_words_scores_full(self):
        raw_text = "On march 5, 2020 the plaintiff filed a complaint alleging fraud against the defendant."
        extraction = "TIMELINE:\n- march 5, 2020 — complaint filed alleging fraud\nCAST OF CHARACTERS:\n- x"
        assert summarize.compute_chronological_integrity(extraction, raw_text) == 1.0

    def test_date_present_but_not_near_event_words_scores_half(self):
        # _proximity_match looks +/-200 chars around the date's position —
        # the original version of this test used a raw_text short enough
        # that the whole document fell inside that window, so the event
        # words matched "nearby" by accident and the test asserted the
        # wrong thing. Padding with >200 chars of unrelated filler between
        # the date and the event words actually exercises the "far apart"
        # case the test claims to check.
        filler = "unrelated filler text about other matters entirely. " * 8  # > 200 chars
        raw_text = f"On march 5, 2020 nothing relevant happened. {filler} Much later, a complaint alleging fraud was filed."
        extraction = "TIMELINE:\n- march 5, 2020 — complaint filed alleging fraud\nCAST OF CHARACTERS:\n- x"
        score = summarize.compute_chronological_integrity(extraction, raw_text)
        assert score == 0.5

    def test_date_not_in_source_at_all_scores_zero_credit(self):
        raw_text = "Nothing about any date appears here."
        extraction = "TIMELINE:\n- january 1, 2099 — some event\nCAST OF CHARACTERS:\n- x"
        assert summarize.compute_chronological_integrity(extraction, raw_text) == 0.0

    def test_no_timeline_section_at_all_scores_full(self):
        assert summarize.compute_chronological_integrity("KEY FACTS:\n- x", "raw") == 1.0


# ── compute_entity_grounding ───────────────────────────────────────────────

class TestComputeEntityGrounding:
    def test_no_named_parties_scores_full(self):
        extraction = "CAST OF CHARACTERS:\nNo named parties found.\nCAUSES OF ACTION:\n- x"
        assert summarize.compute_entity_grounding(extraction, "raw text") == 1.0

    def test_name_present_near_matching_role_scores_full(self):
        raw_text = "Jane Doe, acting as plaintiff, filed the initial complaint in this matter."
        extraction = "CAST OF CHARACTERS:\n- Jane Doe — Plaintiff\nCAUSES OF ACTION:\n- x"
        assert summarize.compute_entity_grounding(extraction, raw_text) == 1.0

    def test_short_name_substring_does_not_false_positive(self):
        """Regression check: 'Roe' must not match inside 'wardrobe'."""
        raw_text = "The witness described the contents of her wardrobe in detail during testimony."
        extraction = "CAST OF CHARACTERS:\n- Roe — Witness\nCAUSES OF ACTION:\n- x"
        score = summarize.compute_entity_grounding(extraction, raw_text)
        assert score == 0.0  # "Roe" genuinely does not appear as a word

    def test_name_absent_entirely_scores_zero_credit(self):
        raw_text = "This document never mentions any specific individual by name."
        extraction = "CAST OF CHARACTERS:\n- John Smith — Defendant\nCAUSES OF ACTION:\n- x"
        assert summarize.compute_entity_grounding(extraction, raw_text) == 0.0


# ── compute_summary_confidence ────────────────────────────────────────────

class TestComputeSummaryConfidence:
    def test_averages_available_signals_and_sets_task_label(self, monkeypatch):
        monkeypatch.setattr(summarize, "compute_chronological_integrity", lambda e, r: 1.0)
        monkeypatch.setattr(summarize, "compute_entity_grounding", lambda e, r: 0.5)
        monkeypatch.setattr(summarize, "compute_semantic_grounding", lambda a, r: 0.8)
        extraction = "KEY FACTS:\n- Something happened.\nCAUSES OF ACTION:\n- Negligence"
        result = summarize.compute_summary_confidence(extraction, "raw text")
        assert result["final_score"] == round((1.0 + 0.5 + 0.8) / 3, 3)
        assert result["task"] == "summarizer"

    def test_missing_semantic_signal_still_produces_a_score(self, monkeypatch):
        monkeypatch.setattr(summarize, "compute_chronological_integrity", lambda e, r: 1.0)
        monkeypatch.setattr(summarize, "compute_entity_grounding", lambda e, r: 1.0)
        # No KEY FACTS/CAUSES OF ACTION content → s_score stays None
        extraction = "TIMELINE:\nNo explicit dates found."
        result = summarize.compute_summary_confidence(extraction, "raw text")
        assert result["s_score"] is None
        assert result["final_score"] == 1.0

    @pytest.mark.parametrize("score,expected_flag", [(0.95, "HIGH INTEGRITY"), (0.8, "MODERATE"), (0.5, "LOW INTEGRITY")])
    def test_rating_bands(self, monkeypatch, score, expected_flag):
        monkeypatch.setattr(summarize, "compute_chronological_integrity", lambda e, r: score)
        monkeypatch.setattr(summarize, "compute_entity_grounding", lambda e, r: score)
        monkeypatch.setattr(summarize, "compute_semantic_grounding", lambda a, r: score)
        extraction = "KEY FACTS:\n- x\nCAUSES OF ACTION:\n- y"
        result = summarize.compute_summary_confidence(extraction, "raw")
        assert result["flag"] == expected_flag


# ── compute_qa_confidence ──────────────────────────────────────────────────

class TestComputeQAConfidence:
    def test_decline_answer_scores_full_and_flags_declined(self):
        result = summarize.compute_qa_confidence(
            "This is not stated in the document.", "irrelevant raw text"
        )
        assert result["final_score"] == 1.0
        assert result["flag"] == "DECLINED"
        assert result["checkable_claims_found"] is False

    def test_no_checkable_content_is_unverifiable_not_silently_high(self, monkeypatch):
        monkeypatch.setattr(summarize, "compute_semantic_grounding", lambda a, r: None)
        # No dates, no proper nouns, no semantic signal
        result = summarize.compute_qa_confidence("yes it does apply here in this case", "raw text")
        assert result["final_score"] is None
        assert result["flag"] == "UNVERIFIABLE"

    def test_date_grounded_in_source_scores_high(self, monkeypatch):
        monkeypatch.setattr(summarize, "compute_semantic_grounding", lambda a, r: 1.0)
        raw_text = "The order was issued on 05/01/2021 and remains in effect."
        answer = "The order was issued on 05/01/2021."
        result = summarize.compute_qa_confidence(answer, raw_text)
        assert result["d_score"] == 1.0

    def test_date_not_in_source_scores_low_for_that_signal(self, monkeypatch):
        monkeypatch.setattr(summarize, "compute_semantic_grounding", lambda a, r: 1.0)
        raw_text = "Nothing about this specific date appears anywhere in the source."
        answer = "The order was issued on 05/01/2021."
        result = summarize.compute_qa_confidence(answer, raw_text)
        assert result["d_score"] == 0.0

    def test_weight_is_dynamic_not_hardcoded_third(self, monkeypatch):
        """Regression check: weight must be 1/(available signal count),
        not a fixed 1/3 — this was a real bug fixed once already."""
        monkeypatch.setattr(summarize, "compute_semantic_grounding", lambda a, r: None)
        # Only proper-noun signal available (no dates, no semantic score)
        raw_text = "Jane Doe appeared as counsel of record in this matter."
        answer = "Jane Doe appeared as counsel."
        result = summarize.compute_qa_confidence(answer, raw_text)
        assert result["d_score"] is None
        assert result["s_score"] is None
        assert result["n_score"] is not None
        assert result["final_score"] == result["n_score"]  # only 1 signal → weight = 1.0


# ── compute_semantic_grounding — logic smoke test with fake embeddings ────

class TestComputeSemanticGrounding:
    def test_empty_answer_returns_none(self):
        assert summarize.compute_semantic_grounding("", "some raw text long enough to chunk") is None

    def test_empty_raw_text_returns_none(self):
        assert summarize.compute_semantic_grounding("A real sentence goes here.", "") is None

    def test_takes_the_weakest_best_match_not_the_average(self, monkeypatch):
        """The docstring's core claim: aggregation is MIN of each sentence's
        best match, not an average — verify with a fake model."""
        import numpy as np

        class FakeTensor:
            def __init__(self, arr):
                self.arr = np.array(arr)
            def max(self):
                return self.arr.max()

        class FakeModel:
            def encode(self, items, convert_to_tensor=True):
                return list(items)  # identity — real values injected via fake cos_sim below

        def fake_cos_sim(sentence_embedding, chunk_embeddings):
            # First sentence matches well (0.9), second sentence matches poorly (0.2)
            if sentence_embedding == "Strong sentence with real support in the source text.":
                return [FakeTensor([0.9, 0.85])]
            return [FakeTensor([0.2, 0.1])]

        monkeypatch.setattr(summarize, "_get_embedding_model", lambda: FakeModel())
        monkeypatch.setattr(summarize.util, "cos_sim", fake_cos_sim)

        answer = (
            "Strong sentence with real support in the source text. "
            "Weak sentence unrelated to anything in the source material."
        )
        score = summarize.compute_semantic_grounding(answer, "x" * 700)
        assert score == 0.2  # weakest of the two best-matches, not their average