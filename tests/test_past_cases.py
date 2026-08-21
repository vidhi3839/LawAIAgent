import pytest
from tasks import past_cases


# ── chunk_text ────────────────────────────────────────────────────────────

class TestChunkText:
    def test_short_text_produces_single_chunk(self):
        text = "one two three four five"
        chunks = past_cases.chunk_text(text, chunk_size=500, overlap=50)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_long_text_produces_overlapping_chunks(self):
        words = [f"word{i}" for i in range(1200)]
        text = " ".join(words)
        chunks = past_cases.chunk_text(text, chunk_size=500, overlap=50)
        assert len(chunks) >= 3
        chunk0_words = chunks[0].split()
        chunk1_words = chunks[1].split()
        assert chunk0_words[-50:] == chunk1_words[:50]

    def test_empty_text_returns_no_chunks(self):
        assert past_cases.chunk_text("", chunk_size=500, overlap=50) == []


# ── _deduplicate_cases (same pattern as mock_court.py's copy) ────────────

class TestDeduplicateCases:
    def test_keeps_highest_similarity_per_citation(self, sample_retrieved_cases_duplicated):
        result = past_cases._deduplicate_cases(sample_retrieved_cases_duplicated)
        citations = [c["citation"] for c in result]
        assert len(citations) == len(set(citations))

    def test_falls_back_to_case_name_when_no_citation(self):
        cases = [
            {"case_name": "Unpublished Case", "citation": "", "similarity_score": 0.3},
            {"case_name": "Unpublished Case", "citation": "", "similarity_score": 0.6},
        ]
        result = past_cases._deduplicate_cases(cases)
        assert len(result) == 1
        assert result[0]["similarity_score"] == 0.6


# ── compute_past_cases_confidence ─────────────────────────────────────────

class TestComputePastCasesConfidence:
    def test_no_results_returns_zero_confidence(self):
        result = past_cases.compute_past_cases_confidence(
            query="anything", retrieved_results=[], llm_answer="", requested_jurisdiction=None
        )
        assert result["final_score"] == 0.0
        assert result["above_threshold"] is False
        assert result["has_citations"] is False

    def test_jurisdiction_score_is_1_when_not_requested(self, sample_retrieved_cases_high_similarity):
        result = past_cases.compute_past_cases_confidence(
            query="q", retrieved_results=sample_retrieved_cases_high_similarity,
            llm_answer="some answer text", requested_jurisdiction=None
        )
        assert result["j_score"] == 1.0

    def test_jurisdiction_score_reflects_partial_match(self):
        results = [
            {"similarity_score": 0.5, "jurisdiction": "federal", "court": "9th Cir.", "text": "x"},
            {"similarity_score": 0.5, "jurisdiction": "state", "court": "NY Ct. App.", "text": "y"},
        ]
        result = past_cases.compute_past_cases_confidence(
            query="q", retrieved_results=results, llm_answer="answer",
            requested_jurisdiction="federal"
        )
        assert result["j_score"] == 0.5

    def test_grounding_score_reflects_answer_word_overlap_with_source(self):
        results = [{"similarity_score": 0.5, "jurisdiction": "federal",
                    "court": "SCOTUS", "text": "the court held negligence requires duty breach causation damages"}]
        grounded_answer = "The court held that negligence requires duty and breach."
        ungrounded_answer = "The court discussed unrelated antitrust merger doctrine entirely."
        g_high = past_cases.compute_past_cases_confidence(
            "q", results, grounded_answer, None)["g_score"]
        g_low = past_cases.compute_past_cases_confidence(
            "q", results, ungrounded_answer, None)["g_score"]
        assert g_high > g_low

    def test_has_citations_false_when_no_case_has_a_citation(self):
        results = [{"similarity_score": 0.5, "jurisdiction": "federal", "court": "x",
                    "text": "y", "citation": ""}]
        result = past_cases.compute_past_cases_confidence("q", results, "answer", None)
        assert result["has_citations"] is False

    def test_has_citations_true_when_at_least_one_case_has_a_citation(
        self, sample_retrieved_cases_high_similarity
    ):
        result = past_cases.compute_past_cases_confidence(
            "q", sample_retrieved_cases_high_similarity, "answer", None
        )
        assert result["has_citations"] is True

