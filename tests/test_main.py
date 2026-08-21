import os
import sys
import importlib
import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture(scope="module")
def main_module(request):
    os.environ.setdefault("DATABASE_URL", "postgresql://fake:fake@localhost/fake")
    os.environ.setdefault("GROQ_API_KEY", "fake-key-for-tests")

    for mod_name in ["main"]:
        if mod_name in sys.modules:
            del sys.modules[mod_name]

    from langgraph.checkpoint.memory import MemorySaver
    fake_pool = MagicMock()
    fake_conn_ctx = MagicMock()
    fake_pool.connection.return_value.__enter__.return_value = fake_conn_ctx
    fake_conn_ctx.execute.return_value.fetchall.return_value = []
    fake_conn_ctx.execute.return_value.fetchone.return_value = None
    fake_checkpointer = MemorySaver()

    with patch("psycopg_pool.ConnectionPool", return_value=fake_pool), \
         patch("langgraph.checkpoint.postgres.PostgresSaver", return_value=fake_checkpointer):
        main = importlib.import_module("main")
        yield main

    if "main" in sys.modules:
        del sys.modules["main"]


# ── _validate_pdf_path ─────────────────────────────────────────────────────

class TestValidatePdfPath:
    def test_empty_path_rejected(self, main_module):
        ok, msg = main_module._validate_pdf_path("")
        assert ok is False
        assert "no file path" in msg.lower()

    def test_path_traversal_rejected(self, main_module):
        ok, msg = main_module._validate_pdf_path("../../etc/passwd.pdf")
        assert ok is False
        assert "traversal" in msg.lower()

    def test_nonexistent_file_rejected(self, main_module):
        ok, msg = main_module._validate_pdf_path("/tmp/definitely_does_not_exist_123.pdf")
        assert ok is False
        assert "not found" in msg.lower()

    def test_non_pdf_extension_rejected(self, tmp_path, main_module):
        f = tmp_path / "document.txt"
        f.write_text("not a pdf")
        ok, msg = main_module._validate_pdf_path(str(f))
        assert ok is False
        assert "expected a .pdf" in msg.lower()

    def test_valid_pdf_path_accepted(self, tmp_path, main_module):
        f = tmp_path / "document.pdf"
        f.write_bytes(b"%PDF-1.4 fake pdf bytes")
        ok, resolved = main_module._validate_pdf_path(str(f))
        assert ok is True
        assert resolved.endswith("document.pdf")

    def test_path_outside_allowed_root_rejected(self, tmp_path, main_module):
        allowed_root = tmp_path / "uploads"
        allowed_root.mkdir()
        outside_file = tmp_path / "elsewhere.pdf"
        outside_file.write_bytes(b"%PDF-1.4")
        ok, msg = main_module._validate_pdf_path(str(outside_file), allowed_root=str(allowed_root))
        assert ok is False
        assert "outside allowed directory" in msg.lower()


# ── track_best_node ────────────────────────────────────────────────────────

class TestTrackBestNode:
    def test_start_node_resets_best_fields_every_turn(self, main_module):
        stale_state_from_a_previous_turn = {
            "best_response": "an answer to a completely different earlier question",
            "best_confidence": 0.9,
            "best_intent": "definition",
        }
        result = main_module.start_node(stale_state_from_a_previous_turn)
        assert result["best_response"] is None
        assert result["best_confidence"] is None
        assert result["best_intent"] is None

    def test_first_attempt_is_always_captured_even_at_zero_confidence(self, main_module):
        """Regression check: best_confidence starts as None specifically
        so a genuine 0.0-confidence first attempt still gets recorded."""
        state = {"final_response": "some answer", "confidence_score": 0.0,
                 "intent": "definition", "best_confidence": None}
        result = main_module.track_best_node(state)
        assert result == {
            "best_response": "some answer",
            "best_confidence": 0.0,
            "best_intent": "definition",
        }

    def test_does_not_overwrite_a_better_earlier_attempt(self, main_module):
        state = {"final_response": "worse answer", "confidence_score": 0.3,
                 "intent": "statute", "best_confidence": 0.9}
        result = main_module.track_best_node(state)
        assert result == {}  # no-op: worse attempt must not replace the better one

    def test_overwrites_when_current_is_strictly_better(self, main_module):
        state = {"final_response": "better answer", "confidence_score": 0.95,
                 "intent": "statute", "best_confidence": 0.9}
        result = main_module.track_best_node(state)
        assert result["best_confidence"] == 0.95
        assert result["best_response"] == "better answer"


# ── verification_node ───────────────────────────────────────────────────────

class TestVerificationNode:
    def test_compound_disclaimer_never_appears_on_error_responses(self, main_module, monkeypatch):
        monkeypatch.setattr(
            main_module, "_verification_node_core",
            lambda state: {"final_response": "Error: something failed", "confidence_score": 0.0}
        )
        state = {"error": "Could not process this request right now.", "is_possibly_compound": True}
        result = main_module.verification_node(state)
        assert "more than one part" not in result["final_response"]

    def test_compound_disclaimer_appears_on_successful_compound_responses(self, main_module, monkeypatch):
        monkeypatch.setattr(
            main_module, "_verification_node_core",
            lambda state: {"final_response": "Here is a real definition.", "confidence_score": 0.9}
        )
        state = {"error": None, "is_possibly_compound": True}
        result = main_module.verification_node(state)
        assert "more than one part" in result["final_response"]

    def test_error_state_short_circuits_to_zero_confidence_response(self, main_module):
        state = {"error": "File not found: /tmp/x.pdf"}
        result = main_module.verification_node(state)
        assert result["confidence_score"] == 0.0
        assert "File not found" in result["final_response"]

    def test_mock_court_missing_result_handled_gracefully(self, main_module):
        state = {"error": None, "intent": "mock_court", "mock_result": {},
                 "raw_api_text": "", "source_url": "", "source_domain": "",
                 "parsed_parameters": {}, "query": "My argument is X"}
        result = main_module.verification_node(state)
        assert result["confidence_score"] == 0.0
        assert "could not run argument analysis" in result["final_response"].lower()

    def test_past_cases_no_results_returns_zero_confidence(self, main_module):
        state = {"error": None, "intent": "past_cases", "retrieved_cases": [],
                  "raw_api_text": "", "source_url": "", "source_domain": "",
                  "parsed_parameters": {}, "query": "any relevant precedent?"}
        result = main_module.verification_node(state)
        assert result["confidence_score"] == 0.0
        assert "no relevant cases found" in result["final_response"].lower()

    def test_statute_error_prefixed_raw_text_short_circuits(self, main_module):
        state = {
            "error": None, "intent": "statute",
            "raw_api_text": "Could not find Title 18 U.S.C. Section 9999 in GovInfo.",
            "source_url": "https://www.govinfo.gov/x", "source_domain": "api.govinfo.gov",
            "parsed_parameters": {"title": 18, "section": "9999"}, "query": "18 USC 9999",
        }
        result = main_module.verification_node(state)
        assert result["confidence_score"] == 0.0


# ── should_retry (currently vestigial — kept for when/if retries return) ──

class TestShouldRetry:
    def test_followup_intent_never_retries(self, main_module):
        assert main_module.should_retry({"intent": "followup"}) == "done"

    def test_permanent_error_returns_done_not_retry(self, main_module):
        state = {"intent": "summarizer", "attempt_count": 0, "error": "File not found: x.pdf",
                 "user_threshold": None}
        assert main_module.should_retry(state) == "done"

    def test_zero_confidence_with_no_error_is_done_not_retry(self, main_module):
        """Confirms the explicit design choice: a hard 0.0 with no error
        is accepted as final, not retried into a loop."""
        state = {"intent": "definition", "attempt_count": 0, "error": None,
                 "confidence_score": 0.0, "user_threshold": None}
        assert main_module.should_retry(state) == "done"

    def test_below_threshold_confidence_with_retries_available_says_retry(self, main_module):
        state = {"intent": "statute", "attempt_count": 0, "error": None,
                 "confidence_score": 0.3, "user_threshold": 0.9}
        assert main_module.should_retry(state) == "retry"

    def test_max_attempts_reached_forces_done(self, main_module):
        state = {"intent": "statute", "attempt_count": 99, "error": None,
                 "confidence_score": 0.1, "user_threshold": 0.9}
        assert main_module.should_retry(state) == "done"


# ── Graph wiring (structural / integration smoke test) ─────────────────────

class TestGraphWiring:
    def test_graph_is_single_pass_after_retry_revert(self, main_module):
        graph = main_module.legal_agent_graph
        assert graph is not None
        node_names = set(main_module.workflow.nodes.keys())
        assert {"start", "detect_compound", "router", "api_runner", "verifier", "track_best"}.issubset(node_names)
        assert "prepare_retry" not in node_names
        assert not hasattr(main_module, "retry_decision_node")
        assert not hasattr(main_module, "retry_prep_node")

    def test_should_retry_and_prepare_retry_node_still_defined_but_unwired(self, main_module):
        """The functions themselves are intentionally kept (not deleted)
        so whoever debugs the attempt_count bug doesn't have to
        rewrite them from scratch — just confirms they still exist and
        are callable in isolation, independent of the graph."""
        assert callable(main_module.should_retry)
        assert callable(main_module.prepare_retry_node)

    def test_detect_compound_question_node_is_wired_and_callable(self, main_module):
        assert callable(main_module.detect_compound_question_node)
        node_names = set(main_module.workflow.nodes.keys())
        assert "detect_compound" in node_names