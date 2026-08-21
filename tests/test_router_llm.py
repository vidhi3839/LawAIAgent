import pytest
from unittest.mock import MagicMock

import router_llm


# ── _citation_matches_query ────────────────────────────────────────────────

class TestCitationMatchesQuery:
    def test_statute_numbers_present_in_query_pass(self):
        assert router_llm._citation_matches_query(
            "See 42 U.S.C. § 1983 for more", "statute", {"title": 42, "section": "1983"}
        ) is True

    def test_statute_hallucinated_number_fails(self):
        # Router extracted 1985, but the query actually says 1983
        assert router_llm._citation_matches_query(
            "See 42 U.S.C. § 1983 for more", "statute", {"title": 42, "section": "1985"}
        ) is False

    def test_rule_number_present_passes(self):
        assert router_llm._citation_matches_query(
            "Under Rule 56 this fails", "rule", {"rule_type": "frcp", "rule_number": "56"}
        ) is True

    def test_rule_number_absent_fails(self):
        assert router_llm._citation_matches_query(
            "Under Rule 56 this fails", "rule", {"rule_type": "frcp", "rule_number": "12"}
        ) is False

    def test_non_citation_intents_always_pass(self):
        assert router_llm._citation_matches_query("anything at all", "definition", {"term": "estoppel"}) is True
        assert router_llm._citation_matches_query("anything at all", "mock_court", {}) is True


# ── router_node_llm (LLM call mocked) ─────────────────────────────────────

class FakeToolCall(dict):
    """Mimics the shape of langchain's tool_calls entries: {"name", "args"}."""
    pass


class FakeRouterResponse:
    def __init__(self, tool_calls):
        self.tool_calls = tool_calls


class TestRouterNodeLLM:
    def test_no_previous_context_never_routes_to_followup(self, monkeypatch):
        response = FakeRouterResponse([FakeToolCall(name="DefinitionLookup", args={"term": "estoppel"})])
        fake_bound_llm = MagicMock(invoke=MagicMock(return_value=response))
        monkeypatch.setattr(router_llm, "router_llm_with_tools", fake_bound_llm)
        result = router_llm.router_node_llm({"query": "What is estoppel?"})
        assert result["intent"] == "definition"
        assert result["error"] is None

    def test_statute_citation_extracted_correctly(self, monkeypatch):
        response = FakeRouterResponse([
            FakeToolCall(name="StatuteLookup", args={"title": 18, "section": "1030"})
        ])
        fake_bound_llm = MagicMock(invoke=MagicMock(return_value=response))
        monkeypatch.setattr(router_llm, "router_llm_with_tools", fake_bound_llm)
        result = router_llm.router_node_llm({"query": "explain 18 U.S.C. 1030"})
        assert result["intent"] == "statute"
        assert result["parsed_parameters"] == {"title": 18, "section": "1030"}

    def test_hallucinated_citation_triggers_reask_and_fails_cleanly_if_still_wrong(self, monkeypatch):
        bad_response = FakeRouterResponse([
            FakeToolCall(name="StatuteLookup", args={"title": 18, "section": "9999"})
        ])
        fake_bound_llm = MagicMock(invoke=MagicMock(return_value=bad_response))
        monkeypatch.setattr(router_llm, "router_llm_with_tools", fake_bound_llm)
        result = router_llm.router_node_llm({"query": "explain 18 U.S.C. 1030"})
        assert result["error"] is not None
        assert "could not reliably extract" in result["error"].lower()

    def test_router_exception_returns_sanitized_rate_limit_message(self, monkeypatch):
        fake_bound_llm = MagicMock()
        fake_bound_llm.invoke.side_effect = RuntimeError("429 rate_limit_exceeded: tokens per minute")
        monkeypatch.setattr(router_llm, "router_llm_with_tools", fake_bound_llm)
        result = router_llm.router_node_llm({"query": "anything"})
        assert result["intent"] == "router_error"
        assert "rate limit" in result["error"].lower()

    def test_router_exception_returns_sanitized_config_error_message(self, monkeypatch):
        fake_bound_llm = MagicMock()
        fake_bound_llm.invoke.side_effect = RuntimeError("404 model_not_found: does not exist")
        monkeypatch.setattr(router_llm, "router_llm_with_tools", fake_bound_llm)
        result = router_llm.router_node_llm({"query": "anything"})
        assert result["intent"] == "router_error"
        assert "configuration" in result["error"].lower()

    def test_followup_reuses_previous_params_with_prior_context(self, monkeypatch):
        response = FakeRouterResponse([FakeToolCall(name="FollowUp", args={})])
        fake_bound_llm = MagicMock(invoke=MagicMock(return_value=response))
        monkeypatch.setattr(router_llm, "router_llm_with_tools", fake_bound_llm)
        state = {
            "query": "what's the penalty under this",
            "last_raw_text": "previously retrieved statute text",
            "last_params": {"title": 18, "section": "1030"},
        }
        result = router_llm.router_node_llm(state)
        assert result["intent"] == "followup"
        assert result["parsed_parameters"] == {"title": 18, "section": "1030"}