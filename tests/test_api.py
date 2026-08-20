"""
Tests for api.py (FastAPI endpoints)

api.py imports Postgres-backed helper functions and the query-handling
functions (run_query_with_retry, run_compound_query, run_multi_tool_query,
detect_compound_question_node) directly from main.py at module level, so
the database connection is mocked before api.py is imported here.

Each /query test mocks run_query_with_retry to return a specific result,
so the test checks how api.py's endpoint HANDLES that result (building
the response body, choosing status codes) rather than whether the
underlying agent produces a good answer. detect_compound_question_node is
also mocked (forced to always report "not compound") so a plain test
query always takes the same code path regardless of its wording.
"""
import os
import sys
import importlib
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    os.environ.setdefault("DATABASE_URL", "postgresql://fake:fake@localhost/fake")
    os.environ.setdefault("GROQ_API_KEY", "fake-key-for-tests")

    for mod_name in ["api", "main"]:
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
        api = importlib.import_module("api")
        yield TestClient(api.app)

    for mod_name in ["api", "main"]:
        if mod_name in sys.modules:
            del sys.modules[mod_name]


class TestHealthAndRoot:
    def test_health_endpoint(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "running"

    def test_root_endpoint(self, client):
        r = client.get("/")
        assert r.status_code == 200


class TestQueryEndpoint:
    def test_query_returns_best_response_when_graph_tracked_a_better_earlier_attempt(self, client, monkeypatch):
        monkeypatch.setenv("ENABLE_MULTI_TOOL", "false")
        with patch("api.detect_compound_question_node", return_value={"is_possibly_compound": False}), \
             patch("api.run_query_with_retry") as mock_run, \
             patch("api.save_thread_metadata"), patch("api.save_message"):
            mock_run.return_value = {
                "final_response": "worse, later attempt",
                "confidence_score": 0.2,
                "intent": "statute",
                "best_response": "the actual best answer",
                "best_confidence": 0.9,
                "best_intent": "statute",
            }
            r = client.post("/query", json={
                "query": "18 U.S.C. 1030",
                "thread_id": "thread-1",
                "lawyer_name": "Jane Doe",
            })
        assert r.status_code == 200
        body = r.json()
        assert body["response"] == "the actual best answer"
        assert body["confidence"] == 0.9

    def test_query_falls_back_to_final_response_if_best_unset(self, client, monkeypatch):
        monkeypatch.setenv("ENABLE_MULTI_TOOL", "false")
        with patch("api.detect_compound_question_node", return_value={"is_possibly_compound": False}), \
             patch("api.run_query_with_retry") as mock_run, \
             patch("api.save_thread_metadata"), patch("api.save_message"):
            mock_run.return_value = {
                "final_response": "only response available",
                "confidence_score": 0.6,
                "intent": "definition",
            }
            r = client.post("/query", json={
                "query": "define estoppel",
                "thread_id": "thread-2",
                "lawyer_name": "Jane Doe",
            })
        assert r.status_code == 200
        assert r.json()["response"] == "only response available"

    def test_recursion_limit_error_returns_graceful_system_limit_response(self, client, monkeypatch):
        monkeypatch.setenv("ENABLE_MULTI_TOOL", "false")
        with patch("api.detect_compound_question_node", return_value={"is_possibly_compound": False}), \
             patch("api.run_query_with_retry") as mock_run, \
             patch("api.save_thread_metadata"), patch("api.save_message"):
            mock_run.side_effect = RuntimeError("GraphRecursionError: recursion limit reached")
            r = client.post("/query", json={
                "query": "anything",
                "thread_id": "thread-3",
                "lawyer_name": "Jane Doe",
            })
        assert r.status_code == 200
        body = r.json()
        assert body["confidence"] == 0.0
        assert body["intent"] == "system_limit"

    def test_document_qa_intent_relabelled_from_summarizer(self, client, monkeypatch):
        monkeypatch.setenv("ENABLE_MULTI_TOOL", "false")
        with patch("api.detect_compound_question_node", return_value={"is_possibly_compound": False}), \
             patch("api.run_query_with_retry") as mock_run, \
             patch("api.save_thread_metadata"), patch("api.save_message"):
            mock_run.return_value = {
                "final_response": "answer to the specific question",
                "confidence_score": 0.8,
                "intent": "summarizer",
                "document_result": {"confidence": {"task": "qa"}},
            }
            r = client.post("/query", json={
                "query": "/tmp/doc.pdf what was the holding?",
                "thread_id": "thread-4",
                "lawyer_name": "Jane Doe",
            })
        assert r.status_code == 200
        assert r.json()["intent"] == "document_qa"


class TestThreadsEndpoint:
    def test_thread_messages_404_for_wrong_lawyer(self, client):
        with patch("api.thread_belongs_to_lawyer", return_value=False):
            r = client.get("/threads/some-thread/messages?lawyer_name=Someone+Else")
        assert r.status_code == 404