"""
Shared fixtures for the LawAIAgent test suite.

Design goal: unit-test the SCORING/LOGIC in this codebase without ever hitting
Groq, GovInfo, Cornell, ChromaDB, or Postgres over the network. Every fixture
here exists to make one of those four things fake and deterministic.

"""
import os
import sys
import types
import pytest
from unittest.mock import MagicMock, patch


PROJECT_ROOT = os.environ.get("LAWAIAGENT_ROOT", os.getcwd())
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


class FakeLLMResponse:
    """Mimics langchain's AIMessage enough for .content access."""
    def __init__(self, content="Mocked LLM response.", tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []


@pytest.fixture
def fake_llm():

    m = MagicMock()
    m.invoke.return_value = FakeLLMResponse("Mocked analysis text.")
    return m


@pytest.fixture
def sample_argument_strong():
    return (
        "Pursuant to 42 U.S.C. § 1983, the defendant acted in good faith "
        "and was not aware of any wrongdoing. This action is time-barred "
        "under the statute of limitations. The plaintiff lacks standing "
        "to sue as there is no injury in fact. The defendant's conduct "
        "falls under qualified immunity as a form of privilege."
    )


@pytest.fixture
def sample_argument_weak():
    return "The defendant did a bad thing and should be punished."


@pytest.fixture
def sample_retrieved_cases_high_similarity():
    return [
        {"case_name": "Smith v. Jones", "citation": "123 F.3d 456", "similarity_score": 0.62,
         "text": "Sample case text.", "jurisdiction": "federal", "court": "9th Cir."},
        {"case_name": "Doe v. Roe", "citation": "789 F.3d 101", "similarity_score": 0.58,
         "text": "Another sample.", "jurisdiction": "federal", "court": "9th Cir."},
    ]


@pytest.fixture
def sample_retrieved_cases_duplicated():
    """Same case (same citation) appearing as three chunks — the exact
    pattern _deduplicate_cases exists to collapse."""
    return [
        {"case_name": "Palsgraf v. LIRR", "citation": "248 N.Y. 339", "similarity_score": 0.41, "text": "chunk 1"},
        {"case_name": "Palsgraf v. LIRR", "citation": "248 N.Y. 339", "similarity_score": 0.55, "text": "chunk 2"},
        {"case_name": "Palsgraf v. LIRR", "citation": "248 N.Y. 339", "similarity_score": 0.33, "text": "chunk 3"},
        {"case_name": "Lucy v. Zehmer", "citation": "196 Va. 493", "similarity_score": 0.47, "text": "other case"},
    ]


def make_fake_pool_and_checkpointer():
    """Builds the two objects test_main.py / test_api.py need to import
    main.py / api.py without touching a real Postgres instance.

    - fake_pool: a MagicMock standing in for psycopg_pool.ConnectionPool.
      Its .connection() context manager returns a mock connection whose
      .execute(...).fetchall()/.fetchone() return empty results — fine,
      since main.py's table-creation SQL doesn't need real behavior here.

    - a REAL langgraph.checkpoint.memory.MemorySaver instance to stand in
      for PostgresSaver's return value. 
    """
    from langgraph.checkpoint.memory import MemorySaver

    fake_pool = MagicMock()
    fake_conn_ctx = MagicMock()
    fake_pool.connection.return_value.__enter__.return_value = fake_conn_ctx
    fake_conn_ctx.execute.return_value.fetchall.return_value = []
    fake_conn_ctx.execute.return_value.fetchone.return_value = None

    return fake_pool, MemorySaver()


@pytest.fixture
def mock_requests_head():
    with patch("requests.request") as mock_req:
        response = MagicMock()
        response.status_code = 200
        mock_req.return_value = response
        yield mock_req


@pytest.fixture(autouse=True)
def block_real_network(request, monkeypatch):
    if os.environ.get("LAWAIAGENT_ALLOW_NETWORK") == "1":
        return
    if request.node.get_closest_marker("integration"):
        return  

    def _blocked(*args, **kwargs):
        raise RuntimeError(
            "A real network call was attempted during a test. "
            "Mock requests.request / requests.get / requests.head explicitly."
        )

    monkeypatch.setattr("requests.request", _blocked)