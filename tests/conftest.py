"""
Shared fixtures for the LawAIAgent test suite.

Design goal: unit-test the SCORING/LOGIC in this codebase without ever hitting
Groq, GovInfo, Cornell, ChromaDB, or Postgres over the network. Every fixture
here exists to make one of those four things fake and deterministic.

Run with:
    pip install pytest
    pytest tests/ -v

Note: mock_court.py, past_cases.py, summarize.py, router_llm.py, and main.py
import langchain_groq / chromadb / sentence_transformers / langgraph /
psycopg_pool at module load time. Those packages must already be installed
in whatever environment runs this app (they're a hard requirement of the
app itself) — this test suite does not install them, it only fakes their
NETWORK/LLM CALLS once imported.
"""
import os
import sys
import types
import pytest
from unittest.mock import MagicMock, patch

# Make sure the project root (where tasks/, main.py, router_llm.py live)
# is importable. Adjust this path if your test folder sits somewhere else.
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
    """A stand-in for a whole ChatGroq-like object (has .invoke()).

    IMPORTANT: newer pydantic (2.x) blocks setting attributes that aren't
    declared model fields directly onto a BaseModel instance — which is
    what ChatGroq / RunnableBinding are. That means
    `monkeypatch.setattr(mock_court.llm, "invoke", fake)` raises
    `ValueError: "ChatGroq" object has no field "invoke"` on recent
    versions. The fix is to never patch an attribute ONTO the real
    pydantic object — instead replace the whole module-level name
    (`mock_court.llm`, `router_llm.router_llm_with_tools`, etc.) with
    this fake object, via `monkeypatch.setattr(module, "llm", fake_llm)`.
    A MagicMock is a plain object, so setting .invoke on IT is fine.
    """
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
      for PostgresSaver's return value. This matters: newer LangGraph
      validates the checkpointer's type at workflow.compile(time) and
      rejects a bare MagicMock with
      `TypeError: Invalid checkpointer provided... Received MagicMock`.
      MemorySaver is a real, in-memory implementation of
      BaseCheckpointSaver, so it satisfies that check and actually works
      for the lifetime of the test process (state just isn't persisted
      anywhere real, which is exactly what a test wants).
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
    """Patches requests.request (used inside _request_with_retry) so
    calculate_legal_confidence's link-reachability check never hits the
    network. Configure status_code on the mock's return_value.

    This explicit `patch()` context manager takes precedence over
    `block_real_network` below for its own duration (unittest.mock.patch
    always wins for the scope it's active in, regardless of what a
    monkeypatch fixture set beforehand) — so tests using this fixture
    work exactly as before, no interaction with the new hard block."""
    with patch("requests.request") as mock_req:
        response = MagicMock()
        response.status_code = 200
        mock_req.return_value = response
        yield mock_req


@pytest.fixture(autouse=True)
def block_real_network(request, monkeypatch):
    """Safety net: if any test forgets to mock a network call, fail loudly
    and fast instead of silently hanging or hitting a real endpoint.

    Blocks `requests.request` for any test NOT marked `integration` —
    tests marked `integration` (test_live_smoke.py,
    test_thread_ownership_integration.py) genuinely need real network/DB
    access and are left alone. Can be disabled entirely for a manual
    debugging session via LAWAIAGENT_ALLOW_NETWORK=1.

    A test that legitimately needs requests.request (e.g. via the
    mock_requests_head fixture above, or an explicit `with
    patch("requests.request")` in the test body) is unaffected — that
    explicit patch overrides this fixture's block for its own duration
    and is restored afterward, same as always.
    """
    if os.environ.get("LAWAIAGENT_ALLOW_NETWORK") == "1":
        return
    if request.node.get_closest_marker("integration"):
        return  # integration/live tests are allowed real network on purpose

    def _blocked(*args, **kwargs):
        raise RuntimeError(
            "A real network call was attempted during a test. "
            "Mock requests.request / requests.get / requests.head explicitly."
        )

    monkeypatch.setattr("requests.request", _blocked)