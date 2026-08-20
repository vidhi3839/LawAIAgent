"""
Integration test for tasks/past_cases.py's search_past_cases() — the one
thing the unit test suite explicitly could NOT cover, since it needs a
real ChromaDB collection + the real embedding model, not mocks.

Excluded from the default `pytest tests/` run (see pytest.ini). Run with:

    pytest tests/test_past_cases_integration.py -m integration -v

WHY A DISPOSABLE COLLECTION, NOT YOUR REAL ONE:
past_cases.py's `collection` variable is bound at import time to your
real production CHROMA_PATH. This test does NOT touch that data or that
path — it swaps `past_cases.collection` for a fresh, in-memory
(non-persistent) ChromaDB collection for the duration of the test only,
seeded with synthetic-but-realistic case chunks. Your real ingested case
data is never read or modified.

WHY THE REAL EMBEDDING MODEL, NOT A FAKE ONE:
The whole point of this test is to check real vector-search behavior
(does a semantically similar query actually retrieve the right chunk,
does dedup collapse the right chunks). A fake/random embedding would make
the ranking meaningless. This uses the same all-MiniLM-L6-v2 model your
app already loads — no new dependency, just the first real model load
taking a few seconds.
"""
import pytest
import chromadb
import uuid

from tasks import past_cases

pytestmark = pytest.mark.integration


@pytest.fixture
def disposable_collection(monkeypatch):
    """Builds a fresh in-memory ChromaDB collection, seeds it with
    synthetic case chunks (including 3 chunks of the SAME case, to
    directly exercise the dedup + over-fetch behavior), and swaps
    past_cases.collection to point at it for the test's duration only.

    IMPORTANT: chromadb.Client() collections are cached by name within the
    same process — reusing a fixed name across tests causes
    `chromadb.errors.InternalError: Collection [...] already exists` on
    the second test, since nothing deleted the first one. Fixed here by
    (a) giving each test run its own unique collection name, and
    (b) explicitly deleting the collection in teardown regardless."""
    client = chromadb.Client()  # in-memory, ephemeral — never touches disk
    collection_name = f"test_past_cases_{uuid.uuid4().hex[:8]}"
    test_collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    # Three chunks of the SAME case (same citation) with deliberately
    # different text, so a query can plausibly match more than one chunk —
    # this is exactly the scenario _deduplicate_cases exists to collapse.
    documents = [
        "Palsgraf v. Long Island Railroad addresses proximate cause and duty of care in negligence claims.",
        "The Palsgraf case established that a defendant owes a duty only to foreseeable plaintiffs.",
        "Palsgraf remains a foundational tort law case on the scope of duty and foreseeability.",
        "Lucy v. Zehmer concerns contract formation and the objective theory of mutual assent.",
    ]
    metadatas = [
        {"case_name": "Palsgraf v. Long Island Railroad Co.", "citation": "248 N.Y. 339",
         "year": "1928", "court": "New York Court of Appeals", "jurisdiction": "state",
         "legal_issues": "negligence, proximate cause"},
        {"case_name": "Palsgraf v. Long Island Railroad Co.", "citation": "248 N.Y. 339",
         "year": "1928", "court": "New York Court of Appeals", "jurisdiction": "state",
         "legal_issues": "negligence, proximate cause"},
        {"case_name": "Palsgraf v. Long Island Railroad Co.", "citation": "248 N.Y. 339",
         "year": "1928", "court": "New York Court of Appeals", "jurisdiction": "state",
         "legal_issues": "negligence, proximate cause"},
        {"case_name": "Lucy v. Zehmer", "citation": "196 Va. 493",
         "year": "1954", "court": "Supreme Court of Virginia", "jurisdiction": "state",
         "legal_issues": "contract formation, mutual assent"},
    ]
    embeddings = past_cases.embedder.encode(documents).tolist()
    test_collection.add(
        ids=[f"test_chunk_{i}" for i in range(len(documents))],
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )

    monkeypatch.setattr(past_cases, "collection", test_collection)
    yield test_collection
    client.delete_collection(name=collection_name)


class TestSearchPastCasesIntegration:
    def test_dedup_collapses_three_chunks_of_same_case_into_one_result(self, disposable_collection):
        result = past_cases.search_past_cases(
            query="What did the court say about duty of care and foreseeability?",
            n_results=5,
        )
        citations = [r["citation"] for r in result["results"]]
        assert citations.count("248 N.Y. 339") == 1  # not 3

    def test_relevant_query_ranks_matching_case_first(self, disposable_collection):
        result = past_cases.search_past_cases(
            query="contract formation and mutual assent between parties",
            n_results=2,
        )
        assert result["results"][0]["citation"] == "196 Va. 493"

    def test_jurisdiction_filter_actually_filters(self, disposable_collection):
        result = past_cases.search_past_cases(
            query="legal doctrine",
            n_results=5,
            jurisdiction_filter="state",
        )
        assert all(r["jurisdiction"] == "state" for r in result["results"])

    def test_empty_collection_returns_no_matching_cases_error(self, monkeypatch):
        client = chromadb.Client()
        collection_name = f"test_empty_{uuid.uuid4().hex[:8]}"
        empty_collection = client.get_or_create_collection(name=collection_name)
        monkeypatch.setattr(past_cases, "collection", empty_collection)

        result = past_cases.search_past_cases(query="anything at all", n_results=5)
        assert result["results"] == []
        assert result.get("error") == "No matching cases found"
        client.delete_collection(name=collection_name)