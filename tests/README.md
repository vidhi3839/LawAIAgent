# LawAIAgent — Test Suite

## What this suite actually verifies, and what it doesn't

This suite checks that the **logic and scoring code** in this repo behaves
correctly — routing decisions, confidence formulas, retry rules, dedup
logic, guardrails against hallucinated citations, and lawyer-data
isolation. It does **not** and cannot verify that any given legal answer
is substantively correct — that needs a human who knows the law reading
the output.

## Setup


```bash
pip install -r tests/requirements-test.txt
pytest tests/ -v
```

## Three tiers of tests — run them differently, for different reasons

### 1. Fast unit tests (run every time, default `pytest tests/`)
Pure logic — scoring formulas, dedup, guardrails, routing — with Groq,
ChromaDB, and Postgres all mocked. No network, no real database, no LLM
tokens spent. Runs in seconds. This is what `pytest.ini`'s
`-m "not integration"` default restricts to.

| File | Covers |
|---|---|
| `test_mock_court.py` | keyword matching, statutory/precedent/vulnerability scoring, strategic strength bands, case dedup, LLM-driven functions (mocked) |
| `test_past_cases.py` | chunking, dedup, confidence scoring (M/J/G) |
| `test_statutory_api.py` | retry/backoff logic, truncation detection, subsection-citation matching, authority/integrity/link confidence scoring |
| `test_summarize.py` | date/entity extraction, decline detection, chronological + entity grounding proximity logic, summary/QA confidence scoring, semantic grounding aggregation logic |
| `test_router_llm.py` | citation-hallucination guardrail, router intent extraction, error sanitization, follow-up param reuse |
| `test_main.py` | PDF path validation guardrail, `track_best_node`, `verification_node` branches, `should_retry` decisions, graph wiring |
| `test_api.py` | `/query`, `/health`, `/threads/{id}/messages` HTTP contract |
| `test_statutory_api_fixtures.py` | Cornell/GovInfo parsing logic replayed against a real captured snapshot (no live network) |

### 2. Integration tests (run deliberately, not on every commit)
Need a real dependency — live network, a real embedding model, or your
actual Postgres — so they're excluded from the default run and marked
`integration` in `pytest.ini`.

```bash
pytest tests/ -m integration -v
```

| File | Needs | Run how often |
|---|---|---|
| `test_live_smoke.py` | Real internet (current Cornell/GovInfo pages) | Weekly, or when you suspect a site changed |
| `test_past_cases_integration.py` | Real ChromaDB + real embedding model (disposable in-memory collection, your production data is never touched) | Whenever `search_past_cases`/ingestion logic changes |
| `test_thread_ownership_integration.py` | Your real `DATABASE_URL` (writes/deletes rows tagged `TEST_ISOLATION_...`, cleans up after itself) | Whenever thread-ownership/privacy logic changes. |

### 3. Manual scripts (run by hand, not part of `pytest` at all)
| File | Purpose | Cost |
|---|---|---|
| `capture_fixtures.py` | Captures fresh real HTML/JSON snapshots from Cornell/GovInfo into `tests/fixtures/` for the fixture tests above to replay. Re-run every few months so fixtures don't go stale. | Free, one-time per run |
| `eval_answers.py` | Runs ~10-20 real, representative legal questions through the actual agent (real Groq calls, real ChromaDB) and writes a timestamped markdown file for **human legal review**. Automated checks here catch obvious breakage (empty responses, zero confidence, prompt-injection success) — they cannot tell you if an answer is legally *correct*. Only a human reading the output can. | Real Groq tokens + network, run deliberately after changing a prompt or model |
