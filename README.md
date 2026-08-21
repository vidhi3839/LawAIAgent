# LawAIAgent

A multi-intent legal research assistant built with **LangGraph, FastAPI, React, ChromaDB, and Groq**.

The system uses Groq-hosted `openai/gpt-oss-120b` for language understanding and generation, while legal facts are grounded in retrieved source material and confidence is calculated programmatically.

---

## Quick Start — Setting Up on a New Device

```bash
# 1. Clone the repo
git clone https://github.com/vidhi3839/LawAIAgent
cd LawAIAgent

# 2. Install backend dependencies
python -m venv .venv
.venv\Scripts\activate        # Windows

pip install -r requirements.txt

# 3. Create your .env file in the project root (see "Environment Variables" further below for the full list) — at minimum:
#    GROQ_API_KEY=your_key
#    DATABASE_URL=your_postgres_connection_string

# 4. Start the backend
uvicorn api:app --reload --port 8000
# API now running at http://localhost:8000

# 5. In a NEW terminal, set up and start the frontend
cd lawaiagent-frontend
npm install
npm run dev
# Frontend now running at http://localhost:5173
```

Open `http://localhost:5173` in a browser — that's the full app running locally.

**Confirm it actually worked:**
```bash
curl http://localhost:8000/health
```
Should return `{"status":"running"}`. If it doesn't, see "Environment Variables" and "Running Locally" further below for the full detail on each step, including what each required credential is for.

**Prefer Docker instead?** Skip steps 2-5 above and see "Running with Docker" further below — it handles both backend and frontend in containers with one command.

---

## Features

- LLM-based intent routing
- U.S. statute lookup
- Federal rule lookup
- U.S. Constitution lookup
- Legal definition lookup
- Past-case hybrid search (semantic embeddings + BM25 keyword search)
- Document summarization and document Q&A
- Mock-court argument analysis
- Follow-up question handling
- Compound-question detection
- Programmatic confidence scoring
- Automatic retry when confidence is below the configured threshold
- Conversation/thread memory
- Lawyer/thread ownership checks
- React frontend
- FastAPI backend
- Docker Compose setup

---

## Architecture

## Architecture

```text
                         ┌──────────────────┐
                         │   React Frontend │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │    FastAPI API   │
                         └────────┬─────────┘
                                  │
                                  ▼
                    ┌──────────────────────────┐
                    │  run_query_with_retry()  │
                    │  max 2 total attempts    │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │       LangGraph          │
                    │       single-pass        │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │ Compound Detection       │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │      LLM Router          │
                    │ Groq / GPT-OSS 120B      │
                    └────────────┬─────────────┘
                                 │
                    ┌────────────┼────────────┐
                    ▼            ▼            ▼
               Legal APIs   Case Search   Documents
                    │            │            │
                    └────────────┼────────────┘
                                 ▼
                    ┌──────────────────────────┐
                    │ Verification & Confidence│
                    │   Programmatic scoring   │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                              Response
```
### Retry Architecture

The LangGraph workflow itself is single-pass and does not contain a retry loop.
When the resulting confidence score is below the configured retry threshold,
`run_query_with_retry()` invokes the compiled graph again, up to a maximum of
two total attempts. Retry bookkeeping is maintained by the surrounding Python
function rather than as a LangGraph state channel.

---

## Project Structure

```text
LawAIAgent/
│
├── main.py
├── api.py
├── router_llm.py
├── requirements.txt
├── CLAUDE.md
├── .gitignore
│
├── tasks/
│   ├── __init__.py
│   ├── statutory_api.py
│   ├── past_cases.py
│   ├── mock_court.py
│   ├── summarize.py
│   └── multi_tool_runner.py
│
├── scripts/
│   ├── test_one_query.py
│   ├── trace_state_history.py
│   └── tune_chunk_size.py
│
├── tests/
│   ├── README.md
│   ├── conftest.py
│   ├── pytest.ini
│   ├── requirements-test.txt
│   ├── test_api.py
│   ├── test_main.py
│   ├── test_router_llm.py
│   ├── test_mock_court.py
│   ├── test_past_cases.py
│   ├── test_past_cases_integration.py
│   ├── test_statutory_api.py
│   ├── test_statutory_api_fixtures.py
│   ├── test_summarize.py
│   ├── test_thread_ownership_integration.py
│   ├── test_live_smoke.py
│   ├── eval_answers.py
│   ├── capture_fixtures.py
│   └── fixtures/
│
├── lawaiagent-frontend/
│   ├── package.json
│   ├── package-lock.json
│   ├── src/
│   └── public/
│
├── docker/
│   ├── Dockerfile.backend
│   ├── Dockerfile.frontend
│   ├── docker-compose.yml
│   ├── nginx.conf
│   ├── .dockerignore
│   └── requirements.txt
│
└── .claude/
    ├── commands/
    ├── hooks/
    └── skills/
```

---

## Technology Stack

### Backend

* Python 3.12
* FastAPI
* Uvicorn
* LangGraph
* LangChain
* Groq
* GPT-OSS 120B (Groq-hosted)
* ChromaDB
* Sentence Transformers
* rank-bm25
* Supabase PostgreSQL
* psycopg
* BeautifulSoup / requests
* pypdf

### Frontend

* React
* Vite
* JavaScript
* CSS

### Deployment

* Docker
* Docker Compose
* Nginx

---

## LLM

The application uses:

```text
Model: openai/gpt-oss-120b
Provider: Groq
Integration: langchain-groq
```

The LLM is used for language-understanding and generation tasks such as:

* Intent routing
* Parameter extraction
* Answer synthesis
* Document analysis
* Mock-court analysis

The application is designed so that legal answers are generated from retrieved source material rather than relying on the model's general training knowledge for legal facts.

---

## Supported Intents

The router currently supports the following main intents:

```text
StatuteLookup
RuleLookup
ConstitutionLookup
DefinitionLookup
PastCasesSearch
MockCourtAnalysis
DocumentSummarize
FollowUp
```

### Statute Lookup

Retrieves U.S. Code content through GovInfo.

### Rule Lookup

Retrieves federal rules such as:

* Federal Rules of Evidence
* Federal Rules of Civil Procedure
* Federal Rules of Appellate Procedure

### Constitution Lookup

Retrieves constitutional amendment text.

### Definition Lookup

Retrieves legal definitions from Cornell Wex.

### Past Cases Search

Searches the firm's uploaded case collection using hybrid retrieval.

### Mock Court Analysis

Analyzes a lawyer's argument by identifying:

* Counterarguments
* Evidence/proof strength
* Potential judicial gaps
* Strategic strength

The strategic-strength score is calculated programmatically.

### Document Summarization

Summarizes or answers questions about lawyer-uploaded documents.

### Follow-Up

Uses conversation/thread context to resolve references such as:

```text
"What about the first one?"
"Can you explain it?"
"Does that apply here?"
```

---

## How Each Tool Sources Data, Prompts the LLM, and Scores Confidence

Every tool follows the same underlying discipline — real data in, LLM only
formats or reasons over that real data, confidence computed against
something checkable — but the specifics differ by tool. This section
covers each one explicitly, since it matters for understanding exactly
what the system can and can't verify.

### Statute / Rule / Constitution / Definition Lookup

**Where the data comes from:** live external sources, fetched fresh per
query — `fetch_statute` calls GovInfo's API directly (searching package
granules for the matching section, then fetching that granule's HTML
content); `fetch_federal_rule`, `fetch_constitutional_amendment`, and
`fetch_legal_definition` scrape the corresponding Cornell Law School
(LII) page with BeautifulSoup. Nothing here comes from training data or
a local database.

**What's fed to the LLM:** the raw fetched text (`raw_api_text`,
truncated to 5,000 characters for statutes or 3,000 for Cornell pages)
plus the source URL, passed into `LEGAL_PRESENTATION_PROMPT` — which
explicitly instructs the model to present that text accurately, add
headers for readability, and never add anything from its own knowledge.

**Confidence formula** (`calculate_legal_confidence` in
`statutory_api.py`): `authority × 0.4 + integrity × 0.4 + link × 0.2`
— authority is 1.0 for `.gov`/`.edu` domains, 0.9 for CourtListener,
0.5 otherwise; integrity is the fraction of significant words (and
specifically "not"/"no", tracked separately to catch a negation flip)
from the raw source that still appear in the LLM's formatted output;
link is whether the source URL is still reachable right now (checked
live via an HTTP HEAD request).

### Past Cases Search

Past-case search uses **hybrid retrieval**, combining semantic embedding
search with BM25 keyword search via Reciprocal Rank Fusion (RRF).


### Semantic Search

A sentence-transformer embedding model (`all-MiniLM-L6-v2`) converts
case chunks and the user's query into vector representations. This
finds conceptually related passages even when the exact words differ —
e.g. a query about "an employer punishing a worker for reporting
discrimination" can match a case actually described using different
wording like "retaliation" or "adverse action."

### Keyword Search

A BM25 index (via `rank-bm25`) over the same case chunks catches exact
citations, party names, and statutory references that semantic search
alone can miss or under-rank.

### Hybrid Retrieval

Both rankings are merged with Reciprocal Rank Fusion — a result that
scores well on *either* signal rises to the top, rather than being
limited to whichever single method was used. 

---


### Mock Court Analysis

**Where the data comes from:** two sources — the lawyer's own submitted
argument text (no external fetch needed for this part), and up to 3
retrieved past cases via the exact same `search_past_cases` hybrid-search
function `PastCasesSearch` uses (shared code, not a separate lookup).

**What's fed to the LLM:** the lawyer's argument text plus each
retrieved case's name, citation, and a 300-character excerpt, across
3 separate prompts (`analyze_counter_arguments`, `evaluate_proof_strength`,
`identify_judicial_gaps`), each capped at 900 output tokens.

**Confidence formula** (`compute_strategic_strength` in `mock_court.py`):
`statutory grounding × 0.4 + precedent support × 0.4 + vulnerability
exposure × 0.2` — statutory grounding is a regex check for a real
citation pattern (or legal terminology as a partial signal); precedent
support is banded off the average similarity score of the retrieved
cases; vulnerability exposure is how many of 5 standard legal defenses
(good faith, statute of limitations, standing, etc.) the argument
already addresses, scaled by argument length. **This score is computed
entirely independently of what the 3 LLM calls write** — the LLM's
analysis and the numeric score are not the same process checking itself.

### Document Summarization / Document Q&A

**Where the data comes from:** the lawyer's own uploaded PDF — its text
is extracted directly from the file the lawyer provided, not fetched
from anywhere external.

**What's fed to the LLM:** the extracted document text, plus the
lawyer's specific question if one was asked (otherwise a general
summarization instruction).

**Confidence formula:** depends on whether a specific question was asked.
For a general summary (`compute_summary_confidence`), it averages
whichever of these signals are actually available — chronological
integrity (do dates mentioned in the summary appear near matching event
language in the source), entity grounding (do named parties mentioned
appear near matching role language in the source), and semantic
grounding (does the summary's meaning actually match the source, via
embedding similarity) — weighting dynamically by how many signals are
available, not a fixed denominator.

### Follow-Up

**Where the data comes from:** nothing is re-fetched — it reuses
`last_raw_text` and `last_source_url` from the previous turn's state,
plus the running `thread_summary`, which is used only to figure out what
"it"/"the first one" refers to, never treated as a source of legal facts
to restate.

**What's fed to the LLM:** the previous turn's raw retrieved text, the
thread summary, and the new question, via `FOLLOWUP_PROMPT` — explicitly
instructed to answer only from the previously retrieved text and say so
clearly if that text doesn't actually cover the new question.

**Confidence formula:** if the answer is an honest decline ("that's not
covered here"), confidence is 100% — declining correctly is scored as a
success, not a failure. Otherwise, confidence is the semantic grounding
score between the new answer and the original retrieved text (or 0.7 as
a fallback if that check can't run).

### Compound Question Handling

**Where the data comes from:** two independent, full lookups — each half
of the split question goes through the exact same tool-routing and
retrieval process as a normal single question, on its own sub-thread.

**What's fed to the LLM:** both halves' retrieved text and sources,
merged via `MULTI_PART_PRESENTATION_PROMPT` into one structured answer.

**Confidence formula:** the **minimum** of three signals — part 1's
confidence, part 2's confidence, and how well the merged answer's words
actually appear in either source's text. Taking the weakest of the three
means a strong first half can't mask a weak second half.

---

## LangGraph Workflow

The main workflow is implemented as a LangGraph `StateGraph`, and is a
**single-pass, linear graph** — it does not loop or cycle internally.

```text
Start
  │
  ▼
Compound Question Detection
  │
  ▼
LLM Router
  │
  ▼
Tool Execution
  │
  ▼
Verification
  │
  ▼
Confidence / Best Result Tracking
  │
  ▼
End (graph invocation returns)
```

---

## Confidence Scoring

Confidence scores are calculated programmatically.

The LLM does **not** decide its own confidence score.

Different tasks use different scoring approaches because the available evidence differs by task.

Examples include:

* Legal citation/source matching
* Retrieval similarity
* Link reachability
* Text grounding
* Past-case retrieval quality
* Mock-court strategic scoring

If the score falls below the configured retry threshold, the surrounding
`run_query_with_retry()` function invokes the graph again. The retry process is
hard-capped at 2 total attempts.

---

## Memory

The application has two different memory mechanisms.

### LangGraph Checkpointer

`PostgresSaver` stores LangGraph execution state using PostgreSQL.

### Thread Summary

A separate conversation summary is maintained for each thread.

It is primarily used to provide conversational context and resolve references in follow-up questions without repeatedly sending the complete conversation history to the LLM.

---

## Multi-Tool Retrieval

`tasks/multi_tool_runner.py` contains a separate multi-tool retrieval path.

It can select:

```text
Primary Tool
+
Secondary Tool
```

and retrieve information from both.

This path is controlled through:

```text
ENABLE_MULTI_TOOL
```

and is disabled by default.

It remains architecturally separate from the primary single-tool path, with
its own routing prompts and confidence-scoring logic. This is documented as
known technical debt and the single-tool LangGraph path remains the primary
application workflow.

---

## Error Handling

The application includes several layers of error handling:

* Network retry handling
* LLM exception handling
* Empty retrieval handling
* Citation validation
* Retry limits
* Graph execution error handling
* PDF path validation
* Thread/lawyer ownership validation

The system should return an explicit failure/no-information response rather than silently inventing an answer.

---

## Requirements

The main dependency file is:

```text
requirements.txt
```

For tests:

```text
tests/requirements-test.txt
```

For Docker:

```text
docker/requirements.txt
```

---

## Environment Variables

Create a `.env` file in the project root.

```env
# Required
GROQ_API_KEY=your_groq_api_key
DATABASE_URL=your_database_url

# Optional — defaults to DEMO_KEY if unset, which is subject to
# GovInfo's stricter rate limits on unregistered API keys
GOVINFO_API_KEY=your_govinfo_api_key

# Optional — toggles the separate multi-tool retrieval path
# (tasks/multi_tool_runner.py). Defaults to false. See "Multi-Tool
# Retrieval" above before enabling — this path is less integrated
# and less tested than the main single-tool path.
ENABLE_MULTI_TOOL=false
```

---


## Running with Docker

**Note:** The native setup above is the primary verified development path.
The Docker setup is provided for containerized deployment and should be
rebuilt with `--build` after configuration or dependency changes.

From the project root:

```bash
docker compose -f docker/docker-compose.yml up --build
```

The Docker setup contains:

* A Python backend container
* A React frontend build container (multi-stage: Node build → nginx serve)
* An Nginx frontend container
* Persistent upload storage
* Persistent ChromaDB storage
* Persistent uploaded-case PDF storage
* Backend health checks (90-second start period — the backend loads
  PyTorch, sentence-transformers, and connects to Postgres before
  opening its port, which realistically takes 30-90+ seconds)

To stop the application:

```bash
docker compose -f docker/docker-compose.yml down
```

The persistent Docker volumes are kept unless explicitly removed.

---

## API Health Check

Once the backend is running:

```text
GET /health
```

Example:

```bash
curl http://localhost:8000/health
```

---

## Testing

Install test dependencies:

```bash
pip install -r tests/requirements-test.txt
```

Run the default test suite:

```bash
pytest tests/ -v
```

The default suite focuses on unit-level logic and mocks external dependencies.

### Integration Tests

Integration tests require real external dependencies and are intentionally separated from the default test run.

Run them with:

```bash
pytest tests/ -m integration -v
```

These include tests involving:

* Live Cornell/GovInfo endpoints
* Real embedding models
* ChromaDB
* PostgreSQL/thread ownership

### Evaluation

`tests/eval_answers.py` runs representative legal questions against the actual system and produces output for human review.

Automated evaluation can detect obvious failures, but it cannot determine whether a generated legal answer is substantively correct.

Legal correctness requires human review.

---

## Important Development Conventions

### Confidence Scores

Confidence calculations must remain programmatic.

### Legal Sources

The system must not fabricate:

* Case law
* Statute text
* Legal citations
* Source content

Generated answers should be grounded in retrieved material.

### Adding a New Intent

A new intent must be wired consistently through the router, execution,
verification, retry configuration, and frontend confidence thresholds.

At minimum, review:

```text
router_llm.py
  ├── TOOL_NAME_TO_INTENT
  └── router system prompt

main.py
  ├── api_execution_node
  ├── verification_node
  └── DEFAULT_RETRY_THRESHOLDS

ConfidenceCard.jsx
  └── DEFAULT_THRESHOLDS
```
---  
  
### Uploaded PDFs

User-supplied PDF paths must pass through the application's PDF path validation logic.

Do not directly open an unvalidated user-supplied path.

---

## Known Architecture Limitations

### Separate Multi-Tool Path

`multi_tool_runner.py` contains routing and confidence logic separate from the
primary LangGraph path. This creates some duplication and should be considered
technical debt before extending either architecture further.

### Compound Question Detection

Compound-question handling currently uses heuristic detection. It should not
be considered a complete general-purpose multi-question reasoning system.

### Provider Rate Limits

Some LLM-heavy intents, particularly mock-court analysis, can approach provider
token-per-minute limits on lower-tier accounts. This can result in rate-limit
responses and additional retry delays.

### Confidence Threshold Calibration

Past-case confidence thresholds are currently heuristic and should be
validated against a larger representative evaluation set for the
`all-MiniLM-L6-v2` embedding model before production use.
