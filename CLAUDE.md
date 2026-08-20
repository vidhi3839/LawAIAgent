# LawAIAgent

LangGraph + FastAPI multi-intent legal research assistant.

## Architecture
- `main.py` — LangGraph StateGraph (`AgentState`): start → detect_compound → router → api_runner → verifier → track_best → END
- `router_llm.py` — single-intent LLM router (statute/rule/constitution/definition/past_cases/summarizer/followup/mock_court)
- `tasks/multi_tool_runner.py` — SEPARATE parallel routing path (primary+secondary tool selection), gated by ENABLE_MULTI_TOOL env var in api.py. Default OFF. Do not merge its confidence logic with the main path without discussion.
- `tasks/statutory_api.py`, `tasks/past_cases.py`, `tasks/mock_court.py`, `tasks/summarize.py` — tool implementations
- `api.py` — FastAPI layer, owns thread/lawyer ownership checks

## Non-negotiable conventions
- Confidence scoring is ALWAYS fully programmatic (weighted formula), never LLM-self-reported. See `compute_strategic_strength`, `calculate_legal_confidence`, `compute_past_cases_confidence` as the pattern to follow for any new scoring.
- Every LLM call response must be passed through `_strip_think_tags` equivalent before use — Groq models sometimes wrap output in `<think>`.
- New tool intents must be added in three places: `router_llm.py` (TOOL_NAME_TO_INTENT), `main.py` (`api_execution_node` branch), `main.py` (`verification_node` branch). Missing one silently breaks the intent.
- `ConfidenceCard.jsx`'s `DEFAULT_THRESHOLDS` must be updated whenever a new intent is added on the backend — it does not auto-derive from backend defaults.
- Never fabricate case law, statute text, or citations not present in retrieved source text — this is enforced in every prompt (`LEGAL_PRESENTATION_PROMPT`, mock_court prompts) and must stay enforced in any new prompt.
- Path handling for uploaded PDFs must go through `_validate_pdf_path` (path traversal guard) — never `open()` a user-supplied path directly.

## Known architecture debt
- `multi_tool_runner.py` duplicates routing/confidence logic that exists elsewhere. Resolve before extending either path further.
- `detect_compound_question_node` is a heuristic disclosure, not real multi-lookup handling — documented limitation, not a bug to "fix" without a scoping conversation first.

## Stack
LangGraph 1.2.9, langchain-groq (llama-3.3-70b-versatile), ChromaDB, Supabase Postgres (via PostgresSaver checkpointer), sentence-transformers, FastAPI, React/Vite frontend, Docker Compose (backend + nginx-served frontend).