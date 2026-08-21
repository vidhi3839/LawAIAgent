---
description: Repo hygiene check — finds leftover junk and risky leftovers before submission
---

Run a cleanup pass over this repo and report findings clearly. Do NOT delete or modify anything automatically — list what you find, and wait for me to confirm before removing anything.

Check for:

1. **Python cache/build junk**: `__pycache__/` folders, `*.pyc` files anywhere in the repo.
2. **Stale model references**: search for any hardcoded `llama-3.3-70b-versatile` (the deprecated Groq model) still left in any file — it should be `openai/gpt-oss-120b` everywhere now.
3. **Secrets that shouldn't be committed**: confirm `.env` is NOT tracked by git (`git status` / `git ls-files | grep .env`), and confirm `.env` appears in `.gitignore` and `_dockerignore`.
4. **Debug leftovers**: any stray `print()` statements outside of intentional CLI scripts (`tests/eval_answers.py`, `tests/capture_fixtures.py` are fine — flag anything in `main.py`, `api.py`, `router_llm.py`, or `tasks/*.py`).
5. **TODO/FIXME markers**: list every `TODO`, `FIXME`, or `XXX` comment left in the codebase, with file and line number, so nothing is silently forgotten.
6. **Orphaned test data**: anything under `data/uploads/`, `data/case_pdfs_uploaded/`, or similar that looks like leftover test files rather than real seeded content.
7. **Unused imports**: flag any obviously unused imports in `main.py`, `api.py`, `router_llm.py`, and `tasks/*.py` (don't remove them, just list them).
8. **Large or accidentally-committed files**: anything over 5MB tracked by git that looks unintentional (model weights, database dumps, etc.).

Summarize as a short checklist: what's clean, what needs a decision from me, and what's safe to delete outright (only truly disposable things like `__pycache__/`). Do not touch `.env`, real case PDF uploads, or anything in `tests/fixtures/` — those are intentional.