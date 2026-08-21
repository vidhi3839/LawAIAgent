---
description: Pre-flight check
---

Run these checks against the running backend and report a clear PASS/FAIL for each. Do not fix anything automatically — just report status.

1. Hit `GET /health` on the backend (default `http://localhost:8000/health`) and confirm it returns `{"status": "running"}`.
2. Check the current value of `ENABLE_MULTI_TOOL` in the running environment (read `.env` or the container's env) and report it explicitly.
3. Confirm the Postgres connection is live by checking backend startup logs (or docker logs for the backend container) for "Connected to Supabase PostgreSQL" and no subsequent connection errors.
4. Confirm ChromaDB has cases in it by hitting `GET /past-cases/list` and reporting the count.
5. Run one real query through `POST /query` with a simple statute lookup (e.g. "What is 18 U.S.C. 1030") and confirm the response has `confidence > 0` and `intent == "statute"`.

Summarize as a short checklist at the end: which items passed, which failed, and what I need to fix before demoing.