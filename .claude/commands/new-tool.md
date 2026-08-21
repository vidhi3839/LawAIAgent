---
description: Scaffold the wiring for a new tool intent in LawAIAgent
argument-hint: <intent_name> <short description>
---

Use the add-tool-intent skill to scaffold a new intent called "$1".

Specifically:
1. Add a placeholder entry to `TOOL_NAME_TO_INTENT` in `router_llm.py` and a stub tool description in the router's system prompt.
2. Add an `elif intent == "$1":` branch (with a `# TODO: implement` marker) in `api_execution_node` in `main.py`.
3. Add the matching branch (with a `# TODO: implement` marker) in `verification_node`  in `main.py`, including a placeholder programmatic confidence score of 0.0 so it's obviously unfinished rather than silently wrong.
4. Add an entry for "$1" to `DEFAULT_THRESHOLDS` in `ConfidenceCard.jsx` (default to 0.5 unless I specify otherwise).
5. Add an entry for "$1" to `DEFAULT_RETRY_THRESHOLDS` in `main.py`.

Do not implement the actual tool logic — just scaffold the four wiring points with clear TODO markers so nothing is silently missing. List out the exact file locations you touched when done.