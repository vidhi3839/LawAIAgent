---
name: add-tool-intent
description: Use when adding a new intent/tool to LawAIAgent's single-tool routing path (e.g. a new lookup type alongside statute/rule/constitution/definition/past_cases/summarizer/mock_court/followup). Ensures all three required wiring points are touched and the frontend threshold map is updated.
---

# Add Tool Intent

LawAIAgent's single-tool path routes every query to exactly one intent. A new intent requires edits in THREE places in the backend plus ONE in the frontend. Missing any of these does not raise an error — it silently misroutes or
mis-scores the response. Confirm all four before considering the task done.

## The four required edits

### 1. `router_llm.py` — teach the router the new tool exists
- Add the new tool to `TOOL_NAME_TO_INTENT` mapping the LLM's tool-call name to your new intent string.
- Add the tool's signature to the system prompt built in `_build_system_prompt` so the LLM knows when to
  pick it and what arguments to extract.
- If the intent needs special param handling (like `statute`/`rule` do), add the matching branch in the parameter-extraction logic.

### 2. `main.py` — `api_execution_node`
- Add an `elif intent == "<new_intent>":` branch that calls the actual tool function and returns the raw result in `AgentState` shape (`raw_api_text`/`source_url`/`source_domain`/`error`, OR a dedicated result
  key if the intent needs a richer structure — follow the pattern used by `mock_result` or `document_result` for complex tasks, not the plain raw-text pattern used by statute/rule/definition).

### 3. `main.py` — `verification_node`
- Add the matching branch that builds `final_response`, computes `confidence_score` PROGRAMMATICALLY (see confidence-formula-review skill — never let the LLM self-report a score), and updates
  `last_raw_text`/`last_source_url`/`last_intent`/`last_params` so `followup` intent can reference this turn later.

### 4. `ConfidenceCard.jsx` — `DEFAULT_THRESHOLDS`
- Add an entry for the new intent string. If skipped, the frontend silently falls back to the generic 0.75 default instead of a task-appropriate threshold.

## Verification checklist before calling it done
- [ ] `router_llm.py`: new entry in `TOOL_NAME_TO_INTENT`
- [ ] `router_llm.py`: tool described in system prompt
- [ ] `main.py`: `api_execution_node` branch added
- [ ] `main.py`: `verification_node` branch added, confidence is programmatic
- [ ] `ConfidenceCard.jsx`: `DEFAULT_THRESHOLDS` entry added
- [ ] Ran a real query through `/query` locally and confirmed `intent` in the response matches the new intent string, not falling through to another branch