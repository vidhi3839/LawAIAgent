---
name: confidence-formula-review
description: Use before merging any new or modified confidence-scoring function in LawAIAgent (anything that returns a final_score/confidence for a task). Checks the function follows the existing programmatic-scoring pattern instead of drifting toward LLM self-rating, and that it's actually wired to both the backend response and the frontend threshold map.
---

# Confidence Formula Review

LawAIAgent's core trust guarantee is that confidence scores are computed by plain Python logic against retrieved/verifiable signals — never by asking the LLM to rate its own answer. This skill is a checklist to run against any new
or edited scoring function before it ships.

## Reference implementations (the pattern to match)
- `tasks/mock_court.py` → `compute_strategic_strength` (statutory grounding × 0.4 + precedent support × 0.4 + vulnerability exposure × 0.2)
- `tasks/statutory_api.py` → `calculate_legal_confidence` (authority × integrity × link-reachability, weighted)
- `tasks/past_cases.py` → `compute_past_cases_confidence` - `tasks/multi_tool_runner.py` → `_compute_confidence` (authority / grounding / coverage — note this is a THIRD, separate formula; don't add a fourth
  without first asking whether it should just extend one of the above)

## Checklist
- [ ] Every sub-score is computed from something checkable: regex/keyword match against source text, similarity score from retrieval, link reachability, word-overlap grounding — NOT an LLM asked to output a number.
- [ ] The function returns a dict with at minimum: `final_score`, a human-readable `rating`/`flag`, and the individual sub-scores used to compute it (so the breakdown can be shown to the lawyer, not just the final number).
- [ ] `final_score` is deterministic given the same inputs — no LLM call in the scoring path itself, even a small one.
- [ ] If this touches `retrieved_cases`, confirm it's deduplicated first (see `_deduplicate_cases` in mock_court.py) — undeduplicated chunks from the same case inflate similarity averages.
