#!/usr/bin/env bash
# Runs after Claude edits main.py or router_llm.py.
# Cross-checks that every intent branch in main.py's api_execution_node /verification_node has a matching entry in router_llm.py's
# TOOL_NAME_TO_INTENT, and warns (does not block) on mismatch.
#
# This is a warning check, not a hard gate — intentionally, since renames mid-edit are normal.

set -uo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
MAIN_PY="$REPO_ROOT/main.py"
ROUTER_PY="$REPO_ROOT/router_llm.py"

if [[ ! -f "$MAIN_PY" || ! -f "$ROUTER_PY" ]]; then
  exit 0
fi

# Intents referenced in main.py's api_execution_node / verification_node
main_intents=$(grep -oE 'intent == "[a-z_]+"' "$MAIN_PY" | sed -E 's/intent == "([a-z_]+)"/\1/' | sort -u)

# Intents registered in router_llm.py's TOOL_NAME_TO_INTENT dict
router_intents=$(grep -oE '"[a-z_]+"' "$ROUTER_PY" | tr -d '"' | sort -u)

missing=""
for intent in $main_intents; do
  if ! grep -q -- "$intent" <<< "$router_intents"; then
    missing="$missing $intent"
  fi
done

if [[ -n "$missing" ]]; then
  echo " check_intent_wiring: main.py references intent(s) [$missing] that don't appear anywhere in router_llm.py." >&2
  echo "    If this is a real new intent, confirm TOOL_NAME_TO_INTENT and the system prompt were both updated (see add-tool-intent skill)." >&2
  echo "    If this is a rename in progress, ignore." >&2
fi

exit 0