#!/usr/bin/env python3
"""
Post-edit smoke check for Python files.

Claude Code provides the hook event as JSON on stdin. For edited Python files, this hook:

1. Performs a fast syntax check with `ast.parse()`.
2. If the repository contains a pytest suite, runs the test suite.

The syntax check runs first because it is cheap and catches syntax/import-stage mistakes before spending time running tests.

A non-zero exit reports a failure to Claude Code. This script does not modify, revert, or delete the edited file.
"""
import ast
import json
import subprocess
import sys
from pathlib import Path

def main():
    try:
        event = json.load(sys.stdin)
    except Exception:
        return 0

    file_path = (
        event.get("tool_input", {}).get("file_path")
        or event.get("tool_input", {}).get("path")
    )
    if not file_path or not file_path.endswith(".py"):
        return 0

    p = Path(file_path)
    if not p.exists():
        return 0

    try:
        ast.parse(p.read_text())
    except SyntaxError as e:
        print(f"  post_edit_smoke_check: {file_path} has a syntax error: {e}", file=sys.stderr)
        return 1

    repo_root = Path(__file__).resolve().parents[2]
    tests_dir = repo_root / "tests"
    if tests_dir.exists() and any(tests_dir.glob("test_*.py")):
        result = subprocess.run(
            ["python", "-m", "pytest", str(tests_dir), "-q"],
            cwd=repo_root, capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            print("  post_edit_smoke_check: pytest failed after this edit:", file=sys.stderr)
            print(result.stdout[-2000:], file=sys.stderr)
            return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())