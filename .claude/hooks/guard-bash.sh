#!/usr/bin/env python3
"""
PreToolUse(Bash) guard — blocks destructive shell patterns.

Reads the tool input as JSON on stdin; exits 2 to deny and stderr is fed
back to Claude.

Heredoc bodies are stripped before pattern-matching so that a commit message
(or PR body) that *mentions* a forbidden flag does not trigger the guard.
"""
from __future__ import annotations

import json
import re
import sys


def strip_heredocs(cmd: str) -> str:
    """Remove heredoc bodies: `<<EOF ... EOF`, `<<'EOF' ... EOF`, etc."""
    pattern = re.compile(
        r"(?P<prefix><<-?\s*['\"]?(?P<tag>[A-Za-z_][A-Za-z0-9_]*)['\"]?)"
        r".*?^\s*(?P=tag)\s*$",
        re.DOTALL | re.MULTILINE,
    )
    return pattern.sub(lambda m: m.group("prefix"), cmd)


RULES: list[tuple[str, str]] = [
    (r"(?:^|\s)--no-verify(?:\s|$)",
     "skipping commit hooks (--no-verify) is not allowed (CLAUDE.md: 'Never skip CI')"),
    (r"(?:^|\s)--no-gpg-sign(?:\s|$)",
     "skipping commit signing is not allowed"),
    (r"-c\s+commit\.gpgsign=false",
     "disabling commit signing via -c is not allowed"),
    # Plain --force: destructive (can overwrite a remote ref that moved).
    # --force-with-lease: allowed — refuses if the remote advanced, which is
    # the standard safe way to update a feature branch after a rebase.
    (r"\bgit\s+push\b[^|;&\n]*--force(?!-with-lease)\b",
     "force-push (--force) is destructive; use --force-with-lease instead"),
    (r"\bgit\s+push\b[^|;&\n]*\s-f(?:\s|$)",
     "force-push (-f) is destructive; use --force-with-lease instead"),
    (r"\bgit\s+push\s+(?:origin\s+)?(?:main|master)(?:\s|$)",
     "direct push to main/master is not allowed; push to a feature branch and open a PR"),
    (r"\bgit\s+reset\s+--hard\b",
     "git reset --hard discards uncommitted work; investigate first"),
    (r"\bgit\s+branch\s+-D\b",
     "force-deleting a branch can lose work; ask the user to confirm"),
    (r"\bgit\s+branch\s+--delete\s+--force\b",
     "force-deleting a branch can lose work; ask the user to confirm"),
    (r"\bgit\s+(?:checkout|switch)\s+(?:main|master)(?:\s|$)",
     "do not switch to main/master mid-task; stay on the feature branch"),
    (r"\bgit\s+clean\s+-[fFdDxX]{1,3}\b",
     "git clean removes untracked files irreversibly"),
    (r"\brm\s+-rf?\s+/(?:\s|$|\*)",
     "rm -rf on the root filesystem is forbidden"),
    (r"\brm\s+-rf?\s+~(?:\s|$|/)",
     "rm -rf on the home directory is forbidden"),
    (r"\brm\s+-rf?\s+\$HOME\b",
     "rm -rf on $HOME is forbidden"),
    (r"\bgit\s+config\s+--global\b",
     "global git config edits are not allowed (CLAUDE.md: 'NEVER update the git config')"),
]


def main() -> int:
    try:
        data = json.load(sys.stdin)
    except Exception:
        return 0  # malformed input — defer to Claude's own handling
    cmd = (data.get("tool_input") or {}).get("command") or ""
    if not cmd:
        return 0
    scrubbed = strip_heredocs(cmd)
    for pat, reason in RULES:
        if re.search(pat, scrubbed):
            first = cmd.splitlines()[0][:200] if cmd else ""
            print(f"BLOCKED by guard-bash: {reason}", file=sys.stderr)
            print(f"Command: {first}", file=sys.stderr)
            print("If this is intentional, ask the user to run it manually.", file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
