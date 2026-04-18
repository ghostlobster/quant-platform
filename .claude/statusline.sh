#!/usr/bin/env bash
# Claude Code status line for quant-platform.
# Output on stdout is rendered as the status line. Keep it one line, ASCII-safe.
set -u

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT" || exit 0

# --- Branch + ahead/behind ---------------------------------------------------
BRANCH="?"
AHEAD="?"
BEHIND="?"
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null)"
  if COUNTS="$(git rev-list --left-right --count origin/main...HEAD 2>/dev/null)"; then
    BEHIND="${COUNTS%%	*}"
    AHEAD="${COUNTS##*	}"
  fi
fi

# --- Python version (stripped) ----------------------------------------------
PY="$(python3 --version 2>/dev/null | awk '{print $2}')"
[ -z "$PY" ] && PY="?"

# --- Coverage from latest coverage.xml (cached read) -------------------------
COV="-"
if [ -f "coverage.xml" ]; then
  COV="$(python3 - <<'PY' 2>/dev/null
import re, sys
try:
    with open("coverage.xml", "r", encoding="utf-8") as f:
        head = f.read(4096)
    m = re.search(r'line-rate="([0-9.]+)"', head)
    if m:
        pct = round(float(m.group(1)) * 100)
        print(f"{pct}%")
    else:
        print("-")
except Exception:
    print("-")
PY
)"
fi

# --- Uncommitted file count --------------------------------------------------
DIRTY=""
if [ "$BRANCH" != "?" ]; then
  N="$(git status --porcelain 2>/dev/null | wc -l | tr -d ' ')"
  [ "$N" != "0" ] && DIRTY=" *${N}"
fi

printf '%s | py%s | cov:%s | %sup/%sdown%s' \
  "$BRANCH" "$PY" "$COV" "$AHEAD" "$BEHIND" "$DIRTY"
