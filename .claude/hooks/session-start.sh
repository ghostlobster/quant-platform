#!/usr/bin/env bash
# SessionStart hook — emits a one-shot context block for the session.
# Output goes into Claude's context, so keep it compact and never echo secrets.
set -u

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT" || exit 0

emit() { printf '%s\n' "$*"; }

emit "## quant-platform session context"
emit ""

# --- Toolchain ---------------------------------------------------------------
PY_VER="$(python3 --version 2>/dev/null || echo 'python3 missing')"
emit "- python: ${PY_VER}"

VENV_STATE="inactive"
[ -n "${VIRTUAL_ENV:-}" ] && VENV_STATE="active (${VIRTUAL_ENV##*/})"
[ -d ".venv" ] && [ -z "${VIRTUAL_ENV:-}" ] && VENV_STATE="present (.venv, not activated)"
emit "- venv: ${VENV_STATE}"

TOOLS=()
for t in ruff pytest bandit pip-audit; do
  if command -v "$t" >/dev/null 2>&1; then
    TOOLS+=("${t}=ok")
  else
    TOOLS+=("${t}=MISSING")
  fi
done
emit "- tooling: ${TOOLS[*]}"

# --- Env / secrets (presence only, never values) -----------------------------
ENV_STATE="absent"
[ -f ".env" ] && ENV_STATE="present"
emit "- .env: ${ENV_STATE}"

# --- Databases ---------------------------------------------------------------
DBS=()
for db in quant.db journal_trades.db data/wf_history.db; do
  if [ -f "$db" ]; then
    DBS+=("${db}=ok")
  else
    DBS+=("${db}=missing")
  fi
done
emit "- databases: ${DBS[*]}"

# --- Git state ---------------------------------------------------------------
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null)"
  HEAD_SHORT="$(git rev-parse --short HEAD 2>/dev/null)"
  HEAD_SUBJ="$(git log -1 --pretty=%s 2>/dev/null)"
  emit "- branch: ${BRANCH}"
  emit "- head: ${HEAD_SHORT} ${HEAD_SUBJ}"

  AHEAD_BEHIND="$(git rev-list --left-right --count "origin/main...HEAD" 2>/dev/null || echo '? ?')"
  BEHIND="${AHEAD_BEHIND%%	*}"
  AHEAD="${AHEAD_BEHIND##*	}"
  emit "- vs origin/main: ${AHEAD} ahead, ${BEHIND} behind"

  DIRTY="$(git status --porcelain 2>/dev/null | wc -l | tr -d ' ')"
  emit "- uncommitted changes: ${DIRTY} file(s)"
fi

emit ""
emit "Reminder: CI gate is 76% coverage + ruff + bandit HIGH + pip-audit. Run the \`pre-push\` skill before pushing."
