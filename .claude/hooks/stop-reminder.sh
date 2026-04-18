#!/usr/bin/env bash
# Stop hook — advisory nudges printed when Claude finishes a turn.
#   1. If the branch has uncommitted .py changes → remind to run the pre-push skill.
#   2. If new implementation work is present (code files added/modified) → remind
#      to associate the change with a ticket before committing.
# Exit 0 always; never block.
set -u

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT" || exit 0

# Quiet when not in a git repo.
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || exit 0

PY_CHANGED="$(git status --porcelain -- '*.py' 2>/dev/null | wc -l | tr -d ' ')"
CODE_CHANGED="$(git status --porcelain -- '*.py' '*.md' '*.yml' '*.yaml' '*.sh' 2>/dev/null | wc -l | tr -d ' ')"

[ "$CODE_CHANGED" = "0" ] && exit 0

printf '\n[stop-reminder]\n'

if [ "$PY_CHANGED" != "0" ]; then
  printf -- '- %s uncommitted .py file(s). Run the `pre-push` skill to mirror CI gates before pushing.\n' "$PY_CHANGED"
fi

# Ticket-association nudge — fires when any tracked-code file was changed.
# Rule: every new implementation must reference a ticket/issue (e.g. #123, QP-45)
# in the branch name, commit message, or PR description.
BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null)"
HAS_TICKET_IN_BRANCH=0
if printf '%s' "$BRANCH" | grep -Eq '(#[0-9]+|[A-Z]{2,}-[0-9]+|issue[-_/][0-9]+)'; then
  HAS_TICKET_IN_BRANCH=1
fi

if [ "$HAS_TICKET_IN_BRANCH" = "0" ]; then
  printf -- '- Rule: associate new implementations with a ticket. Branch `%s` has no ticket ref — include one (e.g. `#123` or `QP-45`) in the commit message or PR body before pushing.\n' "$BRANCH"
fi

exit 0
