---
name: new-branch
description: Enforce the "new implementation = new branch" rule. Use whenever the user asks to start a new implementation, pick up a roadmap ticket, begin work on an issue, or before making the first code edit for a new task. Checks the current branch, proposes a conventional branch name, creates and switches to it, and confirms the working tree is clean before any edits.
---

# new-branch — start every new implementation on its own branch

Every new implementation starts on its own feature branch off `main`. This
skill is the gatekeeper: it refuses to let work begin on `main`, on a stale
feature branch, or with a dirty working tree. It is **read/write git only** —
it never edits code.

## When to run

Invoke at the **start** of any new implementation:
- The user asks to pick up a roadmap ticket (e.g. "let's do P1.1").
- The user references a GitHub issue by number (e.g. "start #139").
- The user describes a new feature / fix / chore that would touch code.
- Before the first `Edit` / `Write` / `NotebookEdit` on a fresh task.

Skip it if the user explicitly says "stay on this branch", "keep working on
the current branch", or is asking for read-only analysis.

## Inputs

One of:
- A GitHub issue number (`#139`, `139`, or URL)
- A short slug describing the work (e.g. `pretrade-guard`)
- A roadmap ticket id (`P1.1`)

## Procedure

Run each Bash step and stop at the first failure.

### 1. Confirm clean working tree

```
git status --porcelain
```

If the output is non-empty: **stop**. Report the uncommitted changes and ask
the user whether to commit, stash, or discard them before proceeding. Never
discard or stash on the user's behalf.

### 2. Fetch and sync with origin

```
git fetch origin main
```

Retry up to 4 times with exponential backoff (2s, 4s, 8s, 16s) on network
failure. If the fetch keeps failing, stop and report.

### 3. Resolve the branch name

Naming rule — one of these, in priority order:

| Input kind | Branch name |
|---|---|
| Issue number `#N` (title known) | `claude/issue-<N>-<kebab-slug-from-title>` |
| Roadmap ticket `PX.Y` | `claude/<pxy>-<kebab-slug-from-plan-heading>` |
| Free-form slug | `claude/<kebab-slug>-<rand4>` |

Where:
- `<kebab-slug>` is lower-case, hyphen-separated, alphanumerics only, max
  40 chars.
- `<rand4>` is 4 random alphanumerics (e.g. `a3kf`) to avoid collisions.
- Always prefix with `claude/` to match the repository convention (see
  recent branches like `claude/quant-platform-comparison-Rta76`).

If given an issue number, resolve the title via the GitHub MCP tool
`mcp__github__issue_read` (owner `ghostlobster`, repo `quant-platform`)
to build the slug. Do not invent a title.

If given a roadmap id, read the matching section header from
`docs/plans/2026-04-19-indie-quant-roadmap.md` to build the slug.

### 4. Check the branch does not already exist

```
git show-ref --verify --quiet refs/heads/<branch-name> && echo EXISTS || echo NEW
git ls-remote --exit-code --heads origin <branch-name>
```

If either exists, regenerate the name with a fresh `<rand4>` suffix (or ask
the user whether to resume the existing branch).

### 5. Create and switch

```
git checkout -b <branch-name> origin/main
```

Always branch off `origin/main`, never off the current branch.

### 6. Confirm

Print:

```
Branch         | <branch-name>
Based on       | origin/main @ <short-sha>
Ticket / issue | <identifier-or-"n/a">
Working tree   | clean
```

State explicitly: "Ready to begin implementation. Run `/pre-push` before
pushing."

## Guardrails

- **Never** run destructive git operations (`reset --hard`, `checkout .`,
  `branch -D`, `clean -f`) as part of this skill.
- **Never** push as part of this skill — pushing is a separate step after
  `/pre-push` gates are green.
- **Never** commit as part of this skill — the skill only creates the
  branch.
- If the user is already on a non-`main` branch with uncommitted in-progress
  work relevant to a *different* ticket, surface that clearly and ask before
  switching.
- If CI gates on the current branch are not green, mention it but do not
  block — that is `/pre-push`'s job.

## Notes

- This skill is the counterpart to `/pre-push` (end-of-cycle) — together
  they bracket every implementation.
- The `claude/` prefix keeps the branch namespace tidy and matches the
  automation convention already present in the repo.
- Branch hygiene is advisory to humans but enforced for Claude runs: if
  Claude is asked to start new implementation work, it should invoke this
  skill as its first action.
