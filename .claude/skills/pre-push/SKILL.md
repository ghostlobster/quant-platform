---
name: pre-push
description: Run the same gates as CI (ruff, pytest with 76% coverage, bandit HIGH, pip-audit) against the current working tree and report pass/fail per stage. Use when the user asks to validate changes before push, check CI locally, run the pre-push checks, or verify the branch is CI-clean.
---

# pre-push — mirror CI gates locally

This skill runs the same four gates that `.github/workflows/ci.yml` enforces, so a failure here means a failure on CI. Run them **in order** and stop at the first failure unless the user said "run all regardless".

## Stages

Run each via Bash, capture stdout/stderr, and report `PASS` or `FAIL` with a one-line summary.

### 1. Lint

```
ruff check .
```

Source: `.github/workflows/ci.yml:22-23`. Fails CI on any violation. On failure, suggest `ruff check . --fix` and show the first 10 lines of the diagnostic.

### 2. Unit tests + coverage (76% combined line+branch gate)

```
pytest tests/ -m "not integration" --cov=. --cov-fail-under=76 --cov-report=term-missing
```

Source: `.github/workflows/ci.yml:73-79`. On failure, report which tests failed (first 5) and, if the failure is coverage-only, print the files with the lowest coverage from the term-missing report.

**Branch coverage is on** (`.coveragerc:branch = True`, #200) so the term-missing report shows partial branches with `1->2` notation — those are conditionals where one arm is unexercised. The `Missing` column lists them; pick a few with the lowest impact before pushing.

### 3. Bandit (HIGH severity only)

```
bandit -r . -ll --exclude ./.git,./tests -f json -o /tmp/bandit-report.json
python -c "import json,sys; r=json.load(open('/tmp/bandit-report.json')); highs=[i for i in r['results'] if i['issue_severity']=='HIGH']; print(f'{len(highs)} HIGH severity issues'); [print(f\"  {i['filename']}:{i['line_number']} {i['test_id']} {i['issue_text']}\") for i in highs]; sys.exit(1 if highs else 0)"
```

Source: `.github/workflows/ci.yml:33-36`. Only HIGH severity fails CI — MEDIUM/LOW are reported but tolerated. Print any HIGH findings with file:line.

### 4. Dependency audit

```
pip-audit -r requirements.txt --ignore-vuln PYSEC-2022-42969
```

Source: `.github/workflows/ci.yml:37-38`. `PYSEC-2022-42969` is intentionally ignored. If a new vulnerability appears, suggest pinning the affected package.

## Reporting

At the end, print a summary table:

```
Stage                    | Result
-------------------------+--------
1. ruff                  | PASS
2. pytest (cov >= 76%)   | PASS
3. bandit HIGH           | PASS
4. pip-audit             | PASS
```

If everything passes, say explicitly: "Ready to push — CI should be green." If anything failed, stop at the first failure, summarise the failure, and propose the specific fix. Do not push, commit, or modify files as part of this skill.

## Notes

- All four commands are in `.claude/settings.json`'s allowlist — they run without a permission prompt.
- The skill is read-only on the codebase. It never runs `ruff check . --fix` on its own — it only suggests it.
- If the user has uncommitted changes, proceed anyway; CI would see them after commit+push.
