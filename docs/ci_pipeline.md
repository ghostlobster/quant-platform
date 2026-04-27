# CI Pipeline Layout

The platform's GitHub Actions pipeline is split into four jobs that run
in parallel on every push to `main` and every PR targeting `main`:

| Job | Owns | Coverage gate |
|---|---|---|
| `lint` | `ruff check .` | n/a |
| `security` | `bandit -ll` (HIGH-only) + `pip-audit` | n/a |
| `Test (Python 3.11)` | unit tests (`-m "not integration and not e2e"`) + integration scaffold | 76% — **gates merges** |
| `E2E (Python 3.11)` | end-to-end regression suite (`-m e2e`) | reported, **not gated** |

Splitting unit and e2e into separate jobs follows three best practices:

1. **Tight feedback loop** — unit tests stay fast (≤1 min) by excluding
   the slower e2e cases. The e2e job runs in parallel, so total wall
   clock doesn't grow.
2. **Distinct failure signal** — when CI goes red, the failing job name
   tells you whether a unit invariant or a cross-module integration
   broke. No more digging through 1500-test logs to find which 16 cases
   matter.
3. **Independent retry** — flaky e2e cases (network mocks, real SQLite,
   real journals) can be re-run without re-running 1500 unit tests.

## Markers

Tests opt into the right pipeline path via pytest markers declared in
`pytest.ini`:

| Marker | Selector | Where it runs |
|---|---|---|
| `@pytest.mark.integration` | `-m integration` | `Test` job, separate step (currently empty scaffold for future live-broker tests) |
| `@pytest.mark.e2e` | `-m e2e` | `E2E` job |
| _no marker_ | `-m "not integration and not e2e"` | `Test` job, unit step |

E2E test files apply the marker module-wide via:

```python
pytestmark = pytest.mark.e2e
```

so every test in the file inherits it without per-function decoration.

## Required checks (branch-protection update needed)

After this PR merges, the branch-protection rule on `main` should be
updated to require **both** of the following checks before a PR can
merge:

- `Test (Python 3.11)`
- `E2E (Python 3.11)`

Without that update, a PR that breaks the e2e chain could still merge
on a green unit job. The CI itself emits both — branch protection just
has to enforce both.

GitHub UI path: **Settings → Branches → main → Edit → Require status
checks to pass before merging → add `E2E (Python 3.11)`.**

## Local mirror

The `/pre-push` skill runs the same gates locally:

- `ruff check .`
- `pytest tests/ -m "not integration" --cov=. --cov-fail-under=76` —
  this still includes e2e tests; the local pass is intentionally
  broader than the CI unit job to catch e2e breakage before push.
- `bandit -r . -ll`
- `pip-audit -r requirements.txt`

Running just the e2e suite locally:

```bash
pytest tests/ -m e2e -v
```

Running just the unit suite (matches CI's unit step):

```bash
pytest tests/ -m "not integration and not e2e"
```

## Adding a new e2e file

1. Create `tests/test_e2e_<chain>.py`.
2. Add `pytestmark = pytest.mark.e2e` immediately after the imports.
3. Reuse fixtures that point `data.db._DB_PATH` / `JOURNAL_DB_PATH` /
   `DUCKDB_PATH` at `tmp_path` so the test never touches operator state.
4. Mock at the network boundary only (HTTP adapter, MLflow registry,
   alert channels). Real SQLite, real journal, real paper trader.
