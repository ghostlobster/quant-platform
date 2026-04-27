# CI Pipeline Layout

The platform's GitHub Actions pipeline runs **five** jobs on every push
to `main` and every PR targeting `main`. The first four run in parallel;
the fifth is the umbrella merge gate that depends on all of them and is
the **only** required check on `main`.

| Job | Owns | Coverage gate |
|---|---|---|
| `lint` | `ruff check .` | n/a |
| `security` | `bandit -ll` (HIGH-only) + `pip-audit` | n/a |
| `Test (Python 3.11)` | unit tests (`-m "not integration and not e2e"`) + integration scaffold | 76% combined line+branch floor (#200) — **gates merges** |
| `E2E (Python 3.11)` | end-to-end regression suite (`-m e2e`) | per-module floor via `scripts/check_e2e_coverage.py` (each cross-module file ≥ 40 %, with hand-tightened bumps) — **gates merges** |
| `Merge gate` | depends on all four above; verifies `needs.*.result` for every parent | n/a — pure dependency aggregator |

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

## Required checks (single umbrella job)

Branch protection on `main` requires exactly **one** check:

```
Merge gate
```

The `merge-gate` job in `.github/workflows/ci.yml` declares
`needs: [lint, security, test, e2e]` with `if: always()` and explicitly
fails when any upstream job's `result` is not `success` or `skipped`.
Adding a future job (e.g. `integration`) to the pipeline only requires
extending the `needs:` array — branch protection never has to change.

GitHub UI path on first setup: **Settings → Branches → main → Edit →
Require status checks to pass before merging → add `Merge gate`.**

The four upstream jobs still publish independent status checks for
visibility (you can see which one failed at a glance on the PR), but
the merge button only listens to `Merge gate`.

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
3. **Reuse the shared fixtures** in `tests/conftest.py`:
   - `e2e_paper_env` — paper-trader + journal isolated to per-test SQLite.
   - `e2e_journal_db` — journal-only isolation when paper_trader is not in scope.
   - `e2e_isolated_caches` — per-test SQLite + DuckDB cache files; resets the DuckDB connection singleton.
   - `E2EFakeBroker` — minimal `BrokerProvider` stand-in returning both `equity` and `total_value` so it works against the live and the paper paths.
4. Mock at the network boundary only (HTTP adapter, MLflow registry,
   alert channels). Real SQLite, real journal, real paper trader.
5. Update `scripts/check_e2e_coverage.py` with any new module the chain
   exercises so the per-module coverage floor catches future drift.

## Best-practice config

| Setting | Value | Why |
|---|---|---|
| `pytest.ini` `addopts` | `--strict-markers --strict-config --tb=short` | Typos in marker names fail loud; short tracebacks keep PR comments readable. |
| `pytest.ini` `timeout` | `60 s` per test (`thread` method) | Catches a hung test before it eats the 15-min job budget. Provided by `pytest-timeout`. |
| `requirements.txt` | `pytest-timeout`, `pytest-xdist` | Available for parallel runs once the suite outgrows a single worker; not enabled by default because xdist startup overhead dominates at the current 16-test size. |
| `tests/conftest.py` env | `OMP_NUM_THREADS=1` etc. | Prevents OpenBLAS background threads from triggering `std::terminate()` during interpreter shutdown on CI. |
