## Summary

<!-- 1-3 bullets describing the change and the why -->

-

## Test plan

<!-- Tick everything that ran clean. CI runs the same gates -->

- [ ] `ruff check .` clean
- [ ] `pytest tests/ -m "not integration"` passes (unit + e2e)
- [ ] Coverage gate (76 % combined line+branch) green
- [ ] If this PR fixes a bug: a regression test exists and carries
      `# regression test for #NNN` (per `CLAUDE.md` § Testing
      Conventions). If a regression test isn't possible, explain why.
- [ ] If this PR touches a critical-path module
      (`broker/` / `journal/` / `risk/` / `audit/` / `bus/`):
      at least one happy-path test AND at least one failure-mode
      test ship together (per `CLAUDE.md` § Negative-test
      discipline).
- [ ] If this PR adds an e2e file: it uses the shared fixtures in
      `tests/conftest.py` and stays under the 3 s per-test budget
      (per `docs/ci_pipeline.md` § Adding a new e2e file).

<!-- Closes #NNN -->
