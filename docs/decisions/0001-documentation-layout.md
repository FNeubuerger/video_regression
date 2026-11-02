# 0001 — Documentation layout

- **Status:** accepted (2026-11-02)

## Context

Eight overlapping status/plan files had accumulated at the repository root
(`BENCHMARK_STATUS.md`, `NEXT_ACTIONS.md`, `TODO.md`, …), each partially stale
and contradicting the others (e.g. runs marked "in progress" that had finished).

## Decision

- `TODO.md` is the only task/status list.
- `docs/RESULTS.md` is the only place for headline numbers and their interpretation.
- `README.md` covers purpose, setup, layout and commands only — no results tables.
- Superseded reports go to `docs/archive/` (unmaintained); one-off launch scripts
  go to `scripts/archive/`. Both are moved with `git mv` to keep history.

## Consequences

New status updates edit `TODO.md`/`docs/RESULTS.md` instead of adding new
root-level files.
