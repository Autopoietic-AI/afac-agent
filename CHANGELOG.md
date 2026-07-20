# CHANGELOG

## 2026-07-20 - M0 + M1 Stabilization

- Added M0/M1 tests for path handling, missing inputs, idempotent history import, idempotent champion registration, validation, doctor, and Windows Chinese/space paths.
- Added project-root-relative path resolver with packaged A1 champion default.
- Added `python -m afac_agent.doctor` preflight.
- Added lightweight contract validation for ProjectState, Tool Registry, Memory records, Trajectory, Anchor Manifest, and A1 champion CSV.
- Added structured `waiting_for_input` handling for missing files and unbound registered tools.
- Ensured failed tool executions do not consume successful experiment rounds.
- Added `pyproject.toml` and `.gitignore`.
- Updated bootstrap scripts to remove hard-coded Python interpreter paths.

No champion CSV, online score, Fold definition, Gate definition, OOF anchor, closed branch, or historical record was changed.
