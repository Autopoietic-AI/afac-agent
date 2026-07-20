# CHANGELOG

## 2026-07-20 - M3A Tool Adapter Foundation

- Added the minimal M3A Tool Adapter protocol and AdapterRunner.
- Added standard Adapter Execution Result Schema.
- Extended Tool Registry metadata with optional Adapter binding fields.
- Added `A1_V53Q1_PATCH_AUDIT`, the first real read-only Adapter.
- Added CLI support for `python -m afac_agent.main run-adapter`.
- Integrated Adapter execution into Orchestrator without removing legacy
  `command_template` execution.
- Added Doctor checks for Adapter schema, registry bindings and output root.
- Added tests for missing inputs, failed validation, duplicate identity,
  read-only frozen-file hashes, stdout/stderr logs, CLI, Orchestrator and
  Windows Chinese/space paths.

No champion CSV, online score, Fold definition, Gate definition, OOF anchor,
Project State, closed branch, or historical record was changed.

## 2026-07-20 - M2 A1 Data Profiler

- Added a read-only, CPU-only, deterministic A1 Data Profiler.
- Added dataset-only, fold-aware-structure, and full-anchor-OOF analysis tiers.
- Added directed graph auditing with self-loop and duplicate directed-edge counts.
- Added in/out/incident/either degree fields and either-neighbor degree buckets.
- Added exact 1/2/3/4-hop visible-train counts with fold-aware visibility when a canonical fold file is provided.
- Added OOF global `train_idx` alignment and prediction Sink/Source based on the OOF confusion matrix.
- Added deterministic core artifact hashing with manifest timestamps excluded from the core hash.
- Added M2 schemas, tests, CLI entrypoint, legacy tool wrapper, doctor integration, and Tool Registry metadata.
- Added generic tool metadata for read-only behavior, experiment-round consumption, prediction mutation, Project State mutation, and GPU requirements.
- Ensured `PROFILE_A1_DATASET` does not consume successful experiment rounds and does not mutate Project State.
- Added Champion Test predicted-label distribution to deterministic M2 artifacts without treating it as Test truth.
- Replaced empty `train_test_shift` placeholders with structured observed/unavailable/not-applicable shift status.
- Explicitly marked directed exact-hop breakdown as `not_generated` for M2 v1 while keeping the primary exact-hop view as either-direction.
- Added `legacy_data_profile_flag_stale` doctor warning for stale legacy data-profile state flags.

No champion CSV, online score, Fold definition, Gate definition, OOF anchor, closed branch, or historical record was changed.

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
