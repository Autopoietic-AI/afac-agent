# AFAC Agent v1.2

Current A1 champion: `v53Q-1`, online `0.7800`.

This repository is the build-time codebase for a bounded AFAC2026 automated
research agent.  It is not a model-training script and M0/M2 does not change
predictions, Fold definitions, Gates, OOF anchors, or the champion CSV.

## Quick start

The package is self-contained for history import and champion registration.

From the project root:

```bash
python -m afac_agent.doctor --project_root .
python -m pytest -vv
```

Windows:

```text
bootstrap_agent.bat
```

Git Bash:

```bash
bash bootstrap_agent.sh
```

The packaged champion is resolved from:

```text
artifacts/A1_v53q1_transition_stable_edge_h2_SAFE.csv
```

External datasets, OOF files and checkpoints should be configured through:

```text
config/paths.local.yaml
```

Create it from:

```text
config/paths.local.example.yaml
```

`config/paths.local.yaml` is intentionally ignored by git.

If Python is not on `PATH`, set `AFAC_PYTHON` before using bootstrap scripts.

## Codex development

Read:

1. `AGENTS.md`
2. `AFAC_AGENT_CODEX_MASTER_EXECUTION_SPEC.md`
3. `CODEX_MASTER_PROMPT_AFAC_AGENT.txt`

Start with milestone `M0 + M1 Stabilization`.

## Current M0/M1 guardrails

- `python -m afac_agent.doctor` validates ProjectState, Tool Registry, Memory records, Trajectory status and champion CSV shape/hash.
- Missing required files return `waiting_for_input`.
- Registered but unbound tools return `waiting_for_input` with `reason=unbound_tool`.
- Failed tools do not consume successful experiment rounds.
- History import and champion registration are idempotent.
- Windows Chinese and space paths are covered by tests.

## M2 A1 Data Profiler

M2 adds a CPU-only, read-only A1 data profiler.  It loads the canonical graph
from the A1 NPZ adjacency CSR and treats `A1_edges.csv` only as an optional
cross-check.  It does not train, generate predictions, create submissions,
write Memory, mutate Project State, or consume successful experiment rounds.

Dataset-only run:

```bash
python -m afac_agent.profilers.a1_data_profiler \
  --npz_path "<path-to-A1.npz>" \
  --edges_csv "<optional-path-to-A1_edges.csv>" \
  --champion_csv artifacts/A1_v53q1_transition_stable_edge_h2_SAFE.csv \
  --out_dir artifacts/data_profile/a1_m2_v1
```

The legacy registered-tool wrapper remains available:

```bash
python tools/profile_a1_dataset.py --npz_path "<path-to-A1.npz>" --out_dir artifacts/data_profile/a1_m2_v1
```

Optional higher tiers require explicit local inputs:

- `--fold_file` for fold-aware structure;
- `--anchor_oof_npz` for full anchor OOF analysis.

If those optional inputs are absent, the profiler completes the lower available
tier and records the missing inputs in warnings.  With `--require_fold` or
`--require_oof`, missing inputs return `waiting_for_input`.

Core deterministic outputs are written under:

```text
artifacts/data_profile/a1_m2_v1/
```

The dataset-only profile records Champion Test predicted-label distribution as
prediction distribution only, never as Test truth.  Shift reporting separates
observed feature/structure shift from OOF-proba shift that is unavailable until
the exact v53Q-1 OOF input is supplied.
