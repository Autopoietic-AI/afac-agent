# PROJECT_STATE.md

## Current Champion

- A1 online: `0.7800`
- Version: `v53Q-1`
- Stack:
  - Graph-visible: `v43C`
  - Isolated: `v46A-1 Balanced Seed Consensus`
  - Micro patch: transition-stable Edge-H2
- A1 file:
  - `A1_v53q1_transition_stable_edge_h2_SAFE.csv`

## Current OOF Anchor

- Full OOF: `0.77538405599`
- Correct: `8530 / 11001`
- v43C Graph-visible: `0.8645881`
- v46A-1 Isolated: `0.4211477632`

## Current Main Contradiction

- Graph-visible has weak-expert complementarity but lacks reliable per-node choice.
- Isolated current raw attributes and current expert probabilities have no safe repair head.
- Existing-signal Isolated route is closed.

## Agent Mode

- Human manual experiment selection: stopped.
- Agent v1:
  - confirmed-history importer
  - A1 dataset profiler
  - online anchor registry
  - standard OOF analyzer
  - hierarchical planner
  - safety gate
  - serial orchestrator
  - append-only memory
  - real-time trajectory

## M0 + M1 Stabilization Status

- Project-root-relative packaged champion default:
  - `artifacts/A1_v53q1_transition_stable_edge_h2_SAFE.csv`
- External data, OOF files and checkpoints:
  - configured by local `config/paths.local.yaml`
  - example: `config/paths.local.example.yaml`
- Doctor entrypoint:
  - `python -m afac_agent.doctor --project_root .`
- Contract validation now covers:
  - ProjectState
  - Tool Registry
  - confirmed Memory records / JSONL Memory
  - Trajectory
  - Online Anchor Manifest
  - A1 champion CSV format
- Missing files and unbound tools return:
  - `waiting_for_input`
- Failed tool executions do not consume successful experiment rounds.
- History import is idempotent by `version`.
- Champion registration is hash-aware and idempotent.
- Tests include Windows Chinese and space paths.

No champion CSV, online score, OOF anchor, Fold definition, Gate definition, closed branch, or confirmed historical record was changed.

## Unique Immediate Action

1. Run `python -m afac_agent.doctor --project_root .`.
2. Run Agent v1 Dry Run.
3. Execute history import.
4. Execute online champion registration.
5. Stop at the next missing input or unbound tool until M2 is explicitly approved.

## Do Not Repeat

- ScaleNet
- MagNet
- v47 unified residual
- v47 pair rules
- v48 reliability atlas
- v48 confidence router
- hard/soft Edge model route
- ModernNCA current route
- semantic KNN/topology
- Isolated OVR
- Isolated pairwise with current signals
- leaderboard-guided node subsets

## Next Research Target

A genuinely new Isolated signal:

- homologous public data/provenance;
- feature-source recovery;
- graph-aligned raw representation.

Agent must audit information gain and compliance before model training.


## Standardized OOF Artifact Status

- Historical v43C/v46A-1 OOF numbers are confirmed.
- Agent v1 standardized OOF parser has not yet generated its own artifact.
- Therefore `anchor_oof_analyzed=false` in Agent state until the NPZ and OOF files are supplied.

## M2 A1 Data Profiler Status

- M2 implements only the A1 Data Profiler framework.
- The profiler is read-only, CPU-only, deterministic and idempotent.
- Canonical graph source is the A1 NPZ adjacency CSR.
- `A1_edges.csv` is optional cross-validation only and is not merged into the graph.
- Dataset-only profile can run with `A1.npz` alone.
- Fold-aware structure requires an explicit canonical fold assignment file.
- Full anchor OOF analysis requires the exact v53Q-1 OOF proba plus global `train_idx` alignment.
- Missing optional Fold/OOF inputs degrade to a lower `analysis_tier` unless `--require_fold` or `--require_oof` is used.
- `PROFILE_A1_DATASET` declares `counts_as_experiment_round=false` and `mutates_project_state=false`.
- Champion Test predicted-label distribution is recorded as prediction distribution only, not Test truth.
- `train_test_shift` now separates observed feature/structure shift from unavailable OOF-proba shift.
- Directed exact-hop breakdown is explicitly marked `not_generated` in M2 v1; the primary exact-hop view remains `either_direction`.
- Doctor reports `legacy_data_profile_flag_stale` when legacy `data_profile_ready=true` is not backed by a valid M2 artifact.
- No champion CSV, online score, OOF anchor, Fold definition, Gate definition, closed branch, or confirmed historical record was changed.
