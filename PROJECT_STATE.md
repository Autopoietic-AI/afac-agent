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

## M3A Tool Adapter Foundation Status

- M3A adds a minimal Adapter protocol, AdapterRunner, Execution Result Schema,
  Registry binding fields and CLI/Orchestrator integration.
- First real Adapter:
  - `A1_V53Q1_PATCH_AUDIT`
  - read-only
  - `counts_as_experiment_round=false`
  - `mutates_predictions=false`
  - `mutates_project_state=false`
  - `requires_gpu=false`
- The Adapter audits v53Q-1 patch evidence only. It does not execute patch
  replay, does not train, does not generate prediction CSVs, and does not
  register a new Champion.
- Adapter run outputs are local artifacts under `artifacts/adapter_runs/` and
  are ignored by git.
- External historical v46A1/v49A paths must come from CLI or
  `config/paths.local.yaml`; no machine-specific absolute path is hard-coded.
- No champion CSV, online score, OOF anchor, Fold definition, Gate definition,
  closed branch, Project State, or confirmed historical record was changed.

## M6R-A v2 status

M6R-A v2 introduces deterministic hierarchical research memory as an implementation-stage artifact, not a committed milestone yet. It preserves the frozen v53Q-1 Champion, 0.7800 online score, current OOF definition, Fold/Gate definitions, confirmed history, and closed branches. Runtime memory and method-research outputs remain ignored until explicitly promoted by a later controlled step.

Safety boundary: no LLM/API calls, no network research, no Adapter execution, no training, no prediction/submission generation, no Project State or History mutation, and `rounds_used` remains 0.
M6R-A v2.1 correction records bucket taxonomy migration through append-only events (`taxonomy_registered`, `scope_reclassified`, `view_superseded`) and keeps runtime views generated from the event log. This correction does not alter Champion, State, History, Provider, Shadow Planner, predictions, folds, or gates.

## M6R-B1 Source-Grounded Method Research Status

M6R-B1 establishes local source-grounded method research infrastructure:
Source Manifest, LocalSourcePackProvider, SourceVerifier, deterministic
SourceChunker, non-LLM Method Card extraction from structured fixtures or
explicit templates, MethodCardValidator, LocalConflictChecker integration, and
policy-weighted MethodRanker. It remains read-only and local-only. It does not
call LLM/API/network paths, execute Adapters, train, generate predictions,
consume rounds, mutate Project State or History, or promote methods to a formal
knowledge base. `network_enabled=false` is a M6R-B1 implementation boundary,
not a permanent Research Agent policy. M6R-B2 will introduce `live_cached`,
`cache_only`, and `disabled` modes: automatic tests stay `disabled`, real
research defaults to `live_cached`, and network/provider failure degrades to
`cache_only`. M6R-B2 is reserved for source-grounded semantic extraction.

## A1 Autonomous Scientific Closed Loop v1 Status

A1 Closed Loop v1 adds a bounded real closed-loop runner for A1. The v1
execution boundary is deliberately narrow: it can select and execute the
registered Fusion Controller over verified OOF assets, record strict OOF
feedback, append an internal research event, update runtime memory views, write
`trajectory_A1.json`, update a runtime best-candidate registry, and make an
automatic stop/continue decision.

Safety boundary:

- max scientific rounds: `3`;
- first real execution uses offline OOF fusion only;
- no Test truth, online submission, Test prediction CSV, Online Champion
  mutation, Project State JSON mutation, confirmed history mutation, Fold/Gate
  mutation, or unregistered Adapter execution;
- runtime artifacts are written under `artifacts/a1_closed_loop_runs/` and are
  ignored by Git;
- the existing `class_weighted_blend_base_composed` fusion candidate starts as
  `portfolio_candidate` and may become `accepted_portfolio`, but is not allowed
  to replace the Online Champion automatically.

Latest verified runtime:

- run id: `a3f0c67c984e0fafc7c97ca9`;
- actual scientific rounds used: `1`;
- round 1 selected `class_weighted_blend_base_composed` through the Fusion
  Controller;
- Overall gain: `+0.000818107444777727`;
- Macro gain: `-0.0009359488044727859`;
- positive folds: `4/5`;
- rescue/damage/net: `38 / 29 / +9`;
- final status: `accepted_portfolio`;
- stop reason:
  `macro_protection_prevents_direct_promotion_and_repeating_same_information_source_is_low_value`;
- `candidate_A1.csv` was not generated.

## A2 Task Integration and Evaluation Foundation Status

- A2 integration module `afac_agent.a2` added (TaskAdapter, Data Profiler,
  Evaluator, Fold validation, dual anchors, portfolio, fusion operators,
  complementarity audit, integration dry-run).
- A2 fold protocol `AFAC_A2_FOLD_V1` validated against the explicit
  `runs/stageA/train_folds.csv`; folds are never re-split.
- A2 Online Deployment Anchor: `A2_ONLINE_CHAMPION_05093` (identity only,
  online score `0.5093`; never an offline OOF artifact).
- A2 Offline Evaluation Anchor: `A2_EVAL_ANCHOR_V1` materialized from the
  verified v42c-DIN OOF route (`runs/C2_v42_merged/oof_scores_full.npz`);
  `deployment_equivalent=false`; Test scores are never used as OOF.
- A2 integration dry-run status: `ready_for_experiment_design`;
  `scientific_rounds_used=0`; no training, no GPU, no Test prediction, no
  submission, no LLM, no network.
- Runtime artifacts: `artifacts/a2_integration/` (ignored by Git).
- No A1 champion, online score, OOF anchor, Fold definition, Gate definition,
  closed branch, Project State JSON, or confirmed history was changed.
