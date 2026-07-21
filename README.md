# AFAC Agent v1.3

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

## M3A Tool Adapter Foundation

M3A adds the minimal Tool Adapter protocol and the first real read-only
adapter, `A1_V53Q1_PATCH_AUDIT`.  It audits the v53Q-1 patch assets, hashes,
4-node migration evidence and v49A meta files without running patch replay,
training, using GPU, generating prediction CSVs, registering a champion, or
consuming successful experiment rounds.

Run with explicit local historical asset paths:

```bash
python -m afac_agent.main run-adapter \
  --tool A1_V53Q1_PATCH_AUDIT \
  --anchor_csv artifacts/A1_v53q1_transition_stable_edge_h2_SAFE.csv \
  --v53q1_base_csv "<local-v46A1-base-csv>" \
  --v49a_oof_meta_csv "<local-v49A-oof-meta-csv>" \
  --v49a_test_meta_csv "<local-v49A-test-meta-csv>" \
  --v53q1_audit_md artifacts/V53Q1_TRANSITION_STABLE_EDGE_H2_AUDIT.md \
  --v53q1_patch_py artifacts/a1_v53q1_transition_stable_edge_h2_patch.py \
  --execute
```

Adapter outputs are ignored by git under:

```text
artifacts/adapter_runs/
```

## M6A LLM Shadow Planner and Bailian provider

M6A keeps the deterministic M5A planner authoritative.  LLM output is only a
shadow proposal: it is schema-normalized, safety-filtered and compared with the
deterministic plan, but it never executes tools, trains models, generates
submissions, registers champions, mutates Project State, or consumes successful
experiment rounds.

The Aliyun Bailian OpenAI-compatible provider is available as:

```bash
python -m afac_agent.main llm-provider-check --provider aliyun_bailian_openai
```

and for advisory shadow planning:

```bash
python -m afac_agent.main shadow-plan --provider aliyun_bailian_openai ...
```

The default model is:

```text
qwen3.6-max-preview
```

Only `qwen3.5-*` and `qwen3.6-*` model names are allowed.  API credentials and
the Bailian base URL are read only from environment variables named in the
ignored local config:

```text
DASHSCOPE_API_KEY
AFAC_BAILIAN_BASE_URL
```

`config/llm.local.json` remains git-ignored.  The committed
`config/llm.local.example.json` contains only non-secret field names and
defaults.  Provider usage artifacts record safe audit metadata such as provider,
model, host, latency, token usage and finish reason; they do not record API
keys, authorization headers, cookies, account data or full environment values.

Framework inspiration / reuse: this provider borrows only the official baseline
idea that model name, endpoint, timeout and credentials should be configurable
and that credentials should come from environment variables.  It does not reuse
baseline behavior where an LLM directly edits model code, chooses final
CONTINUE/PIVOT/STOP decisions, executes unregistered commands, generates
submissions, or treats weak/empty metrics as proof of improvement.

## M6R-A v2 Research Memory Foundation

M6R-A adds a deterministic, read-only research memory layer for hierarchical scientific diagnosis. It separates execution blockers, scientific problems, and evidence gaps, then materializes views from one append-only event log: `research_events.jsonl`. Runtime outputs are written under ignored directories: `artifacts/research_memory/` and `artifacts/method_research/`.

The supported analysis path is Global -> Bucket -> Bucket x Class -> Error Mechanism, with a reverse Cross-Bucket Class audit. Research Queue priority and Top-K brief generation are controlled by `config/research_policy.json`; thresholds and Top-K limits are not hidden in code. This stage does not call LLMs, APIs, adapters, training, prediction, or submission paths. It does not mutate Champion, Project State, or confirmed History.

Framework inspiration record: M6R-A borrows ideas from public baseline-style diagnosis, experiment-memory practices, AIDE-style branch/parent/duplicate concepts, AI-Scientist-style hypothesis/evidence/critique loops, and event-sourcing append-only/materialized-view design. AFAC implements its own Global/Bucket/Bucket-Class decomposition, Cross-Bucket Class audit, mechanism ledger, new-information-source checks, strict OOF safety, frozen Champion boundary, and hierarchical Research Queue. It does not integrate AIDE, AI Scientist, or other framework code, and does not allow LLMs to directly modify code, decide experiments, or execute experiments.
M6R-A v2.1 correction: bucket scopes are now represented as multi-axis signatures. `connectivity_visibility` (`graph_visible`, `isolated`) is separate from `train_label_reachability` (`one_hop_available`, `exact2_only`, `exact3_4_only`, `no_visible_train_within_4_hops`) and `degree_band`; `class_id` remains an independent analysis axis. Research Queue entries expose component-level priority scores and overlap penalties from `config/research_policy.json`. LocalConflictChecker performs deterministic multi-field conflict checks instead of comparing method names only.

## M6R-B1 Source-Grounded Method Research Foundation

M6R-B1 adds a local, deterministic source-grounded method-research pipeline:
Research Brief -> Local Source Pack -> Source Verification -> deterministic
chunking -> Method Card validation -> LocalConflictChecker -> policy-weighted
ranking. Runtime outputs are ignored under `artifacts/method_research_runs/`.
This stage does not call LLMs, APIs, network search, Adapters, training,
prediction, or submission paths, and it does not promote methods into a formal
knowledge base. `network_enabled=false` describes only this local foundation
stage, not a permanent ban on networked research. M6R-B2 will add
`live_cached`, `cache_only`, and `disabled` modes: automated tests use
`disabled`; real research defaults to `live_cached`; provider/network failures
fall back to `cache_only`. M6R-B2 will handle semantic extraction from ordinary
papers or repositories under separate approval.
