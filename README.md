<div align="center">

# AFAC Self-Evolving Autopoietic Research Agent

A bounded, auditable, self-iterating research agent for multi-task data-science competitions

**AFAC2026 · Task 3 · A1 / A2 / B1 / B2**

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Tests](https://img.shields.io/badge/tests-407%20passed-brightgreen)
![Doctor](https://img.shields.io/badge/doctor-PASS-brightgreen)
![Loop](https://img.shields.io/badge/self--iteration-verified-blueviolet)
![License](https://img.shields.io/badge/license-Apache--2.0-green)

</div>

## Abstract

This repository contains an automated research agent that does not merely
execute fixed ML pipelines: it **proposes, runs, critiques, and promotes its
own experiments** inside a strict scientific contract. The agent operates a
closed self-iteration loop — LLM-driven hypothesis proposal, deterministic
admission gates, real fold-based experiments, counterfactual criticism, and
portfolio promotion — across four competition tasks (two graph node
classification tasks, one sequence recommendation task, and one mixed
recommendation task), under frozen champions, frozen folds, and a hard
wall-clock budget.

The central design lesson encoded in the codebase is that **orchestration is
not science**. A loop that calls an LLM, writes artifacts, and produces a
submission file can still be scientifically empty. Every stage of the loop is
therefore guarded by an explicit, machine-checked contract: data provenance,
experiment permissions, paired-fold metric contracts, monotonic budget
deadlines, deployment permissions, and a completion contract that refuses to
declare success without real scientific work.

## Motivation: a run that "completed" without doing science

During development, a formal two-hour run produced a perfectly formed run
manifest — `status: completed`, a submission CSV, a full artifact tree — while
having executed **zero** scientific experiments:

- 3 rounds, all of them deterministic diagnostics;
- `scientific_rounds_used = 0`, yet a submission was deployed;
- wall clock 8340s against a 7200s hard budget;
- item count reported as both 14,065 and 40,011 in different stages (user IDs
  had been counted as items);
- a no-op round (identical predictions) was admitted into the portfolio.

That run is preserved as forensic evidence and marked
`INVALID_SCIENTIFIC_DEPLOYMENT`. The current agent exists because of it: the
repair is not a patch but a contract stack that makes this failure class
unrepresentable.

## The self-iteration loop

Each research round executes the following pipeline. Every arrow is a real,
logged state transition; every stage marked 🔒 is a deterministic gate an LLM
cannot bypass.

```
       hard deadline (monotonic) ── deployment reserve ── safety margin
  ┌───────────────────────────────────────────────────────────────────────┐
  ▼                                                                       │
PROBLEM SELECTION ─▶ M6B LLM PROPOSAL ─▶ PROPOSAL-TO-OPERATOR COMPILER 🔒 ─▶ M5 SAFETY GATE 🔒
 (problem hierarchy)    (hypothesis,      (unknown operator →               (forbidden content,
                          sources,         blocked_missing_adapter,          budget clamp,
                          budget)          never a silent diagnostic)        diagnostic ceiling)
                                                                                │
                                                                                ▼
 PORTFOLIO UPDATE 🔒 ◀─ UNIFIED EVALUATION + NO-OP AUDIT 🔒 ◀─ REAL OPERATOR EXECUTION
  (diagnostic /         (paired panels, rescue/damage,          (retrieval union, candidate
   scientific /          target metric contract 🔒,              ranker, bucket specialist,
   deployment tiers,     identical predictions →                 protected rerank; B1: LP,
   no-op quarantined)    refunded, never promoted)               APPNP, feature baselines)
          │                                                          ▲
          ▼                                                          │
 M6C CRITIC + POSTMORTEM ─▶ NEXT DECISION ───────────────────────────┘
  (revise must change the   (continue / revise / switch / stop,
   semantic genome hash,     gated by measured runtimes and
   not just the name)        the remaining budget)
```

Reading the loop: **problem → proposal → compile → gate → genome → execute →
evaluate → portfolio → critique → next decision**, all inside one monotonic
hard budget, with the portfolio separated into diagnostic / scientific /
deployment tiers so a diagnostic can never be promoted into a submission.

### Proposal-to-Operator Compiler

LLM proposals are free text; experiments are structured objects. The compiler
(`afac_agent/v2/proposal_compiler.py`) maps a proposal to a canonical
`operator_id` from the capability registry, together with retrieval sources,
feature sets, model family, objective, target panel, hyperparameters, and a
runtime estimate. A proposal that asks for an unavailable operator is
**blocked** (`blocked_missing_adapter`) — it is never silently downgraded to a
diagnostic. A `revise` verdict must produce a different *semantic genome
hash*; two consecutive duplicate revisions force a family or problem switch.

### Adaptive fold ladder

Experiments climb a fixed ladder over one hash-stable canonical fold
assignment (never re-randomized):

| Fidelity | Folds | Role | Consumes round | Deployable |
|----------|-------|------|----------------|------------|
| F0 | 0 | deterministic diagnostic | no | **never** |
| F1 | 1–2 | screen | yes | no |
| F2 | 3 | confirm | yes | yes (if promoted) |
| F3 | 5 | full CV (trigger-only) | yes | yes |
| F4 | full | deployment retrain | — | yes |

Promotion requires paired same-fold, same-panel, same-metric, same-evaluator
comparison against the parent; screen evidence alone never promotes an
incumbent.

### Experiment permission model

`afac_agent/v2/experiment_kind.py` assigns every execution a kind with static
permissions. Diagnostics cannot enter the portfolio, cannot be incumbent, and
cannot deploy. Screens enter the scientific portfolio as `screen_only`.
No-op experiments (predictions identical to parent) are refunded and
quarantined to the audit history. This is the contract that the forensic run
violated.

## Contracts that keep the loop honest

- **Data contract** (`data_contract.py`, `b1_data_contract.py`): every count
  carries provenance (`source_file`, `source_column`, `counting_rule`,
  `membership_hash`); full-dataset and profiler-sampled scales are separate
  fields; Input Discovery, Data Intelligence, Experiment Executor, and
  Deployment universes are reconciled, and any mismatch blocks the run.
- **Target metric contract** (`target_metric_contract.py`): every problem node
  declares its panel (e.g. `B2_NOVEL_TARGET_PANEL`), metric, K, and direction;
  parent and candidate must be measured on the same fold, same panel, same
  metric, same evaluator version.
- **Budget contract**: `time.monotonic()` hard deadline from process start,
  a deployment reserve, and a safety margin; no experiment starts when
  `remaining < estimate + reserve + margin`; a fold-level circuit breaker
  aborts slow experiments between folds with partial results instead of
  overrunning the deadline.
- **Deployment permission contract**: format audit, scientific permission,
  budget contract, and data contract must all pass; otherwise no submission
  file is written. With no confirmed candidate, the run falls back to a
  validated anchor (popularity/history baselines re-materialized on the
  canonical folds) and reports `completed_with_anchor_fallback` — never a
  diagnostic dressed up as a model.
- **Completion contract**: `status=completed` requires, among other things,
  at least one real operator execution, `effective_scientific_rounds >= 1`
  (or a declared anchor fallback), no no-op in the portfolio, a deployable
  incumbent, and a wall clock inside the hard budget.
- **Frozen assets & test truth**: champions, folds, confirmed history, and
  anchors are hash-verified before and after every run; test labels are never
  read; test scores are never reported as offline OOF metrics.

## Run supervision and auditability

Every run is driven by the Run Supervisor (`afac_agent/supervisor/`):
`heartbeat.json`, `STATUS.md`, `dashboard.html`, and `run_events.jsonl` are
written live, with transparency fields for the current operator, experiment
kind, semantic genome hash, scientific counters, data-contract status, and the
monotonic deadlines. The dashboard marks non-deployable stages explicitly
(`DIAGNOSTIC — NOT DEPLOYABLE`, `SCREEN ONLY — NOT DEPLOYABLE`). Execution
identity (`execution_id`, includes code commit + nonce) is separated from the
cache identity (`input_fingerprint`), and all LLM calls are recorded in an
append-only ledger (`llm_calls.jsonl`).

## Operator space and capability registry

The capability registry (`afac_agent/v2/capability_registry.py`) records, per
operator, whether it is implemented, available in this environment, and which
fidelity levels it supports — including an honest record of what is missing
(e.g. `lightgbm`/`torch`-based operators are registered as unavailable, and
the registry reports the resulting availability bias rather than claiming a
fallback is optimal).

**B2 (sequence recommendation):** multi-source retrieval union (popularity /
history / repeat / transition / item-CF / attribute / sequence / novel
sources), a candidate-table GBDT ranker over sparse candidate rows (source
scores, RRF, popularity, recency, history/novel, user/item attributes), bucket
specialists (short/long history, history/novel target, long tail), and
protected rerank (top-K set protection, position-10 admission,
fallback-keep-parent).

**B1 (node classification):** feature baselines (logistic / MLP), graph
propagation (label propagation / APPNP over directed-out, directed-in, and
undirected-union views), feature–graph residual blends, and degree-routed
bucket specialists.

## Verified evidence

| Task | Online result | Offline anchor | Loop status |
|------|---------------|----------------|-------------|
| A1 (node classification) | **0.7800** champion `v53Q-1` | OOF 0.775384 | closed loop executed; fusion candidate accepted to portfolio (+0.0008 overall, macro-protected); champion frozen |
| A2 (recommendation) | **0.5093** champion | NDCG@10 0.5925 (v42c OOF) | integration + evaluation foundation complete; real closed loop awaits explicit asset paths |
| B1 (node classification) | **0.37974** (V2) | `B1_EVAL_ANCHOR_V2` | closed loop completed (0.4964 standard / 0.3941 macro); v2.2 wired into the v2 orchestrator with budget circuit breaker |
| B2 (sequence recommendation) | 0.06838 (V1) | `B2_EVAL_ANCHOR_V1` | v1 loop completed; v2.1 scientific-execution repair verified by real-LLM smoke |

Real-LLM self-iteration smoke (900s budget, B2): 12 provider calls
(problem synthesis, M6B, M6C, postmortem per round), 2 real scientific
screen experiments, data contract consistent across all stages
(`n_items = 14,065` from `item.csv` everywhere), no-op isolation verified,
deployment permission correctly refused in smoke mode, completion contract
passed — 599.5s wall clock.

Engineering validation: **407 tests passed**; `doctor` reports PASS;
frozen-asset hashes verified unchanged across every run.

## Repository layout

```text
afac_agent/
├── main.py                    # CLI entry (v2-run, legacy replays, adapters)
├── doctor.py                  # environment & contract validator
├── v2/                        # self-evolving research layer
│   ├── orchestrator.py        # the self-iteration state machine
│   ├── proposal_compiler.py   # LLM proposal → canonical operator
│   ├── experiment_kind.py     # permission model (diagnostic/screen/confirm/…)
│   ├── data_contract.py       # B2 data contract with provenance
│   ├── b1_data_contract.py    # B1 data contract
│   ├── target_metric_contract.py
│   ├── anchor_registry.py     # validated fallback anchors
│   ├── adaptive_fold.py       # canonical folds + F0–F4 ladder + promotion
│   ├── capability_registry.py # operator availability, honestly recorded
│   ├── completion_contract.py
│   ├── budget_scheduler.py    # dynamic budget & premature-stop detection
│   ├── noop_detector.py       # identical-prediction detection & refunds
│   ├── metric_semantics.py    # pool recall vs top-10; error decomposition
│   ├── operators/             # recommendation & classification operators
│   └── smokes.py              # bounded four-task smoke runner
├── supervisor/                # heartbeat, dashboard, stall detection, resume
├── research/                  # hierarchical research memory, method research
├── llm/                       # provider layer (Aliyun Bailian; env-var keys)
├── b1/  b2/  a2/              # per-task adapters, evaluators, loops
└── evaluation/  planning/  profilers/  adapters/
```

## Quick start

```bash
# environment & contract validation
python -m afac_agent.doctor --project_root .

# full test suite
python -m pytest -p no:cacheprovider -q

# bounded four-task smokes (minutes each, never a full loop)
python -m afac_agent.v2.smokes --task all

# a real, LLM-driven self-iteration smoke (B2, ~10 minutes, no deployment)
python -u -m afac_agent.main v2-run --task B2 \
  --data-root "<path-to-B2-data>" \
  --out-root "artifacts/v2_science_smoke/b2" \
  --max-wall-clock-seconds 900 --deployment-reserve-seconds 180 \
  --require-llm --force-new-execution --smoke --no-deployment --no-full-cv
```

LLM access is configured only through environment variables
(`DASHSCOPE_API_KEY`, `AFAC_BAILIAN_BASE_URL`); no credential is ever stored
in the repository. Local data paths live in the git-ignored
`config/paths.local.yaml` (see `config/paths.local.example.yaml`).

## License

Repository code is licensed under the [Apache License 2.0](LICENSE).
Competition datasets, historical model artifacts, and platform results
referenced by the documentation remain subject to their original terms.
