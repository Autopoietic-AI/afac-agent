# A2 Champion 0.5093 — Architecture Review

Source of truth: `afac_agent/a2/anchors.py` `ONLINE_STRATEGY`. The champion is frozen.

## What the champion is

An anchored, guard-railed pipeline over a fixed candidate set:

1. **V23 Anchor** — a fixed, teammate-verified V23 Top10 candidate set. Membership never changes.
2. **v42c-DIN exact-Len3 Top7 rerank** — a bucket specialist that reranks only exact Len3 users,
   strictly inside the original V23 Top7, with alpha=0.50.
3. **C_all Novel rerank** — a specialist that reranks Novel slots only, alpha=0.50.
4. **Position-10 external admission** — at most one external item may enter, only at position 10.

History items and non-Len3 users are protected at every stage. Online score history:
0.5052 → 0.5066 → 0.5068 → 0.5092 → 0.5093.

## Portable principles (the only things that transfer)

- **anchor_first**: freeze a verified anchor candidate set before any model work; all later
  stages operate on the anchor, never regenerate it.
- **bucket_specialist**: deploy narrow specialists per well-defined bucket (e.g. exact Len3 users,
  Novel slots) instead of one global reranker.
- **protected_residual**: everything outside a specialist's bucket is a protected residual —
  untouched by construction, not by convention.
- **novel_only_rerank**: novel slots are reranked by a novel-only specialist; history slots are
  never re-scored.
- **boundary_admission**: external items enter only at a single boundary position (position 10),
  at most one item, so the core ranking is never destabilized.
- **source_consensus_gate**: a new score source influences the system only after passing a
  consensus check against the frozen anchor sources; no mtime/latest-file selection.

## What must NOT transfer

This package forbids migrating any A2-specific raw material into B2: A2 user ids, A2 item ids,
A2 scores, A2 checkpoints, and A2 weights. The assets listed in `artifact_registry.json` live in
the external A2 project, have unknown availability, and are marked `forbidden_direct_transfer`.
Only the six principles above are portable.

## Lessons for B2

- B2's degenerate components (Score Blend → Popularity, Bucket Rerank → Popularity) are exactly
  what bucket_specialist + protected_residual prevent: a specialist must prove it differs from
  the fallback inside its bucket before it ships.
- B2's missing novel-target coverage maps to novel_only_rerank: novel slots need their own
  specialist rather than being squeezed out of a history-only model.
- B1/B2 premature stops and no-op rounds motivate the consensus gate and no-op detection:
  a round that changes nothing is refunded and excluded from the portfolio.
