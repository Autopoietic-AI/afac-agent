#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${AFAC_PYTHON:-python}"
ANCHOR="$PWD/artifacts/A1_v53q1_transition_stable_edge_h2_SAFE.csv"

if ! "$PYTHON" --version >/dev/null 2>&1; then
  echo "[ERROR] Python not available: $PYTHON" >&2
  echo "Set AFAC_PYTHON or activate the environment first." >&2
  exit 2
fi

"$PYTHON" -m afac_agent.doctor --project_root "$PWD"

"$PYTHON" -m afac_agent.auto_loop \
  --project_root "$PWD" \
  --anchor_csv "$ANCHOR" \
  --max_steps 10
