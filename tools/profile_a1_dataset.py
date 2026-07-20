# -*- coding: utf-8 -*-
"""Compatibility wrapper for the M2 A1 Data Profiler."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from afac_agent.profilers.a1_data_profiler import main


if __name__ == "__main__":
    main()
