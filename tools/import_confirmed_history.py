# -*- coding: utf-8 -*-
"""把人工确认的研发历史导入Agent Memory。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from afac_agent.memory import ExperimentMemory
from afac_agent.validation import validate_memory_records_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history_json", required=True)
    parser.add_argument("--memory_jsonl", required=True)
    args = parser.parse_args()

    schema_report = validate_memory_records_file(args.history_json)
    if not schema_report.passed:
        result = {
            "status": "invalid_history",
            "history_schema": schema_report.to_dict(),
            "memory_path": str(
                Path(args.memory_jsonl).resolve()
            ),
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        raise SystemExit(2)

    payload = json.loads(
        Path(args.history_json).read_text(encoding="utf-8")
    )
    memory = ExperimentMemory(args.memory_jsonl)
    imported = memory.import_records(
        payload["experiments"]
    )
    result = {
        "status": "success",
        "imported": imported,
        "total_memory": len(memory.read_all()),
        "history_schema": schema_report.to_dict(),
        "memory_path": str(
            Path(args.memory_jsonl).resolve()
        ),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
