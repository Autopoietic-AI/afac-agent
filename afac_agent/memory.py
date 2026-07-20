# -*- coding: utf-8 -*-
"""实验记忆：历史快照 + 运行期只追加轨迹。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


class ExperimentMemory:
    def __init__(self, jsonl_path: str | Path):
        self.path = Path(jsonl_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: Dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as file:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )

    def read_all(self) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        result = []
        for line in self.path.read_text(
            encoding="utf-8"
        ).splitlines():
            line = line.strip()
            if line:
                result.append(json.loads(line))
        return result

    def versions(self) -> set[str]:
        return {
            str(item.get("version", ""))
            for item in self.read_all()
        }

    def import_records(
        self,
        records: Iterable[Dict[str, Any]],
    ) -> int:
        existing = self.versions()
        imported = 0
        for record in records:
            version = str(record["version"])
            if version in existing:
                continue
            self.append(record)
            existing.add(version)
            imported += 1
        return imported
