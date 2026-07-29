"""Minimal I/O helpers: JSON and CSV persistence."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def save_json(path: str | Path, payload: Any) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def save_csv_rows(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows_list: List[Dict[str, Any]] = list(rows)
    if not rows_list:
        return
    fieldnames: List[str] = []
    seen: set = set()
    for row in rows_list:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows_list:
            writer.writerow(row)
