from __future__ import annotations

SEVERITY_TO_ID: dict[str, int] = {
    "norm": 0,
    "mild": 1,
    "moderate": 2,
    "severe": 3,
}

SEVERITY_TO_TARGET: dict[str, float] = {
    "norm": 1.0,
    "mild": 2.0 / 3.0,
    "moderate": 1.0 / 3.0,
    "severe": 0.0,
}

ID_TO_SEVERITY: dict[int, str] = {v: k for k, v in SEVERITY_TO_ID.items()}

TASK_TO_GROUP: dict[str, str] = {
    "task1": "syllable",
    "task2": "character",
    "task3": "word",
    "task4": "sentence",
    "task5": "sentence",
    "task6": "sentence",
    "task7": "sentence",
    "task8": "sentence",
}
