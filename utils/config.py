"""
YAML experiment config loader.

Experiment YAMLs reference four sub-configs (data, model, loss, train).
This loader resolves them relative to the project root and merges
any inline overrides from the experiment file.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import yaml


def load_yaml(path: str | Path) -> Dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_experiment_config(experiment_path: str | Path) -> Dict:
    """
    Load a structured experiment config from a YAML file.

    The experiment YAML must specify:
      data_config:  path to data config
      model_config: path to model config
      loss_config:  path to loss config
      train_config: path to train config

    Optional top-level keys:
      name, output_dir, seed, overrides (dict of section → key/value updates)
    """
    exp_path = Path(experiment_path).resolve()
    exp = load_yaml(exp_path)

    # Root is two levels above the experiment yaml: configs/experiments/ → root
    root = exp_path.parents[2]

    def _resolve_cfg(cfg_key: str) -> Dict:
        cfg_path = Path(exp[cfg_key])
        if not cfg_path.is_absolute():
            cfg_path = (root / cfg_path).resolve()
        return load_yaml(cfg_path)

    cfg = {
        "experiment": {
            "name":       exp.get("name", exp_path.stem),
            "output_dir": exp.get("output_dir", str(root / "outputs" / exp_path.stem)),
            "seed":       int(exp.get("seed", 42)),
        },
        "data":  _resolve_cfg("data_config"),
        "model": _resolve_cfg("model_config"),
        "loss":  _resolve_cfg("loss_config"),
        "train": _resolve_cfg("train_config"),
    }

    for section, payload in exp.get("overrides", {}).items():
        cfg.setdefault(section, {}).update(payload)

    return cfg
