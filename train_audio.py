#!/usr/bin/env python3
"""
MMDys — training entry point for the audio+text branch.

Usage:
  conda run -n w2v_tts python train_audio.py \
      --config configs/experiments/msdm_audio_text_v1_splitsv2.yaml

Multi-seed example:
  for seed in 17 29 2026; do
    conda run -n w2v_tts python train_audio.py \
        --config configs/experiments/msdm_audio_text_v1_splitsv2.yaml \
        --seed $seed \
        --output-dir outputs/msdm_audio_text_v1_splitsv2/seed_$seed
  done
"""
from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from data import MSDMAudioTextCollator, MSDMAudioTextDataset
from models import AudioTextMMDys
from trainers import SeverityGuidedTrainer
from utils import load_experiment_config, save_json, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MMDys audio+text branch training")
    p.add_argument("--config", type=Path, required=True, help="Path to experiment YAML")
    p.add_argument("--output-dir", type=Path, default=None, help="Override output directory")
    p.add_argument("--seed", type=int, default=None, help="Override random seed")
    return p.parse_args()


def _make_worker_init_fn(base_seed: int):
    def _seed_worker(worker_id: int) -> None:
        worker_seed = int(base_seed + worker_id) % (2 ** 32)
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
    return _seed_worker


def _build_loader(
    dataset: MSDMAudioTextDataset,
    collator: MSDMAudioTextCollator,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    sampler=None,
    generator=None,
    worker_init_fn=None,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collator,
        drop_last=False,
        generator=generator,
        worker_init_fn=worker_init_fn,
    )


def _build_balanced_sampler(
    dataset: MSDMAudioTextDataset,
    mode: str = "severity",
    generator=None,
) -> WeightedRandomSampler:
    severity_ids = [int(item["severity_id"]) for item in dataset.samples]
    if mode == "severity_subject":
        sev_counts = Counter(severity_ids)
        sev_spk = [(int(item["severity_id"]), str(item["speaker_id"])) for item in dataset.samples]
        sev_spk_counts = Counter(sev_spk)
        weights = [
            1.0 / (float(sev_counts[sid]) * float(sev_spk_counts[(sid, spk)]))
            for sid, spk in sev_spk
        ]
    else:
        counts = Counter(severity_ids)
        weights = [1.0 / float(counts[sid]) for sid in severity_ids]
    weight_tensor = torch.tensor(weights, dtype=torch.float32)
    return WeightedRandomSampler(
        weight_tensor,
        num_samples=len(weight_tensor),
        replacement=True,
        generator=generator,
    )


def main() -> None:
    args = parse_args()
    cfg = load_experiment_config(args.config)

    seed = int(args.seed if args.seed is not None else cfg["experiment"].get("seed", 17))
    cfg["experiment"]["seed"] = seed

    train_cfg = cfg["train"]
    deterministic = bool(train_cfg.get("deterministic", True))
    set_seed(
        seed,
        deterministic=deterministic,
        cudnn_benchmark=train_cfg.get("cudnn_benchmark", None),
        deterministic_algorithms=train_cfg.get("deterministic_algorithms", None),
        deterministic_warn_only=bool(train_cfg.get("deterministic_warn_only", True)),
    )

    output_dir = Path(args.output_dir or cfg["experiment"]["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "resolved_config.json", cfg)

    device = torch.device(train_cfg["device"] if torch.cuda.is_available() else "cpu")

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]

    train_ds = MSDMAudioTextDataset(
        split_json_path=data_cfg["train_split_json"],
        wav_root=data_cfg["wav_root"],
        split_name="train",
        target_sample_rate=int(data_cfg["sample_rate"]),
        max_audio_seconds=float(data_cfg["max_audio_seconds"]),
        random_crop_train=bool(data_cfg.get("random_crop_train", True)),
    )
    dev_ds = MSDMAudioTextDataset(
        split_json_path=data_cfg["dev_split_json"],
        wav_root=data_cfg["wav_root"],
        split_name="dev",
        target_sample_rate=int(data_cfg["sample_rate"]),
        max_audio_seconds=float(data_cfg["max_audio_seconds"]),
        random_crop_train=False,
    )
    test_ds = MSDMAudioTextDataset(
        split_json_path=data_cfg["test_split_json"],
        wav_root=data_cfg["wav_root"],
        split_name="test",
        target_sample_rate=int(data_cfg["sample_rate"]),
        max_audio_seconds=float(data_cfg["max_audio_seconds"]),
        random_crop_train=False,
    )

    collator = MSDMAudioTextCollator(
        text_model_name=model_cfg["text_model_name"],
        max_text_length=int(data_cfg["max_text_length"]),
    )

    sampler_gen = torch.Generator().manual_seed(seed + 7)
    train_gen   = torch.Generator().manual_seed(seed + 11)
    dev_gen     = torch.Generator().manual_seed(seed + 23)
    test_gen    = torch.Generator().manual_seed(seed + 37)
    worker_init = _make_worker_init_fn(seed + 1000)

    use_balanced = bool(train_cfg.get("use_balanced_sampler", False))
    balanced_mode = str(train_cfg.get("balanced_sampler_mode", "severity")).strip().lower()
    train_sampler = _build_balanced_sampler(train_ds, mode=balanced_mode, generator=sampler_gen) if use_balanced else None

    num_workers     = int(train_cfg.get("num_workers", 4))
    pin_memory      = bool(train_cfg.get("pin_memory", True))
    batch_size      = int(train_cfg["batch_size"])
    eval_batch_size = int(train_cfg["eval_batch_size"])

    train_loader = _build_loader(
        train_ds, collator, batch_size=batch_size,
        shuffle=not use_balanced, num_workers=num_workers, pin_memory=pin_memory,
        sampler=train_sampler, generator=train_gen, worker_init_fn=worker_init,
    )
    dev_loader = _build_loader(
        dev_ds, collator, batch_size=eval_batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=pin_memory,
        generator=dev_gen, worker_init_fn=worker_init,
    )
    test_loader = _build_loader(
        test_ds, collator, batch_size=eval_batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=pin_memory,
        generator=test_gen, worker_init_fn=worker_init,
    )

    model = AudioTextMMDys(
        audio_model_name=model_cfg["audio_model_name"],
        text_model_name=model_cfg["text_model_name"],
        projection_dim=int(model_cfg["projection_dim"]),
        projection_hidden_dim=int(model_cfg["projection_hidden_dim"]),
        classifier_hidden_dim=int(model_cfg["classifier_hidden_dim"]),
        dropout=float(model_cfg["dropout"]),
        use_text=bool(model_cfg.get("use_text", True)),
    )

    resume_ckpt = str(train_cfg.get("resume_checkpoint", "")).strip()
    if resume_ckpt:
        ckpt_path = Path(resume_ckpt)
        if ckpt_path.exists():
            print(f"[resume] loading checkpoint: {ckpt_path}")
            payload = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(payload["model"], strict=False)

    model = model.to(device)

    trainer = SeverityGuidedTrainer(
        model=model,
        train_loader=train_loader,
        dev_loader=dev_loader,
        test_loader=test_loader,
        cfg=cfg,
        output_dir=output_dir,
        device=device,
    )
    result = trainer.train()
    save_json(output_dir / "final_summary.json", result)


if __name__ == "__main__":
    main()
