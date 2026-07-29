#!/usr/bin/env python3
"""
Video branch training entry point.

Usage:
  python train_video.py --config configs/experiments/video_phase_official.yaml

Multi-seed example:
  for seed in 17 123 42; do
    python train_video.py \
      --config configs/experiments/video_phase_official.yaml \
      --seed $seed \
      --output-dir outputs/video_phase_official/seed_$seed
  done
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, WeightedRandomSampler

from data import MSDMPhaseCollator, MSDMPhaseDataset, VideoUniformCollator, VideoUniformDataset
from models import PhaseVideoMAEModel
from trainers import SeverityGuidedTrainer
from utils import load_experiment_config, save_json, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MMDys-Claude video branch training")
    p.add_argument("--config", type=Path, required=True, help="Path to experiment YAML")
    p.add_argument("--output-dir", type=Path, default=None, help="Override output directory")
    p.add_argument("--seed", type=int, default=None, help="Override random seed")
    return p.parse_args()


def _build_loader(
    dataset: MSDMPhaseDataset,
    collator: MSDMPhaseCollator,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    prefetch_factor: int | None,
    persistent_workers: bool,
    mp_context: str | None,
    sampler=None,
    generator=None,
) -> DataLoader:
    kwargs: dict = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle if sampler is None else False,
        "sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "collate_fn": collator,
        "drop_last": False,
    }
    if generator is not None:
        kwargs["generator"] = generator
    if num_workers > 0:
        if prefetch_factor is not None:
            kwargs["prefetch_factor"] = int(prefetch_factor)
        kwargs["persistent_workers"] = bool(persistent_workers)
        if mp_context:
            kwargs["multiprocessing_context"] = mp_context
    return DataLoader(**kwargs)


def _build_dataset(data_cfg: dict, split_name: str, is_train: bool) -> MSDMPhaseDataset:
    return MSDMPhaseDataset(
        split_json_path=data_cfg[f"{split_name}_split_json"],
        video_root=data_cfg["video_root"],
        flow_feature_root=data_cfg["flow_feature_root"],
        split_name=split_name,
        # Phase segmentation
        pre_seconds=float(data_cfg.get("pre_onset_seconds", 1.0)),
        post_seconds=float(data_cfg.get("post_offset_seconds", 0.3)),
        fallback_pre_seconds=float(data_cfg.get("fallback_pre_seconds", 0.4)),
        fallback_post_seconds=float(data_cfg.get("fallback_post_seconds", 0.2)),
        min_speech_frames=int(data_cfg.get("min_speech_frames", 3)),
        pre_target_frames=int(data_cfg.get("pre_target_frames", 16)),
        speech_target_frames=int(data_cfg.get("speech_target_frames", 32)),
        speech_pad_to_target=bool(data_cfg.get("speech_pad_to_target", False)),
        post_target_frames=int(data_cfg.get("post_target_frames", 5)),
        # Video
        resize_height=int(data_cfg.get("resize_height", 224)),
        resize_half_width=int(data_cfg.get("resize_half_width", 224)),
        max_video_seconds=float(data_cfg.get("max_video_seconds", 5.0)),
        # Flow
        use_flow_descriptors=bool(data_cfg.get("use_flow_descriptors", True)),
        use_flow_as_channel=bool(data_cfg.get("use_flow_as_channel", False)),
        require_flow_ok=bool(data_cfg.get("require_flow_ok", False)),
        # Descriptor cache — always read-only
        descriptor_cache_dir=data_cfg.get("descriptor_cache_dir", None),
        descriptor_cache_write=bool(data_cfg.get("descriptor_cache_write", False)),
        descriptor_cache_miss_policy=str(data_cfg.get("descriptor_cache_miss_policy", "compute")),
        descriptor_eps=float(data_cfg.get("descriptor_eps", 1e-6)),
        deform_descriptor_cache_dir=data_cfg.get("deform_descriptor_cache_dir", None),
        # Augmentation: only for training split
        enable_aug=bool(is_train and data_cfg.get("aug_hflip_prob", 0) > 0),
        aug_hflip_prob=float(data_cfg.get("aug_hflip_prob", 0.5)),
        aug_brightness_delta=float(data_cfg.get("aug_brightness_delta", 0.04)),
        aug_contrast_delta=float(data_cfg.get("aug_contrast_delta", 0.04)),
        aug_phase_jitter_frames=int(data_cfg.get("aug_phase_jitter_frames", 2)),
        # Missing data
        missing_policy=str(data_cfg.get("missing_policy", "skip")),
        runtime_missing_policy=str(data_cfg.get("runtime_missing_policy", "skip")),
        runtime_max_retries=int(data_cfg.get("runtime_max_retries", 8)),
        min_samples_per_split=int(data_cfg.get("min_samples_per_split", 1)),
    )


def _print_feature_availability(output_dir: Path, datasets: dict) -> dict:
    stats = {split: getattr(ds, "feature_availability", {}) for split, ds in datasets.items()}
    save_json(output_dir / "feature_availability.json", stats)
    for split in ["train", "dev", "test"]:
        s = stats.get(split, {})
        print(
            f"[data] split={split} "
            f"total={s.get('total_json_samples', 0)} "
            f"available={s.get('available_feature_samples', 0)} "
            f"missing={s.get('missing_feature_samples', 0)} "
            f"({float(s.get('missing_pct', 0.0)):.1f}%)"
        )
    return stats


def _build_balanced_sampler(
    dataset: MSDMPhaseDataset,
    mode: str = "severity",
    generator=None,
) -> WeightedRandomSampler:
    """
    Weighted sampler for class imbalance.
    mode='severity'         : weight = 1 / class_count
    mode='severity_subject' : weight = 1 / (class_count * speaker_count_per_class)
    """
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


def _build_uniform_dataset(data_cfg: dict, split_name: str, is_train: bool) -> VideoUniformDataset:
    return VideoUniformDataset(
        split_json_path=data_cfg[f"{split_name}_split_json"],
        video_root=data_cfg["video_root"],
        split_name=split_name,
        resize_height=int(data_cfg.get("resize_height", 224)),
        resize_width=int(data_cfg.get("resize_width", data_cfg.get("resize_half_width", 224))),
        max_frames=int(data_cfg.get("max_frames", 512)),
        enable_aug=bool(is_train and data_cfg.get("aug_hflip_prob", 0) > 0),
        aug_hflip_prob=float(data_cfg.get("aug_hflip_prob", 0.5)),
        aug_brightness_delta=float(data_cfg.get("aug_brightness_delta", 0.04)),
        aug_contrast_delta=float(data_cfg.get("aug_contrast_delta", 0.04)),
        missing_policy=str(data_cfg.get("missing_policy", "skip")),
        runtime_missing_policy=str(data_cfg.get("runtime_missing_policy", "skip")),
        runtime_max_retries=int(data_cfg.get("runtime_max_retries", 8)),
        min_samples_per_split=int(data_cfg.get("min_samples_per_split", 1)),
    )


def _compute_ce_weights(train_ds) -> list[float]:
    counts = Counter(int(item["severity_id"]) for item in train_ds.samples)
    total = float(sum(counts.values()))
    num_classes = 4
    raw = [total / (num_classes * max(float(counts.get(c, 0)), 1e-9)) for c in range(num_classes)]
    positive = [w for w in raw if w > 0]
    norm = sum(positive) / max(1, len(positive))
    return [float(w / max(norm, 1e-8)) if w > 0 else 0.0 for w in raw]


def main() -> None:
    args = parse_args()

    if not args.config.exists():
        print(f"[error] Config not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    cfg = load_experiment_config(args.config)

    # Seed
    seed = int(args.seed if args.seed is not None else cfg["experiment"].get("seed", 17))
    cfg["experiment"]["seed"] = seed
    train_cfg = cfg["train"]
    set_seed(
        seed,
        deterministic=bool(train_cfg.get("deterministic", False)),
        cudnn_benchmark=train_cfg.get("cudnn_benchmark", None),
        deterministic_algorithms=train_cfg.get("deterministic_algorithms", None),
        deterministic_warn_only=bool(train_cfg.get("deterministic_warn_only", True)),
    )

    # Output dir
    output_dir = Path(args.output_dir or cfg["experiment"]["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "resolved_config.json", cfg)

    device = torch.device(train_cfg["device"] if torch.cuda.is_available() else "cpu")
    print(f"[init] seed={seed} device={device} output_dir={output_dir}")

    # ---- Datasets -----------------------------------------------------------
    data_cfg  = cfg["data"]
    model_cfg = cfg["model"]

    backbone_type = str(model_cfg.get("backbone_type", "phase_videomae")).lower()
    use_uniform = (backbone_type == "video_uniform")

    if use_uniform:
        train_ds = _build_uniform_dataset(data_cfg, "train", is_train=True)
        dev_ds   = _build_uniform_dataset(data_cfg, "dev",   is_train=False)
        test_ds  = _build_uniform_dataset(data_cfg, "test",  is_train=False)
    else:
        train_ds = _build_dataset(data_cfg, "train", is_train=True)
        dev_ds   = _build_dataset(data_cfg, "dev",   is_train=False)
        test_ds  = _build_dataset(data_cfg, "test",  is_train=False)

    stats = _print_feature_availability(output_dir, {"train": train_ds, "dev": dev_ds, "test": test_ds})

    # Recompute class-balanced CE weights from post-filter train counts
    if bool(train_cfg.get("recompute_ce_weights", True)):
        weights = _compute_ce_weights(train_ds)
        cfg["loss"]["ce_class_weights"] = weights
        save_json(output_dir / "ce_weights.json", {
            "counts": dict(Counter(int(item["severity_id"]) for item in train_ds.samples)),
            "weights": weights,
        })
        print(f"[ce_weights] {weights}")

    # ---- DataLoaders --------------------------------------------------------
    sharing_strategy = str(train_cfg.get("sharing_strategy", "file_system"))
    try:
        mp.set_sharing_strategy(sharing_strategy)
    except RuntimeError:
        pass

    pin_memory           = bool(train_cfg.get("pin_memory", False))
    train_num_workers    = int(train_cfg.get("num_workers", 4))
    eval_num_workers     = int(train_cfg.get("eval_num_workers", 0))   # 0 avoids eval RAM OOM
    prefetch_factor      = int(train_cfg.get("prefetch_factor", 2)) if train_num_workers > 0 else None
    eval_prefetch_factor = int(train_cfg.get("eval_prefetch_factor", 1)) if eval_num_workers > 0 else None
    persistent_workers   = bool(train_cfg.get("persistent_workers", False))
    eval_persistent_workers = bool(train_cfg.get("eval_persistent_workers", False))
    mp_context           = str(train_cfg.get("multiprocessing_context", "spawn")).strip() or None

    collator = VideoUniformCollator() if use_uniform else MSDMPhaseCollator()

    use_balanced = bool(train_cfg.get("use_balanced_sampler", False))
    balanced_mode = str(train_cfg.get("balanced_sampler_mode", "severity")).strip().lower()
    sampler_gen = torch.Generator().manual_seed(seed + 7)
    train_sampler = (
        _build_balanced_sampler(train_ds, mode=balanced_mode, generator=sampler_gen)
        if use_balanced else None
    )
    if use_balanced:
        print(f"[sampler] balanced sampler enabled mode={balanced_mode}")

    train_loader = _build_loader(
        train_ds, collator, int(train_cfg["batch_size"]),
        shuffle=not use_balanced,
        num_workers=train_num_workers, pin_memory=pin_memory,
        prefetch_factor=prefetch_factor, persistent_workers=persistent_workers,
        mp_context=mp_context, sampler=train_sampler,
    )
    dev_loader = _build_loader(
        dev_ds, collator, int(train_cfg["eval_batch_size"]), False,
        eval_num_workers, pin_memory, eval_prefetch_factor, eval_persistent_workers, mp_context,
    )
    test_loader = _build_loader(
        test_ds, collator, int(train_cfg["eval_batch_size"]), False,
        eval_num_workers, pin_memory, eval_prefetch_factor, eval_persistent_workers, mp_context,
    )

    # ---- Model --------------------------------------------------------------
    # backbone_type was already resolved above for the dataset branch
    shared_kwargs = dict(
        hf_backbone_name=str(model_cfg.get("hf_backbone_name", "MCG-NJU/videomae-small-finetuned-kinetics")),
        use_pretrained=bool(model_cfg.get("use_pretrained", True)),
        videomae_num_frames=int(model_cfg.get("videomae_num_frames", 16)),
        image_size=int(model_cfg.get("image_size", data_cfg.get("resize_height", 224))),
        interpolate_pos_emb=bool(model_cfg.get("interpolate_pos_emb", False)),
        in_channels=int(model_cfg.get("in_channels", 3)),
        flow_input_dim=int(model_cfg.get("flow_input_dim", 8)),
        flow_embed_dim=int(model_cfg.get("flow_embed_dim", 64)),
        use_flow_descriptors=bool(model_cfg.get("use_flow_descriptors", data_cfg.get("use_flow_descriptors", True))),
        phase_token_dim=int(model_cfg.get("phase_token_dim", 256)),
        sample_embed_dim=int(model_cfg.get("sample_embed_dim", 256)),
        classifier_hidden_dim=int(model_cfg.get("classifier_hidden_dim", 256)),
        consensus_hidden_dim=int(model_cfg.get("consensus_hidden_dim", 128)),
        num_agg_heads=int(model_cfg.get("num_agg_heads", 4)),
        dropout=float(model_cfg.get("dropout", 0.1)),
    )
    model = PhaseVideoMAEModel(**shared_kwargs)
    print(f"[model] PhaseVideoMAEModel loaded")

    # ---- Resume / init checkpoint ------------------------------------------
    resume_payload = None
    resume_ckpt = str(train_cfg.get("resume_checkpoint", "")).strip()
    if not resume_ckpt and bool(train_cfg.get("auto_resume_latest", False)):
        last = output_dir / "last.pt"
        if last.exists():
            resume_ckpt = str(last)
    if resume_ckpt:
        ckpt_path = Path(resume_ckpt)
        if ckpt_path.exists():
            resume_payload = torch.load(ckpt_path, map_location="cpu")
            if isinstance(resume_payload, dict) and "model" in resume_payload:
                model.load_state_dict(resume_payload["model"], strict=False)
                print(f"[resume] loaded weights from {ckpt_path}")
        else:
            print(f"[resume] checkpoint not found: {ckpt_path} — starting fresh")

    model = model.to(device)

    # ---- Train --------------------------------------------------------------
    trainer = SeverityGuidedTrainer(
        model=model,
        train_loader=train_loader,
        dev_loader=dev_loader,
        test_loader=test_loader,
        cfg=cfg,
        output_dir=output_dir,
        device=device,
        resume_payload=resume_payload,
    )
    result = trainer.train()
    save_json(output_dir / "final_summary.json", result)


if __name__ == "__main__":
    main()
