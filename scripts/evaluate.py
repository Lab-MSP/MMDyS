#!/usr/bin/env python3
"""
evaluate.py — standalone evaluation for a trained MMDys checkpoint.

Loads a saved best.pt (or any .pt checkpoint), runs inference on one or more
splits (test by default), and writes metrics JSON to the checkpoint's output dir
(or to --output-dir if specified).

Usage:
  python scripts/evaluate.py \
    --checkpoint outputs/msdm_video_phase_v3_t16_40gb/seed_17/best.pt \
    --test-split /path/to/msdm_baseline_splits_v2/msdm_test.json \
    [--dev-split  /path/to/msdm_baseline_splits_v2/msdm_dev.json] \
    [--output-dir /path/to/results/]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# Repo root on sys.path so imports match train.py
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data import MSDMPhaseCollator, MSDMPhaseDataset
from models import PhaseVideoMAEModel
from trainers.metrics import compute_eval_metrics
from losses import compute_total_loss
from utils.io import save_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Standalone evaluation for MMDys")
    p.add_argument(
        "--checkpoint", type=Path, required=True,
        help="Path to a .pt checkpoint file (e.g. best.pt)",
    )
    p.add_argument(
        "--test-split", type=Path, default=None,
        help="Override test split JSON path",
    )
    p.add_argument(
        "--dev-split", type=Path, default=None,
        help="Override dev split JSON path (optional; skipped if not given)",
    )
    p.add_argument(
        "--output-dir", type=Path, default=None,
        help="Where to write eval JSON files (default: same dir as checkpoint)",
    )
    p.add_argument(
        "--device", type=str, default=None,
        help="cuda / cpu (default: auto-detect)",
    )
    p.add_argument(
        "--batch-size", type=int, default=8,
        help="Eval batch size (default: 8)",
    )
    p.add_argument(
        "--suffix", type=str, default="",
        help="Optional suffix appended to output filenames, e.g. '_v2'",
    )
    return p.parse_args()


def _build_dataset(data_cfg: dict, split_json: Path) -> MSDMPhaseDataset:
    return MSDMPhaseDataset(
        split_json_path=str(split_json),
        video_root=data_cfg["video_root"],
        flow_feature_root=data_cfg["flow_feature_root"],
        split_name=split_json.stem,
        pre_seconds=float(data_cfg.get("pre_onset_seconds", 1.0)),
        post_seconds=float(data_cfg.get("post_offset_seconds", 0.3)),
        fallback_pre_seconds=float(data_cfg.get("fallback_pre_seconds", 0.4)),
        fallback_post_seconds=float(data_cfg.get("fallback_post_seconds", 0.2)),
        min_speech_frames=int(data_cfg.get("min_speech_frames", 3)),
        pre_target_frames=int(data_cfg.get("pre_target_frames", 16)),
        speech_target_frames=int(data_cfg.get("speech_target_frames", 32)),
        speech_pad_to_target=bool(data_cfg.get("speech_pad_to_target", False)),
        post_target_frames=int(data_cfg.get("post_target_frames", 5)),
        resize_height=int(data_cfg.get("resize_height", 224)),
        resize_half_width=int(data_cfg.get("resize_half_width", 224)),
        max_video_seconds=float(data_cfg.get("max_video_seconds", 5.0)),
        use_flow_descriptors=bool(data_cfg.get("use_flow_descriptors", True)),
        use_flow_as_channel=bool(data_cfg.get("use_flow_as_channel", False)),
        require_flow_ok=bool(data_cfg.get("require_flow_ok", False)),
        descriptor_cache_dir=data_cfg.get("descriptor_cache_dir", None),
        descriptor_cache_write=False,
        descriptor_cache_miss_policy=str(data_cfg.get("descriptor_cache_miss_policy", "compute")),
        descriptor_eps=float(data_cfg.get("descriptor_eps", 1e-6)),
        deform_descriptor_cache_dir=data_cfg.get("deform_descriptor_cache_dir", None),
        enable_aug=False,
        aug_hflip_prob=0.0,
        aug_brightness_delta=0.0,
        aug_contrast_delta=0.0,
        aug_phase_jitter_frames=0,
        missing_policy=str(data_cfg.get("missing_policy", "skip")),
        runtime_missing_policy=str(data_cfg.get("runtime_missing_policy", "skip")),
        runtime_max_retries=int(data_cfg.get("runtime_max_retries", 8)),
        min_samples_per_split=1,
    )


@torch.no_grad()
def evaluate_split(
    model: torch.nn.Module,
    loader: DataLoader,
    split_name: str,
    device: torch.device,
    cfg: dict,
) -> dict:
    model.eval()
    labels, probs, preds, speaker_ids, tasks, similarities, losses = [], [], [], [], [], [], []

    for batch in loader:
        batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        outputs = model(
            pre_left_video=batch["pre_left_video"],
            pre_right_video=batch["pre_right_video"],
            speech_left_video=batch["speech_left_video"],
            speech_right_video=batch["speech_right_video"],
            post_left_video=batch["post_left_video"],
            post_right_video=batch["post_right_video"],
            phase_side_flow_features=batch["phase_side_flow_features"],
            phase_asym_flow_features=batch["phase_asym_flow_features"],
            recovery_ratio=batch["recovery_ratio"],
            speech_num_frames=batch.get("speech_num_frames"),
        )
        loss_dict = compute_total_loss(outputs, batch, cfg)
        losses.append(float(loss_dict["total_loss"].item()))

        logits = outputs["severity_logits"]
        p = torch.softmax(logits, dim=-1)
        labels.extend(batch["severity_ids"].cpu().tolist())
        probs.extend(p.cpu().numpy())
        preds.extend(logits.argmax(dim=-1).cpu().tolist())
        similarities.extend(outputs["similarity"].cpu().tolist())
        speaker_ids.extend(batch["speaker_ids"])
        tasks.extend(batch["tasks"])

    metrics = compute_eval_metrics(
        labels=np.asarray(labels, dtype=np.int64),
        probs=np.asarray(probs, dtype=np.float64),
        preds=np.asarray(preds, dtype=np.int64),
        speaker_ids=speaker_ids,
        tasks=tasks,
        similarities=np.asarray(similarities, dtype=np.float64),
        ignore_labels=None,
    )
    metrics["split"] = split_name
    metrics["eval_loss"] = float(np.mean(losses)) if losses else 0.0
    return metrics


def main() -> None:
    args = parse_args()

    ckpt_path = args.checkpoint.resolve()
    if not ckpt_path.exists():
        print(f"[error] Checkpoint not found: {ckpt_path}", file=sys.stderr)
        sys.exit(1)

    out_dir = (args.output_dir or ckpt_path.parent).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    print(f"[init] checkpoint={ckpt_path}")
    print(f"[init] device={device}  output_dir={out_dir}")

    # ---- Load checkpoint -------------------------------------------------------
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = payload["config"]
    epoch = payload.get("epoch", "?")
    best_metric = payload.get("best_metric", "?")
    print(f"[ckpt] epoch={epoch}  best_metric={best_metric}")

    # ---- Override split paths if provided -------------------------------------
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]

    splits_to_eval: list[tuple[str, Path]] = []
    if args.dev_split is not None:
        splits_to_eval.append(("dev", args.dev_split.resolve()))
    if args.test_split is not None:
        splits_to_eval.append(("test", args.test_split.resolve()))
    if not splits_to_eval:
        # Default: use paths stored in the checkpoint config
        splits_to_eval.append(("test", Path(data_cfg["test_split_json"])))

    # ---- Build model -----------------------------------------------------------
    model = PhaseVideoMAEModel(
        hf_backbone_name=str(model_cfg.get("hf_backbone_name", "MCG-NJU/videomae-small-finetuned-kinetics")),
        use_pretrained=bool(model_cfg.get("use_pretrained", True)),  # needed to get the right HF architecture
        videomae_num_frames=int(model_cfg.get("videomae_num_frames", 16)),
        image_size=int(model_cfg.get("image_size", data_cfg.get("resize_height", 224))),
        interpolate_pos_emb=bool(model_cfg.get("interpolate_pos_emb", False)),
        in_channels=int(model_cfg.get("in_channels", 3)),
        flow_input_dim=int(model_cfg.get("flow_input_dim", 14)),
        flow_embed_dim=int(model_cfg.get("flow_embed_dim", 64)),
        use_flow_descriptors=bool(model_cfg.get("use_flow_descriptors", True)),
        phase_token_dim=int(model_cfg.get("phase_token_dim", 256)),
        sample_embed_dim=int(model_cfg.get("sample_embed_dim", 256)),
        classifier_hidden_dim=int(model_cfg.get("classifier_hidden_dim", 256)),
        consensus_hidden_dim=int(model_cfg.get("consensus_hidden_dim", 128)),
        num_agg_heads=int(model_cfg.get("num_agg_heads", 4)),
        dropout=float(model_cfg.get("dropout", 0.1)),
    )
    model.load_state_dict(payload["model"], strict=True)
    model = model.to(device).eval()
    print(f"[model] weights loaded  params={sum(p.numel() for p in model.parameters()):,}")

    collator = MSDMPhaseCollator()

    # ---- Evaluate each split ---------------------------------------------------
    for split_name, split_json in splits_to_eval:
        print(f"\n[eval] split={split_name}  json={split_json}")
        ds = _build_dataset(data_cfg, split_json)
        print(f"[eval] {len(ds)} samples loaded")

        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collator,
            drop_last=False,
        )

        metrics = evaluate_split(model, loader, split_name, device, cfg)

        # Print key metrics
        print(
            f"[result] f1_final_4cls={metrics.get('f1_final_4cls', 0):.4f}  "
            f"f1_sample_macro_4cls={metrics.get('f1_sample_macro_4cls', 0):.4f}  "
            f"f1_subject_macro_4cls={metrics.get('f1_subject_macro_4cls', 0):.4f}  "
            f"qwk_subject={metrics.get('qwk_subject', 0):.4f}  "
            f"loss={metrics.get('eval_loss', 0):.4f}"
        )

        out_file = out_dir / f"{split_name}_metrics{args.suffix}.json"
        save_json(out_file, metrics)
        print(f"[saved] {out_file}")


if __name__ == "__main__":
    main()
