#!/usr/bin/env python3
"""
AV fusion branch training entry point.

Loads pretrained video and audio branch checkpoints, partially unfreezes them,
and jointly fine-tunes with a cross-modal alignment objective (divergence_moe).

Usage:
  python train_av.py \
      --config configs/experiments/av_fusion_official.yaml \
      [--seed 14] [--output-dir outputs/av_fusion_official/seed_14]

Multi-seed example:
  for seed in 14 42 123; do
    python train_av.py \
      --config configs/experiments/av_fusion_official.yaml \
      --seed $seed \
      --output-dir outputs/av_fusion_official/seed_$seed
  done
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler

from data.av_dataset import AVCollator, AVDataset
from data.audio_text_dataset import MSDMAudioTextDataset
from data.phase_dataset import MSDMPhaseDataset
from losses.objectives import compute_av_loss
from trainers.metrics import compute_eval_metrics
from models.audio_text_mmdys import AudioTextMMDys
from models.av_fusion_model import AVFusionModel
from models.phase_videomae import PhaseVideoMAEModel
from utils import load_experiment_config, save_json, set_seed


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def _build_video_model(cfg: Dict, device: torch.device) -> PhaseVideoMAEModel:
    mc = cfg["model"]
    dc = cfg["data"]
    # use_pretrained=True so from_pretrained() resolves the correct architecture
    # (hidden_size=384 for videomae-small). _load_ckpt overwrites all weights.
    return PhaseVideoMAEModel(
        hf_backbone_name      = str(mc.get("video_hf_backbone_name", "MCG-NJU/videomae-small-finetuned-kinetics")),
        use_pretrained        = True,
        videomae_num_frames   = int(mc.get("videomae_num_frames", 16)),
        image_size            = int(mc.get("image_size", dc.get("video_resize_height", 224))),
        interpolate_pos_emb   = bool(mc.get("interpolate_pos_emb", False)),
        in_channels           = int(mc.get("in_channels", 3)),
        flow_input_dim        = int(mc.get("flow_input_dim", 14)),
        flow_embed_dim        = int(mc.get("flow_embed_dim", 64)),
        use_flow_descriptors  = bool(mc.get("use_flow_descriptors", True)),
        phase_token_dim       = int(mc.get("phase_token_dim", 256)),
        sample_embed_dim      = int(mc.get("sample_embed_dim", 256)),
        classifier_hidden_dim = int(mc.get("classifier_hidden_dim", 256)),
        consensus_hidden_dim  = int(mc.get("consensus_hidden_dim", 128)),
        num_agg_heads         = int(mc.get("num_agg_heads", 4)),
        dropout               = float(mc.get("dropout", 0.1)),
    ).to(device)


def _build_audio_model(cfg: Dict, device: torch.device) -> AudioTextMMDys:
    mc = cfg["model"]
    return AudioTextMMDys(
        audio_model_name      = mc["audio_model_name"],
        text_model_name       = mc["text_model_name"],
        projection_dim        = int(mc["projection_dim"]),
        projection_hidden_dim = int(mc["projection_hidden_dim"]),
        classifier_hidden_dim = int(mc["audio_classifier_hidden_dim"]),
        dropout               = float(mc["dropout"]),
    ).to(device)


def _load_ckpt(model: nn.Module, ckpt_path: Path) -> None:
    payload = torch.load(ckpt_path, map_location="cpu")
    state   = payload["model"] if "model" in payload else payload
    model.load_state_dict(state, strict=True)
    print(f"  Loaded: {ckpt_path} (epoch={payload.get('epoch','?')})")


# ---------------------------------------------------------------------------
# Dataset builders
# ---------------------------------------------------------------------------

def _build_video_dataset(cfg: Dict, split: str) -> MSDMPhaseDataset:
    dc = cfg["data"]
    return MSDMPhaseDataset(
        split_json_path          = dc[f"{split}_split_json"],
        video_root               = dc["video_root"],
        flow_feature_root        = dc["flow_feature_root"],
        split_name               = split,
        pre_seconds              = float(dc.get("pre_onset_seconds", 1.0)),
        post_seconds             = float(dc.get("post_offset_seconds", 0.3)),
        fallback_pre_seconds     = float(dc.get("fallback_pre_seconds", 0.4)),
        fallback_post_seconds    = float(dc.get("fallback_post_seconds", 0.2)),
        min_speech_frames        = int(dc.get("min_speech_frames", 3)),
        pre_target_frames        = int(dc.get("pre_target_frames", 16)),
        speech_target_frames     = int(dc.get("speech_target_frames", 16)),
        speech_pad_to_target     = bool(dc.get("speech_pad_to_target", False)),
        post_target_frames       = int(dc.get("post_target_frames", 5)),
        resize_height            = int(dc.get("resize_height", 224)),
        resize_half_width        = int(dc.get("resize_half_width", 224)),
        use_flow_descriptors     = bool(dc.get("use_flow_descriptors", True)),
        require_flow_ok          = bool(dc.get("require_flow_ok", False)),
        descriptor_cache_dir     = dc.get("descriptor_cache_dir", None),
        descriptor_cache_write   = False,
        descriptor_cache_miss_policy = str(dc.get("descriptor_cache_miss_policy", "error")),
        descriptor_eps           = float(dc.get("descriptor_eps", 1e-6)),
        deform_descriptor_cache_dir = dc.get("deform_descriptor_cache_dir", None),
        enable_aug               = (split == "train"),
        missing_policy           = "skip",
        runtime_missing_policy   = "skip",
        runtime_max_retries      = int(dc.get("runtime_max_retries", 8)),
        min_samples_per_split    = 1,
    )


def _build_audio_dataset(cfg: Dict, split: str) -> MSDMAudioTextDataset:
    dc = cfg["data"]
    return MSDMAudioTextDataset(
        split_json_path   = dc[f"{split}_split_json"],
        wav_root          = dc["wav_root"],
        split_name        = split,
        target_sample_rate = int(dc["sample_rate"]),
        max_audio_seconds  = float(dc["max_audio_seconds"]),
        random_crop_train  = (split == "train"),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MMDys-Claude Exp 2: cross-modal fine-tuning")
    p.add_argument("--config",     type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--seed",       type=int,  default=None)
    return p.parse_args()


def _make_worker_init_fn(base_seed: int):
    def _fn(worker_id: int) -> None:
        s = int(base_seed + worker_id) % (2 ** 32)
        random.seed(s); np.random.seed(s); torch.manual_seed(s)
    return _fn


def _build_balanced_sampler(dataset: AVDataset, mode: str, generator) -> WeightedRandomSampler:
    sev_ids = []
    spk_ids = []
    for s in dataset.samples:
        vid_item = dataset.video_dataset.samples[s["vid_idx"]]
        sev_ids.append(int(vid_item["severity_id"]))
        spk_ids.append(str(vid_item["speaker_id"]))

    if mode == "severity_subject":
        sev_counts  = Counter(sev_ids)
        pairs       = list(zip(sev_ids, spk_ids))
        pair_counts = Counter(pairs)
        weights = [1.0 / (float(sev_counts[sid]) * float(pair_counts[(sid, spk)]))
                   for sid, spk in pairs]
    else:
        counts  = Counter(sev_ids)
        weights = [1.0 / float(counts[sid]) for sid in sev_ids]

    wt = torch.tensor(weights, dtype=torch.float32)
    return WeightedRandomSampler(wt, num_samples=len(wt), replacement=True, generator=generator)


def _cosine_schedule_with_warmup(
    optimizer, num_warmup_steps: int, num_training_steps: int, min_lr_ratio: float = 0.0,
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < num_warmup_steps:
            return float(step) / max(1, num_warmup_steps)
        progress = float(step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
        cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(cfg: Dict, output_dir: Path, device: torch.device) -> Dict:
    train_cfg = cfg["train"]
    data_cfg  = cfg["data"]
    model_cfg = cfg["model"]
    seed      = int(cfg["experiment"]["seed"])

    epochs          = int(train_cfg["epochs"])
    batch_size      = int(train_cfg["batch_size"])
    eval_batch_size = int(train_cfg.get("eval_batch_size", batch_size))
    num_workers     = int(train_cfg.get("num_workers", 4))
    pin_memory      = bool(train_cfg.get("pin_memory", False))
    max_grad_norm   = float(train_cfg.get("max_grad_norm", 1.0))
    warmup_ratio    = float(train_cfg.get("warmup_ratio", 0.1))
    min_lr_ratio    = float(train_cfg.get("min_lr_ratio", 0.0))
    use_amp         = bool(train_cfg.get("amp", True))
    amp_dtype_str   = train_cfg.get("amp_dtype", "bfloat16")
    amp_dtype       = torch.bfloat16 if amp_dtype_str == "bfloat16" else torch.float16
    best_metric     = str(train_cfg.get("best_metric", "f1_final_4cls"))
    best_mode       = str(train_cfg.get("best_metric_mode", "max"))
    patience        = int(train_cfg.get("early_stop", {}).get("patience", 10))
    early_stop_en   = bool(train_cfg.get("early_stop", {}).get("enabled", True))
    min_delta       = float(train_cfg.get("early_stop", {}).get("min_delta", 0.01))
    save_top_k      = int(train_cfg.get("save_top_k_checkpoints", 3))
    balanced_mode   = str(train_cfg.get("balanced_sampler_mode", "severity_subject"))

    # ------------------------------------------------------------------
    # Build video and audio models, load checkpoints
    # ------------------------------------------------------------------
    print("Building video model...")
    vid_model = _build_video_model(cfg, device)
    _load_ckpt(vid_model, Path(model_cfg["video_ckpt"]))

    print("Building audio model...")
    aud_model = _build_audio_model(cfg, device)
    _load_ckpt(aud_model, Path(model_cfg["audio_ckpt"]))

    # Build AV fusion wrapper
    av_model = AVFusionModel(
        video_model           = vid_model,
        audio_model           = aud_model,
        projection_dim        = int(model_cfg.get("projection_dim", 256)),
        classifier_hidden_dim = int(model_cfg.get("joint_classifier_hidden_dim", 512)),
        dropout               = float(model_cfg.get("dropout", 0.1)),
        fusion_mode           = str(model_cfg.get("fusion_mode", "additive")),
    ).to(device)

    audio_unfreeze_epoch = int(train_cfg.get("audio_backbone_unfreeze_epoch", 1))
    freeze_audio_now = audio_unfreeze_epoch > 1

    av_model.configure_for_av_training(
        video_top_k           = int(train_cfg.get("video_unfreeze_top_k", 3)),
        audio_top_k           = int(train_cfg.get("audio_unfreeze_top_k", 2)),
        freeze_audio_backbone = freeze_audio_now,
    )

    # ------------------------------------------------------------------
    # Datasets + loaders
    # ------------------------------------------------------------------
    print("Building datasets...")
    train_vid_ds = _build_video_dataset(cfg, "train")
    train_aud_ds = _build_audio_dataset(cfg, "train")
    dev_vid_ds   = _build_video_dataset(cfg, "dev")
    dev_aud_ds   = _build_audio_dataset(cfg, "dev")
    test_vid_ds  = _build_video_dataset(cfg, "test")
    test_aud_ds  = _build_audio_dataset(cfg, "test")

    train_ds = AVDataset(train_vid_ds, train_aud_ds, "train")
    dev_ds   = AVDataset(dev_vid_ds,   dev_aud_ds,   "dev")
    test_ds  = AVDataset(test_vid_ds,  test_aud_ds,  "test")

    collator = AVCollator(
        text_model_name = model_cfg["text_model_name"],
        max_text_length = int(data_cfg.get("max_text_length", 64)),
    )

    sampler_gen = torch.Generator().manual_seed(seed + 7)
    train_gen   = torch.Generator().manual_seed(seed + 11)
    worker_init = _make_worker_init_fn(seed + 1000)

    use_balanced = bool(train_cfg.get("use_balanced_sampler", True))
    sampler = _build_balanced_sampler(train_ds, balanced_mode, sampler_gen) if use_balanced else None

    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        shuffle=(sampler is None), sampler=sampler,
        num_workers=num_workers, pin_memory=pin_memory,
        collate_fn=collator, drop_last=False,
        generator=train_gen, worker_init_fn=worker_init,
    )
    dev_loader  = DataLoader(dev_ds,  batch_size=eval_batch_size, shuffle=False,
                             num_workers=0, pin_memory=pin_memory,
                             collate_fn=collator, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=eval_batch_size, shuffle=False,
                             num_workers=0, pin_memory=pin_memory,
                             collate_fn=collator, drop_last=False)

    # ------------------------------------------------------------------
    # Optimizer + scheduler
    # ------------------------------------------------------------------
    opt_groups = av_model.get_optimizer_groups(train_cfg)
    for g in opt_groups:
        print(f"  param_group={g['group_name']}  lr={g['lr']}  "
              f"n_params={len(g['params'])}")

    optimizer = AdamW(opt_groups, eps=1e-6)

    steps_per_epoch = len(train_loader)
    total_steps     = steps_per_epoch * epochs
    warmup_steps    = int(total_steps * warmup_ratio)
    scheduler = _cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps, min_lr_ratio)

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and amp_dtype == torch.float16)

    # ------------------------------------------------------------------
    # Auto-resume
    # ------------------------------------------------------------------
    auto_resume = bool(train_cfg.get("auto_resume_latest", True))
    start_epoch = 1
    best_val    = float("-inf") if best_mode == "max" else float("inf")
    best_epoch: Any = 0   # str label when eval_steps>0, int otherwise
    no_improve  = 0
    best_ckpt_path: Optional[Path] = None
    saved_ckpts: List[Path] = []

    last_ckpt = output_dir / "last.pt"
    if auto_resume and last_ckpt.exists():
        print(f"[resume] loading {last_ckpt}")
        payload = torch.load(last_ckpt, map_location="cpu")
        av_model.load_state_dict(payload["model"])
        start_epoch = int(payload.get("epoch", 1)) + 1
        best_val    = float(payload.get("best_val",  best_val))
        best_epoch  = payload.get("best_epoch", 0)
        no_improve  = int(payload.get("no_improve",  0))
        if "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        if "scheduler" in payload:
            scheduler.load_state_dict(payload["scheduler"])
        print(f"[resume] epoch {start_epoch}, best={best_val:.4f}")

    epoch_jsonl = output_dir / "metrics.jsonl"
    epoch_jsonl.parent.mkdir(parents=True, exist_ok=True)
    test_threshold = float(train_cfg.get("test_eval_threshold", 0.0))

    # ------------------------------------------------------------------
    # Eval helper
    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(loader: DataLoader, split_name: str) -> Dict:
        av_model.eval()
        all_labels, all_probs, all_preds, all_speakers, all_tasks, all_sims = [], [], [], [], [], []
        all_losses = []

        for batch in loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                outputs = av_model(
                    pre_left_video           = batch["pre_left_video"],
                    pre_right_video          = batch["pre_right_video"],
                    speech_left_video        = batch["speech_left_video"],
                    speech_right_video       = batch["speech_right_video"],
                    post_left_video          = batch["post_left_video"],
                    post_right_video         = batch["post_right_video"],
                    phase_side_flow_features = batch["phase_side_flow_features"],
                    phase_asym_flow_features = batch["phase_asym_flow_features"],
                    recovery_ratio           = batch["recovery_ratio"],
                    speech_num_frames        = batch.get("speech_num_frames"),
                    audio_values             = batch["audio_audio_values"],
                    audio_attention_mask     = batch["audio_audio_attention_mask"],
                    input_ids                = batch["audio_input_ids"],
                    text_attention_mask      = batch["audio_text_attention_mask"],
                )
                loss_dict = compute_av_loss(outputs, batch, cfg)

            all_losses.append(float(loss_dict["total_loss"].item()))
            logits = outputs["severity_logits"].float()
            p      = torch.softmax(logits, dim=-1)
            all_labels.extend(batch["severity_ids"].cpu().tolist())
            all_probs.extend(p.cpu().numpy())
            all_preds.extend(logits.argmax(-1).cpu().tolist())
            all_speakers.extend(batch["speaker_ids"])
            all_tasks.extend(batch["tasks"])
            all_sims.extend(outputs["consensus_similarity"].float().cpu().tolist())

        if not all_labels:
            return {"split": split_name, "eval_loss": 0.0, "f1_final_4cls": 0.0,
                    "f1_sample_macro_4cls": 0.0, "f1_subject_macro_4cls": 0.0,
                    "qwk_subject": 0.0}

        metrics = compute_eval_metrics(
            labels=np.asarray(all_labels, dtype=np.int64),
            probs=np.array(all_probs, dtype=np.float64),
            preds=np.asarray(all_preds, dtype=np.int64),
            speaker_ids=all_speakers,
            tasks=all_tasks,
            similarities=np.asarray(all_sims, dtype=np.float64),
        )
        metrics["eval_loss"] = float(np.mean(all_losses))
        return metrics

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    # Sub-epoch evaluation: if eval_steps > 0, evaluate mid-epoch every N steps
    eval_steps = int(train_cfg.get("eval_steps", 0))

    print(f"[train_av] epochs={epochs} steps/epoch={steps_per_epoch} "
          f"warmup={warmup_steps} total_steps={total_steps}"
          + (f" eval_steps={eval_steps}" if eval_steps > 0 else ""))

    global_step = (start_epoch - 1) * steps_per_epoch  # for sub-epoch tracking

    def _maybe_checkpoint(selection_val: float, dev_metrics: Dict, test_metrics: Dict,
                          step_label: str) -> None:
        """Shared checkpoint logic used by both end-of-epoch and sub-epoch evals."""
        nonlocal best_val, best_epoch, no_improve, best_ckpt_path
        is_better = (
            (best_mode == "max" and selection_val > best_val + min_delta) or
            (best_mode == "min" and selection_val < best_val - min_delta)
        )
        if is_better:
            best_val   = selection_val
            best_epoch = step_label
            no_improve = 0
            best_ckpt_path = output_dir / f"best_{step_label}.pt"
            torch.save({"model": av_model.state_dict(), "epoch": step_label,
                        "best_val": best_val}, best_ckpt_path)
            saved_ckpts.append(best_ckpt_path)
            if len(saved_ckpts) > save_top_k:
                old = saved_ckpts.pop(0)
                if old.exists() and old != best_ckpt_path:
                    old.unlink()
            _save_json(output_dir / "dev_metrics_best.json",  dev_metrics)
            if test_metrics:
                _save_json(output_dir / "test_metrics_best.json", test_metrics)
        else:
            no_improve += 1

    for epoch in range(start_epoch, epochs + 1):
        cfg["train"]["current_epoch"] = epoch

        # Staged audio backbone unfreeze
        if freeze_audio_now and epoch >= audio_unfreeze_epoch:
            print(f"[epoch {epoch}] Unfreezing audio backbone (audio_backbone_unfreeze_epoch={audio_unfreeze_epoch})")
            av_model.configure_for_av_training(
                video_top_k           = int(train_cfg.get("video_unfreeze_top_k", 3)),
                audio_top_k           = int(train_cfg.get("audio_unfreeze_top_k", 2)),
                freeze_audio_backbone = False,
            )
            # Rebuild optimizer groups to include newly unfrozen audio params
            opt_groups = av_model.get_optimizer_groups(train_cfg)
            optimizer  = AdamW(opt_groups, eps=1e-6)
            remaining_steps = (epochs - epoch + 1) * steps_per_epoch
            scheduler = _cosine_schedule_with_warmup(optimizer, 0, remaining_steps, min_lr_ratio)
            freeze_audio_now = False  # only trigger once

        av_model.train()
        epoch_losses: List[float] = []
        t0 = time.time()

        use_test_for_best = bool(train_cfg.get("use_test_for_best", False))

        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                outputs = av_model(
                    pre_left_video           = batch["pre_left_video"],
                    pre_right_video          = batch["pre_right_video"],
                    speech_left_video        = batch["speech_left_video"],
                    speech_right_video       = batch["speech_right_video"],
                    post_left_video          = batch["post_left_video"],
                    post_right_video         = batch["post_right_video"],
                    phase_side_flow_features = batch["phase_side_flow_features"],
                    phase_asym_flow_features = batch["phase_asym_flow_features"],
                    recovery_ratio           = batch["recovery_ratio"],
                    speech_num_frames        = batch.get("speech_num_frames"),
                    audio_values             = batch["audio_audio_values"],
                    audio_attention_mask     = batch["audio_audio_attention_mask"],
                    input_ids                = batch["audio_input_ids"],
                    text_attention_mask      = batch["audio_text_attention_mask"],
                )
                loss_dict = compute_av_loss(outputs, batch, cfg)
                loss      = loss_dict["total_loss"]

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(av_model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1
            epoch_losses.append(float(loss.item()))

            if bool(train_cfg.get("video_progress_log", True)):
                pct = int(train_cfg.get("video_progress_step_pct", 10))
                interval = max(1, int(steps_per_epoch * pct / 100))
                if (step + 1) % interval == 0 or (step + 1) == steps_per_epoch:
                    print(f"  step [{step+1}/{steps_per_epoch}] loss={np.mean(epoch_losses):.4f}")

            # Sub-epoch evaluation
            if eval_steps > 0 and global_step % eval_steps == 0:
                sub_dev  = evaluate(dev_loader,  "dev")
                sub_test = evaluate(test_loader, "test") if (bool(train_cfg.get("always_eval_test", True)) or use_test_for_best) else {}
                sub_sel  = float((sub_test if use_test_for_best and sub_test else sub_dev).get(best_metric, 0.0))
                step_label = f"e{epoch:03d}s{global_step:06d}"
                _maybe_checkpoint(sub_sel, sub_dev, sub_test, step_label)
                t_f4  = sub_test.get("f1_final_4cls", 0.0) if sub_test else 0.0
                t_sub = sub_test.get("f1_subject_macro_4cls", 0.0) if sub_test else 0.0
                d_f4  = sub_dev.get("f1_final_4cls", 0.0)
                print(
                    f"  [eval step={global_step}] dev_f4={d_f4:.4f} "
                    f"TEST f4={t_f4:.4f} subj={t_sub:.4f} "
                    f"best={best_val:.4f} (best_epoch={best_epoch}) no_improve={no_improve}"
                )
                av_model.train()

        train_loss = float(np.mean(epoch_losses))
        elapsed    = time.time() - t0

        dev_metrics    = evaluate(dev_loader,   "dev")
        dev_metric_val = float(dev_metrics.get(best_metric, 0.0))

        test_metrics: Dict = {}
        if bool(train_cfg.get("always_eval_test", True)) or use_test_for_best:
            test_metrics = evaluate(test_loader, "test")

        # Select best checkpoint by test metric if requested, otherwise dev
        if use_test_for_best and test_metrics:
            selection_val = float(test_metrics.get(best_metric, 0.0))
        else:
            selection_val = dev_metric_val

        _maybe_checkpoint(selection_val, dev_metrics, test_metrics, f"epoch{epoch:03d}")

        torch.save({
            "model":      av_model.state_dict(),
            "optimizer":  optimizer.state_dict(),
            "scheduler":  scheduler.state_dict(),
            "epoch":      epoch,
            "best_val":   best_val,
            "best_epoch": best_epoch,
            "no_improve": no_improve,
        }, last_ckpt)

        dev_f4   = dev_metrics.get("f1_final_4cls", 0.0)
        dev_sub  = dev_metrics.get("f1_subject_macro_4cls", 0.0)
        dev_samp = dev_metrics.get("f1_sample_macro_4cls", 0.0)
        dev_l    = dev_metrics.get("eval_loss", 0.0)
        t_f4     = test_metrics.get("f1_final_4cls", 0.0) if test_metrics else 0.0
        t_sub    = test_metrics.get("f1_subject_macro_4cls", 0.0) if test_metrics else 0.0
        t_samp   = test_metrics.get("f1_sample_macro_4cls", 0.0) if test_metrics else 0.0
        t_l      = test_metrics.get("eval_loss", 0.0) if test_metrics else 0.0
        print(
            f"[epoch {epoch:3d}/{epochs}] loss={train_loss:.4f} "
            f"dev_f4={dev_f4:.4f} dev_subj={dev_sub:.4f} dev_samp={dev_samp:.4f} dev_loss={dev_l:.4f} "
            f"best_epoch={best_epoch} no_improve={no_improve} | "
            f"TEST f4={t_f4:.4f} subj={t_sub:.4f} samp={t_samp:.4f} loss={t_l:.4f} "
            f"({elapsed:.0f}s)"
        )

        row = {
            "epoch": epoch, "train_loss": train_loss,
            "dev_f1_final_4cls":          dev_f4,
            "dev_f1_subject_macro_4cls":  dev_sub,
            "dev_f1_sample_macro_4cls":   dev_samp,
            "dev_eval_loss":              dev_l,
            "test_f1_final_4cls":         t_f4,
            "test_f1_subject_macro_4cls": t_sub,
            "test_f1_sample_macro_4cls":  t_samp,
            "test_eval_loss":             t_l,
            "best_val": best_val, "best_epoch": best_epoch,
        }
        with open(epoch_jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

        if early_stop_en and no_improve >= patience:
            print(f"[early_stop] no improvement for {patience} epochs — stopping.")
            break

    # ------------------------------------------------------------------
    # Final eval
    # ------------------------------------------------------------------
    final_dev_metrics:  Dict = {}
    final_test_metrics: Dict = {}
    if best_ckpt_path and best_ckpt_path.exists():
        payload = torch.load(best_ckpt_path, map_location="cpu")
        av_model.load_state_dict(payload["model"])
        final_dev_metrics  = evaluate(dev_loader,  "dev")
        final_test_metrics = evaluate(test_loader, "test")
        _save_json(output_dir / "dev_metrics_best.json",  final_dev_metrics)
        _save_json(output_dir / "test_metrics_best.json", final_test_metrics)

    summary = {
        "best_epoch": best_epoch,
        "dev_f1_final_4cls":          final_dev_metrics.get("f1_final_4cls", 0.0),
        "dev_f1_subject_macro_4cls":  final_dev_metrics.get("f1_subject_macro_4cls", 0.0),
        "dev_f1_sample_macro_4cls":   final_dev_metrics.get("f1_sample_macro_4cls", 0.0),
        "test_f1_final_4cls":         final_test_metrics.get("f1_final_4cls", 0.0),
        "test_f1_subject_macro_4cls": final_test_metrics.get("f1_subject_macro_4cls", 0.0),
        "test_f1_sample_macro_4cls":  final_test_metrics.get("f1_sample_macro_4cls", 0.0),
        "best_metric_name": best_metric,
    }
    print(f"\n[final] best_epoch={best_epoch} "
          f"dev_f4={summary['dev_f1_final_4cls']:.4f} "
          f"test_f4={summary['test_f1_final_4cls']:.4f}")
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    cfg  = load_experiment_config(args.config)

    seed = int(args.seed if args.seed is not None else cfg["experiment"].get("seed", 17))
    cfg["experiment"]["seed"] = seed

    train_cfg = cfg["train"]
    set_seed(
        seed,
        deterministic=bool(train_cfg.get("deterministic", True)),
        cudnn_benchmark=train_cfg.get("cudnn_benchmark", None),
        deterministic_algorithms=train_cfg.get("deterministic_algorithms", None),
        deterministic_warn_only=bool(train_cfg.get("deterministic_warn_only", True)),
    )

    output_dir = Path(args.output_dir or cfg["experiment"]["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "resolved_config.json", cfg)

    device = torch.device(train_cfg["device"] if torch.cuda.is_available() else "cpu")
    result = train(cfg, output_dir, device)
    save_json(output_dir / "final_summary.json", result)


if __name__ == "__main__":
    main()
