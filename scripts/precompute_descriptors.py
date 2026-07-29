#!/usr/bin/env python
"""
precompute_descriptors.py
=========================
Precompute 14-D per-phase per-hemiface flow descriptors from SEA-RAFT NPZ files.
Run this once before training to build descriptor_cache_v2/.

Output per sample — outputs/descriptor_cache_v2/{sample_id}.npz:
  phase_side_feats  [3, 2, 14]  float32   (phase × side × feature)
  phase_asym_raw    [3, 14]     float32   (left - right per phase)
  recovery_ratio    [2]         float32   (log1p(post_mean / pre_mean) per side)

14-D feature layout (indices 0–13):
  0  mag_mean          mean frame-level magnitude (raw pixel units at 96×96)
  1  mag_std           temporal std of per-frame mean magnitude
  2  mag_peak          peak per-frame mean magnitude
  3  mag_auc           trapezoid AUC normalised to [0, 1] x-axis
  4  smoothness        lag-1 autocorrelation of frame magnitudes ∈ [-1, 1]
  5  temporal_entropy  Shannon entropy of frame-magnitude histogram
  6  early_late_diff   (early - late) / (early + late + eps); speech only ∈ [-1, 1]
  7  slope             linear-fit slope on normalised time axis; speech only
  8  peak_latency      normalised argmax position ∈ [0, 1]; speech only
  9  vertical_bias     mean|v| / (mean_mag + eps)
 10  vert_peak_mag     peak per-frame mean |v|
 11  upper_lower_ratio log1p(upper_mag_mean / (lower_mag_mean + eps))
 12  spatial_entropy   entropy of time-mean 2-D magnitude map
 13  high_active_ratio fraction of frames with mag > 0.5 × peak

Normalisation: raw pixel-space values (no per-sample scaling).
Absolute movement scale is the primary dysarthria severity discriminator;
the model's LayerNorm at the flow projection input handles scale differences.

Usage:
  python scripts/precompute_descriptors.py --workers 8

Phase-bound logic (must match MSDMPhaseDataset._phase_bounds):
  ultra-short (< 0.7 s) → pre/post empty, speech = full video
  short (0.7–1.3 s)     → pre = 0.4 s, post = 0.3 s (from end)
  normal (≥ 1.3 s)      → pre = 1.0 s, post = 0.3 s (from end)
"""
from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

cv2.setNumThreads(0)

# ---------------------------------------------------------------------------
# Phase-bound constants — MUST match MSDMPhaseDataset._phase_bounds
# ---------------------------------------------------------------------------
_PRE_SEC          = 1.0
_POST_SEC         = 0.3
_FALLBACK_PRE_SEC = 0.4
_REDUCED_PRE_SEC  = 0.7   # secondary fallback when normal pre leaves < 4 speech frames
_ULTRA_SHORT      = _FALLBACK_PRE_SEC + _POST_SEC   # 0.7 s
_SHORT            = _PRE_SEC + _POST_SEC             # 1.3 s
_MIN_SPEECH_FRAMES_FOR_REDUCED_PRE = 4
_EPS              = 1e-6

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _probe_fps_frames(path: str) -> Tuple[float, int]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return 25.0, 0
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, max(0, n)


def _phase_bounds(total_frames: int, fps: float) -> Dict[str, slice]:
    fps = max(fps, _EPS)
    dur = total_frames / fps
    if dur < _ULTRA_SHORT:
        return {"pre": slice(0, 0), "speech": slice(0, total_frames), "post": slice(0, 0)}
    short   = dur < _SHORT
    pre_f   = int(round((_FALLBACK_PRE_SEC if short else _PRE_SEC) * fps))
    post_f  = int(round(_POST_SEC * fps))
    pre_end    = min(total_frames, pre_f)
    post_start = max(pre_end, total_frames - post_f)
    # Secondary fallback: if normal pre leaves fewer than 4 speech frames, reduce
    # pre to _REDUCED_PRE_SEC (0.7 s) to recover a usable speech window.
    if (post_start - pre_end) < _MIN_SPEECH_FRAMES_FOR_REDUCED_PRE and not short:
        pre_f   = int(round(_REDUCED_PRE_SEC * fps))
        pre_end = min(total_frames, pre_f)
        post_start = max(pre_end, total_frames - post_f)
    return {
        "pre":    slice(0, pre_end),
        "speech": slice(pre_end, post_start),
        "post":   slice(post_start, total_frames),
    }


def _align_flow(flow: np.ndarray, target: int) -> np.ndarray:
    """Align T_flow (= T_video - 1 by convention) to T_video frames."""
    n = flow.shape[0]
    if n == target:
        return flow
    if n == target - 1:
        return np.concatenate([flow[:1], flow], axis=0)
    idx = np.linspace(0, n - 1, target).round().astype(np.int64)
    return flow[idx]


def _histogram_entropy(values: np.ndarray, bins: int = 16, eps: float = _EPS) -> float:
    if values.size == 0:
        return 0.0
    lo = float(np.min(values))
    hi = float(np.max(values)) + eps
    hist, _ = np.histogram(values, bins=bins, range=(lo, hi), density=False)
    p = hist.astype(np.float64)
    z = p.sum()
    if z <= eps:
        return 0.0
    p /= z
    p = p[p > 0]
    return float(-(p * np.log(p + eps)).sum())


def _segment_14d(
    frame_mag:   np.ndarray,   # [T] per-frame mean magnitude
    frame_vert:  np.ndarray,   # [T] per-frame mean |v|
    upper_mean:  float,        # global mean magnitude in upper-half rows
    lower_mean:  float,        # global mean magnitude in lower-half rows
    spatial_map: np.ndarray,   # [96, 48] time-mean 2-D magnitude
    is_speech:   bool,
) -> np.ndarray:
    """Return 14-D descriptor for one (phase, hemiface) segment."""
    feat = np.zeros(14, dtype=np.float32)
    T = len(frame_mag)
    if T == 0:
        return feat

    x = frame_mag.astype(np.float64)
    mean_m = float(np.mean(x))
    std_m  = float(np.std(x))
    peak_m = float(np.max(x))

    # Normalised AUC: trapz on x-axis [0, 1]
    auc = float(np.trapezoid(x, dx=1.0 / max(1, T - 1))) if T > 1 else float(x[0])

    # Lag-1 autocorrelation
    if T > 2:
        xc    = x - mean_m
        denom = float(np.dot(xc, xc)) + _EPS
        smooth = float(np.clip(np.dot(xc[:-1], xc[1:]) / denom, -1.0, 1.0))
    else:
        smooth = 0.0

    t_ent = _histogram_entropy(x, bins=16)

    # Speech-only features
    early_late_diff = 0.0
    slope           = 0.0
    peak_latency    = 0.0
    if is_speech and T > 1:
        half  = max(1, T // 2)
        early = float(np.mean(x[:half]))
        late  = float(np.mean(x[half:])) if T - half > 0 else early
        early_late_diff = float(np.clip(
            (early - late) / (early + late + _EPS), -1.0, 1.0
        ))
        t_ax  = np.linspace(0.0, 1.0, T, dtype=np.float64)
        slope = float(np.polyfit(t_ax, x, deg=1)[0])
        peak_latency = float(int(np.argmax(x))) / max(1, T - 1)

    # Vertical-motion features
    v = frame_vert.astype(np.float64)
    v_mean = float(np.mean(v))
    v_peak = float(np.max(v))
    v_bias = v_mean / (mean_m + _EPS)

    # Upper / lower jaw ratio
    ul_ratio = math.log1p(upper_mean / (lower_mean + _EPS))

    # Spatial entropy of time-averaged 2-D flow map
    sp_ent = _histogram_entropy(spatial_map.ravel(), bins=32)

    # High-activation ratio
    high_ratio = float(np.mean(x > 0.5 * peak_m)) if peak_m > _EPS else 0.0

    feat[0]  = float(mean_m)
    feat[1]  = float(std_m)
    feat[2]  = float(peak_m)
    feat[3]  = float(auc)
    feat[4]  = float(smooth)
    feat[5]  = float(t_ent)
    feat[6]  = float(early_late_diff)
    feat[7]  = float(slope)
    feat[8]  = float(peak_latency)
    feat[9]  = float(v_bias)
    feat[10] = float(v_peak)
    feat[11] = float(ul_ratio)
    feat[12] = float(sp_ent)
    feat[13] = float(high_ratio)
    return feat


# ---------------------------------------------------------------------------
# Core computation (called in worker processes)
# ---------------------------------------------------------------------------

def _compute_sample(
    video_path: str,
    npz_path:   str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (phase_side_feats [3,2,14], phase_asym_raw [3,14], recovery_ratio [2]).
    Raises on any error — caller handles exception.
    """
    fps, total_frames = _probe_fps_frames(video_path)
    if total_frames <= 0:
        raise ValueError(f"Empty video: {video_path}")

    raw = np.load(npz_path, allow_pickle=True)
    flow = raw["flow"].astype(np.float32)          # [T_flow, 96, 96, 2]
    flow = _align_flow(flow, total_frames)         # [total_frames, 96, 96, 2]

    bounds = _phase_bounds(total_frames, fps)

    phase_order   = ["pre", "speech", "post"]
    col_slices    = [slice(0, 48), slice(48, 96)]  # left, right hemiface columns
    phase_side    = np.zeros((3, 2, 14), dtype=np.float32)

    for p_idx, phase in enumerate(phase_order):
        sl = bounds[phase]
        if sl.start >= sl.stop:
            continue                               # empty phase → keep zeros

        seg   = flow[sl]                           # [T, 96, 96, 2]
        T     = seg.shape[0]
        is_sp = (phase == "speech")

        for s_idx, c_sl in enumerate(col_slices):
            u_h   = seg[:, :, c_sl, 0]            # [T, 96, 48]
            v_h   = seg[:, :, c_sl, 1]            # [T, 96, 48]
            mag_h = np.sqrt(u_h ** 2 + v_h ** 2)  # [T, 96, 48]

            frame_mag  = mag_h.mean(axis=(1, 2))              # [T]
            frame_vert = np.abs(v_h).mean(axis=(1, 2))        # [T]
            upper_mean = float(mag_h[:, :48, :].mean())       # rows 0:48 (eye/cheek)
            lower_mean = float(mag_h[:, 48:, :].mean())       # rows 48:96 (jaw)
            spatial    = mag_h.mean(axis=0)                   # [96, 48]

            phase_side[p_idx, s_idx] = _segment_14d(
                frame_mag, frame_vert,
                upper_mean, lower_mean,
                spatial,
                is_speech=is_sp,
            )

    phase_asym = phase_side[:, 0, :] - phase_side[:, 1, :]   # [3, 14]

    # Recovery ratio: log1p(post_mag_mean / pre_mag_mean) per side
    recovery = np.array([
        math.log1p(float(phase_side[2, s, 0]) / max(float(phase_side[0, s, 0]), _EPS))
        for s in range(2)
    ], dtype=np.float32)

    return phase_side, phase_asym, recovery


# ---------------------------------------------------------------------------
# Worker (top-level for picklability)
# ---------------------------------------------------------------------------

def _worker(args: tuple) -> Tuple[str, str]:
    """(filename, video_path, npz_path, out_path, overwrite) → (filename, status)"""
    filename, video_path, npz_path, out_path, overwrite = args
    out = Path(out_path)
    try:
        if not overwrite and out.exists():
            return filename, "skip"
        psf, par, rr = _compute_sample(video_path, npz_path)
        np.savez_compressed(out, phase_side_feats=psf, phase_asym_raw=par, recovery_ratio=rr)
        return filename, "ok"
    except Exception as exc:
        return filename, f"error:{exc}"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _collect_samples(
    split_dir:  Path,
    video_root: Path,
    npz_root:   Path,
) -> List[Tuple[str, Path, Path]]:
    seen: set = set()
    out: List[Tuple[str, Path, Path]] = []
    for split in ("train", "dev", "test"):
        sj = split_dir / f"msdm_{split}.json"
        if not sj.exists():
            print(f"[warn] {sj} not found — skipping")
            continue
        for row in json.loads(sj.read_text(encoding="utf-8")):
            fn = row["filename"]
            if fn in seen:
                continue
            seen.add(fn)
            vp  = video_root / f"{fn}.avi"
            np_ = npz_root   / f"{fn}.npz"
            if vp.exists() and np_.exists():
                out.append((fn, vp, np_))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Precompute 14-D flow descriptor cache v2")
    p.add_argument("--video-root", type=Path, required=True,
        help="Root directory containing per-utterance .avi files")
    p.add_argument("--npz-root", type=Path, required=True,
        help="Root directory of SEA-RAFT output .npz files (one per utterance)")
    p.add_argument("--split-dir", type=Path, required=True,
        help="Directory containing msdm_train.json, msdm_dev.json, msdm_test.json")
    p.add_argument("--output-dir", type=Path, required=True,
        help="Where to write descriptor cache .npz files")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--overwrite", action="store_true",
        help="Recompute even if output file already exists")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    samples = _collect_samples(args.split_dir, args.video_root, args.npz_root)
    total   = len(samples)
    print(f"[info] {total} samples to process ({args.workers} workers, overwrite={args.overwrite})")

    tasks = [
        (fn, str(vp), str(np_), str(args.output_dir / f"{fn}.npz"), args.overwrite)
        for fn, vp, np_ in samples
    ]

    counts = {"ok": 0, "skip": 0, "error": 0}
    errors: List[Tuple[str, str]] = []
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_worker, t): t[0] for t in tasks}
        for i, fut in enumerate(as_completed(futures), 1):
            fn, status = fut.result()
            if status == "ok":
                counts["ok"] += 1
            elif status == "skip":
                counts["skip"] += 1
            else:
                counts["error"] += 1
                errors.append((fn, status))

            if i % 2000 == 0 or i == total:
                elapsed = time.time() - t0
                rate    = i / elapsed if elapsed > 0 else 0
                eta     = (total - i) / rate if rate > 0 else 0
                print(
                    f"  [{i}/{total}] "
                    f"ok={counts['ok']}  skip={counts['skip']}  "
                    f"error={counts['error']}  "
                    f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s"
                )

    elapsed = time.time() - t0
    print(
        f"\n[done] total={total}  ok={counts['ok']}  "
        f"skip={counts['skip']}  error={counts['error']}  "
        f"time={elapsed:.1f}s"
    )

    if errors:
        print(f"\n[errors] first 20:")
        for fn, msg in errors[:20]:
            print(f"  {fn}: {msg[6:]}")   # strip "error:" prefix

    summary = {
        "total":        total,
        "ok":           counts["ok"],
        "skip":         counts["skip"],
        "error":        counts["error"],
        "elapsed_sec":  round(elapsed, 1),
        "output_dir":   str(args.output_dir),
        "feature_dim":  14,
        "array_shapes": {
            "phase_side_feats": [3, 2, 14],
            "phase_asym_raw":   [3, 14],
            "recovery_ratio":   [2],
        },
        "feature_names": [
            "mag_mean", "mag_std", "mag_peak", "mag_auc",
            "smoothness", "temporal_entropy",
            "early_late_diff", "slope", "peak_latency",
            "vertical_bias", "vert_peak_mag",
            "upper_lower_ratio", "spatial_entropy", "high_active_ratio",
        ],
        "phase_order": ["pre", "speech", "post"],
        "side_order":  ["left", "right"],
        "failed_samples": [{"filename": fn, "error": msg[6:]} for fn, msg in errors],
    }
    out_json = args.output_dir / "precompute_summary.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[info] Summary → {out_json}")


if __name__ == "__main__":
    main()
