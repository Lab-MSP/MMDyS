#!/usr/bin/env python3
"""Batch SEA-RAFT feature extraction for MSDM AVI videos.

Outputs one NPZ per video with:
- flow:        [N, H, W, 2] optical flow vectors (dx, dy)
- info:        [N, H, W, C] SEA-RAFT uncertainty/aux channels
- deform_mag:  [N, H, W] flow magnitude
- summary_*:   scalar stats over deform_mag
- metadata:    run metadata

For single-frame videos, zero-fallback keeps all items in the output set.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


# This script imports RAFT internals directly; SEA-RAFT must be on the path.
# Set SEA_RAFT_ROOT to the root of your SEA-RAFT clone before running:
#   export SEA_RAFT_ROOT=/path/to/SEA-RAFT
_sea_raft_root_env = os.environ.get("SEA_RAFT_ROOT", "")
if not _sea_raft_root_env:
    print(
        "ERROR: SEA_RAFT_ROOT environment variable is not set.\n"
        "Clone SEA-RAFT and set the variable before running this script:\n"
        "  git clone https://github.com/princeton-vl/SEA-RAFT.git /path/to/SEA-RAFT\n"
        "  export SEA_RAFT_ROOT=/path/to/SEA-RAFT",
        file=sys.stderr,
    )
    sys.exit(1)

SEA_RAFT_ROOT = Path(_sea_raft_root_env)
CORE_DIR = SEA_RAFT_ROOT / "core"
if CORE_DIR.exists() and str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))
if str(SEA_RAFT_ROOT) not in sys.path:
    sys.path.insert(0, str(SEA_RAFT_ROOT))

from config.parser import json_to_args
from raft import RAFT
from utils.utils import load_ckpt


@dataclass
class VideoResult:
    video_id: str
    status: str
    num_frames: int
    num_pairs: int
    output_path: str
    elapsed_s: float
    error: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch SEA-RAFT extraction for MSDM AVI videos."
    )
    parser.add_argument("--video-dir", required=True,
        help="Directory containing per-utterance .avi files")
    parser.add_argument("--output-dir", required=True,
        help="Output directory; npz/ and flow_mag/ subdirs are created here")
    parser.add_argument("--landmark-dir", default=None,
        help="Directory of per-video landmark .npz files (required only with --mask-below-jaw)")
    parser.add_argument("--cfg", required=True,
        help="SEA-RAFT model config JSON (e.g. SEA-RAFT/config/eval/spring-M.json)")
    parser.add_argument("--checkpoint", required=True,
        help="SEA-RAFT checkpoint .pth file")
    parser.add_argument("--device", default=None, help="cuda, cpu, mps")
    parser.add_argument(
        "--inference-scale",
        type=str,
        default="auto",
        help=(
            "Override SEA-RAFT scale used at inference, or use 'auto'. "
            "For low-resolution face videos (e.g., 96x96), a value >=1 is often required."
        ),
    )
    parser.add_argument(
        "--single-frame-policy",
        choices=["zero_fallback", "error"],
        default="zero_fallback",
    )
    parser.add_argument(
        "--mask-below-jaw",
        action="store_true",
        help="Mask pixels below jawline landmarks (indices 0..19) to black before flow.",
    )
    parser.add_argument(
        "--jaw-pad",
        type=float,
        default=0.0,
        help="Extra pixels kept below interpolated jaw curve before masking.",
    )
    parser.add_argument(
        "--missing-landmark-policy",
        choices=["error", "disable_mask_for_video"],
        default="error",
        help="Behavior when jaw masking is enabled but landmark file is missing/invalid.",
    )
    parser.add_argument(
        "--fallback-info-channels",
        type=int,
        default=4,
        help="Used only for zero_fallback on single-frame videos.",
    )
    parser.add_argument(
        "--save-dtype",
        choices=["float16", "float32"],
        default="float16",
        help="Storage dtype for flow/info/deform_mag arrays.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Process exactly 3 videos for a quick pipeline check.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--manifest-prefix", default="manifest")
    parser.add_argument(
        "--export-sample-pngs",
        action="store_true",
        help="Export a visualization subset from saved NPY tensors.",
    )
    parser.add_argument(
        "--sample-png-count",
        type=int,
        default=100,
        help="Number of videos to visualize when --export-sample-pngs is enabled.",
    )
    parser.add_argument(
        "--sample-png-seed",
        type=int,
        default=7,
        help="Random seed for selecting visualization subset.",
    )
    return parser.parse_args()


def pick_device(requested: str | None) -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def list_videos(video_dir: Path, limit: int | None) -> List[Path]:
    videos = sorted(video_dir.glob("*.avi"))
    if limit is not None:
        videos = videos[:limit]
    return videos


def load_all_frames(video_path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    frames: List[np.ndarray] = []
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    cap.release()

    if not frames:
        raise RuntimeError(f"No decodable frames: {video_path}")
    return np.asarray(frames, dtype=np.uint8)


def to_storage_dtype(x: np.ndarray, dtype_name: str) -> np.ndarray:
    return x.astype(np.float16 if dtype_name == "float16" else np.float32, copy=False)


def resolve_inference_scale(
    requested: str, height: int, width: int, corr_levels: int
) -> int:
    if requested != "auto":
        return int(requested)

    # CorrBlock downsamples one extra time per level loop; keep feature map size safe.
    min_input = 8 * (2**corr_levels)  # 8x encoder downsample + corr pyramid safety.
    short_side = min(height, width)
    s = 0
    while short_side * (2**s) < min_input:
        s += 1
    return s


def load_landmarks_npz(landmark_path: Path) -> np.ndarray:
    data = np.load(landmark_path, allow_pickle=True)
    if "landmarks" not in data.files:
        raise KeyError(f"'landmarks' key not found in {landmark_path}")
    lmk = data["landmarks"]
    if lmk.ndim != 3 or lmk.shape[1:] != (49, 2):
        raise ValueError(f"Expected landmarks shape [T,49,2], got {lmk.shape} in {landmark_path}")
    return lmk.astype(np.float32)


def jaw_keep_mask_from_landmarks(
    landmarks_49x2: np.ndarray,
    height: int,
    width: int,
    jaw_pad: float,
) -> np.ndarray:
    jaw = landmarks_49x2[:20, :]
    finite = np.isfinite(jaw).all(axis=1)
    jaw = jaw[finite]
    if jaw.shape[0] < 2:
        return np.ones((height, width), dtype=bool)

    xs = np.clip(jaw[:, 0], 0.0, float(width - 1))
    ys = np.clip(jaw[:, 1], 0.0, float(height - 1))
    order = np.argsort(xs)
    xs = xs[order]
    ys = ys[order]

    xs_unique, idx = np.unique(xs, return_index=True)
    ys_unique = ys[idx]
    if xs_unique.shape[0] < 2:
        return np.ones((height, width), dtype=bool)

    x_grid = np.arange(width, dtype=np.float32)
    y_curve = np.interp(x_grid, xs_unique, ys_unique).astype(np.float32) + float(jaw_pad)
    y_curve = np.clip(y_curve, 0.0, float(height - 1))

    yy = np.arange(height, dtype=np.float32)[:, None]
    keep = yy <= y_curve[None, :]
    return keep


def apply_jaw_mask_to_frames(
    frames: np.ndarray,
    landmarks: np.ndarray,
    jaw_pad: float,
) -> np.ndarray:
    t, h, w, _ = frames.shape
    out = frames.copy()
    lt = landmarks.shape[0]
    for i in range(t):
        li = i if i < lt else (lt - 1)
        keep = jaw_keep_mask_from_landmarks(landmarks[li], h, w, jaw_pad)
        out[i][~keep] = 0
    return out


def build_jaw_keep_masks(
    num_frames: int,
    height: int,
    width: int,
    landmarks: np.ndarray,
    jaw_pad: float,
) -> np.ndarray:
    keep_masks = np.ones((num_frames, height, width), dtype=bool)
    lt = landmarks.shape[0]
    for i in range(num_frames):
        li = i if i < lt else (lt - 1)
        keep_masks[i] = jaw_keep_mask_from_landmarks(landmarks[li], height, width, jaw_pad)
    return keep_masks


def run_pair(
    model: torch.nn.Module,
    args_ns: argparse.Namespace,
    inference_scale: int,
    frame1_rgb: np.ndarray,
    frame2_rgb: np.ndarray,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    image1 = torch.from_numpy(frame1_rgb).permute(2, 0, 1).float()[None].to(device)
    image2 = torch.from_numpy(frame2_rgb).permute(2, 0, 1).float()[None].to(device)

    s = inference_scale
    img1 = F.interpolate(image1, scale_factor=2**s, mode="bilinear", align_corners=False)
    img2 = F.interpolate(image2, scale_factor=2**s, mode="bilinear", align_corners=False)
    output = model(img1, img2, iters=args_ns.iters, test_mode=True)

    flow = output["flow"][-1]
    info = output["info"][-1]
    flow_down = (
        F.interpolate(
            flow, scale_factor=0.5**s, mode="bilinear", align_corners=False
        )
        * (0.5**s)
    )
    info_down = F.interpolate(info, scale_factor=0.5**s, mode="area")

    flow_np = flow_down[0].permute(1, 2, 0).detach().cpu().numpy()
    info_np = info_down[0].permute(1, 2, 0).detach().cpu().numpy()
    return flow_np, info_np


@torch.no_grad()
def extract_video(
    video_path: Path,
    model: torch.nn.Module,
    model_args: argparse.Namespace,
    run_args: argparse.Namespace,
    device: torch.device,
    landmark_dir: Path | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float | int | str]]:
    frames = load_all_frames(video_path)
    t, h, w, _ = frames.shape

    inference_scale = resolve_inference_scale(
        requested=run_args.inference_scale,
        height=h,
        width=w,
        corr_levels=int(model_args.corr_levels),
    )
    mask_status = "disabled"
    keep_masks: np.ndarray | None = None

    if run_args.mask_below_jaw:
        landmark_path = None if landmark_dir is None else (landmark_dir / f"{video_path.stem}.npz")
        try:
            if landmark_path is None or not landmark_path.exists():
                raise FileNotFoundError(f"Landmark file not found for {video_path.stem}")
            landmarks = load_landmarks_npz(landmark_path)
            keep_masks = build_jaw_keep_masks(t, h, w, landmarks, run_args.jaw_pad)
            frames = apply_jaw_mask_to_frames(frames, landmarks, run_args.jaw_pad)
            mask_status = "applied"
        except Exception:
            if run_args.missing_landmark_policy == "error":
                raise
            mask_status = "disabled_for_video"

    if t == 1:
        if run_args.single_frame_policy == "error":
            raise RuntimeError("single-frame video and --single-frame-policy=error")
        flow = np.zeros((1, h, w, 2), dtype=np.float32)
        info = np.zeros((1, h, w, run_args.fallback_info_channels), dtype=np.float32)
        deform_mag = np.zeros((1, h, w), dtype=np.float32)
        meta = {
            "status": "single_frame_fallback",
            "num_frames": int(t),
            "num_pairs": 1,
            "height": int(h),
            "width": int(w),
            "inference_scale": inference_scale,
            "jaw_mask": mask_status,
        }
        return flow, info, deform_mag, meta

    flows: List[np.ndarray] = []
    infos: List[np.ndarray] = []
    for i in range(t - 1):
        flow_np, info_np = run_pair(
            model, model_args, inference_scale, frames[i], frames[i + 1], device
        )
        flows.append(flow_np)
        infos.append(info_np)

    flow = np.stack(flows, axis=0)
    info = np.stack(infos, axis=0)

    # Enforce strict zeroed output outside jaw mask to avoid residual motion artifacts.
    if mask_status == "applied" and keep_masks is not None:
        for i in range(t - 1):
            keep_pair = keep_masks[i] & keep_masks[min(i + 1, t - 1)]
            flow[i][~keep_pair] = 0.0
            info[i][~keep_pair] = 0.0

    deform_mag = np.linalg.norm(flow, axis=-1)
    meta = {
        "status": "ok",
        "num_frames": int(t),
        "num_pairs": int(t - 1),
        "height": int(h),
        "width": int(w),
        "inference_scale": inference_scale,
        "jaw_mask": mask_status,
    }
    return flow, info, deform_mag, meta


def write_npz(
    out_path: Path,
    flow: np.ndarray,
    info: np.ndarray,
    deform_mag: np.ndarray,
    meta: Dict[str, float | int | str],
    run_args: argparse.Namespace,
) -> None:
    flow_s = to_storage_dtype(flow, run_args.save_dtype)
    info_s = to_storage_dtype(info, run_args.save_dtype)
    deform_s = to_storage_dtype(deform_mag, run_args.save_dtype)

    payload = {
        "flow": flow_s,
        "info": info_s,
        "deform_mag": deform_s,
        "summary_mean": np.array([float(np.mean(deform_mag))], dtype=np.float32),
        "summary_std": np.array([float(np.std(deform_mag))], dtype=np.float32),
        "summary_max": np.array([float(np.max(deform_mag))], dtype=np.float32),
        "summary_min": np.array([float(np.min(deform_mag))], dtype=np.float32),
        "meta_json": np.array([json.dumps(meta)], dtype=object),
    }
    np.savez_compressed(out_path, **payload)


def write_npy_modalities(
    output_dir: Path,
    video_id: str,
    flow: np.ndarray,
    info: np.ndarray,
    deform_mag: np.ndarray,
    run_args: argparse.Namespace,
) -> None:
    dtype = np.float16 if run_args.save_dtype == "float16" else np.float32

    # flow_mag is the magnitude of the masked flow field.
    flow_mag = np.linalg.norm(flow.astype(np.float32), axis=-1).astype(dtype, copy=False)
    info_s = info.astype(dtype, copy=False)
    deform_s = deform_mag.astype(dtype, copy=False)

    np.save(output_dir / "flow_mag" / f"{video_id}.npy", flow_mag)
    np.save(output_dir / "info" / f"{video_id}.npy", info_s)
    np.save(output_dir / "deform_mag" / f"{video_id}.npy", deform_s)


def normalize_to_uint8(img: np.ndarray) -> np.ndarray:
    x = img.astype(np.float32)
    lo, hi = np.percentile(x, [2.0, 98.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(x))
        hi = float(np.max(x))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(x, dtype=np.uint8)
    x = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return (x * 255.0).astype(np.uint8)


def export_sample_pngs(
    output_dir: Path,
    rows: List[VideoResult],
    count: int,
    seed: int,
) -> None:
    sample_root = output_dir / "sample_imgs"
    flow_dir = sample_root / "flow_mag"
    info_dir = sample_root / "info"
    deform_dir = sample_root / "deform_mag"
    flow_dir.mkdir(parents=True, exist_ok=True)
    info_dir.mkdir(parents=True, exist_ok=True)
    deform_dir.mkdir(parents=True, exist_ok=True)

    ok_ids = [r.video_id for r in rows if r.status == "ok"]
    fallback_ids = [r.video_id for r in rows if r.status == "single_frame_fallback"]
    candidates = ok_ids if ok_ids else fallback_ids
    if not candidates:
        print("No candidates for sample PNG export.")
        return

    rng = random.Random(seed)
    k = min(count, len(candidates))
    picks = rng.sample(candidates, k)

    exported = []
    for video_id in picks:
        flow_path = output_dir / "flow_mag" / f"{video_id}.npy"
        info_path = output_dir / "info" / f"{video_id}.npy"
        deform_path = output_dir / "deform_mag" / f"{video_id}.npy"
        if not (flow_path.exists() and info_path.exists() and deform_path.exists()):
            continue

        flow = np.load(flow_path)      # [T,H,W]
        info = np.load(info_path)      # [T,H,W,C]
        deform = np.load(deform_path)  # [T,H,W]

        t = 0
        flow_img = normalize_to_uint8(flow[t])
        info_img = normalize_to_uint8(info[t, :, :, 0])
        deform_img = normalize_to_uint8(deform[t])

        flow_color = cv2.applyColorMap(flow_img, cv2.COLORMAP_MAGMA)
        info_color = cv2.applyColorMap(info_img, cv2.COLORMAP_VIRIDIS)
        deform_color = cv2.applyColorMap(deform_img, cv2.COLORMAP_PLASMA)

        cv2.imwrite(str(flow_dir / f"{video_id}.png"), flow_color)
        cv2.imwrite(str(info_dir / f"{video_id}.png"), info_color)
        cv2.imwrite(str(deform_dir / f"{video_id}.png"), deform_color)
        exported.append(video_id)

    manifest_path = sample_root / "sample_selection.json"
    with manifest_path.open("w") as f:
        json.dump(
            {"count_requested": count, "count_exported": len(exported), "video_ids": exported},
            f,
            indent=2,
        )
    print(f"Exported sample PNGs: {len(exported)} videos -> {sample_root}")


def build_model(cfg_path: Path, ckpt_path: Path, device: torch.device) -> Tuple[torch.nn.Module, argparse.Namespace]:
    model_args = json_to_args(str(cfg_path))
    model = RAFT(model_args)
    load_ckpt(model, str(ckpt_path))
    model = model.to(device)
    model.eval()
    return model, model_args


def write_manifests(output_dir: Path, prefix: str, rows: List[VideoResult]) -> None:
    ts = int(time.time())
    csv_path = output_dir / f"{prefix}_{ts}.csv"
    json_path = output_dir / f"{prefix}_{ts}.json"

    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["video_id", "status", "num_frames", "num_pairs", "elapsed_s", "output_path", "error"]
        )
        for r in rows:
            writer.writerow(
                [
                    r.video_id,
                    r.status,
                    r.num_frames,
                    r.num_pairs,
                    f"{r.elapsed_s:.4f}",
                    r.output_path,
                    r.error,
                ]
            )

    summary = {
        "total": len(rows),
        "ok": sum(r.status == "ok" for r in rows),
        "single_frame_fallback": sum(r.status == "single_frame_fallback" for r in rows),
        "error": sum(r.status == "error" for r in rows),
        "rows": [r.__dict__ for r in rows],
    }
    with json_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote manifest CSV: {csv_path}")
    print(f"Wrote manifest JSON: {json_path}")


def main() -> None:
    run_args = parse_args()

    if run_args.smoke_test:
        run_args.limit = 3

    video_dir = Path(run_args.video_dir).expanduser().resolve()
    output_dir = Path(run_args.output_dir).expanduser().resolve()
    cfg_path = Path(run_args.cfg).expanduser().resolve()
    ckpt_path = Path(run_args.checkpoint).expanduser().resolve()
    landmark_dir = Path(run_args.landmark_dir).expanduser().resolve() if run_args.landmark_dir else None
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_dir = output_dir / "npz"
    flow_mag_dir = output_dir / "flow_mag"
    info_dir = output_dir / "info"
    deform_mag_dir = output_dir / "deform_mag"
    npz_dir.mkdir(parents=True, exist_ok=True)
    flow_mag_dir.mkdir(parents=True, exist_ok=True)
    info_dir.mkdir(parents=True, exist_ok=True)
    deform_mag_dir.mkdir(parents=True, exist_ok=True)

    if not video_dir.exists():
        raise FileNotFoundError(f"--video-dir not found: {video_dir}")
    if not cfg_path.exists():
        raise FileNotFoundError(f"--cfg not found: {cfg_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"--checkpoint not found: {ckpt_path}")
    if run_args.mask_below_jaw and (landmark_dir is None or not landmark_dir.exists()):
        raise FileNotFoundError(f"--mask-below-jaw requires --landmark-dir; got: {landmark_dir}")

    videos = list_videos(video_dir, run_args.limit)
    if not videos:
        raise RuntimeError(f"No .avi files found in {video_dir}")

    device_name = pick_device(run_args.device)
    device = torch.device(device_name)
    print(f"Using device: {device}")
    print(f"Videos to process: {len(videos)}")

    model, model_args = build_model(cfg_path, ckpt_path, device)
    results: List[VideoResult] = []

    for video_path in tqdm(videos, desc="SEA-RAFT extract", unit="video"):
        video_id = video_path.stem
        out_path = npz_dir / f"{video_id}.npz"
        start = time.time()

        if out_path.exists() and not run_args.overwrite:
            elapsed = time.time() - start
            results.append(
                VideoResult(
                    video_id=video_id,
                    status="exists",
                    num_frames=-1,
                    num_pairs=-1,
                    output_path=str(out_path),
                    elapsed_s=elapsed,
                )
            )
            continue

        try:
            flow, info, deform_mag, meta = extract_video(
                video_path=video_path,
                model=model,
                model_args=model_args,
                run_args=run_args,
                device=device,
                landmark_dir=landmark_dir,
            )
            meta.update(
                {
                    "video_id": video_id,
                    "checkpoint": str(ckpt_path),
                    "cfg": str(cfg_path),
                    "device": device_name,
                    "save_dtype": run_args.save_dtype,
                    "single_frame_policy": run_args.single_frame_policy,
                    "mask_below_jaw": bool(run_args.mask_below_jaw),
                    "jaw_pad": float(run_args.jaw_pad),
                }
            )
            write_npz(out_path, flow, info, deform_mag, meta, run_args)
            write_npy_modalities(output_dir, video_id, flow, info, deform_mag, run_args)
            elapsed = time.time() - start
            results.append(
                VideoResult(
                    video_id=video_id,
                    status=str(meta["status"]),
                    num_frames=int(meta["num_frames"]),
                    num_pairs=int(meta["num_pairs"]),
                    output_path=str(out_path),
                    elapsed_s=elapsed,
                )
            )
        except Exception as exc:
            elapsed = time.time() - start
            results.append(
                VideoResult(
                    video_id=video_id,
                    status="error",
                    num_frames=-1,
                    num_pairs=-1,
                    output_path=str(out_path),
                    elapsed_s=elapsed,
                    error=str(exc),
                )
            )

    write_manifests(output_dir, run_args.manifest_prefix, results)

    total = len(results)
    ok = sum(r.status == "ok" for r in results)
    fallback = sum(r.status == "single_frame_fallback" for r in results)
    exists = sum(r.status == "exists" for r in results)
    err = sum(r.status == "error" for r in results)
    print(
        f"Done. total={total} ok={ok} single_frame_fallback={fallback} "
        f"exists={exists} error={err}"
    )

    if run_args.export_sample_pngs:
        export_sample_pngs(
            output_dir=output_dir,
            rows=results,
            count=run_args.sample_png_count,
            seed=run_args.sample_png_seed,
        )

    if err > 0:
        sys.exit(1)


if __name__ == "__main__":
    # Reduce OpenCV threading overhead for many small videos.
    os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_MSMF", "0")
    cv2.setNumThreads(0)
    main()
