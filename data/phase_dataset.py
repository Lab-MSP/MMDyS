from __future__ import annotations

import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from data.constants import SEVERITY_TO_ID, SEVERITY_TO_TARGET, TASK_TO_GROUP

cv2.setNumThreads(0)


# ---------------------------------------------------------------------------
# Low-level video I/O helpers
# ---------------------------------------------------------------------------

def _load_video_rgb(path: Path) -> Tuple[np.ndarray, float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 25.0
    frames: List[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames in video: {path}")
    return np.stack(frames, axis=0), fps


def _probe_video_meta(path: Path) -> Tuple[float, int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return 25.0, 0
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, max(0, n)


def _sample_or_pad_frames(frames: np.ndarray, num_frames: int) -> np.ndarray:
    total = int(frames.shape[0])
    if total == num_frames:
        return frames
    if total >= num_frames:
        idx = np.linspace(0, total - 1, num_frames).round().astype(np.int64)
        return frames[idx]
    pad_n = num_frames - total
    return np.concatenate([frames, np.repeat(frames[-1:], pad_n, axis=0)], axis=0)


def _resize_video_tensor(video: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
    # video: [T, H, W, 3] float32
    x = video.permute(0, 3, 1, 2)  # [T, 3, H, W]
    x = F.interpolate(x, size=(out_h, out_w), mode="bilinear", align_corners=False)
    return x.permute(0, 2, 3, 1)  # [T, H, W, 3]


# ---------------------------------------------------------------------------
# Flow loading: handles magnitude [T,H,W] or raw XY [T,H,W,2] arrays.
# Returns magnitude array of shape [T, H, W].
# ---------------------------------------------------------------------------

def _load_flow_array(path: Path) -> np.ndarray:
    arr = np.load(path)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    if arr.ndim == 2:
        # [H, W] — single-frame magnitude
        return arr[np.newaxis]  # [1, H, W]

    if arr.ndim == 3:
        # [T, H, W] — already per-frame magnitudes
        return arr

    if arr.ndim == 4:
        last = arr.shape[-1]
        first = arr.shape[0]
        if last == 2:
            # [T, H, W, 2] — raw XY optical flow → compute magnitude
            return np.sqrt(arr[..., 0] ** 2 + arr[..., 1] ** 2)
        if last == 1:
            return arr[..., 0]
        if last == 3:
            return arr[..., 0]
        if first == 1:
            return arr[0]
        if first == 3:
            return arr[0]
        raise ValueError(f"Unsupported 4-D flow array shape: {arr.shape}")

    raise ValueError(f"Unsupported flow array rank {arr.ndim}: {arr.shape}")


def _normalize_flow(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr.astype(np.float32)
    p = float(np.percentile(arr, 99.5))
    if (not np.isfinite(p)) or p <= 1e-6:
        p = max(float(np.max(arr)), 1e-6)
    return np.clip(arr / p, 0.0, 1.0).astype(np.float32)


def _align_flow_len(flow: np.ndarray, target_len: int) -> np.ndarray:
    """Align flow (T-1 or T frames) to exactly target_len frames."""
    n = flow.shape[0]
    if n == target_len:
        return flow
    if n == target_len - 1:
        # Prepend duplicate of first frame
        return np.concatenate([flow[:1], flow], axis=0)
    return _sample_or_pad_frames(flow, target_len)


# ---------------------------------------------------------------------------
# Augmentation helpers (applied consistently across all 6 clips of a sample)
# ---------------------------------------------------------------------------

def _flip_tensor(t: torch.Tensor) -> torch.Tensor:
    """Horizontal flip [T, H, W, 3]."""
    return t.flip(dims=[2])


def _brightness_jitter(t: torch.Tensor, factor: float) -> torch.Tensor:
    """Additive brightness shift, clipped to [0, 1]. factor in [-delta, +delta]."""
    return (t + factor).clamp(0.0, 1.0)


def _contrast_jitter(t: torch.Tensor, factor: float) -> torch.Tensor:
    """Multiplicative contrast around mean. factor in [1-delta, 1+delta]."""
    mean = t.mean()
    return ((t - mean) * factor + mean).clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Flow descriptors
# ---------------------------------------------------------------------------

def _entropy(values: np.ndarray, bins: int = 10, eps: float = 1e-8) -> float:
    if values.size == 0:
        return 0.0
    hi = float(np.max(values)) + eps
    lo = float(np.min(values))
    hist, _ = np.histogram(values, bins=bins, range=(lo, hi), density=False)
    prob = hist.astype(np.float64)
    z = prob.sum()
    if z <= eps:
        return 0.0
    prob /= z
    prob = prob[prob > 0]
    return float(-(prob * np.log(prob + eps)).sum())


def _segment_features(mag: np.ndarray, *, is_speech: bool, eps: float) -> np.ndarray:
    """8-D feature vector for one side of one phase."""
    if mag.size == 0:
        return np.zeros(8, dtype=np.float32)
    x = mag.astype(np.float64)
    mean = float(np.mean(x))
    std = float(np.std(x))
    peak = float(np.max(x))
    auc = float(np.trapz(x, dx=1.0 / max(1, len(x) - 1))) if len(x) > 1 else float(x[0])
    jerk = float(np.var(np.diff(x))) if len(x) > 1 else 0.0
    ent = _entropy(x, bins=10, eps=eps)
    early_late = 0.0
    slope = 0.0
    if is_speech and len(x) > 1:
        half = max(1, len(x) // 2)
        early = float(np.mean(x[:half]))
        late = float(np.mean(x[half:])) if len(x[half:]) > 0 else early
        early_late = early / max(late, eps)
        t_lin = np.linspace(0.0, 1.0, len(x), dtype=np.float64)
        slope = float(np.polyfit(t_lin, x, deg=1)[0]) if len(x) >= 2 else 0.0
    return np.array([mean, std, peak, auc, jerk, ent, early_late, slope], dtype=np.float32)


def _compute_flow_descriptors(
    flow_left: np.ndarray,
    flow_right: np.ndarray,
    phase_bounds: Dict[str, slice],
    eps: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        phase_side_feats : [3, 2, 8]
        phase_asym_raw   : [3, 8]   (left - right)
        recovery_ratio   : [2]      log-space bounded ratio post/pre per side
    """
    phase_order = ["pre", "speech", "post"]
    phase_side_feats = np.zeros((3, 2, 8), dtype=np.float32)

    for p_idx, phase in enumerate(phase_order):
        sl = phase_bounds[phase]
        fl = flow_left[sl]
        fr = flow_right[sl]
        l_mag = fl.mean(axis=(1, 2)) if fl.size > 0 else np.zeros(0, dtype=np.float32)
        r_mag = fr.mean(axis=(1, 2)) if fr.size > 0 else np.zeros(0, dtype=np.float32)
        phase_side_feats[p_idx, 0] = _segment_features(l_mag, is_speech=(phase == "speech"), eps=eps)
        phase_side_feats[p_idx, 1] = _segment_features(r_mag, is_speech=(phase == "speech"), eps=eps)

    phase_asym_raw = phase_side_feats[:, 0, :] - phase_side_feats[:, 1, :]

    # Recovery ratio: log(post_mean / pre_mean + 1), bounded in [0, ~ln(6)] ≈ [0, 1.8]
    # Using log1p(ratio) avoids blow-up when pre is near zero.
    pre_l = float(phase_side_feats[0, 0, 0])   # mean magnitude, pre, left
    pre_r = float(phase_side_feats[0, 1, 0])
    post_l = float(phase_side_feats[2, 0, 0])
    post_r = float(phase_side_feats[2, 1, 0])
    rec_l = math.log1p(post_l / max(pre_l, eps))
    rec_r = math.log1p(post_r / max(pre_r, eps))
    recovery_ratio = np.array([rec_l, rec_r], dtype=np.float32)

    return phase_side_feats, phase_asym_raw, recovery_ratio


# ---------------------------------------------------------------------------
# Main dataset
# ---------------------------------------------------------------------------

class MSDMPhaseDataset(Dataset):
    """
    Phase-aware hemiface video dataset for dysarthria severity classification.

    Flow descriptor cache is opened READ-ONLY (descriptor_cache_write=False by
    default when a pre-built cache is provided) so that ongoing experiments in
    the original directory are not disturbed.
    """

    def __init__(
        self,
        split_json_path: str | Path,
        video_root: str | Path,
        flow_feature_root: str | Path,
        split_name: str,
        # Phase segmentation
        pre_seconds: float = 1.0,
        post_seconds: float = 0.3,
        fallback_pre_seconds: float = 0.4,
        fallback_post_seconds: float = 0.2,
        reduced_pre_seconds: float = 0.7,
        min_speech_frames: int = 3,
        min_speech_frames_for_reduced_pre: int = 4,
        pre_target_frames: int = 16,
        speech_target_frames: int = 32,
        speech_pad_to_target: bool = False,
        post_target_frames: int = 5,
        # Video
        resize_height: int = 224,
        resize_half_width: int = 224,
        max_video_seconds: float = 5.0,
        # Flow
        use_flow_descriptors: bool = True,
        use_flow_as_channel: bool = False,   # append flow magnitude as 4th video channel
        require_flow_ok: bool = False,
        # Descriptor cache — read-only by default
        descriptor_cache_dir: Optional[str | Path] = None,
        descriptor_cache_write: bool = False,
        descriptor_cache_miss_policy: str = "compute",
        descriptor_eps: float = 1e-6,
        # Optional deformation-magnitude descriptor cache (v8).
        # When set, deform descriptors are loaded and concatenated with flow
        # descriptors → flow_input_dim doubles from 14 to 28.
        deform_descriptor_cache_dir: Optional[str | Path] = None,
        # Augmentation (train only)
        enable_aug: bool = False,
        aug_hflip_prob: float = 0.5,
        aug_brightness_delta: float = 0.05,
        aug_contrast_delta: float = 0.05,
        aug_phase_jitter_frames: int = 2,
        # Missing data
        missing_policy: str = "skip",
        runtime_missing_policy: str = "skip",
        runtime_max_retries: int = 8,
        min_samples_per_split: int = 1,
    ) -> None:
        self.split_json_path = Path(split_json_path)
        self.video_root = Path(video_root)
        self.flow_feature_root = Path(flow_feature_root)
        self.split_name = str(split_name)
        self.use_flow_descriptors = bool(use_flow_descriptors)
        self.use_flow_as_channel = bool(use_flow_as_channel)

        self.pre_seconds = float(pre_seconds)
        self.post_seconds = float(post_seconds)
        self.fallback_pre_seconds = float(fallback_pre_seconds)
        self.fallback_post_seconds = float(fallback_post_seconds)
        self.reduced_pre_seconds = float(reduced_pre_seconds)
        self.min_speech_frames = int(min_speech_frames)
        self.min_speech_frames_for_reduced_pre = int(min_speech_frames_for_reduced_pre)
        self.pre_target_frames = int(pre_target_frames)
        self.speech_target_frames = int(speech_target_frames)
        self.speech_pad_to_target = bool(speech_pad_to_target)
        self.post_target_frames = int(post_target_frames)

        self.resize_height = int(resize_height)
        self.resize_half_width = int(resize_half_width)
        self.max_video_seconds = float(max_video_seconds)

        self.require_flow_ok = bool(require_flow_ok)
        self.descriptor_cache_dir = Path(descriptor_cache_dir) if descriptor_cache_dir else None
        self.descriptor_cache_write = bool(descriptor_cache_write)
        self.descriptor_cache_miss_policy = str(descriptor_cache_miss_policy).lower()
        self.descriptor_eps = float(descriptor_eps)
        self.deform_descriptor_cache_dir = (
            Path(deform_descriptor_cache_dir) if deform_descriptor_cache_dir else None
        )

        self.enable_aug = bool(enable_aug)
        self.aug_hflip_prob = float(aug_hflip_prob)
        self.aug_brightness_delta = float(aug_brightness_delta)
        self.aug_contrast_delta = float(aug_contrast_delta)
        self.aug_phase_jitter_frames = int(aug_phase_jitter_frames)

        self.missing_policy = str(missing_policy).lower()
        self.runtime_missing_policy = str(runtime_missing_policy).lower()
        self.runtime_max_retries = max(0, int(runtime_max_retries))
        self.min_samples_per_split = int(min_samples_per_split)

        if self.descriptor_cache_dir is not None and self.descriptor_cache_write:
            self.descriptor_cache_dir.mkdir(parents=True, exist_ok=True)

        self.feature_availability: Dict[str, Any] = {}
        self.samples = self._load_samples()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _phase_bounds(self, total_frames: int, fps: float, jitter: int = 0) -> Dict[str, slice]:
        fps = max(float(fps), 1e-6)
        effective_seconds = total_frames / fps

        # Ultra-short: can't fit even the reduced pre + full post.
        # Treat the entire video as speech; _extract will borrow this segment for
        # the empty pre/post clips so all three phase inputs remain valid tensors.
        # Flow descriptors for pre/post will be zero (no resting-state window exists),
        # and recovery_ratio will be 0 — both are informative signals in this case.
        ultra_short_thresh = self.fallback_pre_seconds + self.post_seconds  # 0.4 + 0.3 = 0.7 s
        if effective_seconds < ultra_short_thresh:
            return {
                "pre":    slice(0, 0),
                "speech": slice(0, total_frames),
                "post":   slice(0, 0),
            }

        # Short (0.7 s – 1.3 s): reduce pre to fallback_pre, keep full post_seconds.
        # fallback_post_seconds is intentionally NOT used here — the post window must
        # always be 300 ms from the true end of the recording regardless of duration.
        short = effective_seconds < (self.pre_seconds + self.post_seconds)
        pre_frames  = int(round((self.fallback_pre_seconds if short else self.pre_seconds) * fps))
        post_frames = int(round(self.post_seconds * fps))

        pre_end    = min(total_frames, max(0, pre_frames + jitter))
        post_start = max(pre_end, total_frames - max(0, post_frames + jitter))

        # Secondary fallback: samples just above the 1.3 s threshold can still have
        # pre_f + post_f ≈ total_frames, leaving < 4 speech frames.  Reduce pre to
        # reduced_pre_seconds (0.7 s) so a usable speech window is recovered.
        # Only applies when not already in the short-fallback regime.
        if (post_start - pre_end) < self.min_speech_frames_for_reduced_pre and not short:
            pre_frames = int(round(self.reduced_pre_seconds * fps))
            pre_end    = min(total_frames, max(0, pre_frames + jitter))
            post_start = max(pre_end, total_frames - max(0, post_frames + jitter))

        return {
            "pre":    slice(0, pre_end),
            "speech": slice(pre_end, post_start),
            "post":   slice(post_start, total_frames),
        }

    def _load_samples(self) -> List[Dict[str, Any]]:
        if not self.split_json_path.exists():
            raise FileNotFoundError(f"Split JSON not found: {self.split_json_path}")
        if not self.video_root.exists():
            raise FileNotFoundError(f"Video root not found: {self.video_root}")
        if self.use_flow_descriptors and not self.flow_feature_root.exists():
            raise FileNotFoundError(f"Flow root not found: {self.flow_feature_root}")

        payload = json.loads(self.split_json_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"Expected list in {self.split_json_path}")

        required = {"filename", "speaker", "task", "severity"}
        items: List[Dict[str, Any]] = []
        skipped_rows: List[Dict[str, str]] = []
        skipped_by_reason: Counter = Counter()
        total_by_severity: Counter = Counter()

        for idx, row in enumerate(payload):
            missing_keys = required - set(row.keys())
            if missing_keys:
                raise KeyError(f"Missing keys {sorted(missing_keys)} at index={idx}")

            severity = str(row["severity"])
            task = str(row["task"])
            speaker = str(row["speaker"])
            filename = str(row["filename"])

            if severity not in SEVERITY_TO_ID:
                raise ValueError(f"Unknown severity '{severity}' at index={idx}")
            if task not in TASK_TO_GROUP:
                raise ValueError(f"Unknown task '{task}' at index={idx}")

            total_by_severity[severity] += 1

            video_path = self.video_root / f"{filename}.avi"
            need_flow = self.use_flow_descriptors or self.use_flow_as_channel
            flow_path = (self.flow_feature_root / f"{filename}.npy") if need_flow else None

            reason = ""
            src_fps, src_frames = 25.0, 0
            if not video_path.exists():
                reason = "missing_video"
            elif need_flow and flow_path is not None and not flow_path.exists():
                reason = "missing_flow"
            elif self.use_flow_descriptors and self.require_flow_ok and ("flow_ok" in row) and not bool(row.get("flow_ok")):
                reason = "flow_ok_false"
            else:
                src_fps, src_frames = _probe_video_meta(video_path)
                if src_frames <= 0:
                    reason = "empty_video"
                else:
                    # Use full video length — pre=first N frames, post=last M frames, speech=middle.
                    # Never truncate: post segment must come from the actual video end.
                    bounds = self._phase_bounds(src_frames, src_fps)
                    speech_len = bounds["speech"].stop - bounds["speech"].start
                    if speech_len < self.min_speech_frames:
                        reason = "too_short_speech"

            if reason:
                if self.missing_policy == "error":
                    raise FileNotFoundError(f"missing_policy=error blocked: {reason} sample={filename}")
                skipped_by_reason[reason] += 1
                skipped_rows.append({"filename": filename, "severity": severity, "reason": reason})
                continue

            items.append({
                "filename": filename,
                "sample_id": filename,
                "video_path": video_path,
                "flow_path": str(flow_path) if flow_path is not None else "",
                "has_flow": flow_path is not None,
                "speaker_id": speaker,
                "task": task,
                "task_group": TASK_TO_GROUP[task],
                "severity": severity,
                "severity_id": SEVERITY_TO_ID[severity],
                "severity_target": SEVERITY_TO_TARGET[severity],
                "dur_sec": float(row.get("dur_sec", 0.0) or 0.0),
                "src_fps": float(src_fps),
                "src_frames": int(src_frames),
            })

        kept_by_severity = Counter(x["severity"] for x in items)
        total = len(payload)
        kept = len(items)
        skipped = len(skipped_rows)
        severe_total = float(total_by_severity.get("severe", 0))
        severe_skipped = float(Counter(x["severity"] for x in skipped_rows).get("severe", 0))

        self.feature_availability = {
            "split": self.split_name,
            "total_json_samples": total,
            "available_feature_samples": kept,
            "missing_feature_samples": skipped,
            "missing_pct": (100.0 * skipped / total) if total > 0 else 0.0,
            "skipped_by_reason": dict(skipped_by_reason),
            "total_by_severity": dict(total_by_severity),
            "kept_by_severity": dict(kept_by_severity),
            "severe_skip_rate": (severe_skipped / severe_total) if severe_total > 0 else 0.0,
            "missing_examples": skipped_rows[:30],
        }

        if kept < self.min_samples_per_split:
            raise ValueError(
                f"Split '{self.split_name}' has only {kept} valid samples "
                f"(min required={self.min_samples_per_split})."
            )
        return items

    # ------------------------------------------------------------------
    # Descriptor cache
    # ------------------------------------------------------------------

    def _cache_path(self, sample_id: str) -> Optional[Path]:
        if self.descriptor_cache_dir is None:
            return None
        return self.descriptor_cache_dir / f"{sample_id}.npz"

    def _load_or_compute_descriptors(
        self,
        sample_id: str,
        flow_left: np.ndarray,
        flow_right: np.ndarray,
        phase_bounds: Dict[str, slice],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        cache_path = self._cache_path(sample_id)

        if cache_path is not None and cache_path.exists():
            payload = np.load(cache_path)
            return (
                payload["phase_side_feats"].astype(np.float32),
                payload["phase_asym_raw"].astype(np.float32),
                payload["recovery_ratio"].astype(np.float32),
            )

        if self.descriptor_cache_miss_policy == "error":
            raise FileNotFoundError(f"Descriptor cache miss for {sample_id} at {cache_path}")

        psf, par, rr = _compute_flow_descriptors(flow_left, flow_right, phase_bounds, self.descriptor_eps)

        if cache_path is not None and self.descriptor_cache_write:
            np.savez_compressed(cache_path, phase_side_feats=psf, phase_asym_raw=par, recovery_ratio=rr)

        return psf, par, rr

    # ------------------------------------------------------------------
    # Item loading
    # ------------------------------------------------------------------

    def _load_item_by_index(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        rgb, src_fps = _load_video_rgb(item["video_path"])
        # Load the full video — never truncate.  Post segment is always taken from the
        # actual end of the recording; truncating would give the wrong recovery window.

        # Optional phase boundary jitter for augmentation
        jitter = 0
        if self.enable_aug and self.aug_phase_jitter_frames > 0:
            jitter = random.randint(-self.aug_phase_jitter_frames, self.aug_phase_jitter_frames)

        phase_bounds = self._phase_bounds(int(rgb.shape[0]), src_fps, jitter=jitter)
        speech_len = phase_bounds["speech"].stop - phase_bounds["speech"].start
        if speech_len < self.min_speech_frames:
            raise RuntimeError(f"speech_too_short after jitter: sample={item['sample_id']} len={speech_len}")

        # Split into left / right hemifaces
        mid = int(rgb.shape[2] // 2)
        rgb_left = rgb[:, :, :mid, :].astype(np.float32) / 255.0
        rgb_right = rgb[:, :, mid:, :].astype(np.float32) / 255.0

        # Flow loading (shared between descriptor and channel modes)
        flow_mag_left: Optional[np.ndarray] = None   # [T, H, W_half] normalised float32
        flow_mag_right: Optional[np.ndarray] = None
        if self.use_flow_descriptors or self.use_flow_as_channel:
            flow = _load_flow_array(Path(item["flow_path"]))
            flow = _align_flow_len(flow, int(rgb.shape[0]))
            flow = _normalize_flow(flow)
            mid_f = int(flow.shape[2] // 2)
            flow_mag_left  = flow[:, :, :mid_f]
            flow_mag_right = flow[:, :, mid_f:]

        # Flow descriptors (8-D / cache-backed 14-D per phase×side)
        if self.use_flow_descriptors:
            phase_side_feats, phase_asym_raw, recovery_ratio = self._load_or_compute_descriptors(
                item["sample_id"], flow_mag_left, flow_mag_right, phase_bounds
            )
        else:
            phase_side_feats = np.zeros((3, 2, 8), dtype=np.float32)
            phase_asym_raw = np.zeros((3, 8), dtype=np.float32)
            recovery_ratio = np.zeros(2, dtype=np.float32)

        # Optional deform descriptors (v8): load from cache and concatenate with flow.
        if self.deform_descriptor_cache_dir is not None:
            deform_path = self.deform_descriptor_cache_dir / f"{item['sample_id']}.npz"
            if deform_path.exists():
                d = np.load(deform_path)
                d_psf = d["phase_side_feats"].astype(np.float32)   # [3, 2, 14]
                d_par = d["phase_asym_raw"].astype(np.float32)      # [3, 14]
                phase_side_feats = np.concatenate([phase_side_feats, d_psf], axis=2)  # [3, 2, 28]
                phase_asym_raw   = np.concatenate([phase_asym_raw,   d_par], axis=1)  # [3, 28]

        def _extract(arr: np.ndarray, phase: str, target_t: int) -> torch.Tensor:
            """Fixed-length extraction for pre/post (resample to exact target).
            For ultra-short videos the pre/post slices are empty; borrow the speech
            segment so all three phase inputs are valid tensors.  The model receives
            identical visual content for all phases and can learn from that signal
            together with the zero-valued flow features for the missing windows."""
            sl = phase_bounds[phase]
            seg = arr[sl]
            if seg.shape[0] == 0:
                seg = arr[phase_bounds["speech"]]
            seg = _sample_or_pad_frames(seg, target_t)
            t = torch.from_numpy(seg).to(torch.float32)
            return _resize_video_tensor(t, self.resize_height, self.resize_half_width).contiguous()

        def _extract_speech(arr: np.ndarray) -> torch.Tensor:
            """Speech-phase extraction.

            speech_pad_to_target=True  (T=32 experiment): resample/pad to exactly
            speech_target_frames so every sample has identical T, matching the
            videomae_num_frames=32 model which processes the full clip in one pass.

            speech_pad_to_target=False (T=16 / variable experiment): keep all frames,
            subsample only if longer than speech_target_frames.  Short syllables stay
            short and are encoded as a single VideoMAE window (T≤16, native size).
            """
            sl = phase_bounds["speech"]
            seg = arr[sl]
            if self.speech_pad_to_target:
                seg = _sample_or_pad_frames(seg, self.speech_target_frames)
            elif seg.shape[0] > self.speech_target_frames:
                seg = _sample_or_pad_frames(seg, self.speech_target_frames)
            t = torch.from_numpy(seg.copy()).to(torch.float32)
            return _resize_video_tensor(t, self.resize_height, self.resize_half_width).contiguous()

        pre_left = _extract(rgb_left, "pre", self.pre_target_frames)
        pre_right = _extract(rgb_right, "pre", self.pre_target_frames)
        speech_left = _extract_speech(rgb_left)
        speech_right = _extract_speech(rgb_right)
        post_left = _extract(rgb_left, "post", self.post_target_frames)
        post_right = _extract(rgb_right, "post", self.post_target_frames)

        speech_num_frames = int(speech_left.shape[0])  # actual T before collator padding

        # Append flow magnitude as 4th channel [T, H, W, 3] → [T, H, W, 4]
        if self.use_flow_as_channel:
            def _extract_flow_chan(farr: np.ndarray, phase: str, target_t: int) -> torch.Tensor:
                """Extract one phase from [T, H, W] flow magnitude → [target_t, H, W, 1]."""
                sl = phase_bounds[phase]
                seg = farr[sl]
                if seg.shape[0] == 0:
                    seg = farr[phase_bounds["speech"]]
                seg = _sample_or_pad_frames(seg, target_t)
                # add channel dim → [T, H, W, 1] so _resize_video_tensor can handle it
                t = torch.from_numpy(seg).to(torch.float32).unsqueeze(-1)
                return _resize_video_tensor(t, self.resize_height, self.resize_half_width).contiguous()

            def _extract_speech_flow(farr: np.ndarray) -> torch.Tensor:
                sl = phase_bounds["speech"]
                seg = farr[sl]
                if self.speech_pad_to_target:
                    seg = _sample_or_pad_frames(seg, self.speech_target_frames)
                elif seg.shape[0] > self.speech_target_frames:
                    seg = _sample_or_pad_frames(seg, self.speech_target_frames)
                t = torch.from_numpy(seg.copy()).to(torch.float32).unsqueeze(-1)
                return _resize_video_tensor(t, self.resize_height, self.resize_half_width).contiguous()

            pre_left   = torch.cat([pre_left,    _extract_flow_chan(flow_mag_left,  "pre",  self.pre_target_frames)],  dim=-1)
            pre_right  = torch.cat([pre_right,   _extract_flow_chan(flow_mag_right, "pre",  self.pre_target_frames)],  dim=-1)
            speech_left  = torch.cat([speech_left,  _extract_speech_flow(flow_mag_left)],  dim=-1)
            speech_right = torch.cat([speech_right, _extract_speech_flow(flow_mag_right)], dim=-1)
            post_left  = torch.cat([post_left,   _extract_flow_chan(flow_mag_left,  "post", self.post_target_frames)], dim=-1)
            post_right = torch.cat([post_right,  _extract_flow_chan(flow_mag_right, "post", self.post_target_frames)], dim=-1)

        # Augmentation applied consistently across all 6 clips
        if self.enable_aug:
            if random.random() < self.aug_hflip_prob:
                # Horizontal flip: swap left<->right to preserve asymmetry semantics
                pre_left, pre_right = _flip_tensor(pre_right), _flip_tensor(pre_left)
                speech_left, speech_right = _flip_tensor(speech_right), _flip_tensor(speech_left)
                post_left, post_right = _flip_tensor(post_right), _flip_tensor(post_left)
                # Also mirror the asymmetry features
                phase_asym_raw = -phase_asym_raw
                phase_side_feats = phase_side_feats[:, [1, 0], :]

            if self.aug_brightness_delta > 0:
                b = random.uniform(-self.aug_brightness_delta, self.aug_brightness_delta)
                for clip in [pre_left, pre_right, speech_left, speech_right, post_left, post_right]:
                    # only jitter RGB channels; leave any extra channel (e.g. flow) unchanged
                    clip[..., :3].add_(b).clamp_(0.0, 1.0)

            if self.aug_contrast_delta > 0:
                c = random.uniform(1.0 - self.aug_contrast_delta, 1.0 + self.aug_contrast_delta)
                for clip in [pre_left, pre_right, speech_left, speech_right, post_left, post_right]:
                    rgb_slice = clip[..., :3]
                    mean = rgb_slice.mean()
                    clip[..., :3] = (rgb_slice - mean).mul_(c).add_(mean).clamp_(0.0, 1.0)

        out = dict(item)
        out.update({
            "pre_left_video": pre_left,
            "pre_right_video": pre_right,
            "speech_left_video": speech_left,
            "speech_right_video": speech_right,
            "post_left_video": post_left,
            "post_right_video": post_right,
            "speech_num_frames": speech_num_frames,
            "phase_side_flow_features": torch.from_numpy(phase_side_feats).float(),
            "phase_asym_flow_features": torch.from_numpy(phase_asym_raw).float(),
            "recovery_ratio": torch.from_numpy(recovery_ratio).float(),
        })
        return out

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        local_skipped = 0
        current = int(idx)
        max_attempts = max(1, self.runtime_max_retries + 1)

        for _ in range(max_attempts):
            try:
                out = self._load_item_by_index(current)
                out["runtime_skipped_count"] = local_skipped
                return out
            except Exception as exc:
                if self.runtime_missing_policy == "error":
                    raise
                local_skipped += 1
                if len(self.samples) <= 1:
                    raise RuntimeError(f"Failed to load idx={idx}: {exc}") from exc
                # Brief sleep so transient NFS hiccups can recover before retrying
                import time as _time
                _time.sleep(0.5)
                current = random.randint(0, len(self.samples) - 1)

        raise RuntimeError(f"Failed after {max_attempts} attempts at idx={idx}")
