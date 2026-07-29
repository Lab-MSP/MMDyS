from __future__ import annotations

import json
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


def _resize_frames(frames: np.ndarray, h: int, w: int) -> torch.Tensor:
    """frames: [T, H, W, 3] float32 → [T, H_out, W_out, 3]"""
    t = torch.from_numpy(frames).float()             # [T, H, W, 3]
    x = t.permute(0, 3, 1, 2)                        # [T, 3, H, W]
    x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
    return x.permute(0, 2, 3, 1).contiguous()         # [T, H, W, 3]


class VideoUniformDataset(Dataset):
    """
    Full-video RGB dataset with no phase segmentation.

    Loads the entire video (full frame, not split into hemifaces), resizes to
    (resize_height, resize_width), normalises to [0,1], and returns it as a
    single tensor [T, H, W, 3]. The model splits it into uniform T=16 windows.
    """

    def __init__(
        self,
        split_json_path: str | Path,
        video_root: str | Path,
        split_name: str,
        resize_height: int = 224,
        resize_width: int = 224,
        max_frames: int = 512,          # hard cap to avoid OOM on very long videos
        enable_aug: bool = False,
        aug_hflip_prob: float = 0.5,
        aug_brightness_delta: float = 0.04,
        aug_contrast_delta: float = 0.04,
        missing_policy: str = "skip",
        runtime_missing_policy: str = "skip",
        runtime_max_retries: int = 8,
        min_samples_per_split: int = 1,
    ) -> None:
        self.split_json_path = Path(split_json_path)
        self.video_root = Path(video_root)
        self.split_name = str(split_name)
        self.resize_height = int(resize_height)
        self.resize_width = int(resize_width)
        self.max_frames = int(max_frames)
        self.enable_aug = bool(enable_aug)
        self.aug_hflip_prob = float(aug_hflip_prob)
        self.aug_brightness_delta = float(aug_brightness_delta)
        self.aug_contrast_delta = float(aug_contrast_delta)
        self.missing_policy = str(missing_policy).lower()
        self.runtime_missing_policy = str(runtime_missing_policy).lower()
        self.runtime_max_retries = max(0, int(runtime_max_retries))
        self.min_samples_per_split = int(min_samples_per_split)

        self.samples = self._load_samples()

    def _load_samples(self) -> List[Dict[str, Any]]:
        if not self.split_json_path.exists():
            raise FileNotFoundError(f"Split JSON not found: {self.split_json_path}")
        if not self.video_root.exists():
            raise FileNotFoundError(f"Video root not found: {self.video_root}")

        payload = json.loads(self.split_json_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"Expected list in {self.split_json_path}")

        required = {"filename", "speaker", "task", "severity"}
        items: List[Dict[str, Any]] = []
        skipped_by_reason: Counter = Counter()

        for idx, row in enumerate(payload):
            missing_keys = required - set(row.keys())
            if missing_keys:
                raise KeyError(f"Missing keys {sorted(missing_keys)} at index={idx}")

            severity = str(row["severity"])
            task     = str(row["task"])
            speaker  = str(row["speaker"])
            filename = str(row["filename"])

            if severity not in SEVERITY_TO_ID:
                raise ValueError(f"Unknown severity '{severity}' at index={idx}")
            if task not in TASK_TO_GROUP:
                raise ValueError(f"Unknown task '{task}' at index={idx}")

            video_path = self.video_root / f"{filename}.avi"
            reason = ""
            src_fps, src_frames = 25.0, 0
            if not video_path.exists():
                reason = "missing_video"
            else:
                src_fps, src_frames = _probe_video_meta(video_path)
                if src_frames <= 0:
                    reason = "empty_video"

            if reason:
                if self.missing_policy == "error":
                    raise FileNotFoundError(f"missing_policy=error blocked: {reason} sample={filename}")
                skipped_by_reason[reason] += 1
                continue

            items.append({
                "filename":        filename,
                "sample_id":       filename,
                "video_path":      video_path,
                "speaker_id":      speaker,
                "task":            task,
                "task_group":      TASK_TO_GROUP[task],
                "severity":        severity,
                "severity_id":     SEVERITY_TO_ID[severity],
                "severity_target": SEVERITY_TO_TARGET[severity],
                "dur_sec":         float(row.get("dur_sec", 0.0) or 0.0),
                "src_fps":         float(src_fps),
                "src_frames":      int(src_frames),
            })

        kept = len(items)
        total = len(payload)
        print(
            f"[VideoUniformDataset:{self.split_name}] "
            f"kept={kept}/{total}  skipped={dict(skipped_by_reason)}"
        )
        if kept < self.min_samples_per_split:
            raise ValueError(
                f"Split '{self.split_name}' has only {kept} valid samples "
                f"(min required={self.min_samples_per_split})."
            )
        return items

    def _load_item_by_index(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        rgb, _ = _load_video_rgb(item["video_path"])

        # Hard cap
        if rgb.shape[0] > self.max_frames:
            step = rgb.shape[0] / self.max_frames
            indices = np.round(np.arange(self.max_frames) * step).astype(np.int64)
            rgb = rgb[indices]

        # Normalise and resize
        frames = rgb.astype(np.float32) / 255.0          # [T, H, W, 3]
        video = _resize_frames(frames, self.resize_height, self.resize_width)  # [T, H, W, 3]

        # Augmentation
        if self.enable_aug:
            if random.random() < self.aug_hflip_prob:
                video = video.flip(dims=[2])
            if self.aug_brightness_delta > 0:
                b = random.uniform(-self.aug_brightness_delta, self.aug_brightness_delta)
                video = (video + b).clamp(0.0, 1.0)
            if self.aug_contrast_delta > 0:
                c = random.uniform(1.0 - self.aug_contrast_delta, 1.0 + self.aug_contrast_delta)
                mean = video.mean()
                video = ((video - mean) * c + mean).clamp(0.0, 1.0)

        out = dict(item)
        out["full_video"] = video           # [T, H, W, 3]
        out["num_frames"] = int(video.shape[0])
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
                import time as _time
                _time.sleep(0.5)
                current = random.randint(0, len(self.samples) - 1)
        raise RuntimeError(f"Failed after {max_attempts} attempts at idx={idx}")


class VideoUniformCollator:
    """
    Pads variable-length full_video tensors to the batch maximum T.
    Last-frame repeat padding (same as speech collator in phase_collate.py).
    Output full_video: [B, C, T_max, H, W]
    """

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        clips = [item["full_video"] for item in batch]   # each [T, H, W, 3]
        max_t = max(c.shape[0] for c in clips)
        padded: List[torch.Tensor] = []
        for c in clips:
            if c.shape[0] < max_t:
                pad_n = max_t - c.shape[0]
                tail = c[-1:].expand(pad_n, -1, -1, -1)
                c = torch.cat([c, tail], dim=0)
            padded.append(c)

        # [B, T, H, W, 3] → [B, 3, T, H, W]
        full_video = torch.stack(padded, dim=0).permute(0, 4, 1, 2, 3).contiguous()

        runtime_skipped = int(sum(item.get("runtime_skipped_count", 0) for item in batch))
        return {
            "sample_ids":    [item["sample_id"]  for item in batch],
            "speaker_ids":   [item["speaker_id"] for item in batch],
            "tasks":         [item["task"]       for item in batch],
            "task_groups":   [item["task_group"] for item in batch],
            "severity_ids":  torch.tensor([item["severity_id"] for item in batch], dtype=torch.long),
            "full_video":    full_video,                   # [B, 3, T_max, H, W]
            "num_frames":    torch.tensor([item["num_frames"] for item in batch], dtype=torch.long),
            "runtime_skipped_count": runtime_skipped,
        }
