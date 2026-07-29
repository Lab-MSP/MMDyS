"""
Audio+text dataset for MMDys-Claude speech branch.
Adapted from MMDys-CL/data/dataset.py to use msdm_baseline_splits_v2.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List

import torch
import torchaudio
from torch.utils.data import Dataset

from data.constants import SEVERITY_TO_ID, SEVERITY_TO_TARGET, TASK_TO_GROUP


class MSDMAudioTextDataset(Dataset):
    def __init__(
        self,
        split_json_path: str | Path,
        wav_root: str | Path,
        split_name: str,
        target_sample_rate: int = 16000,
        max_audio_seconds: float = 8.0,
        random_crop_train: bool = True,
    ) -> None:
        self.split_json_path = Path(split_json_path)
        self.wav_root = Path(wav_root)
        self.split_name = split_name
        self.target_sample_rate = int(target_sample_rate)
        self.max_audio_seconds = float(max_audio_seconds)
        self.random_crop_train = bool(random_crop_train)
        self.samples = self._load_samples()

    def _load_samples(self) -> List[Dict[str, Any]]:
        if not self.split_json_path.exists():
            raise FileNotFoundError(f"Split json not found: {self.split_json_path}")
        if not self.wav_root.exists():
            raise FileNotFoundError(f"Wav root not found: {self.wav_root}")

        with self.split_json_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        if not isinstance(payload, list):
            raise ValueError(f"Expected JSON list in {self.split_json_path}")

        required = {"filename", "speaker", "task", "severity", "transcript"}
        items: List[Dict[str, Any]] = []
        skipped = 0

        for idx, row in enumerate(payload):
            missing = required - set(row.keys())
            if missing:
                raise KeyError(f"Missing keys {sorted(missing)} at index={idx}")

            severity = str(row["severity"])
            if severity not in SEVERITY_TO_ID:
                raise ValueError(f"Unsupported severity '{severity}' at index={idx}")

            task = str(row["task"])
            if task not in TASK_TO_GROUP:
                raise ValueError(f"Unsupported task '{task}' at index={idx}")

            filename = str(row["filename"])
            wav_path = self.wav_root / f"{filename}.wav"
            if not wav_path.exists():
                skipped += 1
                continue

            transcript = str(row["transcript"])
            speaker = str(row["speaker"])
            dur_sec = float(row.get("dur_sec", 0.0))

            items.append(
                {
                    "filename": filename,
                    "sample_id": filename,
                    "wav_path": wav_path,
                    "speaker_id": speaker,
                    "task": task,
                    "task_group": TASK_TO_GROUP[task],
                    "severity": severity,
                    "severity_id": SEVERITY_TO_ID[severity],
                    "severity_target": SEVERITY_TO_TARGET[severity],
                    "transcript": transcript,
                    "dur_sec": dur_sec,
                }
            )

        if skipped > 0:
            print(f"[MSDMAudioTextDataset:{self.split_name}] skipped {skipped}/{len(payload)} entries (wav not found)")
        print(f"[MSDMAudioTextDataset:{self.split_name}] loaded {len(items)} samples")
        return items

    def _resample_if_needed(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        if sample_rate == self.target_sample_rate:
            return waveform
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.target_sample_rate)
        return resampler(waveform)

    def _crop_if_needed(self, waveform: torch.Tensor) -> torch.Tensor:
        max_samples = int(self.max_audio_seconds * self.target_sample_rate)
        if waveform.shape[-1] <= max_samples:
            return waveform
        if self.split_name == "train" and self.random_crop_train:
            start = random.randint(0, waveform.shape[-1] - max_samples)
        else:
            start = (waveform.shape[-1] - max_samples) // 2
        return waveform[..., start: start + max_samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        waveform, sample_rate = torchaudio.load(item["wav_path"])
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = self._resample_if_needed(waveform, int(sample_rate))
        waveform = self._crop_if_needed(waveform)
        waveform = waveform.squeeze(0).to(torch.float32)

        out = dict(item)
        out["waveform"] = waveform
        out["audio_num_samples"] = int(waveform.shape[0])
        out["sample_rate"] = self.target_sample_rate
        return out
