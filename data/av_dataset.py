"""
AVDataset — joint audio-visual dataset for Exp 2 (cross-modal fine-tuning).

Each sample delivers both video phase data and audio+text data for the SAME
filename key, enabling within-batch cross-modal contrastive losses.

Internally it wraps MSDMPhaseDataset and MSDMAudioTextDataset; at __getitem__
it looks up the same sample index in both sub-datasets (using a shared filename
index built at construction time) and returns a combined dict.

The collate_fn (AVCollator) delegates video collation to MSDMPhaseCollator and
audio collation to MSDMAudioTextCollator, then merges the two dicts with
"video_" and "audio_" key prefixes for non-shared keys.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from data.audio_text_collate import MSDMAudioTextCollator
from data.audio_text_dataset import MSDMAudioTextDataset
from data.phase_collate import MSDMPhaseCollator
from data.phase_dataset import MSDMPhaseDataset


class AVDataset(Dataset):
    """
    Joint audio-visual dataset for Exp 2.

    Builds the intersection of filenames available in both sub-datasets.
    Missing-modality samples (one present, other absent) are silently dropped.

    Args:
        video_dataset:  A pre-constructed MSDMPhaseDataset.
        audio_dataset:  A pre-constructed MSDMAudioTextDataset.
        split_name:     For logging only.
    """

    def __init__(
        self,
        video_dataset: MSDMPhaseDataset,
        audio_dataset: MSDMAudioTextDataset,
        split_name: str = "train",
    ) -> None:
        self.video_dataset = video_dataset
        self.audio_dataset = audio_dataset
        self.split_name    = str(split_name)
        self.samples       = self._build_joint_index()

    def _build_joint_index(self) -> List[Dict[str, int]]:
        # Build filename → index maps for both sub-datasets
        vid_map: Dict[str, int] = {}
        for i, s in enumerate(self.video_dataset.samples):
            fn = str(s.get("sample_id", s.get("filename", "")))
            vid_map[fn] = i

        aud_map: Dict[str, int] = {}
        for i, s in enumerate(self.audio_dataset.samples):
            fn = str(s.get("sample_id", s.get("filename", "")))
            aud_map[fn] = i

        common = sorted(set(vid_map.keys()) & set(aud_map.keys()))
        vid_only = len(vid_map) - len(common)
        aud_only = len(aud_map) - len(common)
        print(
            f"[AVDataset/{self.split_name}] joint={len(common)}  "
            f"vid_only={vid_only}  aud_only={aud_only}"
        )
        return [{"filename": fn, "vid_idx": vid_map[fn], "aud_idx": aud_map[fn]}
                for fn in common]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s     = self.samples[idx]
        v_item = self.video_dataset[s["vid_idx"]]
        a_item = self.audio_dataset[s["aud_idx"]]
        # Merge: video items prefixed with "video_", audio with "audio_",
        # shared keys (severity_id, severity_target, speaker_id, task, filename)
        # taken from the video side (both should be identical).
        merged: Dict[str, Any] = {}
        shared = {"severity_id", "severity_target", "speaker_id", "task",
                  "task_group", "sample_id", "filename"}
        for k, v in v_item.items():
            if k in shared:
                merged[k] = v
            else:
                merged[f"video_{k}"] = v
        for k, v in a_item.items():
            if k not in shared:
                merged[f"audio_{k}"] = v
        return merged


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------

class AVCollator:
    """
    Collate joint video+audio batches for AVDataset.

    Uses MSDMPhaseCollator for the video keys (strip "video_" prefix),
    and MSDMAudioTextCollator for the audio keys (strip "audio_" prefix),
    then merges both output dicts.  Shared keys (severity_ids, etc.) come
    from the video collator output.
    """

    def __init__(
        self,
        text_model_name: str,
        max_text_length: int = 64,
        tokenizer: Optional[Any] = None,
    ) -> None:
        self.video_collator = MSDMPhaseCollator()
        self.audio_collator = MSDMAudioTextCollator(
            text_model_name=text_model_name,
            max_text_length=int(max_text_length),
            tokenizer=tokenizer,
        )

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Split the merged dicts back into video-only and audio-only sub-dicts
        shared_keys = {"severity_id", "severity_target", "speaker_id", "task",
                       "task_group", "sample_id", "filename"}

        video_batch: List[Dict[str, Any]] = []
        audio_batch: List[Dict[str, Any]] = []

        for item in batch:
            v_item: Dict[str, Any] = {}
            a_item: Dict[str, Any] = {}
            for k, v in item.items():
                if k in shared_keys:
                    # Both sub-collators need severity_id etc. under plain names
                    v_item[k] = v
                    a_item[k] = v
                elif k.startswith("video_"):
                    v_item[k[len("video_"):]] = v
                elif k.startswith("audio_"):
                    a_item[k[len("audio_"):]] = v
            video_batch.append(v_item)
            audio_batch.append(a_item)

        v_out = self.video_collator(video_batch)
        a_out = self.audio_collator(audio_batch)

        # Merge: all video keys pass through unchanged; audio keys get "audio_" prefix
        # except for keys already present in v_out (severity_ids etc.)
        merged = dict(v_out)
        for k, v in a_out.items():
            if k not in merged:
                merged[f"audio_{k}"] = v
        return merged
