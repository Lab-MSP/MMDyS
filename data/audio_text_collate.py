"""
Batch collator for the audio+text branch.
"""
from __future__ import annotations

from typing import Any, Dict, List

import torch
from transformers import AutoTokenizer


class MSDMAudioTextCollator:
    def __init__(
        self,
        text_model_name: str,
        max_text_length: int = 64,
        tokenizer: Any = None,
    ) -> None:
        self.max_text_length = int(max_text_length)
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(text_model_name)
        if getattr(self.tokenizer, "pad_token", None) is None:
            if getattr(self.tokenizer, "eos_token", None) is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            elif getattr(self.tokenizer, "unk_token", None) is not None:
                self.tokenizer.pad_token = self.tokenizer.unk_token
            else:
                raise ValueError("Tokenizer requires pad/eos/unk token.")

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch_size = len(batch)
        max_audio_len = max(item["waveform"].shape[0] for item in batch)

        audio_values = torch.zeros((batch_size, max_audio_len), dtype=torch.float32)
        audio_attention_mask = torch.zeros((batch_size, max_audio_len), dtype=torch.long)
        for i, item in enumerate(batch):
            wav = item["waveform"]
            audio_values[i, : wav.shape[0]] = wav
            audio_attention_mask[i, : wav.shape[0]] = 1

        transcripts = [item["transcript"] for item in batch]
        text_batch = self.tokenizer(
            transcripts,
            padding=True,
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="pt",
        )

        severity_ids = torch.tensor([item["severity_id"] for item in batch], dtype=torch.long)
        severity_targets = torch.tensor([item["severity_target"] for item in batch], dtype=torch.float32)

        return {
            "sample_ids": [item["sample_id"] for item in batch],
            "speaker_ids": [item["speaker_id"] for item in batch],
            "tasks": [item["task"] for item in batch],
            "task_groups": [item["task_group"] for item in batch],
            "transcripts": transcripts,
            "audio_values": audio_values,
            "audio_attention_mask": audio_attention_mask,
            "input_ids": text_batch["input_ids"],
            "text_attention_mask": text_batch["attention_mask"],
            "severity_ids": severity_ids,
            "severity_targets": severity_targets,
        }
