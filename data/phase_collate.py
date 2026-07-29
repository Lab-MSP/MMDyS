from __future__ import annotations

from typing import Any, Dict, List

import torch


class MSDMPhaseCollator:
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        def _stack_video(key: str) -> torch.Tensor:
            # items are [T, H, W, 3]; stack to [B, T, H, W, 3] then permute to [B, 3, T, H, W]
            return torch.stack([item[key] for item in batch], dim=0).permute(0, 4, 1, 2, 3).contiguous()

        def _pad_speech(key: str) -> torch.Tensor:
            """Pad variable-length speech clips to the batch maximum T.
            Padding uses last-frame repeat (not zeros) so VideoMAE sees a frozen face
            rather than black frames — attention pooling can still learn to down-weight
            the repeated tail, but it won't be confused by out-of-distribution black input.
            Output: [B, 3, T_max, H, W]
            """
            clips: List[torch.Tensor] = [item[key] for item in batch]  # each [T, H, W, 3]
            max_t = max(c.shape[0] for c in clips)
            padded: List[torch.Tensor] = []
            for c in clips:
                if c.shape[0] < max_t:
                    pad_n = max_t - c.shape[0]
                    tail = c[-1:].expand(pad_n, -1, -1, -1)
                    c = torch.cat([c, tail], dim=0)
                padded.append(c)
            return torch.stack(padded, dim=0).permute(0, 4, 1, 2, 3).contiguous()

        runtime_skipped = int(sum(item.get("runtime_skipped_count", 0) for item in batch))

        return {
            "sample_ids": [item["sample_id"] for item in batch],
            "speaker_ids": [item["speaker_id"] for item in batch],
            "tasks": [item["task"] for item in batch],
            "task_groups": [item["task_group"] for item in batch],
            "severity_ids": torch.tensor([item["severity_id"] for item in batch], dtype=torch.long),
            "pre_left_video": _stack_video("pre_left_video"),
            "pre_right_video": _stack_video("pre_right_video"),
            "speech_left_video": _pad_speech("speech_left_video"),
            "speech_right_video": _pad_speech("speech_right_video"),
            "speech_num_frames": torch.tensor(
                [item["speech_num_frames"] for item in batch], dtype=torch.long
            ),  # [B] — actual valid frames before padding
            "post_left_video": _stack_video("post_left_video"),
            "post_right_video": _stack_video("post_right_video"),
            "phase_side_flow_features": torch.stack(
                [item["phase_side_flow_features"] for item in batch], dim=0
            ),  # [B, 3, 2, 8]
            "phase_asym_flow_features": torch.stack(
                [item["phase_asym_flow_features"] for item in batch], dim=0
            ),  # [B, 3, 8]
            "recovery_ratio": torch.stack(
                [item["recovery_ratio"] for item in batch], dim=0
            ),  # [B, 2]
            "runtime_skipped_count": runtime_skipped,
        }
