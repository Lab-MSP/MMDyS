#!/usr/bin/env python3
"""Pipeline smoke test — no real MSDM data or pretrained weights required.

Generates one synthetic utterance per severity class (4 total), runs the full
data-loading and model forward-pass for all three branches (video, audio+text,
AV fusion), and checks that output shapes are correct and values are finite.

HuggingFace backbones are replaced with randomly-initialised tiny models so
the test runs offline without downloading any weights.

Usage (from repo root):
    python scripts/smoke_test.py

All artefacts are written to a temp directory and cleaned up on exit.
Exit code 0 = pass, non-zero = failure.
"""

from __future__ import annotations

import json
import shutil
import struct
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Synthetic data parameters — match real MSDM video/audio format
# ---------------------------------------------------------------------------

SEVERITIES = ["norm", "mild", "moderate", "severe"]
SAMPLE_RATE = 16_000
FPS = 30
DUR_SEC = 1.5
NUM_FRAMES = int(FPS * DUR_SEC)  # 45
VIDEO_H = VIDEO_W = 96
AUDIO_SAMPLES = int(SAMPLE_RATE * DUR_SEC)
FLOW_T = NUM_FRAMES - 1  # one flow vec per adjacent frame pair


# ---------------------------------------------------------------------------
# Tiny random HF model configs — used to replace real pretrained backbones
# ---------------------------------------------------------------------------

def _tiny_videomae_config(num_frames: int = 16, image_size: int = 96):
    from transformers import VideoMAEConfig
    return VideoMAEConfig(
        num_frames=num_frames,
        image_size=image_size,
        patch_size=16,
        num_channels=3,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
        tubelet_size=2,
    )


def _tiny_wav2vec2_config():
    from transformers import Wav2Vec2Config
    return Wav2Vec2Config(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
        feat_extract_norm="layer",
        feat_proj_dropout=0.0,
        layerdrop=0.0,
        apply_spec_augment=False,
        conv_dim=(32, 32, 32, 32, 32, 32, 32),
        conv_stride=(5, 2, 2, 2, 2, 2, 2),
        conv_kernel=(10, 3, 3, 3, 3, 2, 2),
        conv_bias=False,
    )


def _tiny_roberta_config():
    from transformers import RobertaConfig
    return RobertaConfig(
        vocab_size=100,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
        max_position_embeddings=64,
        type_vocab_size=1,
    )


@contextmanager
def _patch_pretrained_loading():
    """Replace all HF from_pretrained calls with tiny random models/tokenizers."""
    from transformers import AutoModel, AutoTokenizer, VideoMAEModel, Wav2Vec2Model

    def fake_videomae(name_or_path, num_frames=16, image_size=96, **kw):
        cfg = _tiny_videomae_config(num_frames=num_frames, image_size=image_size)
        return VideoMAEModel(cfg)

    def fake_wav2vec2(name_or_path, **kw):
        return Wav2Vec2Model(_tiny_wav2vec2_config())

    def fake_automodel(name_or_path, **kw):
        from transformers import RobertaModel
        return RobertaModel(_tiny_roberta_config())

    def fake_tokenizer(name_or_path, **kw):
        # Use a real PreTrainedTokenizerFast with a tiny in-memory vocab so no
        # download is needed and all tokenizer_utils_base attributes are present.
        from tokenizers import Tokenizer
        from tokenizers.models import WordPiece
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import PreTrainedTokenizerFast

        vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "测": 4, "试": 5}
        fast_tok = Tokenizer(WordPiece(vocab=vocab, unk_token="[UNK]"))
        fast_tok.pre_tokenizer = Whitespace()
        tok = PreTrainedTokenizerFast(
            tokenizer_object=fast_tok,
            unk_token="[UNK]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            sep_token="[SEP]",
            model_max_length=64,
        )
        return tok

    with (
        patch.object(VideoMAEModel,  "from_pretrained", staticmethod(fake_videomae)),
        patch.object(Wav2Vec2Model,  "from_pretrained", staticmethod(fake_wav2vec2)),
        patch.object(AutoModel,      "from_pretrained", staticmethod(fake_automodel)),
        patch.object(AutoTokenizer,  "from_pretrained", staticmethod(fake_tokenizer)),
    ):
        yield


# ---------------------------------------------------------------------------
# Synthetic fixture creation
# ---------------------------------------------------------------------------

def _write_avi(path: Path, num_frames: int, h: int, w: int, fps: float) -> None:
    import cv2
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    rng = np.random.default_rng(0)
    for _ in range(num_frames):
        out.write(rng.integers(0, 256, (h, w, 3), dtype=np.uint8))
    out.release()


def _write_wav(path: Path, num_samples: int, sample_rate: int) -> None:
    data = (np.random.default_rng(0).standard_normal(num_samples) * 0.05 * 32767).astype(np.int16)
    data_bytes = data.tobytes()
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + len(data_bytes), b"WAVE",
        b"fmt ", 16, 1, 1, sample_rate,
        sample_rate * 2, 2, 16,
        b"data", len(data_bytes),
    )
    path.write_bytes(header + data_bytes)


def _write_flow_npy(path: Path, flow_t: int, h: int, w: int) -> None:
    arr = np.random.default_rng(0).random((flow_t, h, w), dtype=np.float32).astype(np.float16)
    np.save(path, arr)


def _write_descriptor_npz(path: Path) -> None:
    rng = np.random.default_rng(0)
    np.savez(
        path,
        phase_side_feats=rng.random((3, 2, 14), dtype=np.float32),
        phase_asym_raw=rng.random((3, 14), dtype=np.float32),
        recovery_ratio=rng.random(2, dtype=np.float32),
    )


def build_fixture(tmp: Path) -> Dict[str, Path]:
    video_dir = tmp / "video"
    wav_dir   = tmp / "audio" / "wav"
    flow_dir  = tmp / "flow_mag"
    desc_dir  = tmp / "descriptor_cache"
    split_dir = tmp / "splits"
    for d in (video_dir, wav_dir, flow_dir, desc_dir, split_dir):
        d.mkdir(parents=True)

    samples = []
    for i, sev in enumerate(SEVERITIES):
        spk = f"SMOKE_{i:04d}"
        fn  = f"{spk}_task1_1_S00000"
        _write_avi(video_dir / f"{fn}.avi", NUM_FRAMES, VIDEO_H, VIDEO_W, float(FPS))
        _write_wav(wav_dir   / f"{fn}.wav", AUDIO_SAMPLES, SAMPLE_RATE)
        _write_flow_npy(flow_dir / f"{fn}.npy", FLOW_T, VIDEO_H, VIDEO_W)
        _write_descriptor_npz(desc_dir / f"{fn}.npz")
        samples.append({
            "filename":   fn,
            "speaker":    spk,
            "task":       "task1",
            "severity":   sev,
            "dur_sec":    DUR_SEC,
            "transcript": "测试",
        })

    for split in ("train", "dev", "test"):
        (split_dir / f"msdm_{split}.json").write_text(json.dumps(samples))

    return {
        "video_dir": video_dir,
        "wav_dir":   wav_dir,
        "flow_dir":  flow_dir,
        "desc_dir":  desc_dir,
        "split_dir": split_dir,
    }


# ---------------------------------------------------------------------------
# Per-branch tests
# ---------------------------------------------------------------------------

def _assert_finite(t: torch.Tensor, name: str) -> None:
    assert torch.isfinite(t).all(), f"{name} contains non-finite values"


def test_video_branch(paths: Dict[str, Path]) -> None:
    print("  building dataset...")
    from data import MSDMPhaseCollator, MSDMPhaseDataset

    ds = MSDMPhaseDataset(
        split_json_path=str(paths["split_dir"] / "msdm_train.json"),
        video_root=str(paths["video_dir"]),
        flow_feature_root=str(paths["flow_dir"]),
        descriptor_cache_dir=str(paths["desc_dir"]),
        split_name="train",
        resize_height=VIDEO_H,
        resize_half_width=VIDEO_W,
        use_flow_descriptors=True,
        require_flow_ok=False,
        missing_policy="skip",
        min_samples_per_split=1,
    )
    assert len(ds) > 0, "video dataset loaded 0 samples"
    batch = MSDMPhaseCollator()([ds[i] for i in range(len(ds))])

    print("  building model (random weights)...")
    with _patch_pretrained_loading():
        from models import PhaseVideoMAEModel
        model = PhaseVideoMAEModel(use_pretrained=True, videomae_num_frames=16, image_size=VIDEO_H, flow_input_dim=14)

    model.eval()
    with torch.no_grad():
        out = model(
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
    logits = out["severity_logits"]
    assert logits.shape == (len(ds), 4), f"unexpected shape {logits.shape}"
    _assert_finite(logits, "video logits")
    print(f"  OK — logits {tuple(logits.shape)}")


def test_audio_branch(paths: Dict[str, Path]) -> None:
    print("  building dataset...")
    from data import MSDMAudioTextCollator, MSDMAudioTextDataset

    ds = MSDMAudioTextDataset(
        split_json_path=str(paths["split_dir"] / "msdm_train.json"),
        wav_root=str(paths["wav_dir"]),
        split_name="train",
    )
    assert len(ds) > 0, "audio dataset loaded 0 samples"

    print("  building model (random weights)...")
    with _patch_pretrained_loading():
        from data import MSDMAudioTextCollator
        from models import AudioTextMMDys
        collator = MSDMAudioTextCollator(text_model_name="dummy")
        model = AudioTextMMDys(
            audio_model_name="dummy",
            text_model_name="dummy",
        )

    batch = collator([ds[i] for i in range(len(ds))])
    model.eval()
    with torch.no_grad():
        out = model(
            audio_values=batch["audio_values"],
            audio_attention_mask=batch["audio_attention_mask"],
            input_ids=batch.get("input_ids"),
            text_attention_mask=batch.get("text_attention_mask"),
        )

    logits = out["severity_logits"]
    assert logits.shape == (len(ds), 4), f"unexpected shape {logits.shape}"
    _assert_finite(logits, "audio logits")
    print(f"  OK — logits {tuple(logits.shape)}")


def test_av_branch(paths: Dict[str, Path]) -> None:
    print("  building datasets...")
    from data import (
        AVCollator, AVDataset,
        MSDMAudioTextCollator, MSDMAudioTextDataset,
        MSDMPhaseCollator, MSDMPhaseDataset,
    )

    vid_ds = MSDMPhaseDataset(
        split_json_path=str(paths["split_dir"] / "msdm_train.json"),
        video_root=str(paths["video_dir"]),
        flow_feature_root=str(paths["flow_dir"]),
        descriptor_cache_dir=str(paths["desc_dir"]),
        split_name="train",
        resize_height=VIDEO_H,
        resize_half_width=VIDEO_W,
        use_flow_descriptors=True,
        require_flow_ok=False,
        missing_policy="skip",
        min_samples_per_split=1,
    )
    aud_ds = MSDMAudioTextDataset(
        split_json_path=str(paths["split_dir"] / "msdm_train.json"),
        wav_root=str(paths["wav_dir"]),
        split_name="train",
    )
    av_ds = AVDataset(vid_ds, aud_ds, split_name="train")
    assert len(av_ds) > 0, "av dataset loaded 0 samples"

    print("  building model and collating batch (random weights)...")
    with _patch_pretrained_loading():
        from models import AudioTextMMDys, AVFusionModel, PhaseVideoMAEModel
        collator = AVCollator(text_model_name="dummy")
        vid_model = PhaseVideoMAEModel(
            use_pretrained=True, videomae_num_frames=16, image_size=VIDEO_H, flow_input_dim=14,
        )
        aud_model = AudioTextMMDys(audio_model_name="dummy", text_model_name="dummy")
        model = AVFusionModel(
            video_model=vid_model,
            audio_model=aud_model,
            fusion_mode="divergence_moe",
        )

    batch = collator([av_ds[i] for i in range(len(av_ds))])
    model.eval()
    with torch.no_grad():
        out = model(
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
            audio_values=batch["audio_audio_values"],
            audio_attention_mask=batch["audio_audio_attention_mask"],
            input_ids=batch.get("audio_input_ids"),
            text_attention_mask=batch.get("audio_text_attention_mask"),
        )

    logits = out["severity_logits"]
    assert logits.shape == (len(av_ds), 4), f"unexpected shape {logits.shape}"
    _assert_finite(logits, "av logits")
    print(f"  OK — logits {tuple(logits.shape)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

    print("=== MMDyS smoke test ===\n")
    tmp = Path(tempfile.mkdtemp(prefix="mmdys_smoke_"))
    try:
        print(f"Generating synthetic fixture in {tmp} ...")
        paths = build_fixture(tmp)
        print(f"  {len(SEVERITIES)} utterances ({', '.join(SEVERITIES)})\n")

        branches = [
            ("video branch  (PhaseVideoMAEModel)",       test_video_branch),
            ("audio branch  (AudioTextMMDys)",           test_audio_branch),
            ("av branch     (AVFusionModel divergence_moe)", test_av_branch),
        ]
        failures = []
        for label, fn in branches:
            print(f"-- {label} --")
            try:
                fn(paths)
            except Exception as exc:
                import traceback
                print(f"  FAIL: {exc}")
                traceback.print_exc()
                failures.append((label, exc))
            print()

        if failures:
            for label, exc in failures:
                print(f"FAIL  {label}: {exc}")
            sys.exit(1)
        else:
            print("All checks passed.")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
