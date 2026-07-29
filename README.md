# MM_Dys_Repo — Multimodal Dysarthria Severity Classification

Code for 4-class dysarthria severity classification (normal / mild / moderate / severe) from the MSDM dataset using audio, text, and video modalities.

## Overview

**Task**: Predict dysarthria severity from short speech samples across 8 tasks (syllable, character, word, sentences). 4 severity classes with ordinal targets mapped to [1.0, 2/3, 1/3, 0.0].

**Primary metric**: `f1_final_4cls = 10 × subject-level macro-F1 + sample-level macro-F1`.  Subject-level prediction averages per-utterance softmax probabilities across all utterances per speaker, then takes argmax.

**Best approach**: Audio-Visual Divergence MoE fusion. Three clinically-motivated difference-vector experts (speech-text divergence, bilateral lateralization, cross-modal divergence), gated by their L2 norms.

| Metric | Value |
|--------|-------|
| `f1_final_4cls` | 10.009 (seed 14) |
| `f1_subject_macro_4cls` | 0.922 |
| `f1_sample_macro_4cls` | 0.787 |
| `qwk_subject` | — |

---

## Repository layout

```
MM_Dys_Repo/
├── data/
│   ├── constants.py            # Severity → id / ordinal target mappings
│   ├── audio_text_dataset.py   # Wav + transcript dataset loader
│   ├── audio_text_collate.py   # Wav2Vec2 / RoBERTa tokenisation collator
│   ├── phase_dataset.py        # Video phase dataset (pre/speech/post × left/right)
│   ├── phase_collate.py        # Video + flow descriptor collator
│   ├── av_dataset.py           # Joint audio-video dataset + collator
│   └── __init__.py
├── models/
│   ├── audio_text_mmdys.py     # AudioTextMMDys: Wav2Vec2 + RoBERTa fusion head
│   ├── phase_videomae.py       # PhaseVideoMAEModel: VideoMAE-Small + bilateral hemiface
│   ├── av_fusion_model.py      # AVFusionModel: divergence_moe and other fusion modes
│   └── __init__.py
├── losses/
│   ├── objectives.py           # All loss functions + compute_*_loss entry points
│   └── __init__.py
├── trainers/
│   ├── trainer.py              # SeverityGuidedTrainer (video + audio branches)
│   ├── metrics.py              # compute_eval_metrics: f1_final_4cls, QWK, confusion matrices
│   └── __init__.py
├── utils/
│   ├── config.py               # YAML config loader with sub-config merging
│   ├── io.py                   # save_json / save_csv_rows
│   ├── runtime.py              # set_seed with deterministic mode
│   └── __init__.py
├── scripts/
│   ├── precompute_descriptors.py  # Build 14-D flow descriptor cache from SEA-RAFT NPZ
│   ├── evaluate.py                # Standalone evaluation from a saved checkpoint
│   └── plot_metrics.py            # Plot confusion matrices and per-task F1 from metrics JSON
├── configs/
│   ├── data/                   # video_phase.yaml, audio_text.yaml, av_joint.yaml
│   ├── model/                  # video_phase_full.yaml, audio_text.yaml, av_fusion_divergence_moe.yaml
│   ├── loss/                   # video_phase.yaml, audio_text.yaml, av_fusion.yaml
│   ├── train/                  # video_phase.yaml, audio_text.yaml, av_fusion.yaml
│   └── experiments/            # video_phase_official.yaml, audio_text_official.yaml, av_fusion_official.yaml
├── train_video.py              # Entry point: video/flow branch
├── train_audio.py              # Entry point: speech/text branch
└── train_av.py                 # Entry point: AV Divergence MoE fusion
```

---

## Prerequisites

### Dataset access

The MSDM dataset is available **by request only**. To obtain access, contact the dataset authors and follow the access procedure described in the original MSDM paper.

Once access is granted, organise the data in the following layout (exact directory names are your choice — you will point to them in the config files):

```
/path/to/msdm/
├── splits/
│   ├── msdm_train.json
│   ├── msdm_dev.json
│   └── msdm_test.json
├── video/
│   ├── N_F_10001_G1_task1_1_S00000.avi
│   ├── N_F_10001_G1_task1_1_S00001.avi
│   └── ...                               # one .avi per utterance
└── audio/
    └── wav/
        ├── N_F_10001_G1_task1_1_S00000.wav
        └── ...                           # one .wav per utterance
```

**Split JSON format** — each file is a JSON array where every element has:

```json
{
  "filename":   "N_F_10001_G1_task1_1_S00000",
  "speaker":    "N_F_10001",
  "task":       "task1",
  "severity":   "norm",
  "dur_sec":    1.23,
  "transcript": "哎"
}
```

`filename` is the bare stem used to locate `video/<filename>.avi` and `audio/wav/<filename>.wav`. `severity` must be one of `norm`, `mild`, `moderate`, `severe`. `transcript` is required for the audio+text and AV branches.

### Requirements

```bash
pip install torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

### Preprocessing (video and AV branches)

The video and AV fusion branches require two preprocessing steps before training. Audio-only training can skip this section.

#### Step 1 — Optical flow extraction with SEA-RAFT

Clone and set up [SEA-RAFT](https://github.com/princeton-vl/SEA-RAFT):

```bash
git clone https://github.com/princeton-vl/SEA-RAFT.git /path/to/SEA-RAFT
cd /path/to/SEA-RAFT
# follow the SEA-RAFT README to install dependencies and download model weights
```

Run SEA-RAFT on the MSDM videos to produce per-utterance `.npz` flow files. Each output file (`<filename>.npz`) contains:
- `flow`: `[T, H, W, 2]` — optical flow vectors (dx, dy)
- `deform_mag`: `[T, H, W]` — per-pixel flow magnitude

Use `scripts/extract_flow_features.py` (included in this repo), which wraps SEA-RAFT's RAFT model and writes one `.npz` per utterance. It must be able to import from the SEA-RAFT repo — set the `SEA_RAFT_ROOT` environment variable to point to your SEA-RAFT clone:

```bash
export SEA_RAFT_ROOT=/path/to/SEA-RAFT

python scripts/extract_flow_features.py \
    --video-dir    /path/to/msdm/video \
    --output-dir   /path/to/sea_raft_output \
    --cfg          /path/to/SEA-RAFT/config/eval/spring-M.json \
    --checkpoint   /path/to/SEA-RAFT/checkpoints/SEA-RAFT-models/Tartan-C-T-TSKH-spring540x960-M.pth \
    --overwrite
```

The `.npz` files are written to `<output-dir>/npz/` (referred to as `<sea_raft_npz_dir>` below).

#### Step 2 — Precompute 14-D flow descriptors

Run once after Step 1, before training:

```bash
python scripts/precompute_descriptors.py \
    --video-root /path/to/msdm/video \
    --npz-root   <sea_raft_npz_dir> \
    --split-dir  /path/to/msdm/splits \
    --output-dir /path/to/descriptor_cache \
    --workers 8
```

This reads each SEA-RAFT `.npz` alongside its `.avi` file and writes a single per-utterance `.npz` into `--output-dir` containing the 14-D descriptor vector for each phase × hemiface combination.

#### Update config paths

After completing both preprocessing steps, fill in the `/path/to/...` placeholders in the data configs:

| Config file | Keys to update |
|---|---|
| `configs/data/video_phase.yaml` | `train/dev/test_split_json`, `video_root`, `flow_feature_root`, `descriptor_cache_dir` |
| `configs/data/audio_text.yaml` | `train/dev/test_split_json`, `wav_root` |
| `configs/data/av_joint.yaml` | `train/dev/test_split_json`, `video_root`, `flow_feature_root`, `wav_root`, `descriptor_cache_dir` |

Set `flow_feature_root` to `<output-dir>/npz` (the npz subdirectory from Step 1) and `descriptor_cache_dir` to the `--output-dir` from Step 2.

---

## Training (3-step pipeline)

### Step 1 — Video branch

```bash
for seed in 17 123 42; do
  python train_video.py \
    --config configs/experiments/video_phase_official.yaml \
    --seed $seed \
    --output-dir outputs/video_phase_official/seed_$seed
done
```

Architecture: VideoMAE-Small (16 frames) encodes 6 clips (pre/speech/post × left/right hemiface). 14-D optical flow descriptors projected and concatenated with visual tokens. Phase-relation tokens aggregated by multi-head attention → sample embedding → 4-class severity head. Only VideoMAE blocks 9–11 unfrozen (no progressive unfreeze — prevents overfitting).

### Step 2 — Audio + text branch

```bash
for seed in 17 42 29; do
  python train_audio.py \
    --config configs/experiments/audio_text_official.yaml \
    --seed $seed \
    --output-dir outputs/audio_text_official/seed_$seed
done
```

Architecture: Wav2Vec2-large-XLSR-53-Chinese + Chinese RoBERTa-WWM-ext-large. Masked attentive pooling on each encoder, projected to shared 256-D space, concat fusion `[z_audio; z_text; |z_audio − z_text|]` → 4-class head. Top-6 Wav2Vec2 blocks fine-tuned; RoBERTa fully frozen.

### Step 3 — AV Divergence MoE fusion

First update `configs/model/av_fusion_divergence_moe.yaml` with the checkpoint paths from the best single-seed runs of each branch:

```yaml
video_ckpt: outputs/video_phase_official/seed_123/best_epochXXX.pt
audio_ckpt: outputs/audio_text_official/seed_42/best_epochXXX.pt
```

Then run:

```bash
for seed in 14 42 123; do
  python train_av.py \
    --config configs/experiments/av_fusion_official.yaml \
    --seed $seed \
    --output-dir outputs/av_fusion_official/seed_$seed
done
```

The AV model loads both branch checkpoints, freezes the audio backbone for the first 2 epochs, then unfreezes the top-2 Wav2Vec2 blocks from epoch 3. Sub-epoch evaluation runs every 500 steps.

---

## Evaluation

```bash
python scripts/evaluate.py \
  --checkpoint outputs/av_fusion_official/seed_14/best_e001s000500.pt \
  --test-split /path/to/msdm_official_splits/msdm_test.json
```

Metrics are written to `test_metrics_<suffix>.json` in the checkpoint directory. Confusion matrices (sample-level and subject-level, 4×4) are included under `sample_confusion_matrix` and `subject_confusion_matrix`.

### Plot confusion matrices and task F1

```bash
python scripts/plot_metrics.py \
  --json outputs/av_fusion_official/seed_14/test_metrics_best.json
```

---

## Loss functions

| Loss | Applies to |
|------|-----------|
| `intra_similarity_l1_loss` | Video branch (consensus score → ordinal target L1) |
| `intra_similarity_mse_loss` | Audio branch (A↔T similarity → ordinal target MSE) |
| `soft_diagonal_phase_consensus_loss` | Video branch (per-phase consensus KL) |
| `ordinal_ranking_loss` | All branches (pairwise severity margin) |
| `severity_ce_loss` | All branches (cross-entropy with label smoothing) |
| `soft_diagonal_audio_text_loss` | Audio branch (A↔T soft-diagonal InfoNCE) |
| `cross_modal_alignment_loss` | AV fusion (video ↔ audio InfoNCE) |
| `cross_modal_ordinal_loss` | AV fusion (cross-modal ordinal ranking) |

Top-level loss functions per branch:
- `compute_total_loss` — video branch
- `compute_audio_text_loss` — audio/text branch
- `compute_av_loss` — AV fusion (handles `divergence_moe` auxiliary per-expert CE)

---

## Key design notes

**Phase segmentation**: Each video split into pre-speech (1.0 s), speech, and post-speech (0.3 s) windows. Ultra-short clips (< 0.7 s) use the full clip as speech. Short clips (0.7–1.3 s) use 0.4 s for pre.

**Hemiface splitting**: Videos cropped to 224×224, split left/right along the vertical midline. Each hemiface processed independently for bilateral comparison.

**14-D flow descriptors** (per phase × hemiface): temporal magnitude statistics (mean, std, peak, AUC, smoothness, entropy), speech-phase dynamics (early–late diff, slope, peak latency), vertical motion (bias, peak), upper/lower jaw ratio, spatial entropy, high-activation ratio.

**Divergence MoE fusion**: Three experts each receive a clinically-motivated difference vector — (1) z_audio − z_text captures speech-text divergence, (2) mean(emb_L − emb_R) across phases captures bilateral lateralization, (3) z_video − z_audio captures cross-modal disagreement. A scalar-norm gate (softmax over L2 norms of each expert input) determines the mixture weights. Each expert is also supervised by an auxiliary CE loss (weight 0.3) to prevent gate collapse.

**Evaluation**: Subject-level prediction averages per-utterance softmax probabilities per speaker then takes argmax. `f1_final_4cls = 10 × f1_subject_macro + f1_sample_macro` is the primary metric used for checkpoint selection.
