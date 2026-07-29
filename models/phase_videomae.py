from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import VideoMAEConfig, VideoMAEModel


def _inflate_patch_embed_channels(
    model: "VideoMAEModel",
    new_in_channels: int,
) -> None:
    """
    Inflate VideoMAE patch-embedding Conv3d from 3 → new_in_channels in-place.

    The extra channel weights are zero-initialised so the model's initial
    behaviour is identical to the 3-channel pretrained weights; the new
    channel(s) are learned from scratch during fine-tuning.
    """
    proj = model.embeddings.patch_embeddings.projection  # Conv3d(3, D, ...)
    old_w = proj.weight.data                             # [D, C_old, kt, kh, kw]
    D, C_old = old_w.shape[0], old_w.shape[1]
    if C_old == new_in_channels:
        return  # nothing to do

    # Build new weight: copy existing channels, zero-init extras
    new_w = torch.zeros(D, new_in_channels, *old_w.shape[2:],
                        dtype=old_w.dtype, device=old_w.device)
    new_w[:, :C_old] = old_w

    # Replace the projection layer
    new_proj = nn.Conv3d(
        new_in_channels, D,
        kernel_size=proj.kernel_size,
        stride=proj.stride,
        padding=proj.padding,
        bias=(proj.bias is not None),
    )
    with torch.no_grad():
        new_proj.weight.copy_(new_w)
        if proj.bias is not None:
            new_proj.bias.copy_(proj.bias.data)

    model.embeddings.patch_embeddings.projection = new_proj
    # Update both the config (for serialisation) and the cached instance attribute
    # that VideoMAEPatchEmbeddings.forward() checks at line 170 of modeling_videomae.py.
    model.config.num_channels = new_in_channels
    model.embeddings.patch_embeddings.num_channels = new_in_channels
    print(f"[patch_embed_inflate] {C_old}ch → {new_in_channels}ch "
          f"(extra {new_in_channels - C_old} channel(s) zero-initialised)")


def _interpolate_videomae_pos_emb(
    src_model: VideoMAEModel,
    tgt_model: VideoMAEModel,
    src_num_frames: int,
    tgt_num_frames: int,
    patch_size: int = 16,
    tubelet_size: int = 2,
) -> None:
    """
    Copy all weights from src_model into tgt_model, bilinearly interpolating any
    positional-embedding tensors whose temporal dimension has changed.

    VideoMAE variants may store a learnable position table of shape
    [1, (T/tubelet)*n_spatial, D] or [(T/tubelet)*n_spatial, D].
    We split out the temporal axis, interpolate, and write the result back.
    If no such parameter is found (e.g. pure sine-cosine PE), this is a no-op
    and the target model already handles arbitrary T correctly.
    """
    hidden_size  = src_model.config.hidden_size
    image_size   = src_model.config.image_size
    n_spatial    = (image_size // patch_size) ** 2
    n_t_src      = src_num_frames  // tubelet_size
    n_t_tgt      = tgt_num_frames  // tubelet_size
    n_tok_src    = n_t_src * n_spatial

    src_state = src_model.state_dict()
    tgt_state = tgt_model.state_dict()
    new_state: Dict[str, torch.Tensor] = {}

    for name, tgt_w in tgt_state.items():
        if name not in src_state:
            new_state[name] = tgt_w          # keep random init
            continue
        src_w = src_state[name]
        if src_w.shape == tgt_w.shape:
            new_state[name] = src_w.clone()  # shapes match — direct copy
        elif n_tok_src in src_w.shape and src_w.dim() >= 2:
            # Positional-embedding tensor; reshape → interpolate temporal → reshape back.
            w = src_w.float()
            squeezed = (w.dim() == 2)
            if squeezed:
                w = w.unsqueeze(0)                              # [1, n_tok_src, D]
            w = w.view(1, n_t_src, n_spatial, hidden_size)
            w = w.permute(0, 3, 1, 2)                          # [1, D, n_t_src, n_spatial]
            w = F.interpolate(w, size=(n_t_tgt, n_spatial), mode="bilinear", align_corners=False)
            w = w.permute(0, 2, 3, 1)                          # [1, n_t_tgt, n_spatial, D]
            w = w.reshape(1, n_t_tgt * n_spatial, hidden_size)
            if squeezed:
                w = w.squeeze(0)
            new_state[name] = w.to(src_w.dtype)
            print(f"[pos_emb_interp] interpolated {name}: "
                  f"{list(src_w.shape)} → {list(new_state[name].shape)}")
        else:
            new_state[name] = tgt_w          # mismatch not due to PE — keep random

    tgt_model.load_state_dict(new_state, strict=False)


class PhaseVideoMAEModel(nn.Module):
    """
    Phase-aware hemiface VideoMAE model for dysarthria severity classification.

    Key fixes vs original:
      - Mean-pool encoder outputs (VideoMAEModel has no CLS token).
      - Speech phase uses attention pooling over windows instead of mean.
      - Phase position embeddings inform the model which phase each token is from.
      - Attention-based phase aggregation (learned query) instead of flat gated MLP.
      - Recovery ratio bounded via log1p in the dataset; no change needed here.
      - Proper 2-layer GLU for the gated aggregation.
    """

    def __init__(
        self,
        hf_backbone_name: str = "MCG-NJU/videomae-small-finetuned-kinetics",
        use_pretrained: bool = True,
        videomae_num_frames: int = 16,
        image_size: int = 224,
        interpolate_pos_emb: bool = False,
        in_channels: int = 3,       # 3=RGB, 4=RGBF (flow magnitude as 4th channel)
        flow_input_dim: int = 8,
        flow_embed_dim: int = 64,
        use_flow_descriptors: bool = True,
        phase_token_dim: int = 256,
        sample_embed_dim: int = 256,
        classifier_hidden_dim: int = 256,
        consensus_hidden_dim: int = 128,
        num_agg_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.hf_backbone_name = str(hf_backbone_name)
        self.videomae_num_frames = int(videomae_num_frames)
        self.use_flow_descriptors = bool(use_flow_descriptors)
        self.in_channels = int(in_channels)

        # ---- VideoMAE encoder ------------------------------------------------
        if use_pretrained:
            native_t = 16  # VideoMAE-Small pretrained frame count
            if interpolate_pos_emb and self.videomae_num_frames != native_t:
                # Load at native T=16 to get the correctly pretrained weights,
                # then bilinearly interpolate the temporal positional embeddings
                # to the target T so no tokens are misaligned or truncated.
                print(f"[pos_emb_interp] loading pretrained at T={native_t}, "
                      f"interpolating to T={self.videomae_num_frames}")
                src = VideoMAEModel.from_pretrained(
                    self.hf_backbone_name,
                    num_frames=native_t,
                    image_size=int(image_size),
                    ignore_mismatched_sizes=False,
                )
                self.video_encoder = VideoMAEModel.from_pretrained(
                    self.hf_backbone_name,
                    num_frames=self.videomae_num_frames,
                    image_size=int(image_size),
                    ignore_mismatched_sizes=True,   # size mismatch expected for PE
                )
                _interpolate_videomae_pos_emb(
                    src, self.video_encoder, native_t, self.videomae_num_frames
                )
                del src
            else:
                self.video_encoder = VideoMAEModel.from_pretrained(
                    self.hf_backbone_name,
                    num_frames=self.videomae_num_frames,
                    image_size=int(image_size),
                    ignore_mismatched_sizes=False,
                )
        else:
            cfg = VideoMAEConfig(
                num_frames=self.videomae_num_frames,
                image_size=int(image_size),
            )
            self.video_encoder = VideoMAEModel(cfg)

        # Inflate patch embed to in_channels (no-op when in_channels==3)
        if self.in_channels != 3:
            _inflate_patch_embed_channels(self.video_encoder, self.in_channels)

        self.visual_dim = int(self.video_encoder.config.hidden_size)  # 384 for videomae-small
        self.flow_embed_dim = int(flow_embed_dim)
        self.side_dim = self.visual_dim + self.flow_embed_dim  # 384+64=448

        # ---- Flow projections ------------------------------------------------
        # Terminal layer is LN (not GELU) so flow embedding norm stays bounded
        # during training — prevents flow from dominating vis in phase_rel_proj.
        self.side_flow_proj = nn.Sequential(
            nn.Linear(int(flow_input_dim), 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, self.flow_embed_dim),
            nn.LayerNorm(self.flow_embed_dim),
        )
        self.asym_flow_proj = nn.Sequential(
            nn.Linear(int(flow_input_dim), 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, self.flow_embed_dim),
            nn.LayerNorm(self.flow_embed_dim),
        )

        # ---- Pre-fusion visual normalisation ---------------------------------
        # VideoMAE residual stream norm grows with sqrt(hidden_size) (~19.6);
        # flow embeddings are ~3–4 at init but can grow during training at 10×
        # the backbone LR. A shared LN here keeps both branches on equal footing
        # before they enter phase_rel_proj.
        self.visual_side_ln = nn.LayerNorm(self.visual_dim)

        # ---- Speech-phase temporal attention pooling -------------------------
        # Scores each window's contribution; single linear → scalar weight.
        self.speech_window_attn = nn.Linear(self.visual_dim, 1)

        # ---- Phase relation projection ---------------------------------------
        # Input: [emb_L, emb_R, |L-R|, L*R, flow_asym] = 4*side_dim + flow_embed_dim
        phase_in_dim = 4 * self.side_dim + self.flow_embed_dim
        self.phase_rel_proj = nn.Sequential(
            nn.Linear(phase_in_dim, phase_token_dim),
            nn.LayerNorm(phase_token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ---- Phase position embeddings ---------------------------------------
        # 3 learnable positions: 0=pre, 1=speech, 2=post
        self.phase_pos_emb = nn.Embedding(3, phase_token_dim)

        # ---- Attention-based phase aggregation ------------------------------
        # Learned query attends over the 3 phase tokens.
        self.phase_agg_query = nn.Parameter(torch.randn(1, 1, phase_token_dim))
        self.phase_agg_attn = nn.MultiheadAttention(
            phase_token_dim, num_heads=int(num_agg_heads), batch_first=True, dropout=dropout
        )

        # ---- Sample projection -----------------------------------------------
        self.sample_proj = nn.Sequential(
            nn.Linear(phase_token_dim, sample_embed_dim),
            nn.LayerNorm(sample_embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ---- Classification head --------------------------------------------
        # Inputs: sample embedding + recovery_ratio (2-D)
        cls_in_dim = sample_embed_dim + 2
        self.severity_head = nn.Sequential(
            nn.Linear(cls_in_dim, classifier_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, 4),
        )

        # ---- Consensus head (scalar severity proxy for intra-sample loss) ----
        # Maps per-phase consensus embedding [side_dim] → scalar ∈ (0,1)
        self.consensus_head = nn.Sequential(
            nn.Linear(self.side_dim, consensus_hidden_dim),
            nn.GELU(),
            nn.Linear(consensus_hidden_dim, 1),
        )

        # ---- Unfreeze schedule (VideoMAE-Small: 12 blocks 0–11) --------------
        self._video_blocks: List[nn.Module] = list(self.video_encoder.encoder.layer)
        self._initial_range = (9, 11)   # last 3 blocks initially
        self._epoch8_range  = (5, 8)
        self._epoch18_range = (1, 4)
        self._epoch8_unfrozen  = False
        self._epoch18_unfrozen = False

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------

    def _encode_clip(self, clip: torch.Tensor) -> torch.Tensor:
        """
        clip : [B, C, T, H, W]  (C=3 for RGB, C=4 for RGBF)
        Returns mean-pooled patch representation [B, D].

        VideoMAEModel.last_hidden_state has shape [B, num_patches, D];
        there is NO CLS token, so we mean-pool all patch tokens.
        """
        t = clip.shape[2]
        if t != self.videomae_num_frames:
            clip = F.interpolate(
                clip,
                size=(self.videomae_num_frames, clip.shape[-2], clip.shape[-1]),
                mode="trilinear",
                align_corners=False,
            )
        # VideoMAE expects [B, T, C, H, W]
        x = clip.permute(0, 2, 1, 3, 4).contiguous()
        out = self.video_encoder(pixel_values=x)
        # Mean-pool all patch tokens — correct for VideoMAE (no CLS token)
        return out.last_hidden_state.mean(dim=1)  # [B, D]

    def _encode_speech(
        self,
        speech_clip: torch.Tensor,
        valid_t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Speech clip [B, 3, T, H, W]; T may vary due to last-frame padding in the collator.
        Split into videomae_num_frames windows, encode each, then attention-pool.

        valid_t: [B] int tensor — actual speech frames per sample before padding.
                 Padding windows (start >= valid_t[b]) are masked to -inf before softmax
                 so they don't contribute to the weighted sum.
        """
        t = speech_clip.shape[2]
        if t <= self.videomae_num_frames:
            return self._encode_clip(speech_clip)

        window_embs: List[torch.Tensor] = []
        window_starts: List[int] = []
        start = 0
        while start < t:
            end = min(t, start + self.videomae_num_frames)
            seg = speech_clip[:, :, start:end]
            if seg.shape[2] < self.videomae_num_frames:
                pad = self.videomae_num_frames - seg.shape[2]
                seg = F.pad(seg, (0, 0, 0, 0, 0, pad))
            window_embs.append(self._encode_clip(seg))  # [B, D]
            window_starts.append(start)
            start += self.videomae_num_frames

        # Stack: [B, n_windows, D]
        stacked = torch.stack(window_embs, dim=1)
        # Attention pooling: score each window → softmax → weighted sum
        scores = self.speech_window_attn(stacked)  # [B, n_windows, 1]

        if valid_t is not None:
            # Mask windows whose start >= valid_t[b] (pure padding)
            # window_starts: [n_windows], valid_t: [B] → mask: [B, n_windows, 1]
            starts = torch.tensor(window_starts, device=speech_clip.device)  # [n_windows]
            vt = valid_t.to(speech_clip.device).unsqueeze(1)                 # [B, 1]
            padding_mask = starts.unsqueeze(0) >= vt                         # [B, n_windows]
            scores = scores.masked_fill(padding_mask.unsqueeze(-1), float("-inf"))

        attn_w = torch.softmax(scores, dim=1)   # [B, n_windows, 1]
        return (stacked * attn_w).sum(dim=1)    # [B, D]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        pre_left_video:   torch.Tensor,
        pre_right_video:  torch.Tensor,
        speech_left_video:  torch.Tensor,
        speech_right_video: torch.Tensor,
        post_left_video:  torch.Tensor,
        post_right_video: torch.Tensor,
        phase_side_flow_features: torch.Tensor,             # [B, 3, 2, 8]
        phase_asym_flow_features: torch.Tensor,             # [B, 3, 8]
        recovery_ratio: torch.Tensor,                       # [B, 2]
        speech_num_frames: Optional[torch.Tensor] = None,  # [B] valid frames before padding
    ) -> Dict[str, torch.Tensor]:

        # ---- Visual encoding (6 clips) --------------------------------------
        vis_pre_l  = self._encode_clip(pre_left_video)
        vis_pre_r  = self._encode_clip(pre_right_video)
        vis_sp_l   = self._encode_speech(speech_left_video,  valid_t=speech_num_frames)
        vis_sp_r   = self._encode_speech(speech_right_video, valid_t=speech_num_frames)
        vis_post_l = self._encode_clip(post_left_video)
        vis_post_r = self._encode_clip(post_right_video)

        # ---- Flow projections -----------------------------------------------
        if self.use_flow_descriptors:
            flow_side = self.side_flow_proj(phase_side_flow_features)   # [B, 3, 2, flow_embed_dim]
            flow_asym = self.asym_flow_proj(phase_asym_flow_features)   # [B, 3, flow_embed_dim]
            rec = recovery_ratio.to(vis_pre_l.dtype)
        else:
            B, dev, dt = vis_pre_l.shape[0], vis_pre_l.device, vis_pre_l.dtype
            flow_side = torch.zeros((B, 3, 2, self.flow_embed_dim), device=dev, dtype=dt)
            flow_asym = torch.zeros((B, 3, self.flow_embed_dim), device=dev, dtype=dt)
            rec = torch.zeros((B, 2), device=dev, dtype=dt)

        # ---- Assemble side embeddings per phase [B, 3, side_dim] ------------
        vis_left  = torch.stack([vis_pre_l, vis_sp_l, vis_post_l], dim=1)   # [B, 3, 384]
        vis_right = torch.stack([vis_pre_r, vis_sp_r, vis_post_r], dim=1)

        vis_left_n  = self.visual_side_ln(vis_left)   # [B, 3, 384] — normalise before fusion
        vis_right_n = self.visual_side_ln(vis_right)
        emb_left  = torch.cat([vis_left_n,  flow_side[:, :, 0, :]], dim=-1)   # [B, 3, 448]
        emb_right = torch.cat([vis_right_n, flow_side[:, :, 1, :]], dim=-1)

        # ---- Relational features per phase ----------------------------------
        rel = torch.cat([
            emb_left,
            emb_right,
            torch.abs(emb_left - emb_right),
            emb_left * emb_right,
            flow_asym,
        ], dim=-1)  # [B, 3, 4*448+64]

        phase_tokens = self.phase_rel_proj(rel)  # [B, 3, phase_token_dim]

        # ---- Phase position embeddings --------------------------------------
        phase_idx = torch.arange(3, device=phase_tokens.device)
        phase_tokens = phase_tokens + self.phase_pos_emb(phase_idx).unsqueeze(0)  # [B, 3, D]

        # ---- Attention-based aggregation ------------------------------------
        B = phase_tokens.shape[0]
        query = self.phase_agg_query.expand(B, -1, -1)  # [B, 1, D]
        z_agg, _ = self.phase_agg_attn(query, phase_tokens, phase_tokens)  # [B, 1, D]
        z_sample = self.sample_proj(z_agg.squeeze(1))  # [B, sample_embed_dim]

        # ---- Classification -------------------------------------------------
        cls_in = torch.cat([z_sample, rec], dim=-1)
        severity_logits = self.severity_head(cls_in)  # [B, 4]

        # ---- Consensus embeddings for contrastive loss ----------------------
        c_phase = 0.5 * (emb_left + emb_right)  # [B, 3, side_dim]
        c_sample = c_phase.mean(dim=1)           # [B, side_dim]

        # Scalar severity proxy (1.0=norm → 0.0=severe) for ranking & intra loss
        consensus_similarity = torch.sigmoid(self.consensus_head(c_sample).squeeze(-1))  # [B]

        # Expected severity score for ordinal ranking
        severity_levels = torch.tensor(
            [1.0, 2.0 / 3.0, 1.0 / 3.0, 0.0],
            device=severity_logits.device,
            dtype=severity_logits.dtype,
        )
        severity_score = torch.softmax(severity_logits, dim=-1).matmul(severity_levels)

        return {
            "severity_logits":           severity_logits,
            "similarity":                consensus_similarity,  # alias for trainer compat
            "consensus_similarity":      consensus_similarity,
            "severity_score":            severity_score,
            "phase_side_embeddings":     torch.stack([emb_left, emb_right], dim=2),  # [B,3,2,448]
            "phase_consensus_embeddings": c_phase,                                    # [B,3,448]
            "phase_tokens":              phase_tokens,
            "z_sample":                  z_sample,
            "recovery_ratio":            rec,
        }

    # ------------------------------------------------------------------
    # Training configuration
    # ------------------------------------------------------------------

    def _final_norm_modules(self) -> List[nn.Module]:
        mods: List[nn.Module] = []
        for attr in ("layernorm", "fc_norm"):
            m = getattr(self.video_encoder, attr, None)
            if isinstance(m, nn.Module):
                mods.append(m)
        return mods

    def _set_block_range_trainable(self, start: int, end: int, trainable: bool) -> None:
        for idx in range(start, end + 1):
            if 0 <= idx < len(self._video_blocks):
                for p in self._video_blocks[idx].parameters():
                    p.requires_grad = trainable

    @staticmethod
    def _params_from_blocks(blocks: List[nn.Module], start: int, end: int) -> List[nn.Parameter]:
        params: List[nn.Parameter] = []
        for idx in range(start, end + 1):
            if 0 <= idx < len(blocks):
                params.extend(blocks[idx].parameters())
        return params

    def configure_single_loop_training(self, train_cfg: Dict | None = None) -> None:
        train_cfg = train_cfg or {}
        # Freeze entire backbone
        for p in self.video_encoder.parameters():
            p.requires_grad = False

        # Unfreeze initial range of blocks + final norms
        self._set_block_range_trainable(self._initial_range[0], self._initial_range[1], True)
        for m in self._final_norm_modules():
            for p in m.parameters():
                p.requires_grad = True

        # All heads are trainable
        head_modules = [
            self.side_flow_proj, self.asym_flow_proj,
            self.visual_side_ln,
            self.speech_window_attn,
            self.phase_rel_proj, self.phase_pos_emb,
            self.phase_agg_attn, self.sample_proj,
            self.severity_head, self.consensus_head,
        ]
        for m in head_modules:
            for p in m.parameters():
                p.requires_grad = True
        self.phase_agg_query.requires_grad = True

        if bool(train_cfg.get("gradient_checkpointing", False)):
            self.video_encoder.gradient_checkpointing_enable()

        self._epoch8_unfrozen  = False
        self._epoch18_unfrozen = False

    def get_initial_optimizer_groups(self, train_cfg: Dict) -> List[Dict]:
        lr_head = float(train_cfg["lr_head"])
        lr_backbone = float(train_cfg["lr_backbone"])
        wd = float(train_cfg["weight_decay"])

        head_params: List[nn.Parameter] = []
        for m in [
            self.side_flow_proj, self.asym_flow_proj,
            self.speech_window_attn,
            self.phase_rel_proj, self.phase_pos_emb,
            self.phase_agg_attn, self.sample_proj,
            self.severity_head, self.consensus_head,
        ]:
            head_params.extend(m.parameters())
        head_params.append(self.phase_agg_query)

        norm_params: List[nn.Parameter] = []
        for m in self._final_norm_modules():
            norm_params.extend(m.parameters())

        backbone_params = (
            self._params_from_blocks(self._video_blocks, self._initial_range[0], self._initial_range[1])
            + norm_params
        )

        return [
            {"params": head_params, "lr": lr_head, "weight_decay": wd, "group_name": "phase_heads"},
            {"params": backbone_params, "lr": lr_backbone, "weight_decay": wd,
             "group_name": f"video_block_{self._initial_range[0]}_{self._initial_range[1]}"},
        ]

    def get_epoch_unfreeze_groups(self, epoch: int, train_cfg: Dict) -> List[Dict]:
        wd = float(train_cfg["weight_decay"])
        groups: List[Dict] = []

        # Set disable_progressive_unfreeze: true in train config to keep only the
        # initial block range (9-11) frozen — prevents overfitting on smaller splits.
        if train_cfg.get("disable_progressive_unfreeze", False):
            return groups

        if epoch >= 5 and not self._epoch8_unfrozen:
            s, e = self._epoch8_range
            self._set_block_range_trainable(s, e, True)
            groups.append({
                "params": self._params_from_blocks(self._video_blocks, s, e),
                "lr": float(train_cfg.get("lr_unfreeze_epoch5", 5e-6)),
                "weight_decay": wd,
                "group_name": f"video_block_{s}_{e}",
            })
            self._epoch8_unfrozen = True

        if epoch >= 12 and not self._epoch18_unfrozen:
            s, e = self._epoch18_range
            self._set_block_range_trainable(s, e, True)
            groups.append({
                "params": self._params_from_blocks(self._video_blocks, s, e),
                "lr": float(train_cfg.get("lr_unfreeze_epoch12", 2e-6)),
                "weight_decay": wd,
                "group_name": f"video_block_{s}_{e}",
            })
            self._epoch18_unfrozen = True

        return groups

    def named_trainable_parameters(self) -> Iterable[tuple[str, nn.Parameter]]:
        for n, p in self.named_parameters():
            if p.requires_grad:
                yield n, p
