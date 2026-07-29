"""
AVFusionModel — Exp 2: Cross-Modal Soft Alignment Fine-tuning.

Supports three fusion modes (fusion_mode arg):

  "additive"     (default, backward-compatible)
      z_video_proj + z_audio → joint_head → 4 logits

  "moe"          (Idea 1 — Mixture of Experts)
      Three expert heads + a gating network:
        Expert A: video head on z_video_proj
        Expert B: audio head on z_audio
        Expert C: interaction head on z_video_proj ⊙ z_audio (element-wise product)
      Gate: softmax(Linear([z_video_proj; z_audio; sim_scalar])) → 3 weights
      Logits: w_A*logit_A + w_B*logit_B + w_C*logit_C

  "disagreement" (Idea 3 — Disagreement as feature)
      Drops InfoNCE. Uses the difference vector as the fusion signal:
        [z_video_proj; z_audio; z_video_proj - z_audio; cosine_sim] → joint_head → 4 logits
      The divergence between modalities is itself informative for severity.

  "divergence_moe" (Idea 4 — Divergence experts with scalar gate)
      Three experts, each operating on a clinically-motivated difference vector:
        Expert 1 (speech-text):   z_audio − z_text          [B, 256]
        Expert 2 (lateralization): mean_phase(emb_L − emb_R) [B, 448] → proj 256
        Expert 3 (cross-modal):   z_video_proj − z_audio    [B, 256]
      Gate input: three scalar norms ||e1||, ||e2||, ||e3|| → softmax(3 weights)
      Logits: w1*logit_1 + w2*logit_2 + w3*logit_3
      Each expert also supervised by auxiliary CE (lower weight) to prevent collapse.

Freeze strategy (applied by configure_for_av_training):
  - VideoMAE: only top-k blocks unfrozen
  - Wav2Vec2: only top-k blocks unfrozen (unless freeze_audio_backbone=True)
  - RoBERTa:  fully frozen always
  - All heads and new fusion modules: trainable
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.audio_text_mmdys import AudioTextMMDys
from models.phase_videomae import PhaseVideoMAEModel


class AVFusionModel(nn.Module):
    """
    Joint audio-visual model for Exp 2 cross-modal alignment fine-tuning.

    Args:
        video_model:            Pre-built PhaseVideoMAEModel.
        audio_model:            Pre-built AudioTextMMDys.
        projection_dim:         Shared projection space dimension (default 256).
        classifier_hidden_dim:  Hidden dim of joint severity head (default 512).
        dropout:                Dropout for new heads (default 0.1).
    """

    FUSION_MODES = ("additive", "moe", "disagreement", "divergence_moe")

    def __init__(
        self,
        video_model: PhaseVideoMAEModel,
        audio_model: AudioTextMMDys,
        projection_dim: int = 256,
        classifier_hidden_dim: int = 512,
        dropout: float = 0.1,
        fusion_mode: str = "additive",
    ) -> None:
        super().__init__()
        assert fusion_mode in self.FUSION_MODES, \
            f"fusion_mode must be one of {self.FUSION_MODES}, got '{fusion_mode}'"
        self.fusion_mode    = fusion_mode
        self.projection_dim = projection_dim
        self.video_model    = video_model
        self.audio_model    = audio_model

        # sample_proj is an nn.Sequential; find output dim from its last Linear layer
        video_sample_dim = projection_dim
        if hasattr(video_model, "sample_proj"):
            for m in video_model.sample_proj.modules():
                if isinstance(m, nn.Linear):
                    video_sample_dim = m.out_features

        # Cross-modal projection: maps z_sample → shared 256-D space
        self.cross_proj = nn.Sequential(
            nn.Linear(video_sample_dim, projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim, projection_dim),
        )

        # ---- Fusion-mode-specific heads ----
        if fusion_mode == "additive":
            # z_video_proj + z_audio → head (backward-compatible default)
            self.joint_head = nn.Sequential(
                nn.Linear(projection_dim, classifier_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(classifier_hidden_dim, 4),
            )

        elif fusion_mode == "moe":
            # Expert A: video-only
            self.expert_video = nn.Sequential(
                nn.Linear(projection_dim, classifier_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(classifier_hidden_dim, 4),
            )
            # Expert B: audio-only
            self.expert_audio = nn.Sequential(
                nn.Linear(projection_dim, classifier_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(classifier_hidden_dim, 4),
            )
            # Expert C: interaction (element-wise product captures agreement)
            self.expert_interact = nn.Sequential(
                nn.Linear(projection_dim, classifier_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(classifier_hidden_dim, 4),
            )
            # Gate: [z_video; z_audio; sim_scalar] → 3 expert weights
            self.moe_gate = nn.Sequential(
                nn.Linear(projection_dim * 2 + 1, 64),
                nn.GELU(),
                nn.Linear(64, 3),
            )

        elif fusion_mode == "disagreement":
            # [z_video; z_audio; z_video - z_audio; sim] → head
            # Disagreement vector + scalar are the primary fusion signals
            self.joint_head = nn.Sequential(
                nn.Linear(projection_dim * 3 + 1, classifier_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(classifier_hidden_dim, 4),
            )

        elif fusion_mode == "divergence_moe":
            # Expert 1: speech-text divergence  (z_audio − z_text)  [B, 256]
            self.div_expert_1 = nn.Sequential(
                nn.Linear(projection_dim, classifier_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(classifier_hidden_dim, 4),
            )
            # Expert 2: bilateral lateralization  mean_phase(emb_L − emb_R) → proj [B, 256]
            # phase_side_embeddings is [B, 3, 2, 448]; emb_L = [:,:,0,:], emb_R = [:,:,1,:]
            lat_dim = 448
            self.lat_proj = nn.Sequential(
                nn.Linear(lat_dim, projection_dim),
                nn.LayerNorm(projection_dim),
            )
            self.div_expert_2 = nn.Sequential(
                nn.Linear(projection_dim, classifier_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(classifier_hidden_dim, 4),
            )
            # Expert 3: cross-modal divergence  (z_video_proj − z_audio)  [B, 256]
            self.div_expert_3 = nn.Sequential(
                nn.Linear(projection_dim, classifier_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(classifier_hidden_dim, 4),
            )
            # Gate: scalar norms [||e1||, ||e2||, ||e3||] → softmax(3 weights)
            self.div_gate = nn.Sequential(
                nn.Linear(3, 16),
                nn.GELU(),
                nn.Linear(16, 3),
            )

        self._init_new_heads()

    def _init_new_heads(self) -> None:
        new_modules = [self.cross_proj]
        if self.fusion_mode == "additive":
            new_modules += [self.joint_head]
        elif self.fusion_mode == "moe":
            new_modules += [self.expert_video, self.expert_audio,
                            self.expert_interact, self.moe_gate]
        elif self.fusion_mode == "disagreement":
            new_modules += [self.joint_head]
        elif self.fusion_mode == "divergence_moe":
            new_modules += [self.lat_proj, self.div_expert_1, self.div_expert_2,
                            self.div_expert_3, self.div_gate]
        for mod in new_modules:
            for m in mod.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Freeze configuration
    # ------------------------------------------------------------------

    def configure_for_av_training(
        self,
        video_top_k: int = 3,
        audio_top_k: int = 2,
        freeze_audio_backbone: bool = False,
    ) -> None:
        """
        Freeze everything, then selectively unfreeze:
          - Top `video_top_k` VideoMAE transformer blocks (blocks 9-11 for k=3)
          - Top `audio_top_k` Wav2Vec2 transformer blocks (unless freeze_audio_backbone=True)
          - RoBERTa: fully frozen always
          - All non-backbone heads: trainable

        Args:
            freeze_audio_backbone: If True, keep audio backbone frozen (heads still train).
                                   Call again with False to unfreeze for staged training.
        """
        # Freeze everything
        for p in self.parameters():
            p.requires_grad = False

        # --- Video backbone: unfreeze top-k transformer blocks ---
        vid_layers = self._get_videomae_layers(self.video_model)
        boundary = max(0, len(vid_layers) - video_top_k)
        for idx, layer in enumerate(vid_layers):
            for p in layer.parameters():
                p.requires_grad = idx >= boundary

        # --- Audio backbone: optionally unfreeze top-k blocks ---
        if not freeze_audio_backbone:
            aud_layers = self.audio_model._get_transformer_layers(self.audio_model.audio_encoder)
            aud_boundary = max(0, len(aud_layers) - audio_top_k)
            for idx, layer in enumerate(aud_layers):
                for p in layer.parameters():
                    p.requires_grad = idx >= aud_boundary

        # --- Text encoder: fully frozen always ---

        # --- Heads: always trainable ---
        trainable_modules = [
            self.video_model.phase_agg_attn,
            self.video_model.sample_proj,
            self.video_model.severity_head,
            self.video_model.consensus_head,
            self.audio_model.audio_pool,
            self.audio_model.text_pool,
            self.audio_model.audio_proj,
            self.audio_model.text_proj,
            self.audio_model.severity_head,
            self.cross_proj,
        ]
        if self.fusion_mode == "additive":
            trainable_modules += [self.joint_head]
        elif self.fusion_mode == "moe":
            trainable_modules += [self.expert_video, self.expert_audio,
                                  self.expert_interact, self.moe_gate]
        elif self.fusion_mode == "disagreement":
            trainable_modules += [self.joint_head]
        elif self.fusion_mode == "divergence_moe":
            trainable_modules += [self.lat_proj, self.div_expert_1, self.div_expert_2,
                                  self.div_expert_3, self.div_gate]
        for mod in trainable_modules:
            for p in mod.parameters():
                p.requires_grad = True

        total  = sum(p.numel() for p in self.parameters())
        active = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_str = " [audio backbone frozen]" if freeze_audio_backbone else ""
        print(f"[AVFusionModel] trainable={active:,}  frozen={total - active:,}  "
              f"total={total:,}  ({100*active/total:.1f}%){frozen_str}")

    @staticmethod
    def _get_videomae_layers(model: PhaseVideoMAEModel) -> List[nn.Module]:
        """Return the list of VideoMAE transformer encoder layers."""
        vit = model.video_encoder  # VideoMAEModel
        if hasattr(vit, "encoder") and hasattr(vit.encoder, "layer"):
            return list(vit.encoder.layer)
        raise ValueError("Cannot find VideoMAE transformer layers in PhaseVideoMAEModel.video_encoder")

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        # Video inputs
        pre_left_video:           torch.Tensor,
        pre_right_video:          torch.Tensor,
        speech_left_video:        torch.Tensor,
        speech_right_video:       torch.Tensor,
        post_left_video:          torch.Tensor,
        post_right_video:         torch.Tensor,
        phase_side_flow_features: torch.Tensor,
        phase_asym_flow_features: torch.Tensor,
        recovery_ratio:           torch.Tensor,
        speech_num_frames:        Optional[torch.Tensor] = None,
        # Audio inputs
        audio_values:             torch.Tensor = None,
        audio_attention_mask:     torch.Tensor = None,
        input_ids:                torch.Tensor = None,
        text_attention_mask:      torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:

        # ---- Video branch ----
        v_out = self.video_model(
            pre_left_video           = pre_left_video,
            pre_right_video          = pre_right_video,
            speech_left_video        = speech_left_video,
            speech_right_video       = speech_right_video,
            post_left_video          = post_left_video,
            post_right_video         = post_right_video,
            phase_side_flow_features = phase_side_flow_features,
            phase_asym_flow_features = phase_asym_flow_features,
            recovery_ratio           = recovery_ratio,
            speech_num_frames        = speech_num_frames,
        )

        # ---- Audio branch ----
        a_out = self.audio_model(
            audio_values         = audio_values,
            audio_attention_mask = audio_attention_mask,
            input_ids            = input_ids,
            text_attention_mask  = text_attention_mask,
        )

        # ---- Cross-modal projection ----
        z_video_proj = F.normalize(self.cross_proj(v_out["z_sample"]), dim=-1)  # [B, 256]
        z_audio      = a_out["z_audio"]                                           # [B, 256]
        z_text       = a_out["z_text"]                                            # [B, 256]
        sim_scalar   = F.cosine_similarity(z_video_proj, z_audio, dim=-1).unsqueeze(-1)  # [B, 1]

        # ---- Joint severity head (fusion-mode-dependent) ----
        if self.fusion_mode == "additive":
            fused           = z_video_proj + z_audio
            severity_logits = self.joint_head(fused)

        elif self.fusion_mode == "moe":
            logit_v = self.expert_video(z_video_proj)                             # [B, 4]
            logit_a = self.expert_audio(z_audio)                                  # [B, 4]
            logit_i = self.expert_interact(z_video_proj * z_audio)                # [B, 4]
            gate_in = torch.cat([z_video_proj, z_audio, sim_scalar], dim=-1)      # [B, 513]
            gate_w  = torch.softmax(self.moe_gate(gate_in), dim=-1)               # [B, 3]
            severity_logits = (gate_w[:, 0:1] * logit_v
                             + gate_w[:, 1:2] * logit_a
                             + gate_w[:, 2:3] * logit_i)                          # [B, 4]

        elif self.fusion_mode == "disagreement":
            diff  = z_video_proj - z_audio                                        # [B, 256]
            fused = torch.cat([z_video_proj, z_audio, diff, sim_scalar], dim=-1)  # [B, 769]
            severity_logits = self.joint_head(fused)

        elif self.fusion_mode == "divergence_moe":
            # Expert 1: speech-text divergence
            e1 = z_audio - z_text                                                 # [B, 256]
            logit_1 = self.div_expert_1(e1)                                       # [B, 4]

            # Expert 2: bilateral lateralization (mean across 3 phases)
            pse = v_out["phase_side_embeddings"]                                  # [B, 3, 2, 448]
            lat_diff = pse[:, :, 0, :] - pse[:, :, 1, :]                         # [B, 3, 448]
            lat_mean = lat_diff.mean(dim=1)                                       # [B, 448]
            e2 = self.lat_proj(lat_mean)                                          # [B, 256]
            logit_2 = self.div_expert_2(e2)                                       # [B, 4]

            # Expert 3: cross-modal divergence
            e3 = z_video_proj - z_audio                                           # [B, 256]
            logit_3 = self.div_expert_3(e3)                                       # [B, 4]

            # Scalar-norm gate
            norms = torch.stack([
                e1.norm(dim=-1),
                e2.norm(dim=-1),
                e3.norm(dim=-1),
            ], dim=-1)                                                             # [B, 3]
            gate_w = torch.softmax(self.div_gate(norms), dim=-1)                  # [B, 3]
            severity_logits = (gate_w[:, 0:1] * logit_1
                             + gate_w[:, 1:2] * logit_2
                             + gate_w[:, 2:3] * logit_3)                          # [B, 4]

        out = {
            # Joint classification
            "severity_logits":            severity_logits,
            # Cross-modal embeddings
            "z_video_proj":               z_video_proj,
            "z_audio":                    z_audio,
            "z_text":                     z_text,
            # For intra-sample alignment losses
            "similarity":                 a_out["similarity"],
            "consensus_similarity":       v_out["consensus_similarity"],
            "severity_score":             v_out["severity_score"],
            # For soft-diagonal / phase losses (from video branch)
            "phase_side_embeddings":      v_out["phase_side_embeddings"],
            "phase_consensus_embeddings": v_out["phase_consensus_embeddings"],
            # Individual branch logits (for monitoring)
            "video_severity_logits":      v_out["severity_logits"],
            "audio_severity_logits":      a_out["severity_logits"],
        }
        # Auxiliary per-expert logits for divergence_moe (used by loss for aux CE)
        if self.fusion_mode == "divergence_moe":
            out["div_logits_1"] = logit_1
            out["div_logits_2"] = logit_2
            out["div_logits_3"] = logit_3
            out["div_gate_weights"] = gate_w
        return out

    # ------------------------------------------------------------------

    def named_trainable_parameters(self) -> Iterable[tuple]:
        for n, p in self.named_parameters():
            if p.requires_grad:
                yield n, p

    def get_optimizer_groups(self, train_cfg: Dict) -> List[Dict]:
        """
        Returns AdamW parameter groups:
          - new heads (cross_proj, joint_head): lr_head
          - video backbone top-k: lr_backbone_video
          - audio backbone top-k: lr_backbone_audio
          - audio/video existing heads: lr_head
        """
        lr_head           = float(train_cfg["lr_head"])
        lr_backbone_video = float(train_cfg.get("lr_backbone_video", train_cfg.get("lr_backbone", 5e-6)))
        lr_backbone_audio = float(train_cfg.get("lr_backbone_audio", train_cfg.get("lr_backbone", 5e-6)))
        weight_decay      = float(train_cfg["weight_decay"])

        new_head_params:   List[nn.Parameter] = []
        vid_backbone_params: List[nn.Parameter] = []
        aud_backbone_params: List[nn.Parameter] = []
        other_head_params: List[nn.Parameter] = []

        new_head_mods = [self.cross_proj]
        if self.fusion_mode == "additive":
            new_head_mods += [self.joint_head]
        elif self.fusion_mode == "moe":
            new_head_mods += [self.expert_video, self.expert_audio,
                              self.expert_interact, self.moe_gate]
        elif self.fusion_mode == "disagreement":
            new_head_mods += [self.joint_head]
        elif self.fusion_mode == "divergence_moe":
            new_head_mods += [self.lat_proj, self.div_expert_1, self.div_expert_2,
                              self.div_expert_3, self.div_gate]
        new_head_names = {id(p) for m in new_head_mods for p in m.parameters()}
        vid_layers = set()
        try:
            for layer in self._get_videomae_layers(self.video_model):
                for p in layer.parameters():
                    vid_layers.add(id(p))
        except ValueError:
            pass
        aud_layers = set()
        for layer in self.audio_model._get_transformer_layers(self.audio_model.audio_encoder):
            for p in layer.parameters():
                aud_layers.add(id(p))

        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if id(p) in new_head_names:
                new_head_params.append(p)
            elif id(p) in vid_layers:
                vid_backbone_params.append(p)
            elif id(p) in aud_layers:
                aud_backbone_params.append(p)
            else:
                other_head_params.append(p)

        groups: List[Dict] = []
        if new_head_params:
            groups.append({"params": new_head_params,    "lr": lr_head,           "weight_decay": weight_decay, "group_name": "new_heads"})
        if other_head_params:
            groups.append({"params": other_head_params,  "lr": lr_head,           "weight_decay": weight_decay, "group_name": "existing_heads"})
        if vid_backbone_params:
            groups.append({"params": vid_backbone_params, "lr": lr_backbone_video, "weight_decay": weight_decay, "group_name": "video_backbone"})
        if aud_backbone_params:
            groups.append({"params": aud_backbone_params, "lr": lr_backbone_audio, "weight_decay": weight_decay, "group_name": "audio_backbone"})
        return groups
