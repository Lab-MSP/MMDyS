"""
Audio + text severity classification model for MMDys.
Architecture: Wav2Vec2 (audio) + RoBERTa (text), masked attentive pooling,
concat fusion with abs-diff, 4-class severity head.

Architecture hyperparameters:
  audio: jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn
  text:  hfl/chinese-roberta-wwm-ext-large
  projection_dim: 256, classifier_hidden_dim: 512, dropout: 0.1
  Top-6 wav2vec2 transformer blocks (18-23) unfrozen during training.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, Wav2Vec2Model


class _ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _MaskedAttentivePooling(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # hidden: [B, T, D], mask: [B, T] in {0, 1}
        logits = self.score(hidden).squeeze(-1)
        logits = logits.masked_fill(mask <= 0, -1e9)
        attn = torch.softmax(logits, dim=-1)
        return (hidden * attn.unsqueeze(-1)).sum(dim=1)


class AudioTextMMDys(nn.Module):
    """
    Audio + text fusion model for dysarthria severity classification.

    Outputs (used by compute_audio_text_loss):
      severity_logits   : [B, 4]
      z_audio           : [B, projection_dim]  — L2-normalised
      z_text            : [B, projection_dim]  — L2-normalised
      similarity        : [B]                  — cosine similarity
    """

    def __init__(
        self,
        audio_model_name: str,
        text_model_name: str,
        projection_dim: int = 256,
        projection_hidden_dim: int = 256,
        classifier_hidden_dim: int = 512,
        dropout: float = 0.1,
        use_text: bool = True,
    ) -> None:
        super().__init__()

        self.use_text = bool(use_text)

        self.audio_encoder = Wav2Vec2Model.from_pretrained(audio_model_name)
        # Disable internal SpecAugment — we apply no augmentation at the feature level;
        # the CUDA indexing kernel for masked_spec_embed is incompatible with bfloat16 AMP.
        self.audio_encoder.config.apply_spec_augment = False
        self.audio_pool = _MaskedAttentivePooling(int(self.audio_encoder.config.hidden_size))
        self.audio_proj = _ProjectionHead(
            int(self.audio_encoder.config.hidden_size), projection_hidden_dim, projection_dim, dropout
        )

        if self.use_text:
            self.text_encoder = AutoModel.from_pretrained(text_model_name)
            text_hidden = int(self.text_encoder.config.hidden_size)
            self.text_proj = _ProjectionHead(text_hidden, projection_hidden_dim, projection_dim, dropout)
            self.text_pool = _MaskedAttentivePooling(text_hidden)
            # concat(z_audio, z_text, |z_audio - z_text|) → 3 * projection_dim
            fused_dim = projection_dim * 3
        else:
            self.text_encoder = None
            self.text_proj = None
            self.text_pool = None
            # audio-only: z_audio only → projection_dim
            fused_dim = projection_dim

        self.severity_head = nn.Sequential(
            nn.Linear(fused_dim, classifier_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, 4),
        )

    def _audio_feature_mask(self, audio_attention_mask: torch.Tensor, feature_len: int) -> torch.Tensor:
        return self.audio_encoder._get_feature_vector_attention_mask(  # pylint: disable=protected-access
            feature_len, audio_attention_mask
        )

    def forward(
        self,
        audio_values: torch.Tensor,
        audio_attention_mask: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        text_attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        audio_out = self.audio_encoder(input_values=audio_values, attention_mask=audio_attention_mask)
        audio_hidden = audio_out.last_hidden_state
        audio_feat_mask = self._audio_feature_mask(audio_attention_mask, audio_hidden.shape[1])
        audio_pooled = self.audio_pool(audio_hidden, audio_feat_mask)
        z_audio = F.normalize(self.audio_proj(audio_pooled), dim=-1)

        if self.use_text:
            text_out = self.text_encoder(input_ids=input_ids, attention_mask=text_attention_mask)
            text_pooled = self.text_pool(text_out.last_hidden_state, text_attention_mask)
            z_text = F.normalize(self.text_proj(text_pooled), dim=-1)
            similarity = F.cosine_similarity(z_audio, z_text, dim=-1)
            fused = torch.cat([z_audio, z_text, (z_audio - z_text).abs()], dim=-1)
        else:
            z_text = torch.zeros_like(z_audio)
            similarity = torch.ones(z_audio.shape[0], device=z_audio.device, dtype=z_audio.dtype)
            fused = z_audio

        severity_logits = self.severity_head(fused)

        return {
            "severity_logits": severity_logits,
            "z_audio": z_audio,
            "z_text": z_text,
            "similarity": similarity,
        }

    # ------------------------------------------------------------------
    # Parameter setup
    # ------------------------------------------------------------------

    @staticmethod
    def _get_transformer_layers(module: nn.Module) -> List[nn.Module]:
        if hasattr(module, "encoder") and hasattr(module.encoder, "layers"):
            return list(module.encoder.layers)
        if hasattr(module, "encoder") and hasattr(module.encoder, "layer"):
            return list(module.encoder.layer)
        if hasattr(module, "roberta") and hasattr(module.roberta, "encoder"):
            return list(module.roberta.encoder.layer)
        raise ValueError(f"Cannot resolve transformer layers for {type(module).__name__}")

    def configure_single_loop_training(self, train_cfg: Optional[Dict] = None) -> None:
        # Freeze all parameters first.
        for p in self.parameters():
            p.requires_grad = False

        # Unfreeze top-6 wav2vec2 transformer blocks (indices 18-23).
        audio_layers = self._get_transformer_layers(self.audio_encoder)
        for idx, layer in enumerate(audio_layers):
            for p in layer.parameters():
                p.requires_grad = idx >= 18

        # Optionally unfreeze top-k text transformer blocks (default 0 = frozen).
        text_top_k = 0
        if train_cfg is not None:
            text_top_k = max(0, int(train_cfg.get("text_unfreeze_top_k", 0)))
        if text_top_k > 0:
            text_layers = self._get_transformer_layers(self.text_encoder)
            boundary = max(0, len(text_layers) - text_top_k)
            for idx, layer in enumerate(text_layers):
                for p in layer.parameters():
                    p.requires_grad = idx >= boundary

        # Always train: audio pooling, projection, classifier.
        trainable_mods = [self.audio_pool, self.audio_proj, self.severity_head]
        if self.use_text:
            trainable_mods += [self.text_pool, self.text_proj]
        for mod in trainable_mods:
            for p in mod.parameters():
                p.requires_grad = True

    def get_initial_optimizer_groups(self, train_cfg: Dict) -> List[Dict]:
        lr_head = float(train_cfg["lr_head"])
        lr_backbone = float(train_cfg["lr_backbone"])
        weight_decay = float(train_cfg["weight_decay"])

        head_params, audio_params, text_params = [], [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if n.startswith("audio_encoder"):
                audio_params.append(p)
            elif n.startswith("text_encoder"):
                text_params.append(p)
            else:
                head_params.append(p)

        groups: List[Dict] = []
        if head_params:
            groups.append({"params": head_params, "lr": lr_head, "weight_decay": weight_decay, "group_name": "heads"})
        if audio_params:
            groups.append({"params": audio_params, "lr": lr_backbone, "weight_decay": weight_decay, "group_name": "audio_top6"})
        if text_params:
            groups.append({"params": text_params, "lr": lr_backbone, "weight_decay": weight_decay, "group_name": "text_top_k"})
        return groups

    def get_epoch_unfreeze_groups(self, epoch: int, train_cfg: Dict) -> List[Dict]:
        del epoch, train_cfg
        return []

    def named_trainable_parameters(self) -> Iterable[tuple[str, nn.Parameter]]:
        for n, p in self.named_parameters():
            if p.requires_grad:
                yield n, p
