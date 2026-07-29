from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Severity target helpers
# ---------------------------------------------------------------------------

def severity_alignment_targets(severity_ids: torch.Tensor, num_classes: int = 4) -> torch.Tensor:
    """Map class id → scalar target: norm=1.0, mild=2/3, moderate=1/3, severe=0.0."""
    return 1.0 - severity_ids.to(torch.float32) / float(max(1, num_classes - 1))


# ---------------------------------------------------------------------------
# Intra-sample alignment loss
# ---------------------------------------------------------------------------

def intra_similarity_l1_loss(
    similarity: torch.Tensor,
    target_similarity: torch.Tensor,
) -> torch.Tensor:
    """L1 loss pushing consensus_similarity toward severity_alignment_targets."""
    return F.l1_loss(similarity, target_similarity)


# ---------------------------------------------------------------------------
# Soft-diagonal phase-consensus contrastive loss
# ---------------------------------------------------------------------------

def soft_diagonal_phase_consensus_loss(
    phase_side_embeddings: torch.Tensor,
    phase_consensus_embeddings: torch.Tensor,
    severity_targets: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Pulls each side embedding toward its own phase consensus, with the pull
    weighted by severity_target (1.0 = norm, 0.0 = severe).

    phase_side_embeddings    : [B, 3, 2, D]
    phase_consensus_embeddings: [B, 3, D]
    severity_targets          : [B]

    Anchors: 6B  (3 phases × 2 sides × B samples)
    Targets: 3B  (3 phases × B samples)
    """
    bsz = int(phase_side_embeddings.shape[0])
    if bsz <= 1:
        return phase_side_embeddings.new_tensor(0.0)

    anchors = F.normalize(phase_side_embeddings.reshape(bsz * 6, -1), dim=-1)   # [6B, D]
    targets = F.normalize(phase_consensus_embeddings.reshape(bsz * 3, -1), dim=-1)  # [3B, D]

    logits = torch.matmul(anchors, targets.T) / float(max(temperature, 1e-6))  # [6B, 3B]
    log_probs = F.log_softmax(logits, dim=1)

    # For anchor i (sample s, phase p, side l/r): positive target index is s*3 + p
    sample_idx = torch.arange(bsz, device=anchors.device).repeat_interleave(6)   # [6B]
    phase_idx  = torch.arange(3,   device=anchors.device).repeat_interleave(2).repeat(bsz)  # [6B]
    pos_idx    = sample_idx * 3 + phase_idx  # [6B]

    num_targets = targets.shape[0]  # 3B
    diag_mass = severity_targets[sample_idx].clamp(0.0, 1.0)            # [6B]
    off_mass  = (1.0 - diag_mass) / float(max(1, num_targets - 1))     # [6B]

    target_probs = torch.zeros_like(log_probs)
    target_probs += off_mass.unsqueeze(1)
    target_probs.scatter_(1, pos_idx.unsqueeze(1), diag_mass.unsqueeze(1))
    row_sums = target_probs.sum(dim=1, keepdim=True).clamp(min=1e-8)
    target_probs = target_probs / row_sums

    return F.kl_div(log_probs, target_probs, reduction="batchmean")


# ---------------------------------------------------------------------------
# Ordinal pairwise ranking loss
# ---------------------------------------------------------------------------

def ordinal_ranking_loss(
    similarity: torch.Tensor,
    severity_ids: torch.Tensor,
    severity_targets: torch.Tensor,
    base_margin: float = 0.05,
    gap_margin_scale: float = 0.05,
) -> torch.Tensor:
    """
    Enforce: sim(lower-severity) > sim(higher-severity) + margin.
    margin = base_margin + gap_margin_scale * (target_i - target_j)
    """
    if similarity.shape[0] <= 1:
        return similarity.new_tensor(0.0)

    sim_i = similarity.unsqueeze(1)
    sim_j = similarity.unsqueeze(0)
    sev_i = severity_ids.unsqueeze(1)
    sev_j = severity_ids.unsqueeze(0)
    tgt_i = severity_targets.unsqueeze(1)
    tgt_j = severity_targets.unsqueeze(0)

    valid = sev_i < sev_j
    if valid.sum() == 0:
        return similarity.new_tensor(0.0)

    gap = (tgt_i - tgt_j).clamp(min=0.0)
    margin = base_margin + gap_margin_scale * gap
    violations = F.relu(margin - (sim_i - sim_j))
    return violations[valid].mean()


# ---------------------------------------------------------------------------
# Cross-entropy with optional class weights + label smoothing
# ---------------------------------------------------------------------------

def severity_ce_loss(
    severity_logits: torch.Tensor,
    severity_ids: torch.Tensor,
    label_smoothing: float = 0.0,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    return F.cross_entropy(
        severity_logits,
        severity_ids,
        weight=class_weights,
        label_smoothing=float(max(0.0, min(1.0, label_smoothing))),
    )


# ---------------------------------------------------------------------------
# Loss weight schedule resolver
# ---------------------------------------------------------------------------

def _resolve_weight(loss_cfg: Dict, cfg: Dict, base_key: str, default: float = 0.0) -> float:
    base_value = float(loss_cfg.get(base_key, default))
    schedule = loss_cfg.get(f"{base_key}_schedule", None)
    if not schedule:
        return base_value
    epoch = int(cfg.get("train", {}).get("current_epoch", 1))
    for item in schedule:
        if int(item.get("start_epoch", 1)) <= epoch <= int(item.get("end_epoch", 10 ** 9)):
            return float(item["value"])
    return base_value


def _resolve_ce_weights(outputs: Dict[str, torch.Tensor], loss_cfg: Dict) -> torch.Tensor | None:
    w = loss_cfg.get("ce_class_weights", None)
    if w is None:
        return None
    return torch.as_tensor(w, dtype=outputs["severity_logits"].dtype, device=outputs["severity_logits"].device)


# ---------------------------------------------------------------------------
# Main loss entry point
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Audio-text soft-diagonal contrastive loss (cross-modal)
# ---------------------------------------------------------------------------

def _soft_diagonal_target_matrix(
    severity_targets: torch.Tensor,
    batch_size: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    target = severity_targets.new_zeros((batch_size, batch_size))
    if batch_size == 1:
        target[0, 0] = 1.0
        return target
    diag_mass = severity_targets.clamp(0.0, 1.0)
    off_mass = (1.0 - diag_mass) / float(batch_size - 1)
    target = target + off_mass.unsqueeze(1)
    target.fill_diagonal_(0.0)
    target = target + torch.diag(diag_mass)
    row_sums = target.sum(dim=1, keepdim=True).clamp(min=eps)
    return target / row_sums


def soft_diagonal_audio_text_loss(
    z_audio: torch.Tensor,
    z_text: torch.Tensor,
    severity_targets: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Cross-modal soft-diagonal contrastive loss (audio→text and text→audio)."""
    if z_audio.shape[0] <= 1:
        return z_audio.new_tensor(0.0)
    logits = torch.matmul(z_audio, z_text.T) / float(max(temperature, 1e-6))
    log_probs_a2t = F.log_softmax(logits, dim=1)
    log_probs_t2a = F.log_softmax(logits.T, dim=1)
    target_probs = _soft_diagonal_target_matrix(severity_targets, logits.shape[0]).to(logits.dtype)
    loss_a2t = F.kl_div(log_probs_a2t, target_probs, reduction="batchmean")
    loss_t2a = F.kl_div(log_probs_t2a, target_probs, reduction="batchmean")
    return 0.5 * (loss_a2t + loss_t2a)


def intra_similarity_mse_loss(
    similarity: torch.Tensor,
    target_similarity: torch.Tensor,
) -> torch.Tensor:
    """MSE loss pushing audio-text cosine similarity toward severity_alignment_targets."""
    return F.mse_loss(similarity, target_similarity)


# ---------------------------------------------------------------------------
# Audio-text total loss entry point
# ---------------------------------------------------------------------------

def compute_audio_text_loss(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    cfg: Dict,
) -> Dict[str, torch.Tensor]:
    """
    Loss for the audio+text branch (AudioTextMMDys).

    Expects outputs keys: severity_logits, z_audio, z_text, similarity
    Uses MSE intra-similarity and cross-modal soft-diagonal InfoNCE.
    Ordinal loss uses cosine similarity as rank signal.
    """
    loss_cfg = cfg["loss"]
    severity_ids = batch["severity_ids"]
    severity_targets = severity_alignment_targets(severity_ids, num_classes=4)

    # -- Intra-sample alignment (MSE, matching run5) --------------------------
    l_intra = intra_similarity_mse_loss(outputs["similarity"], severity_targets)

    # -- Cross-modal soft-diagonal contrastive --------------------------------
    l_soft_diag = soft_diagonal_audio_text_loss(
        outputs["z_audio"],
        outputs["z_text"],
        severity_targets=severity_targets,
        temperature=float(loss_cfg.get("soft_diag_temperature", 0.06)),
    )

    # -- Ordinal ranking (on cosine similarity as severity proxy) -------------
    l_ordinal = ordinal_ranking_loss(
        outputs["similarity"],
        severity_ids,
        severity_targets=severity_targets,
        base_margin=float(loss_cfg.get("ordinal_base_margin", 0.10)),
        gap_margin_scale=float(loss_cfg.get("ordinal_gap_margin_scale", 0.14)),
    )

    # -- Cross-entropy -------------------------------------------------------
    label_smoothing = float(cfg.get("train", {}).get("label_smoothing", loss_cfg.get("label_smoothing", 0.0)))
    l_ce = severity_ce_loss(
        outputs["severity_logits"],
        severity_ids,
        label_smoothing=label_smoothing,
        class_weights=_resolve_ce_weights(outputs, loss_cfg),
    )

    # -- Weighted sum --------------------------------------------------------
    w_intra     = _resolve_weight(loss_cfg, cfg, "lambda_intra",     default=0.9)
    w_soft_diag = _resolve_weight(loss_cfg, cfg, "lambda_soft_diag", default=0.6)
    w_ordinal   = _resolve_weight(loss_cfg, cfg, "lambda_ordinal",   default=0.9)
    w_ce        = _resolve_weight(loss_cfg, cfg, "lambda_ce",        default=0.15)

    total = w_intra * l_intra + w_soft_diag * l_soft_diag + w_ordinal * l_ordinal + w_ce * l_ce

    def _t(v: float, ref: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(v, dtype=ref.dtype, device=ref.device)

    return {
        "total_loss":   total,
        "l_intra":      l_intra,
        "l_soft_diag":  l_soft_diag,
        "l_ordinal":    l_ordinal,
        "l_ce":         l_ce,
        "w_intra":      _t(w_intra,     total),
        "w_soft_diag":  _t(w_soft_diag, total),
        "w_ordinal":    _t(w_ordinal,   total),
        "w_ce":         _t(w_ce,        total),
    }


# ---------------------------------------------------------------------------
# Cross-modal alignment loss (Exp 2)
# ---------------------------------------------------------------------------

def cross_modal_alignment_loss(
    z_video_proj: torch.Tensor,
    z_audio: torch.Tensor,
    severity_targets: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Soft-diagonal InfoNCE between z_video_proj and z_audio.
    Pulls same-sample representations together; soft target mass is
    distributed based on severity_targets (higher mass on diagonal for
    severe samples, softer for normal — consistent with the audio-text loss).
    """
    if z_video_proj.shape[0] <= 1:
        return z_video_proj.new_tensor(0.0)
    logits = torch.matmul(z_video_proj, z_audio.T) / float(max(temperature, 1e-6))
    log_v2a = F.log_softmax(logits,   dim=1)
    log_a2v = F.log_softmax(logits.T, dim=1)
    target_probs = _soft_diagonal_target_matrix(severity_targets, logits.shape[0]).to(logits.dtype)
    loss_v2a = F.kl_div(log_v2a, target_probs, reduction="batchmean")
    loss_a2v = F.kl_div(log_a2v, target_probs, reduction="batchmean")
    return 0.5 * (loss_v2a + loss_a2v)


def cross_modal_ordinal_loss(
    z_video_proj: torch.Tensor,
    z_audio: torch.Tensor,
    severity_ids: torch.Tensor,
    severity_targets: torch.Tensor,
    base_margin: float = 0.05,
    gap_margin_scale: float = 0.05,
) -> torch.Tensor:
    """
    Enforce that the cross-modal similarity sim(z_video_proj_i, z_audio_i)
    follows the same ordinal ordering as severity: lower severity → higher sim.
    """
    if z_video_proj.shape[0] <= 1:
        return z_video_proj.new_tensor(0.0)
    # Per-sample diagonal similarity as the ordinal signal
    diag_sim = (z_video_proj * z_audio).sum(dim=-1)   # [B]
    return ordinal_ranking_loss(
        diag_sim, severity_ids, severity_targets,
        base_margin=base_margin, gap_margin_scale=gap_margin_scale,
    )


def compute_av_loss(
    outputs: dict,
    batch: dict,
    cfg: dict,
) -> dict:
    """
    Total loss for Exp 2 AVFusionModel.

    Combines:
      1. Video phase losses (phase_side_embeddings, consensus_similarity, severity_score)
      2. Audio-text cross-modal soft-diagonal + intra + ordinal
      3. Joint CE on joint severity_logits
      4. Cross-modal alignment: video↔audio InfoNCE + ordinal
    """
    loss_cfg         = cfg["loss"]
    severity_ids     = batch["severity_ids"]
    severity_targets = severity_alignment_targets(severity_ids, num_classes=4)
    label_smoothing  = float(cfg.get("train", {}).get("label_smoothing",
                             loss_cfg.get("label_smoothing", 0.0)))

    # 1. Video phase intra + soft-diag + ordinal
    l_vid_intra = intra_similarity_l1_loss(outputs["consensus_similarity"], severity_targets)
    l_vid_softdiag = soft_diagonal_phase_consensus_loss(
        outputs["phase_side_embeddings"],
        outputs["phase_consensus_embeddings"],
        severity_targets=severity_targets,
        temperature=float(loss_cfg.get("soft_diag_temperature", 0.07)),
    )
    l_vid_ordinal = ordinal_ranking_loss(
        outputs["severity_score"],
        severity_ids, severity_targets,
        base_margin=float(loss_cfg.get("ordinal_base_margin", 0.05)),
        gap_margin_scale=float(loss_cfg.get("ordinal_gap_margin_scale", 0.05)),
    )

    # 2. Audio-text intra + soft-diag + ordinal
    l_aud_intra    = intra_similarity_mse_loss(outputs["similarity"], severity_targets)
    l_aud_softdiag = soft_diagonal_audio_text_loss(
        outputs["z_audio"], outputs["z_text"],
        severity_targets=severity_targets,
        temperature=float(loss_cfg.get("soft_diag_temperature", 0.07)),
    )
    l_aud_ordinal  = ordinal_ranking_loss(
        outputs["similarity"],
        severity_ids, severity_targets,
        base_margin=float(loss_cfg.get("ordinal_base_margin", 0.10)),
        gap_margin_scale=float(loss_cfg.get("ordinal_gap_margin_scale", 0.14)),
    )

    # 3. Joint CE
    l_ce = severity_ce_loss(
        outputs["severity_logits"],
        severity_ids,
        label_smoothing=label_smoothing,
        class_weights=_resolve_ce_weights(outputs, loss_cfg),
    )

    # 4. Cross-modal alignment
    l_xm_info = cross_modal_alignment_loss(
        outputs["z_video_proj"], outputs["z_audio"],
        severity_targets=severity_targets,
        temperature=float(loss_cfg.get("cross_modal_temperature", 0.07)),
    )
    l_xm_ord = cross_modal_ordinal_loss(
        outputs["z_video_proj"], outputs["z_audio"],
        severity_ids, severity_targets,
        base_margin=float(loss_cfg.get("cross_modal_ordinal_margin", 0.05)),
        gap_margin_scale=float(loss_cfg.get("cross_modal_ordinal_gap_scale", 0.05)),
    )

    # Weights
    w_vid_intra     = _resolve_weight(loss_cfg, cfg, "lambda_vid_intra",     default=1.0)
    w_vid_softdiag  = _resolve_weight(loss_cfg, cfg, "lambda_vid_softdiag",  default=0.6)
    w_vid_ordinal   = _resolve_weight(loss_cfg, cfg, "lambda_vid_ordinal",   default=0.6)
    w_aud_intra     = _resolve_weight(loss_cfg, cfg, "lambda_aud_intra",     default=0.9)
    w_aud_softdiag  = _resolve_weight(loss_cfg, cfg, "lambda_aud_softdiag",  default=0.6)
    w_aud_ordinal   = _resolve_weight(loss_cfg, cfg, "lambda_aud_ordinal",   default=0.9)
    w_ce            = _resolve_weight(loss_cfg, cfg, "lambda_ce",            default=0.4)
    w_xm_info       = _resolve_weight(loss_cfg, cfg, "lambda_xm_info",       default=0.5)
    w_xm_ord        = _resolve_weight(loss_cfg, cfg, "lambda_xm_ord",        default=0.3)

    total = (
        w_vid_intra    * l_vid_intra    +
        w_vid_softdiag * l_vid_softdiag +
        w_vid_ordinal  * l_vid_ordinal  +
        w_aud_intra    * l_aud_intra    +
        w_aud_softdiag * l_aud_softdiag +
        w_aud_ordinal  * l_aud_ordinal  +
        w_ce           * l_ce           +
        w_xm_info      * l_xm_info      +
        w_xm_ord       * l_xm_ord
    )

    loss_dict = {
        "total_loss":     total,
        "l_vid_intra":    l_vid_intra,
        "l_vid_softdiag": l_vid_softdiag,
        "l_vid_ordinal":  l_vid_ordinal,
        "l_aud_intra":    l_aud_intra,
        "l_aud_softdiag": l_aud_softdiag,
        "l_aud_ordinal":  l_aud_ordinal,
        "l_ce":           l_ce,
        "l_xm_info":      l_xm_info,
        "l_xm_ord":       l_xm_ord,
    }

    # Auxiliary per-expert CE for divergence_moe (prevents expert collapse)
    w_aux_ce = _resolve_weight(loss_cfg, cfg, "lambda_aux_ce", default=0.0)
    if w_aux_ce > 0.0 and "div_logits_1" in outputs:
        l_aux_1 = severity_ce_loss(outputs["div_logits_1"], severity_ids,
                                   label_smoothing=label_smoothing)
        l_aux_2 = severity_ce_loss(outputs["div_logits_2"], severity_ids,
                                   label_smoothing=label_smoothing)
        l_aux_3 = severity_ce_loss(outputs["div_logits_3"], severity_ids,
                                   label_smoothing=label_smoothing)
        l_aux = (l_aux_1 + l_aux_2 + l_aux_3) / 3.0
        loss_dict["total_loss"] = loss_dict["total_loss"] + w_aux_ce * l_aux
        loss_dict["l_aux_expert_ce"] = l_aux

    return loss_dict


# ---------------------------------------------------------------------------
# Video branch total loss entry point
# ---------------------------------------------------------------------------

def compute_total_loss(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    cfg: Dict,
) -> Dict[str, torch.Tensor]:
    loss_cfg = cfg["loss"]
    severity_ids = batch["severity_ids"]
    severity_targets = severity_alignment_targets(severity_ids, num_classes=4)

    # -- Intra-sample alignment ----------------------------------------------
    l_intra = intra_similarity_l1_loss(outputs["consensus_similarity"], severity_targets)

    # -- Phase-consensus contrastive -----------------------------------------
    l_soft_diag = soft_diagonal_phase_consensus_loss(
        outputs["phase_side_embeddings"],
        outputs["phase_consensus_embeddings"],
        severity_targets=severity_targets,
        temperature=float(loss_cfg.get("soft_diag_temperature", 0.07)),
    )

    # -- Ordinal ranking (uses severity_score: softmax-weighted expected class) -
    rank_signal = outputs.get("severity_score", outputs["consensus_similarity"])
    l_ordinal = ordinal_ranking_loss(
        rank_signal,
        severity_ids,
        severity_targets=severity_targets,
        base_margin=float(loss_cfg.get("ordinal_base_margin", 0.05)),
        gap_margin_scale=float(loss_cfg.get("ordinal_gap_margin_scale", 0.05)),
    )

    # -- Cross-entropy -------------------------------------------------------
    label_smoothing = float(cfg.get("train", {}).get("label_smoothing", loss_cfg.get("label_smoothing", 0.0)))
    l_ce = severity_ce_loss(
        outputs["severity_logits"],
        severity_ids,
        label_smoothing=label_smoothing,
        class_weights=_resolve_ce_weights(outputs, loss_cfg),
    )

    # -- Weighted sum --------------------------------------------------------
    w_intra    = _resolve_weight(loss_cfg, cfg, "lambda_intra",    default=1.0)
    w_soft_diag = _resolve_weight(loss_cfg, cfg, "lambda_soft_diag", default=0.5)
    w_ordinal  = _resolve_weight(loss_cfg, cfg, "lambda_ordinal",  default=0.4)
    w_ce       = _resolve_weight(loss_cfg, cfg, "lambda_ce",       default=0.4)

    total = w_intra * l_intra + w_soft_diag * l_soft_diag + w_ordinal * l_ordinal + w_ce * l_ce

    def _t(v: float, ref: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(v, dtype=ref.dtype, device=ref.device)

    return {
        "total_loss":   total,
        "l_intra":      l_intra,
        "l_soft_diag":  l_soft_diag,
        "l_ordinal":    l_ordinal,
        "l_ce":         l_ce,
        "w_intra":      _t(w_intra,     total),
        "w_soft_diag":  _t(w_soft_diag, total),
        "w_ordinal":    _t(w_ordinal,   total),
        "w_ce":         _t(w_ce,        total),
    }


def compute_uniform_video_loss(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    cfg: Dict,
) -> Dict[str, torch.Tensor]:
    """
    Simplified loss for VideoUniformWindowModel — no phase embeddings required.

    Uses CE + optional intra + ordinal. soft_diag is always 0 (needs phase embeddings).
    """
    loss_cfg = cfg["loss"]
    severity_ids = batch["severity_ids"]
    severity_targets = severity_alignment_targets(severity_ids, num_classes=4)

    l_intra = intra_similarity_l1_loss(outputs["consensus_similarity"], severity_targets)

    rank_signal = outputs.get("severity_score", outputs["consensus_similarity"])
    l_ordinal = ordinal_ranking_loss(
        rank_signal,
        severity_ids,
        severity_targets=severity_targets,
        base_margin=float(loss_cfg.get("ordinal_base_margin", 0.05)),
        gap_margin_scale=float(loss_cfg.get("ordinal_gap_margin_scale", 0.05)),
    )

    label_smoothing = float(cfg.get("train", {}).get("label_smoothing", loss_cfg.get("label_smoothing", 0.0)))
    l_ce = severity_ce_loss(
        outputs["severity_logits"],
        severity_ids,
        label_smoothing=label_smoothing,
        class_weights=_resolve_ce_weights(outputs, loss_cfg),
    )

    w_intra   = _resolve_weight(loss_cfg, cfg, "lambda_intra",   default=0.0)
    w_ordinal = _resolve_weight(loss_cfg, cfg, "lambda_ordinal", default=0.4)
    w_ce      = _resolve_weight(loss_cfg, cfg, "lambda_ce",      default=1.0)

    total = w_intra * l_intra + w_ordinal * l_ordinal + w_ce * l_ce

    def _t(v: float, ref: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(v, dtype=ref.dtype, device=ref.device)

    zero = total.detach() * 0.0
    return {
        "total_loss":   total,
        "l_intra":      l_intra,
        "l_soft_diag":  zero,
        "l_ordinal":    l_ordinal,
        "l_ce":         l_ce,
        "w_intra":      _t(w_intra,   total),
        "w_soft_diag":  _t(0.0,       total),
        "w_ordinal":    _t(w_ordinal, total),
        "w_ce":         _t(w_ce,      total),
    }
