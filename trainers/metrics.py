from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score

from data.constants import ID_TO_SEVERITY, SEVERITY_TO_TARGET


# ---------------------------------------------------------------------------
# Rank-order helpers
# ---------------------------------------------------------------------------

def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.zeros_like(values, dtype=np.float64)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        rank = 0.5 * (i + j) + 1.0
        ranks[order[i: j + 1]] = rank
        i = j + 1
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    rx, ry = _rankdata(x) - _rankdata(x).mean(), _rankdata(y) - _rankdata(y).mean()
    denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / denom) if denom > 1e-12 else 0.0


# ---------------------------------------------------------------------------
# Subject-level aggregation
# ---------------------------------------------------------------------------

def _subject_preds_from_probs(
    probs: np.ndarray,
    labels: np.ndarray,
    speaker_ids: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    by_spk_probs: Dict[str, List[np.ndarray]] = defaultdict(list)
    by_spk_labels: Dict[str, List[int]] = defaultdict(list)
    for prob, label, spk in zip(probs, labels, speaker_ids):
        by_spk_probs[spk].append(prob)
        by_spk_labels[spk].append(int(label))
    s_true, s_pred = [], []
    for spk in sorted(by_spk_probs):
        mean_p = np.stack(by_spk_probs[spk]).mean(axis=0)
        s_pred.append(int(np.argmax(mean_p)))
        votes = np.asarray(by_spk_labels[spk], dtype=np.int64)
        s_true.append(int(np.bincount(votes).argmax()))
    return np.asarray(s_true), np.asarray(s_pred)


def _subject_preds_mode(
    preds: np.ndarray,
    labels: np.ndarray,
    speaker_ids: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    by_spk_preds: Dict[str, List[int]] = defaultdict(list)
    by_spk_labels: Dict[str, List[int]] = defaultdict(list)
    for pred, label, spk in zip(preds, labels, speaker_ids):
        by_spk_preds[spk].append(int(pred))
        by_spk_labels[spk].append(int(label))
    s_true, s_pred = [], []
    for spk in sorted(by_spk_preds):
        s_pred.append(int(np.bincount(np.asarray(by_spk_preds[spk], dtype=np.int64)).argmax()))
        s_true.append(int(np.bincount(np.asarray(by_spk_labels[spk], dtype=np.int64)).argmax()))
    return np.asarray(s_true), np.asarray(s_pred)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _f1_macro(y_true: np.ndarray, y_pred: np.ndarray, labels: List[int]) -> float:
    return float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))


def _qwk(y_true: np.ndarray, y_pred: np.ndarray, labels: List[int]) -> float:
    if len(y_true) < 2:
        return 0.0
    score = cohen_kappa_score(y_true, y_pred, labels=labels, weights="quadratic")
    return float(score) if (score is not None and np.isfinite(score)) else 0.0


def _baseline_total_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[int],
) -> Tuple[float, float, float, float]:
    """Weighted precision/recall harmonic mean (matches KGMV-Net evaluation)."""
    cm = confusion_matrix(y_true, y_pred, labels=labels).astype(np.float64)
    support = cm.sum(axis=1)
    tp = np.diag(cm)
    pred_support = cm.sum(axis=0)
    precision = np.divide(tp, pred_support, out=np.zeros_like(tp), where=pred_support > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    total_sup = float(support.sum())
    if total_sup <= 0:
        return 0.0, 0.0, 0.0, 0.0
    wp = float(np.sum(precision * support) / total_sup)
    wr = float(np.sum(recall * support) / total_sup)
    denom = wp + wr
    wf1 = float(2.0 * wp * wr / denom) if denom > 0 else 0.0
    acc = float(tp.sum() / total_sup)
    return wf1, wp, wr, acc


# ---------------------------------------------------------------------------
# Main evaluation — 4-class only (norm/mild/moderate/severe)
# ---------------------------------------------------------------------------

def compute_eval_metrics(
    labels: np.ndarray,
    probs: np.ndarray,
    preds: np.ndarray,
    speaker_ids: List[str],
    tasks: List[str],
    similarities: np.ndarray,
    ignore_labels: Optional[List[int]] = None,  # kept for API compat, no longer used
) -> Dict:
    y_true = labels.astype(np.int64)
    y_pred = preds.astype(np.int64)

    labels_4cls = [0, 1, 2, 3]

    # ----- Sample-level F1 (4-class) -----------------------------------------
    f1_sample_macro_4cls = _f1_macro(y_true, y_pred, labels=labels_4cls)
    sample_cm = confusion_matrix(y_true, y_pred, labels=labels_4cls).tolist()

    # ----- Subject-level (mean-prob aggregation) -----------------------------
    s_true, s_pred = _subject_preds_from_probs(probs, y_true, speaker_ids)
    f1_subject_macro_4cls = _f1_macro(s_true, s_pred, labels=labels_4cls)
    subject_cm = confusion_matrix(s_true, s_pred, labels=labels_4cls).tolist()

    qwk_subject = _qwk(s_true, s_pred, labels=labels_4cls)

    # ----- f1_final_4cls: primary metric (10×subject + sample) ---------------
    f1_final_4cls = 10.0 * f1_subject_macro_4cls + f1_sample_macro_4cls

    # ----- Baseline-compatible weighted F1 (4-class) -------------------------
    baseline_labels = sorted(set(int(v) for v in y_true.tolist()))
    bl_sample_f1, bl_sample_p, bl_sample_r, bl_sample_acc = _baseline_total_f1(
        y_true, y_pred, baseline_labels
    )
    s_true_mode, s_pred_mode = _subject_preds_mode(y_pred, y_true, speaker_ids)
    bl_subj_labels = sorted(set(int(v) for v in s_true_mode.tolist())) if len(s_true_mode) > 0 else labels_4cls
    bl_subj_f1, bl_subj_p, bl_subj_r, bl_subj_acc = _baseline_total_f1(
        s_true_mode, s_pred_mode, bl_subj_labels
    )
    baseline_f1_final_mode = 10.0 * bl_subj_f1 + bl_sample_f1

    # ----- Per-task and per-task-group F1 (4-class) --------------------------
    task_group_map = {
        "task1": "syllable", "task2": "character", "task3": "word",
        "task4": "sentence", "task5": "sentence", "task6": "sentence",
        "task7": "sentence", "task8": "sentence",
    }
    f1_by_task: Dict[str, float] = {}
    for task in sorted(set(tasks)):
        idx = np.asarray([i for i, t in enumerate(tasks) if t == task], dtype=np.int64)
        if len(idx) > 0:
            f1_by_task[task] = _f1_macro(y_true[idx], y_pred[idx], labels=labels_4cls)

    by_group: Dict[str, List[int]] = defaultdict(list)
    for i, task in enumerate(tasks):
        by_group[task_group_map.get(task, "other")].append(i)
    f1_by_task_group: Dict[str, float] = {}
    for grp, idx_list in by_group.items():
        idx = np.asarray(idx_list, dtype=np.int64)
        f1_by_task_group[grp] = _f1_macro(y_true[idx], y_pred[idx], labels=labels_4cls)

    # ----- Similarity / distance calibration ---------------------------------
    sev_targets = np.asarray(
        [SEVERITY_TO_TARGET[ID_TO_SEVERITY[int(y)]] for y in y_true], dtype=np.float64
    )
    sim_spearman_vs_target = _spearman(similarities.astype(np.float64), sev_targets)
    sim_spearman_vs_id = _spearman(similarities.astype(np.float64), y_true.astype(np.float64))

    distances = np.clip(1.0 - similarities.astype(np.float64), 0.0, 1.0)
    dist_targets = y_true.astype(np.float64) / 3.0
    dist_abs_err = np.abs(distances - dist_targets)
    distance_mae_overall = float(np.mean(dist_abs_err))

    sim_by_sev: Dict[str, dict] = {}
    dist_mae_by_sev: Dict[str, dict] = {}
    dist_mean_by_sev: Dict[str, dict] = {}
    ordered = ["norm", "mild", "moderate", "severe"]
    for sev_id in range(4):
        mask = y_true == sev_id
        sev_name = ID_TO_SEVERITY[sev_id]
        if mask.sum() == 0:
            sim_by_sev[sev_name] = {"count": 0, "mean": 0.0, "std": 0.0}
            dist_mae_by_sev[sev_name] = {"count": 0, "mae": 0.0}
            dist_mean_by_sev[sev_name] = {"count": 0, "mean": 0.0, "std": 0.0}
            continue
        vals = similarities[mask]
        dvals = distances[mask]
        derr = dist_abs_err[mask]
        sim_by_sev[sev_name] = {"count": int(mask.sum()), "mean": float(np.mean(vals)), "std": float(np.std(vals))}
        dist_mae_by_sev[sev_name] = {"count": int(mask.sum()), "mae": float(np.mean(derr))}
        dist_mean_by_sev[sev_name] = {"count": int(mask.sum()), "mean": float(np.mean(dvals)), "std": float(np.std(dvals))}

    present = [k for k in ordered if dist_mean_by_sev[k]["count"] > 0]
    mono_violations = [
        f"{l}>={r}"
        for l, r in zip(present[:-1], present[1:])
        if dist_mean_by_sev[l]["mean"] >= dist_mean_by_sev[r]["mean"]
    ]
    distance_monotonic = len(mono_violations) == 0

    ref = next((k for k in ["norm", "mild", "moderate"] if dist_mean_by_sev[k]["count"] > 0), None)
    norm_severe_gap = (
        float(dist_mean_by_sev["severe"]["mean"] - dist_mean_by_sev[ref]["mean"])
        if ref is not None and dist_mean_by_sev["severe"]["count"] > 0
        else 0.0
    )

    return {
        # Primary metric
        "f1_sample_macro_4cls":     f1_sample_macro_4cls,
        "f1_subject_macro_4cls":    f1_subject_macro_4cls,
        "f1_final_4cls":            f1_final_4cls,
        # Ordinal agreement
        "qwk_subject":              qwk_subject,
        # Baseline-compatible weighted F1
        "baseline_sample_total_f1": bl_sample_f1,
        "baseline_sample_acc":      bl_sample_acc,
        "baseline_subject_total_f1_mode": bl_subj_f1,
        "baseline_subject_acc_mode": bl_subj_acc,
        "baseline_f1_final_mode":   baseline_f1_final_mode,
        # Per-task
        "f1_by_task":               f1_by_task,
        "f1_by_task_group":         f1_by_task_group,
        # Confusion matrices
        "sample_confusion_matrix":  sample_cm,
        "subject_confusion_matrix": subject_cm,
        # Similarity/distance calibration
        "similarity_spearman_vs_target":      sim_spearman_vs_target,
        "similarity_spearman_vs_severity_id": sim_spearman_vs_id,
        "similarity_by_severity":             sim_by_sev,
        "distance_mae_overall":               distance_mae_overall,
        "distance_mae_by_severity":           dist_mae_by_sev,
        "distance_mean_by_severity":          dist_mean_by_sev,
        "norm_severe_alignment_gap":          norm_severe_gap,
        "distance_monotonic_increasing":      distance_monotonic,
        "distance_monotonic_violations":      mono_violations,
    }
