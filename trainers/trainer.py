from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from losses import compute_audio_text_loss, compute_total_loss, compute_uniform_video_loss
from trainers.metrics import compute_eval_metrics
from utils.io import save_csv_rows, save_json


class SeverityGuidedTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        dev_loader: DataLoader,
        test_loader: DataLoader,
        cfg: Dict,
        output_dir: str | Path,
        device: torch.device,
        resume_payload: Optional[Dict] = None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader
        self.cfg = cfg
        self.output_dir = Path(output_dir)
        self.device = device

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.train_history_path = self.output_dir / "train_history.csv"
        self.epoch_jsonl_path   = self.output_dir / "metrics_per_epoch.jsonl"
        self.enable_epoch_jsonl = bool(cfg["train"].get("enable_epoch_jsonl", True))
        if self.enable_epoch_jsonl and self.epoch_jsonl_path.exists():
            self.epoch_jsonl_path.unlink()

        train_cfg = cfg["train"]
        self.grad_accum_steps = int(train_cfg.get("gradient_accumulation_steps", 1))
        self.max_grad_norm    = float(train_cfg.get("max_grad_norm", 1.0))

        # ---- Checkpoint selection -------------------------------------------
        self.best_metric_name          = str(train_cfg.get("best_metric", "f1_final_4cls"))
        self.best_metric_mode          = str(train_cfg.get("best_metric_mode", "max")).lower()
        self.best_metric_tiebreaker    = str(train_cfg.get("best_metric_tiebreaker", ""))
        self.best_metric_tiebreaker_mode = str(train_cfg.get("best_metric_tiebreaker_mode", "min")).lower()
        self.best_metric_eps           = float(train_cfg.get("best_metric_eps", 1e-6))
        self.save_top_k                = max(1, int(train_cfg.get("save_top_k_checkpoints", 1)))

        # ---- Evaluation frequency -------------------------------------------
        eval_policy = train_cfg.get("eval_policy", {})
        self.eval_first_n   = int(eval_policy.get("first_n_epochs", 10))
        self.eval_freq_first = int(eval_policy.get("every_n_epochs_first", train_cfg.get("eval_every_epochs", 1)))
        self.eval_freq_after = int(eval_policy.get("every_n_epochs_after", train_cfg.get("eval_every_epochs_after", 2)))

        # ---- Early stopping -------------------------------------------------
        es = train_cfg.get("early_stop", {})
        self.es_enabled  = bool(es.get("enabled", False))
        self.es_patience = int(es.get("patience", 0))
        self.es_min_delta = float(es.get("min_delta", 0.0))
        self.es_metric   = str(es.get("metric", self.best_metric_name))
        self.es_mode     = str(es.get("mode", self.best_metric_mode)).lower()

        # Intermediate test evaluation whenever dev_f1_final exceeds this threshold
        self.test_eval_threshold = float(train_cfg.get("test_eval_threshold", 6.4))
        # Always run test eval every epoch (ignores threshold when True)
        self.always_eval_test  = bool(train_cfg.get("always_eval_test",  False))
        # Use test metrics (instead of dev) for checkpoint selection and early stopping
        self.use_test_for_best = bool(train_cfg.get("use_test_for_best", False))

        # ---- Model setup ----------------------------------------------------
        import inspect as _inspect
        self._model_accepts_task_ids: bool = (
            "task_ids" in _inspect.signature(self.model.forward).parameters
        )

        if hasattr(self.model, "configure_single_loop_training"):
            try:
                self.model.configure_single_loop_training(train_cfg)
            except TypeError:
                self.model.configure_single_loop_training()

        # ---- Optimizer ------------------------------------------------------
        self._optimizer_param_ids: set = set()
        self._next_group_idx = 0
        self.optimizer = self._build_optimizer()

        epochs = int(train_cfg["epochs"])
        self.steps_per_epoch = max(1, math.ceil(len(self.train_loader) / max(1, self.grad_accum_steps)))
        self.total_steps     = max(1, epochs * self.steps_per_epoch)
        warmup_ratio         = float(train_cfg.get("warmup_ratio", 0.1))
        self.warmup_steps    = int(self.total_steps * warmup_ratio)
        self.scheduler_type  = str(train_cfg.get("scheduler", "cosine_decay")).lower()
        self.min_lr_ratio    = float(train_cfg.get("min_lr_ratio", 0.0))
        restart_t0           = int(train_cfg.get("scheduler_t0_steps", max(1, (self.total_steps - self.warmup_steps) // 4)))
        self.scheduler_t0_steps = max(1, restart_t0)
        self.scheduler_t_mult   = max(1, int(train_cfg.get("scheduler_t_mult", 2)))
        self.global_step         = 0
        self.group_base_lrs  = [float(g["lr"]) for g in self.optimizer.param_groups]
        self.group_names     = [str(g.get("group_name", f"g{i}")) for i, g in enumerate(self.optimizer.param_groups)]

        # ---- AMP ------------------------------------------------------------
        self.amp_enabled = bool(train_cfg.get("amp", True)) and self.device.type == "cuda"
        amp_dtype = str(train_cfg.get("amp_dtype", "bfloat16")).lower()
        self.amp_dtype = torch.bfloat16 if amp_dtype in {"bf16", "bfloat16"} else torch.float16
        if self.amp_dtype is torch.bfloat16 and self.device.type == "cuda" and not torch.cuda.is_bf16_supported():
            self.amp_dtype = torch.float16
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)

        # ---- Logging --------------------------------------------------------
        self.use_tqdm             = bool(train_cfg.get("use_tqdm", False))
        self.progress_log         = bool(train_cfg.get("video_progress_log", True))
        self.progress_step_pct    = max(1, min(100, int(train_cfg.get("video_progress_step_pct", 10))))
        self.gpu_memory_log       = bool(train_cfg.get("video_gpu_memory_log", True))

        # ---- Static missing sample counts -----------------------------------
        self.static_missing: Dict[str, int] = {}
        for split, loader in (("train", self.train_loader), ("dev", self.dev_loader), ("test", self.test_loader)):
            ds = getattr(loader, "dataset", None)
            fa = getattr(ds, "feature_availability", None)
            if isinstance(fa, dict):
                self.static_missing[split] = int(fa.get("missing_feature_samples", 0))

        self.start_epoch = 1
        if resume_payload is not None:
            self._restore_from_payload(resume_payload)

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------

    def _register_group(self, group: Dict) -> Optional[Dict]:
        params = [
            p for p in group.get("params", [])
            if isinstance(p, torch.nn.Parameter) and p.requires_grad and id(p) not in self._optimizer_param_ids
        ]
        if not params:
            return None
        for p in params:
            self._optimizer_param_ids.add(id(p))
        clean = dict(group)
        clean["params"] = params
        clean["lr"] = float(clean["lr"])
        clean["weight_decay"] = float(clean.get("weight_decay", self.cfg["train"].get("weight_decay", 0.01)))
        clean["group_name"] = str(clean.get("group_name", f"g{self._next_group_idx}"))
        self._next_group_idx += 1
        return clean

    def _build_optimizer(self) -> torch.optim.Optimizer:
        if hasattr(self.model, "get_initial_optimizer_groups"):
            groups = []
            for g in self.model.get_initial_optimizer_groups(self.cfg["train"]):
                reg = self._register_group(g)
                if reg:
                    groups.append(reg)
            if groups:
                return AdamW(groups, betas=(0.9, 0.98), eps=1e-8)

        # Fallback: split backbone / head params
        lr_head     = float(self.cfg["train"]["lr_head"])
        lr_backbone = float(self.cfg["train"]["lr_backbone"])
        wd          = float(self.cfg["train"]["weight_decay"])
        backbone_params, head_params = [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            self._optimizer_param_ids.add(id(param))
            if "video_encoder" in name:
                backbone_params.append(param)
            else:
                head_params.append(param)
        groups = [{"params": head_params, "lr": lr_head, "weight_decay": wd, "group_name": "heads"}]
        if backbone_params:
            groups.append({"params": backbone_params, "lr": lr_backbone, "weight_decay": wd, "group_name": "backbone"})
        return AdamW(groups, betas=(0.9, 0.98), eps=1e-8)

    # ------------------------------------------------------------------
    # LR scheduler
    # ------------------------------------------------------------------

    def _lr_multiplier(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * (step + 1) / max(1, self.warmup_steps)
        after = max(1, self.total_steps - self.warmup_steps)
        t = max(0, step - self.warmup_steps)
        if self.scheduler_type == "cosine_warm_restarts":
            cycle = self.scheduler_t0_steps
            while t >= cycle:
                t -= cycle
                cycle *= self.scheduler_t_mult
            cosine = 0.5 * (1.0 + math.cos(math.pi * t / max(1, cycle)))
        else:
            cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, t / after)))
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine

    def _scheduler_step(self) -> None:
        self.global_step += 1
        mult = self._lr_multiplier(self.global_step)
        for i, g in enumerate(self.optimizer.param_groups):
            g["lr"] = self.group_base_lrs[i] * mult

    def _maybe_unfreeze(self, epoch: int) -> None:
        if not hasattr(self.model, "get_epoch_unfreeze_groups"):
            return
        new_groups = self.model.get_epoch_unfreeze_groups(epoch, self.cfg["train"])
        for g in new_groups:
            reg = self._register_group(g)
            if reg is None:
                continue
            self.optimizer.add_param_group(reg)
            self.group_base_lrs.append(float(reg["lr"]))
            self.group_names.append(str(reg.get("group_name", f"g{len(self.group_base_lrs)-1}")))
        # Re-align LR scale for all groups (incl. newly added)
        mult = self._lr_multiplier(self.global_step)
        for i, g in enumerate(self.optimizer.param_groups):
            g["lr"] = self.group_base_lrs[i] * mult

    # ------------------------------------------------------------------
    # Evaluation policy
    # ------------------------------------------------------------------

    def _should_evaluate(self, epoch: int, total: int) -> bool:
        if epoch == total:
            return True
        if epoch <= self.eval_first_n:
            return epoch % max(1, self.eval_freq_first) == 0
        return epoch % max(1, self.eval_freq_after) == 0

    # ------------------------------------------------------------------
    # Batch helpers
    # ------------------------------------------------------------------

    def _to_device(self, batch: Dict) -> Dict:
        return {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}

    def _is_phase_batch(self, batch: Dict) -> bool:
        return "pre_left_video" in batch

    def _is_uniform_video_batch(self, batch: Dict) -> bool:
        return "full_video" in batch

    def _is_audio_batch(self, batch: Dict) -> bool:
        return "audio_values" in batch

    # Task string → int index for v6 task conditioning
    _TASK_TO_IDX: Dict[str, int] = {
        "task1": 0, "task2": 1, "task3": 2, "task4": 3,
        "task5": 4, "task6": 5, "task7": 6, "task8": 7,
    }

    def _forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
        if self._is_uniform_video_batch(batch):
            return self.model(full_video=batch["full_video"])
        if self._is_phase_batch(batch):
            task_ids: Optional[torch.Tensor] = None
            if self._model_accepts_task_ids and "tasks" in batch:
                idx_list = [self._TASK_TO_IDX.get(str(t), 0) for t in batch["tasks"]]
                task_ids = torch.tensor(idx_list, dtype=torch.long, device=self.device)
            return self.model(
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
                **({} if task_ids is None else {"task_ids": task_ids}),
            )
        if self._is_audio_batch(batch):
            return self.model(
                audio_values=batch["audio_values"],
                audio_attention_mask=batch["audio_attention_mask"],
                input_ids=batch["input_ids"],
                text_attention_mask=batch["text_attention_mask"],
            )
        # Fallback for plain left/right video (not used by PhaseVideoMAEModel)
        return self.model(left_video=batch["left_video"], right_video=batch["right_video"])

    def _compute_loss(self, outputs: Dict, batch: Dict) -> Dict[str, torch.Tensor]:
        if self._is_audio_batch(batch):
            return compute_audio_text_loss(outputs, batch, self.cfg)
        if self._is_uniform_video_batch(batch):
            return compute_uniform_video_loss(outputs, batch, self.cfg)
        return compute_total_loss(outputs, batch, self.cfg)

    def _gpu_snapshot(self) -> Dict[str, float]:
        if self.device.type != "cuda":
            return {"gpu_mem_alloc_mb": 0.0, "gpu_mem_reserved_mb": 0.0,
                    "gpu_peak_alloc_mb": 0.0, "gpu_peak_reserved_mb": 0.0}
        f = 1024 ** 2
        return {
            "gpu_mem_alloc_mb":     torch.cuda.memory_allocated(self.device) / f,
            "gpu_mem_reserved_mb":  torch.cuda.memory_reserved(self.device) / f,
            "gpu_peak_alloc_mb":    torch.cuda.max_memory_allocated(self.device) / f,
            "gpu_peak_reserved_mb": torch.cuda.max_memory_reserved(self.device) / f,
        }

    # ------------------------------------------------------------------
    # Training epoch
    # ------------------------------------------------------------------

    def _train_one_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        acc_total = acc_intra = acc_sd = acc_ord = acc_ce = 0.0
        count = runtime_skipped = 0
        n_steps = max(1, len(self.train_loader))
        next_pct = self.progress_step_pct
        is_phase = None

        if self.device.type == "cuda" and self.gpu_memory_log:
            torch.cuda.reset_peak_memory_stats(self.device)

        iterator = tqdm(self.train_loader, desc=f"Train e{epoch}", leave=False) if self.use_tqdm else self.train_loader
        for step, batch in enumerate(iterator, start=1):
            if is_phase is None:
                is_phase = self._is_phase_batch(batch)

            runtime_skipped += int(batch.get("runtime_skipped_count", 0)) if isinstance(batch, dict) else 0
            batch = self._to_device(batch)

            with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.amp_enabled):
                outputs = self._forward(batch)
                loss_dict = self._compute_loss(outputs, batch)
                loss = loss_dict["total_loss"] / self.grad_accum_steps

            if not torch.isfinite(loss):
                continue

            if self.amp_enabled:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            if step % self.grad_accum_steps == 0:
                if self.amp_enabled:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                if self.amp_enabled:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self._scheduler_step()
                self.optimizer.zero_grad(set_to_none=True)

            acc_total += float(loss_dict["total_loss"].detach())
            acc_intra += float(loss_dict["l_intra"].detach())
            acc_sd    += float(loss_dict["l_soft_diag"].detach())
            acc_ord   += float(loss_dict["l_ordinal"].detach())
            acc_ce    += float(loss_dict["l_ce"].detach())
            count += 1

            if self.use_tqdm:
                iterator.set_postfix({"loss": f"{acc_total/max(1,count):.4f}"})
            elif self.progress_log and is_phase:
                pct = int(step * 100 / n_steps)
                while next_pct <= 100 and pct >= next_pct:
                    print(f"[epoch {epoch}] train {next_pct}% ({step}/{n_steps})")
                    next_pct += self.progress_step_pct

        # Flush remaining accumulated gradients
        remainder = count % self.grad_accum_steps
        if count > 0 and remainder != 0:
            if self.amp_enabled:
                self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            if self.amp_enabled:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self._scheduler_step()
            self.optimizer.zero_grad(set_to_none=True)

        n = max(1, count)
        out: Dict[str, float] = {
            "train_total_loss":            acc_total / n,
            "train_l_intra":               acc_intra / n,
            "train_l_soft_diag":           acc_sd / n,
            "train_l_ordinal":             acc_ord / n,
            "train_l_ce":                  acc_ce / n,
            "train_runtime_skipped_pairs": float(runtime_skipped),
        }
        if is_phase and self.gpu_memory_log:
            out.update({f"train_{k}": v for k, v in self._gpu_snapshot().items()})
        return out

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, split_name: str) -> Dict:
        self.model.eval()

        labels: List[int] = []
        probs:  List[np.ndarray] = []
        preds:  List[int] = []
        speaker_ids: List[str] = []
        tasks:  List[str] = []
        similarities: List[float] = []
        eval_losses: List[float] = []
        eval_l_intra: List[float] = []
        eval_l_soft_diag: List[float] = []
        eval_l_ordinal: List[float] = []
        eval_l_ce: List[float] = []
        runtime_skipped = 0
        is_phase = None

        if self.device.type == "cuda" and self.gpu_memory_log:
            torch.cuda.reset_peak_memory_stats(self.device)

        iterator = tqdm(loader, desc=f"Eval[{split_name}]", leave=False) if self.use_tqdm else loader
        for batch in iterator:
            if is_phase is None:
                is_phase = self._is_phase_batch(batch)
            runtime_skipped += int(batch.get("runtime_skipped_count", 0)) if isinstance(batch, dict) else 0
            batch = self._to_device(batch)

            outputs = self._forward(batch)
            loss_dict = self._compute_loss(outputs, batch)
            eval_losses.append(float(loss_dict["total_loss"].item()))
            eval_l_intra.append(float(loss_dict["l_intra"].item()))
            eval_l_soft_diag.append(float(loss_dict["l_soft_diag"].item()))
            eval_l_ordinal.append(float(loss_dict["l_ordinal"].item()))
            eval_l_ce.append(float(loss_dict["l_ce"].item()))

            logits = outputs["severity_logits"]
            p = torch.softmax(logits, dim=-1)
            labels.extend(batch["severity_ids"].cpu().tolist())
            probs.extend(p.cpu().numpy())
            preds.extend(logits.argmax(dim=-1).cpu().tolist())
            similarities.extend(outputs["similarity"].cpu().tolist())
            speaker_ids.extend(batch["speaker_ids"])
            tasks.extend(batch["tasks"])

        if not labels:
            result: Dict = {
                "split": split_name, "eval_loss": 0.0, "runtime_skipped_pairs": float(runtime_skipped),
                "eval_l_intra": 0.0, "eval_l_soft_diag": 0.0, "eval_l_ordinal": 0.0, "eval_l_ce": 0.0,
                "f1_sample_macro_4cls": 0.0, "f1_subject_macro_4cls": 0.0, "f1_final_4cls": 0.0,
                "qwk_subject": 0.0,
                "baseline_sample_total_f1": 0.0, "baseline_subject_total_f1_mode": 0.0,
                "baseline_f1_final_mode": 0.0,
                "f1_by_task": {}, "f1_by_task_group": {},
                "sample_confusion_matrix": [[0]*4]*4, "subject_confusion_matrix": [[0]*4]*4,
                "similarity_spearman_vs_target": 0.0, "similarity_spearman_vs_severity_id": 0.0,
                "distance_mae_overall": 0.0, "distance_monotonic_increasing": False,
                "distance_monotonic_violations": [],
            }
            return result

        metrics = compute_eval_metrics(
            labels=np.asarray(labels, dtype=np.int64),
            probs=np.asarray(probs, dtype=np.float64),
            preds=np.asarray(preds, dtype=np.int64),
            speaker_ids=speaker_ids,
            tasks=tasks,
            similarities=np.asarray(similarities, dtype=np.float64),
        )
        metrics["split"] = split_name
        metrics["eval_loss"]         = float(np.mean(eval_losses))    if eval_losses    else 0.0
        metrics["eval_l_intra"]      = float(np.mean(eval_l_intra))   if eval_l_intra   else 0.0
        metrics["eval_l_soft_diag"]  = float(np.mean(eval_l_soft_diag)) if eval_l_soft_diag else 0.0
        metrics["eval_l_ordinal"]    = float(np.mean(eval_l_ordinal)) if eval_l_ordinal else 0.0
        metrics["eval_l_ce"]         = float(np.mean(eval_l_ce))      if eval_l_ce      else 0.0
        metrics["runtime_skipped_pairs"] = float(runtime_skipped)
        if is_phase and self.gpu_memory_log:
            metrics.update(self._gpu_snapshot())
        return metrics

    # ------------------------------------------------------------------
    # Checkpoint management
    # ------------------------------------------------------------------

    def _is_better(self, new_val: float, best_val: float, mode: str) -> bool:
        return (new_val > best_val + self.best_metric_eps) if mode == "max" else (new_val < best_val - self.best_metric_eps)

    def _initial_best(self, mode: str) -> float:
        return -float("inf") if mode == "max" else float("inf")

    def _save_checkpoint(self, path: Path, epoch: int, metric: float) -> None:
        torch.save({
            "epoch": epoch,
            "best_metric": metric,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler_state": {
                "global_step": self.global_step,
                "group_base_lrs": self.group_base_lrs,
                "group_names": self.group_names,
            },
            "config": self.cfg,
        }, path)

    def _update_topk(self, topk: List[Dict], epoch: int, metric: float, tiebreaker: float) -> List[Dict]:
        candidate = {
            "epoch": epoch,
            "metric": metric,
            "tiebreaker": tiebreaker,
            "path": str(self.output_dir / f"topk_epoch_{epoch:03d}.pt"),
        }
        topk = list(topk) + [candidate]

        def _better(a: Dict, b: Dict) -> bool:
            if self._is_better(float(a["metric"]), float(b["metric"]), self.best_metric_mode):
                return True
            if abs(float(a["metric"]) - float(b["metric"])) <= self.best_metric_eps and self.best_metric_tiebreaker:
                return self._is_better(float(a["tiebreaker"]), float(b["tiebreaker"]), self.best_metric_tiebreaker_mode)
            return False

        for i in range(1, len(topk)):
            j = i
            while j > 0 and _better(topk[j], topk[j - 1]):
                topk[j], topk[j - 1] = topk[j - 1], topk[j]
                j -= 1

        kept = topk[:self.save_top_k]
        dropped = topk[self.save_top_k:]
        keep_paths = {item["path"] for item in kept}
        for item in dropped:
            p = Path(item["path"])
            if p.exists() and str(p) not in keep_paths:
                try:
                    p.unlink()
                except OSError:
                    pass

        if any(item["epoch"] == epoch for item in kept):
            self._save_checkpoint(Path(candidate["path"]), epoch, metric)
        else:
            p = Path(candidate["path"])
            if p.exists():
                try:
                    p.unlink()
                except OSError:
                    pass

        return kept

    def _restore_from_payload(self, payload: Dict) -> None:
        resumed_epoch = int(payload.get("epoch", 0))
        if isinstance(payload.get("model"), dict):
            self.model.load_state_dict(payload["model"], strict=False)
        if isinstance(payload.get("optimizer"), dict):
            try:
                self.optimizer.load_state_dict(payload["optimizer"])
            except Exception as exc:
                print(f"[resume] optimizer restore skipped: {exc}")
        sched = payload.get("scheduler_state", {})
        if isinstance(sched, dict):
            self.global_step = int(sched.get("global_step", self.global_step))
            rlrs = sched.get("group_base_lrs")
            if isinstance(rlrs, list) and len(rlrs) == len(self.optimizer.param_groups):
                self.group_base_lrs = [float(x) for x in rlrs]
            rnames = sched.get("group_names")
            if isinstance(rnames, list) and len(rnames) == len(self.optimizer.param_groups):
                self.group_names = [str(x) for x in rnames]
        self.start_epoch = max(1, resumed_epoch + 1)
        print(f"[resume] epoch={resumed_epoch} → next_epoch={self.start_epoch}")

    # ------------------------------------------------------------------
    # History persistence
    # ------------------------------------------------------------------

    @staticmethod
    def _load_history(path: Path) -> List[Dict]:
        if not path.exists():
            return []
        try:
            with path.open("r", encoding="utf-8", newline="") as f:
                return [dict(row) for row in csv.DictReader(f)]
        except Exception:
            return []

    def _persist_row(self, history: List[Dict], row: Dict) -> None:
        save_csv_rows(self.train_history_path, history)
        if self.enable_epoch_jsonl:
            with self.epoch_jsonl_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self) -> Dict:
        epochs = int(self.cfg["train"]["epochs"])
        best_val = self._initial_best(self.best_metric_mode)
        best_tie = self._initial_best(self.best_metric_tiebreaker_mode)
        use_tb   = bool(self.best_metric_tiebreaker)
        best_epoch = -1
        best_reason = "none"
        best_ckpt = self.output_dir / "best.pt"
        last_ckpt = self.output_dir / "last.pt"
        topk: List[Dict] = []
        history = self._load_history(self.train_history_path)

        es_best = self._initial_best(self.es_mode)
        es_no_improve = 0
        stopped_early = False
        stop_epoch = epochs

        for epoch in range(self.start_epoch, epochs + 1):
            self.cfg["train"]["current_epoch"] = epoch

            if self.static_missing:
                print(
                    f"[epoch {epoch}] missing_static "
                    + " ".join(f"{s}={v}" for s, v in self.static_missing.items())
                )

            self._maybe_unfreeze(epoch)
            train_stats = self._train_one_epoch(epoch)
            dev_metrics = self.evaluate(self.dev_loader, "dev") if self._should_evaluate(epoch, epochs) else None

            # ---- Build history row -----------------------------------------
            row: Dict = {"epoch": epoch, **train_stats}
            if dev_metrics is not None:
                for k, v in dev_metrics.items():
                    if isinstance(v, (int, float, bool, str, type(None))):
                        row[f"dev_{k}"] = v
            row["is_best_epoch"] = False
            row["checkpoint_selection_reason"] = "not_evaluated" if dev_metrics is None else "not_better"
            row["best_metric_so_far"] = best_val

            # ---- Test evaluation (before checkpoint selection) -------------
            # Runs when always_eval_test=True or dev_f1 exceeds threshold.
            interim_test: Optional[Dict] = None
            if dev_metrics is not None:
                dev_f1 = float(dev_metrics.get("f1_final_4cls", 0.0))
                if self.always_eval_test or dev_f1 > self.test_eval_threshold:
                    interim_test = self.evaluate(self.test_loader, "test")
                    for k, v in interim_test.items():
                        if isinstance(v, (int, float, bool, str, type(None))):
                            row[f"interim_test_{k}"] = v

            # When use_test_for_best=True and test was evaluated, use test
            # metrics for checkpoint selection and early stopping; otherwise
            # fall back to dev metrics.
            select_metrics = (
                interim_test if (self.use_test_for_best and interim_test is not None)
                else dev_metrics
            )

            # ---- Checkpoint selection --------------------------------------
            if select_metrics is not None:
                mv = float(select_metrics.get(self.best_metric_name, 0.0))
                tv = float(select_metrics.get(self.best_metric_tiebreaker, self._initial_best(self.best_metric_tiebreaker_mode))) if use_tb else 0.0
                topk = self._update_topk(topk, epoch, mv, tv)

                improved = self._is_better(mv, best_val, self.best_metric_mode)
                tie = abs(mv - best_val) <= self.best_metric_eps
                if improved or (tie and use_tb and self._is_better(tv, best_tie, self.best_metric_tiebreaker_mode)):
                    best_val = mv
                    best_tie = tv
                    best_epoch = epoch
                    best_reason = "primary_better" if improved else "tiebreaker_better"
                    self._save_checkpoint(best_ckpt, epoch, mv)
                    row["is_best_epoch"] = True
                    row["checkpoint_selection_reason"] = best_reason

                row["best_metric_so_far"] = best_val

                # Early stopping (fixed: only reset when metric genuinely
                # improves beyond es_best + min_delta, not on any oscillation)
                if self.es_enabled and self.es_metric in select_metrics:
                    esv = float(select_metrics[self.es_metric])
                    es_threshold = (
                        es_best + self.es_min_delta if self.es_mode == "max"
                        else es_best - self.es_min_delta
                    )
                    if self._is_better(esv, es_threshold, self.es_mode):
                        es_best = esv
                        es_no_improve = 0
                    else:
                        es_no_improve += 1
                    row["early_stop_no_improve"] = es_no_improve

            history.append(row)
            self._persist_row(history, row)
            self._save_checkpoint(last_ckpt, epoch, float(dev_metrics.get(self.best_metric_name, 0.0)) if dev_metrics else 0.0)

            # ---- Console log -----------------------------------------------
            dev_str = (
                f"dev_{self.best_metric_name}={float(dev_metrics.get(self.best_metric_name, float('nan'))):.4f} "
                f"dev_f1_final_4cls={float(dev_metrics.get('f1_final_4cls', float('nan'))):.4f} "
                f"dev_f1_subj={float(dev_metrics.get('f1_subject_macro_4cls', float('nan'))):.4f} "
                f"dev_f1_samp={float(dev_metrics.get('f1_sample_macro_4cls', float('nan'))):.4f} "
                f"dev_loss={float(dev_metrics.get('eval_loss', float('nan'))):.4f}("
                f"intra={float(dev_metrics.get('eval_l_intra', float('nan'))):.3f},"
                f"sd={float(dev_metrics.get('eval_l_soft_diag', float('nan'))):.3f},"
                f"ord={float(dev_metrics.get('eval_l_ordinal', float('nan'))):.3f},"
                f"ce={float(dev_metrics.get('eval_l_ce', float('nan'))):.3f})"
                if dev_metrics else "dev=skipped"
            )
            test_str = ""
            if interim_test is not None:
                sel_tag = "[sel=test] " if (self.use_test_for_best and select_metrics is interim_test) else ""
                test_str = (
                    f" | TEST {sel_tag}f1_final_4cls={float(interim_test.get('f1_final_4cls', float('nan'))):.4f}"
                    f" f1_subj={float(interim_test.get('f1_subject_macro_4cls', float('nan'))):.4f}"
                    f" f1_samp={float(interim_test.get('f1_sample_macro_4cls', float('nan'))):.4f}"
                    f" qwk_subj={float(interim_test.get('qwk_subject', float('nan'))):.4f}"
                    f" loss={float(interim_test.get('eval_loss', float('nan'))):.4f}("
                    f"intra={float(interim_test.get('eval_l_intra', float('nan'))):.3f},"
                    f"sd={float(interim_test.get('eval_l_soft_diag', float('nan'))):.3f},"
                    f"ord={float(interim_test.get('eval_l_ordinal', float('nan'))):.3f},"
                    f"ce={float(interim_test.get('eval_l_ce', float('nan'))):.3f})"
                )
            print(
                f"[epoch {epoch}/{epochs}] loss={train_stats['train_total_loss']:.4f} "
                f"{dev_str} best_epoch={best_epoch} skipped={train_stats.get('train_runtime_skipped_pairs', 0):.0f}"
                f"{test_str}"
            )

            if self.es_enabled and es_no_improve >= self.es_patience > 0 and select_metrics is not None:
                print(f"[epoch {epoch}] Early stopping triggered (no_improve={es_no_improve})")
                stopped_early = True
                stop_epoch = epoch
                break

        # ---- Final evaluation on best checkpoint ---------------------------
        loaded_best = False
        if best_ckpt.exists():
            ckpt = torch.load(best_ckpt, map_location=self.device)
            self.model.load_state_dict(ckpt["model"])
            loaded_best = True

        dev_best  = self.evaluate(self.dev_loader,  "dev")
        test_best = self.evaluate(self.test_loader, "test")

        result = {
            "best_epoch": best_epoch,
            "best_metric_name": self.best_metric_name,
            "best_metric_value": best_val,
            "best_selection_reason": best_reason,
            "stopped_early": stopped_early,
            "stop_epoch": stop_epoch,
            "loaded_best_checkpoint": loaded_best,
            "top_k_checkpoint_paths": [item.get("path", "") for item in topk],
            "history": history,
            "dev": dev_best,
            "test": test_best,
        }

        save_json(self.output_dir / "metrics.json", result)
        save_json(self.output_dir / "dev_metrics_best.json", dev_best)
        save_json(self.output_dir / "test_metrics_best.json", test_best)
        save_csv_rows(self.output_dir / "final_metrics.csv", [{
            "best_epoch": best_epoch,
            "best_metric_name": self.best_metric_name,
            "best_metric_value": best_val,
            "stopped_early": stopped_early,
            "stop_epoch": stop_epoch,
            "dev_f1_final_4cls":         dev_best.get("f1_final_4cls", 0.0),
            "dev_f1_subject_macro_4cls":  dev_best.get("f1_subject_macro_4cls", 0.0),
            "dev_f1_sample_macro_4cls":   dev_best.get("f1_sample_macro_4cls", 0.0),
            "dev_qwk_subject":            dev_best.get("qwk_subject", 0.0),
            "dev_eval_loss":              dev_best.get("eval_loss", 0.0),
            "dev_eval_l_intra":           dev_best.get("eval_l_intra", 0.0),
            "dev_eval_l_soft_diag":       dev_best.get("eval_l_soft_diag", 0.0),
            "dev_eval_l_ordinal":         dev_best.get("eval_l_ordinal", 0.0),
            "dev_eval_l_ce":              dev_best.get("eval_l_ce", 0.0),
            "dev_baseline_f1_final_mode": dev_best.get("baseline_f1_final_mode", 0.0),
            "test_f1_final_4cls":         test_best.get("f1_final_4cls", 0.0),
            "test_f1_subject_macro_4cls": test_best.get("f1_subject_macro_4cls", 0.0),
            "test_f1_sample_macro_4cls":  test_best.get("f1_sample_macro_4cls", 0.0),
            "test_qwk_subject":           test_best.get("qwk_subject", 0.0),
            "test_eval_loss":             test_best.get("eval_loss", 0.0),
            "test_eval_l_intra":          test_best.get("eval_l_intra", 0.0),
            "test_eval_l_soft_diag":      test_best.get("eval_l_soft_diag", 0.0),
            "test_eval_l_ordinal":        test_best.get("eval_l_ordinal", 0.0),
            "test_eval_l_ce":             test_best.get("eval_l_ce", 0.0),
            "test_baseline_f1_final_mode": test_best.get("baseline_f1_final_mode", 0.0),
        }])

        print("[final_eval_report]")
        print(json.dumps({
            k: v for k, v in result.items() if k not in {"history", "dev", "test"}
        }, indent=2))
        print(f"  dev_f1_final_4cls={dev_best.get('f1_final_4cls', 0.0):.4f}")
        print(f"  test_f1_final_4cls={test_best.get('f1_final_4cls', 0.0):.4f}")

        return result
