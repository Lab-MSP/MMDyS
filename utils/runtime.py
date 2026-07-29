"""Reproducibility helper: set all relevant random seeds."""
from __future__ import annotations

from typing import Optional

import numpy as np
import random
import torch


def set_seed(
    seed: int,
    *,
    deterministic: bool = False,
    cudnn_benchmark: Optional[bool] = None,
    deterministic_algorithms: Optional[bool] = None,
    deterministic_warn_only: bool = True,
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False if cudnn_benchmark is None else bool(cudnn_benchmark)
    elif cudnn_benchmark is not None:
        torch.backends.cudnn.benchmark = bool(cudnn_benchmark)

    use_det_algos = deterministic if deterministic_algorithms is None else bool(deterministic_algorithms)
    if use_det_algos:
        try:
                torch.use_deterministic_algorithms(True, warn_only=bool(deterministic_warn_only))
        except TypeError:
            torch.use_deterministic_algorithms(True)
