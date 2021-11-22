# -*- coding: utf-8 -*-

from dataclasses import dataclass, field
from inspect import signature
from typing import Any, Dict, List

import torch.nn as nn
import torch.optim as optim


@dataclass
class OptimizerConf:
    algo: str = "Adam"  # enum: SGD, Adam
    lr: float = 0.0028125
    weight_decay: float = 0.0
    kwargs: Dict[str, Any] = field(default_factory=lambda: {})


def create_optimizer(cfg: OptimizerConf, *models: List[nn.Module]):
    algo_cls = getattr(optim, cfg.algo, None)
    if algo_cls is None:
        raise KeyError("Unknown optimization algorithm: {cfg.algorithm}")

    parameters = nn.ModuleList(models).parameters()
    kwargs = dict(**cfg.kwargs, lr=cfg.lr)
    if "weight_decay" in signature(algo_cls).parameters:
        kwargs["weight_decay"] = cfg.weight_decay
    return algo_cls(parameters, **kwargs)
