# -*- coding: utf-8 -*-

from dataclasses import dataclass
from math import cos, pi

import torch.nn as nn
import torch.optim as optim
from omegaconf import OmegaConf as oc


@dataclass
class SchedulerConf:
    mode: str = "linear"  # enum: linear cosine, cosine_pi
    cycle: int = 0
    scale_min: float = 0.0


def create_scheduler(
    cfg: SchedulerConf,
    optimizer: optim.Optimizer,
    num_steps: int,
):
    T = cfg.cycle if cfg.cycle > 0 else num_steps
    s_min = cfg.scale_min
    scale = lambda r: r * (1 - s_min) + s_min
    if cfg.mode == "cosine_pi":  # 0 -> T => 0 -> pi
        lr_lambda = lambda t: scale(0.5 + 0.5 * cos(t / T * pi))
    elif cfg.mode == "cosine":  # 0 -> T => 0 -> 0.5*pi
        lr_lambda = lambda t: scale(cos(t / T * pi / 2))
    else:  # default is linear
        lr_lambda = lambda t: scale(1 - t / T)
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
