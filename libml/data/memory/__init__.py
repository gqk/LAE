# -*- coding: utf-8 -*-

from dataclasses import dataclass

from omegaconf import OmegaConf as oc

from .base import Memory
from .herding import Herding


@dataclass
class MemoryConf:
    algo_name: str = "herding"
    max_size: int = 0
    num_samples_per_class: int = 0
    norm_feature: bool = False


def create_memory(cfg: MemoryConf):
    kwargs = oc.to_container(cfg)
    algo_name = kwargs.pop("algo_name")
    if algo_name not in ["herding"]:
        raise KeyError(f"Unknown memory algo name: {algo_name}")
    return Herding(**kwargs)
