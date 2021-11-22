# -*- coding: utf-8 -*-

import os
from abc import ABC, abstractmethod
from packaging import version
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Tuple,
    TypeVar,
)

import torch
import torch.nn as nn
from continuum.scenarios.base import _BaseScenario as Scenario
from omegaconf import DictConfig
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from lightning import Fabric
from lightning.fabric.strategies import STRATEGY_REGISTRY

from ..data.memory import Memory
from ..utils.tensorboard import TensorboardWriter


class DataModuleABC(ABC):
    current_task: int
    start_task: int
    num_tasks: int
    scenario_train: Scenario
    scenario_val: Scenario
    memory: Optional[Memory] = None
    class_names: List[str]

    @property
    def num_new_classes(self):
        return self.scenario_train.increments[self.current_task]

    @property
    def num_old_classes(self):
        return sum(self.scenario_train.increments[: self.current_task])

    @property
    def num_seen_classes(self):
        return sum(self.scenario_train.increments[: self.current_task + 1])

    def num_classes_of(self, task_id: int):
        return self.scenario_train.increments[task_id]

    @abstractmethod
    def train_dataloader(self) -> DataLoader:
        ...

    @abstractmethod
    def val_dataloader(self) -> DataLoader:
        ...

    def reset_transform(self, transform: Callable, train: bool):
        pass

    def on_load_checkpoint(self, ckpt: Dict[str, Any]) -> None:
        pass

    def on_save_checkpoint(self, ckpt: Dict[str, Any]) -> None:
        pass

    def on_save_config(self, cfg: DictConfig) -> None:
        pass


TrainerType = TypeVar("Trainer", bound="TrainerABC")


class ModuleABC(nn.Module):
    current_epoch: int
    global_step: int
    device: torch.DeviceObjType
    logger: TensorboardWriter
    datamodule: DataModuleABC
    trainer: TrainerType

    @property
    def current_task(self) -> int:
        return self.datamodule.current_task

    @property
    def start_task(self) -> int:
        return self.datamodule.start_task

    @property
    def global_rank(self) -> int:
        return self.trainer.global_rank

    def setup(self) -> None:
        pass

    def wrap(self, model: nn.Module, *optimizers: List[Optimizer], **kwargs):
        return self.trainer.setup(model, *optimizers, **kwargs)

    def run(self, mode: str) -> None:
        pass

    def backward(self, *args, **kwargs) -> None:
        return self.trainer.backward(*args, **kwargs)

    def autocast(self, *args, **kwargs) -> Generator[None, None, None]:
        return self.trainer.autocast()

    def to_device(self, *args, **kwargs) -> Any:
        return self.trainer.to_device(*args, **kwargs)

    def on_load_checkpoint(self, ckpt: Dict[str, Any]) -> None:
        pass

    def on_save_checkpoint(self, ckpt: Dict[str, Any]) -> None:
        pass

    def on_save_config(self, cfg: DictConfig) -> None:
        pass

    def pre_train_epoch(self) -> None:
        pass

    def train_epoch(self, loader: DataLoader, num_batches: int) -> None:
        pass

    def post_train_epoch(self) -> None:
        pass

    def pre_train_step(self) -> None:
        pass

    def train_step(
        self, batch, batch_idx
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        pass

    def post_train_step(self) -> None:
        pass

    def pre_eval_epoch(self) -> None:
        pass

    def eval_epoch(self, loader: DataLoader) -> Any:
        pass

    def eval_step(self, batch, batch_idx) -> torch.Tensor:
        pass

    def post_eval_epoch(self, output: Any = None) -> None:
        pass


class TrainerABC(Fabric):
    datamodule: DataModuleABC
    module: ModuleABC

    @property
    def current_task(self):
        return self.datamodule.current_task

    @property
    def start_task(self):
        return self.datamodule.start_task

    @abstractmethod
    def save_checkpoint(self) -> None:
        ...

    def attach(self, module: ModuleABC, datamodule: DataModuleABC):
        self.datamodule = datamodule
        self.module = module


def _fix_ddp_unused_params():
    key = "find_unused_parameters"
    val = os.getenv("DDP_FIND_UNUSED_PARAMETERS", True)
    for name in STRATEGY_REGISTRY:
        if "ddp" in name:
            STRATEGY_REGISTRY[name]["init_params"][key] = val


_fix_ddp_unused_params()


def _fix_f32_mp():
    try:
        f32mp = os.getenv("TORCH_FLOAT32_MATMUL_PRECISION", "medium")
        torch.set_float32_matmul_precision(f32mp)
    except:
        pass


_fix_f32_mp()
