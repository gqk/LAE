# -*- coding: utf-8 -*-

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from absl import flags, logging
from omegaconf import OmegaConf as oc


from ..utils.config import build_config
from ..utils.tensorboard import TensorboardWriter
from .abc import DataModuleABC, ModuleABC, TrainerABC
from .datamodule import DataConf
from .module import ModuleConf


@dataclass
class TrainerConf:
    seed_everything: int = 42
    accelerator: str = "gpu"
    strategy: Optional[str] = None
    devices: int = 1
    num_nodes: int = 1
    precision: int = 32
    benchmark: bool = True
    deterministic: bool = True
    checkpoint: Optional[str] = None


_FABRIC_FLAGS = ["accelerator", "strategy", "devices", "num_nodes", "precision"]


def get_lite_flags(cfg: TrainerConf) -> Dict[str, Any]:
    return {k: getattr(cfg, k) for k in _FABRIC_FLAGS}


class Trainer(TrainerABC):
    log_dir: str
    ckpt_dir: str

    def __init__(self, cfg: TrainerConf, base_dir: str, **kwargs):
        kwargs = dict(**get_lite_flags(cfg), **kwargs)
        if kwargs.get("devices", 1) < 2 and kwargs.get("num_nodes", 1) < 2:
            strategy = kwargs.get("strategy", None)
            if strategy is not None:
                logging.warn(
                    f"Strategy is set as {strategy}, but there is no more "
                    f"than 1 device, reset it as None."
                )
            kwargs["strategy"] = None
        super().__init__(**kwargs)

        self.cfg = cfg
        self.base_dir = base_dir

    def setup_task(self):
        task_id = self.datamodule.current_task
        self.log_dir = Path(self.base_dir) / f"task_{task_id}"
        self.ckpt_dir = self.log_dir / "checkpoints"

        if self.is_global_zero:
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)
            logging.get_absl_handler().use_absl_log_file("log", self.log_dir)
            self.module.logger = TensorboardWriter(self.log_dir)
            self.save_config()

        self.module.setup()
        self.module.current_epoch, self.module.global_step = 0, 0
        if task_id == self.start_task and self.cfg.checkpoint:
            self.load_checkpoint()

    def load_checkpoint(self):
        ckpt = self.load(self.cfg.checkpoint)
        self.module.on_load_checkpoint(ckpt)
        self.datamodule.on_load_checkpoint(ckpt)

    def save_checkpoint(self):
        if self.is_global_zero:
            ckpt, current_epoch = dict(), self.module.current_epoch
            self.module.on_save_checkpoint(ckpt)
            self.datamodule.on_save_checkpoint(ckpt)
            self.save(ckpt, self.ckpt_dir / f"epoch_{current_epoch}.ckpt")

    def save_config(self):
        cfg = oc.create(oc.to_container(self.cfg))
        self.datamodule.on_save_config(cfg)
        self.module.on_save_config(cfg)
        oc.save(cfg, self.log_dir / "config.yaml")

    def run_task(self, mode: str = "train"):
        self.module.run(mode)

    def run(self, mode: str = "train"):
        self.launch()
        self.seed_everything(self.cfg.seed_everything)

        torch.backends.cudnn.benchmark = self.cfg.benchmark
        torch.backends.cudnn.deterministic = self.cfg.deterministic

        flags.FLAGS.alsologtostderr = True
        flags.FLAGS.showprefixforinfo = False
        if not self.is_global_zero:
            flags.FLAGS.verbosity = -1000  # disable logging
            Path(self.base_dir).rmdir()  # ddp launch issue
        self.base_dir = self.broadcast(self.base_dir)  # ddp launch issue

        self.module.datamodule = self.datamodule
        self.module.trainer = self
        self.module.device = self.device

        dm = self.datamodule
        for dm.current_task in range(dm.current_task, dm.num_tasks):
            self.setup_task()
            self.run_task(mode)
            if mode == "eval":
                break
