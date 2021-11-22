# -*- coding: utf-8 -*-

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

from continuum import ClassIncremental
from continuum.datasets import CIFAR100, ImageNet100, DomainNet
from continuum.datasets.base import _ContinuumDataset as Dataset
from continuum.scenarios.base import _BaseScenario as Scenario
from omegaconf import DictConfig
from omegaconf import OmegaConf as oc
from torch.utils.data import DataLoader

from ..data.dataloader import DataLoaderConf, create_dataloader
from ..data.imagenet_r import ImageNetR
from ..data.memory import Memory, MemoryConf, create_memory
from ..data.utils import split_train_val
from .abc import DataModuleABC


@dataclass
class DataConf:
    current_task: int = 0
    dataset: str = "cifar100"
    root: str = "./data"
    num_init_classes: int = 0
    num_increment_classes: int = 10
    class_order: Optional[List[int]] = None
    memory: MemoryConf = MemoryConf()
    dataloader: DataLoaderConf = DataLoaderConf()
    eval_on: str = "seen"  # enum seen current all
    val_split_ratio: float = 0.0
    train_val_split_seed: int = 42


class DataModule(DataModuleABC):
    dataset_train: Dataset = None
    dataset_val: Dataset = None

    def __init__(self, cfg: DataConf):
        self.cfg = cfg
        self.current_task = self.cfg.current_task
        self.start_task = self.cfg.current_task
        self.setup()

    @property
    def num_tasks(self):
        return self.scenario_train.nb_tasks

    def setup(self):
        kwargs = dict(
            initial_increment=self.cfg.num_init_classes,
            increment=self.cfg.num_increment_classes,
            class_order=self.cfg.class_order,
        )

        self.dataset_train = self.configure_dataset(train=True)
        if self.cfg.val_split_ratio > 0:
            seed = self.cfg.train_val_split_seed
            args = (self.dataset_train, self.cfg.val_split_ratio, seed)
            self.dataset_train, self.dataset_val = split_train_val(*args)
        else:
            self.dataset_val = self.configure_dataset(train=False)
        self.scenario_train = ClassIncremental(
            self.dataset_train,
            transformations=self.train_transform(),
            **kwargs,
        )
        self.scenario_val = ClassIncremental(
            self.dataset_val,
            transformations=self.val_transform(),
            **kwargs,
        )
        if self.cfg.memory and self.cfg.memory.max_size > 0:
            self.memory = create_memory(self.cfg.memory)

    def on_load_checkpoint(self, ckpt):
        pass

    def on_save_checkpoint(self, ckpt):
        pass

    def on_save_config(self, cfg: DictConfig):
        cfg.data = oc.create(oc.to_container(self.cfg))
        cfg.data.current_task = self.current_task

    def configure_dataset(self, train: bool):
        name = self.cfg.dataset.lower()
        root = str(Path(self.cfg.root) / name)
        if name == "cifar100":
            dataset = CIFAR100(root, train=train, download=True)
        elif name == "imagenet-r":
            dataset = ImageNetR(root, train=train, download=True)
        elif name == "domainnet":
            dataset = DomainNet(root, train=train, download=True)
        else:
            raise ValueError(f"Unknown dataset {name}.")
        return dataset

    def reset_transform(self, transform: Callable, train: bool):
        if train:
            self.scenario_train.trsf = transform
        else:
            self.scenario_val.trsf = transform

    def train_transform(self):
        return self.dataset_train.transformations

    def val_transform(self):
        return self.dataset_val.transformations

    def train_dataloader(
        self,
        mock_test: bool = False,
        with_memory: bool = True,
    ) -> DataLoader:
        if not isinstance(self.scenario_train, ClassIncremental):
            raise NotImplementedError

        dataset = self.scenario_train[self.current_task]
        if with_memory and self.memory is not None and self.current_task > 0:
            dataset = deepcopy(dataset)
            memset_args = (self.scenario_train, self.current_task)
            dataset.concat(self.memory.to_dataset(*memset_args))

        kwargs = dict(config=self.cfg.dataloader, train=not mock_test)
        return create_dataloader(dataset, **kwargs)

    def val_dataloader(self) -> DataLoader:
        if not isinstance(self.scenario_train, ClassIncremental):
            raise NotImplementedError
        if self.cfg.eval_on == "current":
            dataset = self.scenario_val[self.current_task]
        elif self.cfg.eval_on == "seen":
            dataset = self.scenario_val[: self.current_task + 1]
        else:
            dataset = self.scenario_val[:]

        return create_dataloader(dataset, self.cfg.dataloader, train=False)
