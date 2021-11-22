# -*- coding: utf-8 -*-

from dataclasses import dataclass

from omegaconf import OmegaConf as oc
from torch.utils.data import DataLoader


@dataclass
class DataLoaderConf:
    batch_size: int = 24
    batch_size_val: int = 0
    num_workers: int = 4
    pin_memory: bool = True
    drop_last: bool = True


def create_dataloader(
    dataset,
    config: DataLoaderConf,
    train: bool = True,
):
    kwargs = oc.to_container(config)
    batch_size_val = kwargs.pop("batch_size_val")
    if not train and batch_size_val > 0:
        kwargs["batch_size"] = batch_size_val
    kwargs["drop_last"] = kwargs["drop_last"] and train
    return DataLoader(dataset, **kwargs, shuffle=train)
