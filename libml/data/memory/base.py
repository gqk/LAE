# -*- coding: utf-8 -*-

from collections import defaultdict
from copy import deepcopy
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from absl import logging
from continuum.scenarios.base import _BaseScenario as Scenario
from torch.utils.data.dataloader import DataLoader


class Memory:
    def __init__(
        self,
        max_size: int,
        num_samples_per_class: int,
        norm_feature: bool,
    ):
        super().__init__()
        self.max_size = max_size
        self.num_samples_per_class = num_samples_per_class
        self.norm_feature = norm_feature
        self.container = []

    def __getitem__(self, idx):
        return self.container[idx]

    def select(
        self, features: torch.Tensor, num_samples: int
    ) -> List[List[int]]:
        raise NotImplementedError

    def delete(
        self,
        indices: List[int],
        num_samples: int,
    ) -> Tuple[List[int], List[int]]:
        return indices[:num_samples], indices[num_samples:]

    def update(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        dry_run: bool = False,
    ) -> List[List[int]]:
        cls_features, cls_indices = defaultdict(list), defaultdict(list)
        offset, device = 0, next(model.parameters()).device
        for _, batch in enumerate(dataloader):
            feature, target = model(batch[0].to(device)), batch[1]
            indices = torch.arange(target.shape[0]) + offset
            for t in target.unique().tolist():
                selected = target == t
                cls_features[t].append(feature[selected])
                cls_indices[t].append(indices[selected])
        cls_features = {k: torch.cat(v) for k, v in cls_features.items()}
        cls_indices = {k: torch.cat(v) for k, v in cls_indices.items()}

        n_old, n_new = min(cls_features.keys()), len(cls_features)
        container = self.container[:n_old]
        num_samples_per_class = self.num_samples_per_class
        if num_samples_per_class < 1:  # dynamic adjust
            num_samples_per_class = self.max_size // (n_new + n_old)
            container = [
                self.delete(item, num_samples_per_class)[0]
                for item in container
            ]

        container = container + [[] for _ in range(n_new)]
        for t in sorted(cls_features.keys()):
            features = cls_features[t]
            if self.norm_feature:
                features = F.normalize(features, dim=1)
            selected = self.select(features, num_samples_per_class)
            container[t] = cls_indices[t][selected].tolist()
            logging.info(f"Selected {len(selected)} samples for class {t}")

        if not dry_run:
            self.container = container

        return container

    def to_dataset(self, scenario: Scenario, current_task: int):
        dataset = deepcopy(scenario[0])
        dataset._x = dataset._x[:0]
        dataset._y = dataset._y[:0]
        dataset._t = dataset._t[:0]
        container = self.container[:]
        for t, n_classes in enumerate(scenario.increments[:current_task]):
            indices = sum(container[:n_classes], [])
            container = container[n_classes:]
            dataset.add_samples(*scenario[t].get_raw_samples(indices))
        return dataset
