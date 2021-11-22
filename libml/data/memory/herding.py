# -*- coding: utf-8 -*-

from typing import List

import torch

from .base import Memory


class Herding(Memory):
    def select(self, features: torch.Tensor, num_samples: int) -> List[int]:
        num_samples = min(features.shape[0], num_samples)

        assert num_samples > 0

        mu_t = mu = features.mean(dim=0)
        selection, num_selected = torch.zeros_like(features[:, 0]), 0

        num_iters = 0
        while num_selected < num_samples and num_iters < 1000:
            num_iters += 1
            index = features.matmul(mu_t).argmax()
            if selection[index] == 0:
                num_selected += 1
                selection[index] = num_selected
            mu_t = mu_t + mu - features[index]

        selection[selection == 0] = 10000
        selected_indices = selection.argsort()[:num_samples].tolist()

        return selected_indices
