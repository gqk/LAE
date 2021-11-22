# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class AvgToken(nn.Module):
    """Reducing tokens to feature like AvgPool

    Examples:
        - cls_token: AvgToken(0, 1)
        - first 5 tokens: AvgToken(end=5)
        - tokens except cls_token: AvgToken(start=1)
    """

    def __init__(self, start: int = 0, end: int = -1):
        super().__init__()
        self.start, self.end = start, end

    def forward(self, tokens: torch.Tensor):
        return tokens[:, self.start : self.end].mean(dim=1)
