# -*- coding: utf-8 -*-

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..misc.scaler import Scaler


class Prompt(nn.Module):
    def __init__(
        self,
        length: int = 5,
        dim: int = 512,
        position: int = 1,
        reducible: bool = False,
        scale: Optional[float] = 1.0,
    ):
        super().__init__()

        self.length = length
        self.dim = dim
        self.position = position
        self.reducible = reducible

        tokens = nn.Parameter(torch.zeros(length, dim))
        self.register_parameter("tokens", tokens)
        nn.init.uniform_(self.tokens.data, 0, 0.01)

        self.scaler = Scaler(scale)

    def forward(self, x: torch.Tensor):
        assert x.shape[-1] == self.dim

        tokens = self.scaler(self.tokens).expand(x.shape[0], -1, -1)
        if self.position > 0:
            x1, x2 = x[:, : self.position], x[:, self.position :]
            return torch.cat([x1, tokens, x2], dim=1)
        return torch.cat([tokens, x], dim=1)

    def reduce(self, x: torch.Tensor):
        if not self.reducible:
            return x

        if self.position > 0:
            x1, x2 = x[:, : self.position], x[:, self.position + self.length :]
            return torch.cat([x1, x2], dim=1)
        return x[:, self.length :]

    def extra_repr(self):
        tpl = "length={}, dim={}, position={}, reducible={}"
        return tpl.format(self.length, self.dim, self.position, self.reducible)
