# -*- coding: utf-8 -*-

import math
from typing import Optional, Type, Union

import torch
import torch.nn as nn

from ..misc.scaler import Scaler


class Adapter(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        down_sample: Union[float, int] = 5,
        mode: str = "parallel",  # enum before, after, parallel
        scale: Optional[float] = None,
        act_layer: Type[nn.Module] = nn.GELU,
    ):
        super().__init__()

        assert mode in ["before", "after", "parallel"], f"Unknown mode {mode}"

        hidden_dim = down_sample
        if isinstance(down_sample, float):
            hidden_dim = int(embed_dim * down_sample)

        self.layer = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            act_layer(),
            nn.Linear(hidden_dim, embed_dim),
            Scaler(scale),
        )
        self.mode = mode

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.layer[0].weight, a=math.sqrt(5))
        nn.init.zeros_(self.layer[0].bias)
        nn.init.zeros_(self.layer[2].weight)
        nn.init.zeros_(self.layer[2].bias)

    def forward(self, module, input, **kwargs):
        if self.mode == "before":
            return module(self.layer(input) + input, **kwargs)
        if self.mode == "after":
            return self.layer(module(input, **kwargs)) + input
        return module(input, **kwargs) + self.layer(input)


class Conv2dAdapter(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        down_sample: Union[float, int] = 5,
        mode: str = "before",  # enum before, after, parallel
        scale: Optional[float] = None,
        act_layer: Type[nn.Module] = nn.GELU,
    ):
        super().__init__()
        assert mode in ["before", "after", "parallel"], f"Unknown mode {mode}"

        hidden_dim = down_sample
        if isinstance(down_sample, float):
            hidden_dim = int(in_channels * down_sample)

        if out_channels is None:
            out_channels = in_channels

        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
            act_layer(),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1),
            Scaler(scale),
        )
        self.mode = mode

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.layer[0].weight, a=math.sqrt(5))
        nn.init.zeros_(self.layer[0].bias)
        nn.init.zeros_(self.layer[2].weight)
        nn.init.zeros_(self.layer[2].bias)

    def forward(self, module, input, **kwargs):
        if self.mode == "before":
            return module(self.layer(input) + input, **kwargs)
        if self.mode == "after":
            return self.layer(module(input, **kwargs)) + input
        return module(input, **kwargs) + self.layer(input)
