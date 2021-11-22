# -*- coding: utf-8 -*-

from typing import List

import torch.nn as nn


def freeze(module: nn.Module, *submodules: List[str]):
    if submodules:
        module = nn.ModuleList(
            [m for n, m in module.named_modules() if n in submodules]
        )
    for param in module.parameters():
        param.requires_grad_(False)
        param.grad = None


def unfreeze(module: nn.Module, *submodules: List[str]):
    if submodules:
        module = nn.ModuleList(
            [m for n, m in module.named_modules() if n in submodules]
        )
    for param in module.parameters():
        param.requires_grad_(True)
