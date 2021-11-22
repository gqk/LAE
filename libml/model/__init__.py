# -*- coding: utf-8 -*-

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List

import torch.nn as nn

from .avg_token import AvgToken
from .backbone import vit, swin, convnext
from .head.dynamic_simple import DynamicSimpleHead
from .utils import freeze, unfreeze


_BACKBONE = {
    "ViT-B_16": vit.vit_b16_in21k,
    "Swin-B": swin.swin_base_patch4_window7_224_in22k,
    "ConvNeXt-B": convnext.convnext_base_in22k,
}


@dataclass
class ModelConf:
    backbone: str = "ViT-B_16"
    backbone_kwargs: Dict[str, Any] = field(default_factory=lambda: {})
    head_kwargs: Dict[str, Any] = field(default_factory=lambda: {})


class Model(nn.Module):
    backbone: nn.Module
    head: nn.Module

    def __init__(self, cfg: ModelConf):
        super().__init__()

        self.cfg = cfg
        self.setup()

    def setup(self):
        self.backbone = self.create_backbone()
        self.head = self.create_head()

    def forward(self, *args, **kwargs):
        return self.head(self.backbone(*args, **kwargs))

    def create_backbone(self):
        if self.cfg.backbone in _BACKBONE:
            return _BACKBONE[self.cfg.backbone](**self.cfg.backbone_kwargs)

        if self.cfg.backbone.startswith("swin_"):
            module, fn_name = swin, self.cfg.backbone
        elif self.cfg.backbone.startswith("convnext_"):
            module, fn_name = vit, self.cfg.backbone
        elif self.cfg.backbone.startswith("vit_"):
            module, fn_name = vit, self.cfg.backbone
        else:
            raise KeyError(f"Unknown backbone {self.cfg.backbone}")

        if not hasattr(module, fn_name):
            raise KeyError(f"Unknown backbone {self.cfg.backbone}")

        return getattr(module, fn_name)(**self.cfg.backbone_kwargs)

    def create_head(self):
        assert self.backbone is not None, "Create backbone first"

        kwargs = dict(**self.cfg.head_kwargs)
        kwargs["num_features"] = self.backbone.num_features
        backbone = self.cfg.backbone.lower()
        if backbone.startswith("vit"):
            start = kwargs.pop("pool_start", 0)
            end = kwargs.pop("pool_end", 1)
            kwargs["pool"] = AvgToken(start=start, end=end)
        elif backbone.startswith("swin"):
            start = kwargs.pop("pool_start", 0)
            end = kwargs.pop("pool_end", -1)
            kwargs["pool"] = AvgToken(start=start, end=end)

        return DynamicSimpleHead(**kwargs)

    def freeze(self, *submodules: List[str]):
        return freeze(self, *submodules)

    def unfreeze(self, *submodules: List[str]):
        return unfreeze(self, *submodules)

    def clone(self, freeze: bool = False):
        model = deepcopy(self)
        if freeze:
            model.freeze()
        return model
