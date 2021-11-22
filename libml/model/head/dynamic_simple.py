# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from .simple import SimpleHead


class DynamicSimpleHead(SimpleHead):
    def __init__(
        self,
        num_classes: int = 0,
        num_features: int = 2048,
        bias: bool = True,
        pool: nn.Module = nn.AdaptiveAvgPool2d(1),
        neck: nn.Module = nn.Identity(),
    ):
        super().__init__(num_classes, num_features, bias, pool, neck)

    @property
    def embeddings(self):
        weights = [classifier.weight for classifier in self.classifiers]
        return torch.cat(weights)

    def setup(
        self,
        pool: nn.Module = nn.AdaptiveAvgPool2d(1),
        neck: nn.Module = nn.Identity(),
    ):
        self.pool, self.neck = pool, neck

        self.classifiers = nn.ModuleList()

        if self.num_classes > 0:
            self.append(self.num_classes)

    def append(self, num_classes: int):
        args = [self.num_features, num_classes]
        classifier = self._create_classifier(*args, bias=self.bias)

        self.classifiers.append(classifier)
        if len(self.classifiers) > 1 or self.num_classes == 0:
            self.num_classes += num_classes

    def classify(self, input: torch.Tensor):
        output = [classifier(input) for classifier in self.classifiers]
        output = torch.cat(output, dim=1)
        return output

    def forward(self, input):
        output = self.pool(input).flatten(1)
        output = self.neck(output)

        if self.feature_mode:
            return output

        output = self.classify(output)

        return output
