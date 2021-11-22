# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class SimpleHead(nn.Module):
    feature_mode: bool = False

    def __init__(
        self,
        num_classes: int,
        num_features: int = 2048,
        bias: bool = True,
        pool: nn.Module = nn.AdaptiveAvgPool2d(1),
        neck: nn.Module = nn.Identity(),
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_features = num_features
        self.bias = bias

        self.setup(pool, neck)

    @property
    def embeddings(self):
        return self.classifier.weight

    def setup(
        self,
        pool: nn.Module = nn.AdaptiveAvgPool2d(1),
        neck: nn.Module = nn.Identity(),
    ):
        self.pool, self.neck = pool, neck

        args = [self.num_features, self.num_classes]
        self.classifier = self._create_classifier(*args, bias=self.bias)

    def classify(self, input: torch.Tensor):
        return self.classifier(input)

    def forward(self, input):
        output = self.pool(input).flatten(1)
        output = self.neck(output)

        if self.feature_mode:
            return output

        output = self.classify(output)
        return output

    @staticmethod
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find("Linear") != -1:
            nn.init.normal_(m.weight, std=0.001)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def _create_classifier(
        self,
        num_features: int,
        num_classes: int,
        bias=True,
    ):
        assert num_features > 0 and num_classes > 0

        classifier = nn.Linear(num_features, num_classes, bias=bias)
        self.weights_init(classifier)

        return classifier
