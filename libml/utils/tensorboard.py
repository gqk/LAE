# -*- coding: utf-8 -*-

"""MetricWriter for Pytorch summary files.
Use this writer for the Pytorch-based code.

based on:
https://github.com/google/CommonLoopUtils/blob/main/clu/metric_writers/torch_tensorboard_writer.py

"""

from typing import Any, Mapping, Optional, Union

import numpy as np
import torch
from torch.utils import tensorboard

Array = Union[torch.Tensor, np.ndarray]
Scalar = Union[int, float, np.number, np.ndarray]


def to_numpy(a: Array):
    if isinstance(a, torch.Tensor):
        return a.cpu().numpy()
    return a


class TensorboardWriter:
    """MetricWriter that writes Pytorch summary files."""

    def __init__(self, logdir: str):
        super().__init__()
        self._writer = tensorboard.SummaryWriter(log_dir=logdir)

    def write_scalars(self, step: int, scalars: Mapping[str, Scalar]):
        for key, value in scalars.items():
            self._writer.add_scalar(key, value, global_step=step)

    def write_images(self, step: int, images: Mapping[str, Array]):
        for key, value in images.items():
            self._writer.add_image(
                key, to_numpy(value), global_step=step, dataformats="HWC"
            )

    def write_texts(self, step: int, texts: Mapping[str, str]):
        for key, value in texts.items():
            self._writer.text(key, value, global_step=step)

    def write_histograms(
        self,
        step: int,
        arrays: Mapping[str, Array],
        num_buckets: Optional[Mapping[str, int]] = None,
    ):
        for tag, values in arrays.items():
            bins = None if num_buckets is None else num_buckets.get(tag)
            self._writer.add_histogram(
                tag, values, global_step=step, bins="auto", max_bins=bins
            )

    def write_hparams(self, hparams: Mapping[str, Any]):
        self._writer.add_hparams(hparams, {})

    def flush(self):
        self._writer.flush()

    def close(self):
        self._writer.close()
