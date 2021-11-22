import os
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from continuum.datasets import ImageFolderDataset, _ContinuumDataset
from continuum.datasets import SVHN, MNIST, CIFAR10, FashionMNIST
from continuum.tasks import TaskType
from torchvision import transforms
from PIL import Image

from .not_minst import NotMNIST

dataset_class_names = {
    CIFAR10: "cifar10",
    MNIST: "mnist",
    SVHN: "svhn",
    NotMNIST: "not-mnist",
    FashionMNIST: "fashion-mnist",
}


class FiveDatasets(_ContinuumDataset):
    """The dataset composed of 5 small datasetsi.
    - 10 classes per dataset
    - 500 images per class
    - size 64x64
    """

    num_classes = 50

    @property
    def transformations(self):
        """Default transformations if nothing is provided to the scenario."""
        return [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]

    def get_data(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, Union[np.ndarray, None]]:
        kwargs = dict(train=self.train, download=True)
        img_size = 64

        def resize_to_rgb(img):
            img = Image.fromarray(img).resize((img_size, img_size))
            img = np.array(img.convert("RGB"))
            return img

        xs, ys, offset = [], [], 0
        for klass in [CIFAR10, MNIST, SVHN, NotMNIST, FashionMNIST]:
            klass_name = dataset_class_names[klass]
            data_path = os.path.join(self.data_path, klass_name)
            x, y, _ = klass(data_path, **kwargs).get_data()
            x = np.array([resize_to_rgb(x[i]) for i in range(x.shape[0])])
            xs.append(x)
            ys.append(y + offset)
            offset += np.unique(y).size
            print(f"{klass_name} {self.train}: {y.size}")
        data = (np.concatenate(xs), np.concatenate(ys), None)
        return data
