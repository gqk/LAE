import os
from typing import Tuple, Union

import numpy as np
from continuum.datasets import ImageFolderDataset, _ContinuumDataset
from continuum.download import download, untar
from continuum.tasks import TaskType
from torchvision import transforms


class ImageNetR(_ContinuumDataset):
    """ImageNet-R dataset.
    - 200 classes
    """

    url = "https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar"
    train_subset_url = "https://gist.githubusercontent.com/gqk/e127fe18bf179bdcbdf5e29a8c1ae523/raw/train_list.txt"
    test_subset_url = "https://gist.githubusercontent.com/gqk/e127fe18bf179bdcbdf5e29a8c1ae523/raw/val_list.txt"

    num_classes = 200

    @property
    def transformations(self):
        """Default transformations if nothing is provided to the scenario."""
        return [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]

    def _download(self):
        path = os.path.join(self.data_path, "imagenet-r")
        if not os.path.exists(path):
            if not os.path.exists(f"{path}.tar"):
                download(self.url, self.data_path)
            untar(f"{path}.tar")

        filename = "val_list.txt"
        self.subset_url = self.test_subset_url
        if self.train:
            filename = "train_list.txt"
            self.subset_url = self.train_subset_url
        self.data_subset = os.path.join(self.data_path, filename)
        if not os.path.exists(self.data_subset):
            print("Downloading subset indexes...", end=" ")
            download(self.subset_url, self.data_path)
            print("Done!")

    @property
    def data_type(self) -> TaskType:
        return TaskType.IMAGE_PATH

    def get_data(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, Union[np.ndarray, None]]:
        data = self._parse_subset(self.data_subset, train=self.train)  # type: ignore
        return (*data, None)

    def _parse_subset(
        self,
        subset: Union[Tuple[np.array, np.array], str, None],
        train: bool = True,
    ) -> Tuple[np.array, np.array]:
        if not isinstance(subset, str):
            return subset

        x, y = [], []
        with open(subset, "r") as f:
            for line in f:
                split_line = line.split(" ")
                path = split_line[0].strip()
                x.append(os.path.join(self.data_path, path))
                y.append(int(split_line[1].strip()))
        x, y = np.array(x), np.array(y)
        return x, y
