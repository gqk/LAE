import os
from typing import Tuple, Union

import numpy as np
from continuum.datasets import ImageFolderDataset, _ContinuumDataset
from continuum.download import download, unzip
from continuum.tasks import TaskType
from torchvision import transforms
from PIL import Image


class NotMNIST(_ContinuumDataset):
    """Smaller version of ImageNet.
    - 10 classes
    - 52000 images per class
    - size 64x64
    """

    url = "https://github.com/facebookresearch/Adversarial-Continual-Learning/raw/main/data/notMNIST.zip"
    num_classes = 10

    @property
    def transformations(self):
        """Default transformations if nothing is provided to the scenario."""
        return [
            transforms.ToTensor(),
        ]

    def _download(self):
        path = os.path.join(self.data_path, "notMNIST")
        if not os.path.exists(path):
            if not os.path.exists(f"{path}.zip"):
                download(self.url, self.data_path)
            unzip(f"{path}.zip")
            print("Done!")
        self.data_subset = "Train" if self.train else "Test"

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

        fpath = os.path.join(self.data_path, "notMNIST", subset)
        x, y = [], []
        for folder in os.listdir(fpath):
            folder_path = os.path.join(fpath, folder)
            for ims in os.listdir(folder_path):
                try:
                    img_path = os.path.join(folder_path, ims)
                    x.append(np.array(Image.open(img_path).convert("RGB")))
                    # Folders are A-J so labels will be 0-9
                    y.append(ord(folder) - 65)
                except:
                    print("File {}/{} is broken".format(folder, ims))
        x, y = np.array(x), np.array(y)
        return x, y
