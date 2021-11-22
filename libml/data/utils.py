from copy import deepcopy

import numpy as np
from continuum.datasets import _ContinuumDataset


def split_train_val(
    dataset: _ContinuumDataset, val_split: float, seed: int = 42
):
    original_get_data = dataset.get_data
    random_state = np.random.RandomState(seed=seed)

    def get_data_train():
        x, y, t = original_get_data()

        x_train, y_train, t_train = [], [], []
        for c in np.unique(y):
            indexes = np.nonzero(y == c)[0]
            random_state.shuffle(indexes)
            sel = indexes[int(val_split * len(indexes)) :]
            x_train.append(x[sel])
            y_train.append(y[sel])
            if t:
                t_train.append(t[sel])
        x = np.concatenate(x_train)
        y = np.concatenate(y_train)
        if t:
            t = np.concatenate(t_train)
        return x, y, t

    def get_data_val():
        x, y, t = original_get_data()

        x_val, y_val, t_val = [], [], []
        for c in np.unique(y):
            indexes = np.nonzero(y == c)[0]
            random_state.shuffle(indexes)
            sel = indexes[: int(val_split * len(indexes))]
            x_val.append(x[sel])
            y_val.append(y[sel])
            if t:
                t_val.append(t[sel])
        x = np.concatenate(x_val)
        y = np.concatenate(y_val)
        if t:
            t = np.concatenate(t_val)
        return x, y, t

    dataset_train, dataset_val = deepcopy(dataset), deepcopy(dataset)
    dataset_train.get_data = get_data_train
    dataset_val.get_data = get_data_val

    return dataset_train, dataset_val
