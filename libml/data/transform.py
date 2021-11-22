# -*- coding: utf-8 -*-


import torchvision.transforms as T

IMGNET_MEAN = [0.485, 0.456, 0.406]
IMGNET_STD = [0.229, 0.224, 0.225]


def build_imagenet_transform(
    train: bool = True,
    interpolation: T.InterpolationMode = T.InterpolationMode.BILINEAR,
    norm: bool = False,
):
    transforms = [T.ToTensor()]
    if norm:
        transforms.append(T.Normalize(mean=IMGNET_MEAN, std=IMGNET_STD))

    if train:
        train_transforms = [
            T.RandomResizedCrop(224, interpolation=interpolation),
            T.RandomHorizontalFlip(),
        ]
        return train_transforms + transforms

    test_transforms = [
        T.Resize(256, interpolation=interpolation),
        T.CenterCrop(224),
    ]
    return test_transforms + transforms
