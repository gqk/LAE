# <div align="center"><b> A Unified Continual Learning Framework with General <br /> Parameter-Efficient Tuning </b></div>

<div align="center">

[Qiankun Gao](https://github.com/gqk), [Chen Zhao](https://zhao-chen.com/), [Yifan Sun](https://yifansun-reid.github.io), Teng Xi, Gang Zhang, [Bernard Ghanem](https://www.bernardghanem.com/), [Jian Zhang](https://github.com/jianzhangcs)

[[`Paper`](https://github.com/gqk/LAE/files/12387816/03318.pdf)] [[`Supp`](https://github.com/gqk/LAE/files/12387826/03318-supp.pdf)] [[`arXiv`](https://arxiv.org/abs/2303.10070)] [[`BibTex`](#citation)] 

<br />

<img src='https://github.com/gqk/LAE/assets/73707470/4db6b2d1-7c2e-4211-a4e4-6bf8380d0132' width="80%" />

</div>

## News
- [2023/08/19] Camera ready is submitted. 
- [2023/07/14] Accepted to ICCV 2023 as poster presentation, code is released to the public!

## Installation

- Install all dependencies via `pip`
    ```shell
    pip install -r requirements.txt
    ```

    :warning: Remove `torch` and `torchvision` from `requirements.txt` first if another version of pytorch have already installed.

## Dataset

1. Create a dataset root diretory, _e.g._, `data`.
2. `CIFAR100` and `ImageNet-R` datasets will be automatically downloaded, while [`DomainNet`](https://ai.bu.edu/M3SDA) requires manual download.
3. Overview of dataset root diretory

    ```shell
    ├── cifar100
    │   └── cifar-100-python
    ├── domainnet
    │   ├── clipart
    │   ├── infograph
    │   ├── painting
    │   ├── quickdraw
    │   ├── real
    │   └── sketch
    └── imagenet-r
        ├── imagenet-r
        ├── train_list.txt
        └── val_list.txt
    ```

    :warning: The **train-validation split of ImageNet-R dataset** are consistent with the [L2P JAX code](https://github.com/google-research/l2p), replace the `train_list.txt` and `val_list.txt` with [train_list_coda-p.txt](https://github.com/gqk/LAE/files/12387932/train_list_coda-p.txt) and [val_list_coda-p.txt](https://github.com/gqk/LAE/files/12387934/val_list_coda-p.txt) if you want to use the train-validation splitation of [CODA-Prompt](https://github.com/GT-RIPL/CODA-Prompt).

## Experiment

- Generate config file (replace `<root>` with your dataset root path)

    ```shell
    python main.py data.root=<root> data.dataset=cifar100 --print_config > cifar100.yaml
    ```

- Run code with an experiment config file
    ```shell
    python main.py --config=cifar100.yaml
    ```

- Reproduce results in the paper

  We provide [configs](./config) and [Makefile](./Makefile) to quickly reproduce the ten-tasks experimental results reported in the paper, run the following command if the `make` has been installed:

    ```shell
    make vit_adapter
    make vit_lora
    make vit_prefix
    make swin_adapter
    make convnext_adapter
    ```

    Run `make` command with `BASE` arg (default is `base/cifar100_order1.yaml`) to reproduce other experiments, _e.g._:
    ```
    make BASE="base/imagenet-r_order1.yaml" vit_adapter
    ```

    Modifiy `data.num_increment_classes` (`5/10` for CIFAR100/ImageNet-R) in base config files to reproduce `20-task` experiments.

## Acknowledgement

- PyTorch implementation of [L2P](https://github.com/JH-LEE-KR/l2p-pytorch) and [DualPrompt](https://github.com/JH-LEE-KR/dualprompt-pytorch).
- JAX implementation of L2P and DualPrompt: https://github.com/google-research/l2p.
- [CODA-Prompt ](https://github.com/GT-RIPL/CODA-Prompt), state-of-the-art work from CVPR 2023.
- [ESN](https://github.com/iamwangyabin/ESN), state-of-the-art work from AAAI 2023.
- [Continumm](https://github.com/Continvvm/continuum), awesome data loading library for Continual Learning.

## Citation

```
@article{gao2023lae,
  title = {A Unified Continual Learning Framework with General Parameter-Efficient Tuning},
  author = {Qiankun Gao, Chen Zhao, Yifan Sun, Teng Xi, Gang Zhang, Bernard Ghanem, Jian Zhang},
  journal = {International Conference on Computer Vision (ICCV)},
  year = {2023}
}
```
