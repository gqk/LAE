# -*- coding: utf-8 -*-

import sys
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Union

from filelock import FileLock
from omegaconf import DictConfig, ListConfig
from omegaconf import OmegaConf as oc


def create_version_dir(base_dir: Path, version: Optional[int] = None):
    # TODO create_version_dir after torch ddp initializing
    with FileLock(base_dir / "._version_lock"):
        if version is None:
            stems = map(lambda d: str(d.stem), base_dir.iterdir())
            stems = filter(lambda s: s.startswith("version_"), stems)
            stems = [-1] + [int(s[8:]) for s in stems if s[8:].isdigit()]
            version = max(stems) + 1
        base_dir = base_dir / f"version_{version}"
        base_dir.mkdir(exist_ok=False)
    return base_dir


def to_abs_path(path: Path, root: Path):
    if path.is_absolute():
        return path
    return root / path


def resolve_extends(extends: Union[DictConfig, ListConfig, list], root: Path):
    assert isinstance(extends, (DictConfig, ListConfig, list))

    cfg = oc.create({})
    if isinstance(extends, DictConfig):
        # special case: root extends mixed in sub extends
        cfg.merge_with(resolve_extends(extends.pop("_", []), root))
        for key, ext in extends.items():
            file = to_abs_path(Path(ext), root)
            value = oc.load(file.with_suffix(".yaml"))
            nest_extends = value.pop("extends", None)
            if nest_extends is not None:
                nest_cfg_ext = resolve_extends(nest_extends, file.parent)
                value = oc.merge(nest_cfg_ext, value)
            cfg.merge_with({key: value})
        return cfg

    for ext in extends:
        file = to_abs_path(Path(ext), root)
        value = oc.load(file.with_suffix(".yaml"))
        nest_extends = value.pop("extends", None)
        if nest_extends is not None:
            nest_cfg_ext = resolve_extends(nest_extends, file.parent)
            value = oc.merge(nest_cfg_ext, value)
        cfg.merge_with(value)
    return cfg


def build_config(
    root: Path,
    name: str = "",
    argv: List[str] = None,
    structure: Any = None,
    print_exit: bool = False,
    log_dir: Optional[str] = "logs",
    version: Union[int, str] = "auto",  # set to none if don't need version dir
):
    """Build config from command line and config file


    Examples:
        - python main.py --print_config
        - python main.py --config=cifar100_base
        - python main.py --config=cifar100_extend_base
        - python main.py --config=cifar100_extend_base_and_data

        cifar100_base:
            A: 123
            B: 345
            C:
              D: 789
        data/cifar100_10_10:
            E: [1, 2, 3]

        cifar100_extend_base:
            extends:
              - cifar100_base

        cifar100_extend_base_and_data:
            extends:
              - cifar100_base
              - data/cifar100_10_10

        cifar100_extend_base_and_data:
            extends:
              _:
                - cifar100_base
                - data/cifar100_10_10
    """

    args = oc.from_cli(argv)
    cfg = oc.create({}) if structure is None else oc.structured(structure)

    if name:
        name = Path(name).with_suffix("")
        cfg_tmp = resolve_extends([name.with_suffix(".yaml")], root)
        cfg.merge_with(cfg_tmp)
    name = name if name else datetime.today().strftime("%Y-%m-%d")

    if "extends" in args:
        cfg_tmp = resolve_extends(args.pop("extends"), root)
        cfg.merge_with(cfg_tmp)

    overrides = oc.masked_copy(args, [k for k in cfg if k in args])
    args = oc.masked_copy(args, [k for k in args if k not in cfg])
    cfg = oc.merge(cfg, overrides)
    if print_exit:
        print(oc.to_yaml(cfg))
        sys.exit(0)

    if log_dir is not None:
        log_dir = log_dir / name
        log_dir.mkdir(parents=True, exist_ok=True)
        if version != "none":
            v_no = version if isinstance(version, int) else None
            log_dir = create_version_dir(log_dir, v_no)
        args.log_dir = log_dir

    return cfg, args
