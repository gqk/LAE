# -*- coding: utf-8 -*-

from dataclasses import dataclass
from pathlib import Path
from typing import Type

from absl import app, flags

from ..utils.config import build_config
from .datamodule import DataConf, DataModule
from .module import Module, ModuleConf
from .trainer import Trainer, TrainerConf

FLAGS = flags.FLAGS
flags.DEFINE_string("config", "", "config file name")
flags.DEFINE_string("config_dir", "config", "config directory")
flags.DEFINE_bool("print_config", False, "print config and exit")
flags.DEFINE_bool("evaluate", False, "evaluate mode")
flags.DEFINE_bool("debug", False, "debug mode")


@dataclass
class Conf(TrainerConf):
    data: DataConf = DataConf()
    module: ModuleConf = ModuleConf()


class App:
    def __init__(
        self,
        trainer_cls: Type[Trainer] = Trainer,
        datamodule_cls: Type[DataModule] = DataModule,
        module_cls: Type[Module] = Module,
        config_cls: Type[Conf] = Conf,
        root: Path = Path.cwd(),
        **kwargs,
    ):
        self.trainer_cls = trainer_cls
        self.datamodule_cls = datamodule_cls
        self.module_cls = module_cls
        self.config_cls = config_cls
        self.root = root
        self.kwargs = kwargs

    def main(self, argv):
        kwargs = dict(
            root=self.root / FLAGS.config_dir,
            name=FLAGS.config,
            argv=argv[1:],
            structure=self.config_cls(),
            print_exit=FLAGS.print_config,
            log_dir=self.root / "logs",
        )
        config, args = build_config(**kwargs)
        datamodule = self.datamodule_cls(config.data)
        module = self.module_cls(config.module)
        trainer = self.trainer_cls(config, args.log_dir, **self.kwargs)
        trainer.attach(module, datamodule)
        trainer.run(mode="eval" if FLAGS.evaluate else "train")

    def run(self):
        app.run(self.main)
