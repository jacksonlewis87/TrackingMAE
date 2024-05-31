import dataclasses
import json
import os
import pytorch_lightning as pl
from datetime import datetime
from pytorch_lightning.loggers import TensorBoardLogger
from typing import List


class ModelDriver:
    def __init__(
        self,
        full_config,
        model_config,
        model: pl.LightningModule,
        data_module: pl.LightningDataModule,
    ):
        self.full_config = full_config
        self.model_config = model_config
        self.model = model
        self.data_module = data_module

    def save_configs(self) -> None:
        time = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
        os.makedirs(self.model_config.experiment_path, exist_ok=True)
        with open(os.path.join(self.model_config.experiment_path, f"full_config_{time}.json"), "w") as f:
            f.write(json.dumps(dataclasses.asdict(self.full_config)))

    def run_training(self):
        self.save_configs()
        trainer = self.setup_trainer()
        trainer.fit(
            model=self.model,
            train_dataloaders=self.data_module.train_dataloader(),
            val_dataloaders=self.data_module.val_dataloader(),
            ckpt_path=self.model_config.checkpoint_path,
        )

    @staticmethod
    def get_callbacks() -> List[pl.Callback] | None:
        return [pl.callbacks.LearningRateMonitor(logging_interval="step")]

    def setup_trainer(self) -> pl.Trainer:
        logger = TensorBoardLogger(self.model_config.experiment_path, name="lightning_logs")
        os.makedirs(self.model_config.experiment_path, exist_ok=True)
        return pl.Trainer(
            default_root_dir=self.model_config.experiment_path,
            max_epochs=self.model_config.epochs,
            accelerator="cpu",
            logger=logger,
            callbacks=self.get_callbacks(),
        )
