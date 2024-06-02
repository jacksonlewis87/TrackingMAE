import os
import torch

from constants import ROOT_DIR
from data.mae.data_config import DataConfig
from data.mae.data_module import setup_data_module
from model.mae.masking import MaskingStrategy
from model.mae.model import TrackingMaskedAutoEncoder
from model.mae.model_config import FullConfig, ModelConfig
from visualization.tracking import TrackingVisualization


def run_eval(config: FullConfig):
    os.makedirs(config.model_config.experiment_path, exist_ok=True)

    data_module = setup_data_module(config=config.data_config)
    model = TrackingMaskedAutoEncoder.load_from_checkpoint(config.model_config.checkpoint_path, config=config)
    model.eval()

    with torch.no_grad():
        for batch in data_module.val_dataloader():
            loss, pred, mask = model.forward(batch)
            # print(pred.size())
            x = batch.permute(0, 3, 1, 2)
            print(x.size())
            print(pred.size())
            x_patched = model.patchify(x)
            x_unpatched = model.unpatchify(x_patched)
            print(x_unpatched.size())
            print(torch.all(torch.eq(x, x_unpatched)))
            x = x.permute(0, 2, 3, 1)
            pred = model.unpatchify(pred).permute(0, 2, 3, 1)
            for i in range(batch.size(dim=0)):
                print("x")
                tv = TrackingVisualization.from_tensor(tracking_tensor=x[i])
                tv.execute()
                print("pred")
                tv = TrackingVisualization.from_tensor(tracking_tensor=pred[i])
                tv.execute()


def do_work():
    experiment_name = "mae_v0"

    checkpoint_version = 26
    checkpoint_epoch = 359
    checkpoint_step = 24480

    checkpoint_version = 23
    checkpoint_epoch = 209
    checkpoint_step = 112980

    config = FullConfig(
        data_config=DataConfig(
            tensor_path=f"{ROOT_DIR}/data/tensors",
            data_split_path=f"{ROOT_DIR}/data/training/{experiment_name}/data_split.json",
            batch_size=8,
            train_size=0.8,
            shuffle_players=True,
            num_frames=50,
            include_z=False,
        ),
        model_config=ModelConfig(
            experiment_path=f"{ROOT_DIR}/data/training/{experiment_name}",
            learning_rate=0.0001,
            epochs=100,
            checkpoint_path=f"{ROOT_DIR}/data/training/{experiment_name}/lightning_logs/version_{checkpoint_version}/checkpoints/epoch={checkpoint_epoch}-step={checkpoint_step}.ckpt",
            num_players=11,
            num_sequence_patches=5,
            embedding_dimension=64,
            encoder_depth=24,
            num_encoder_heads=4,
            decoder_embedding_dimension=512,
            decoder_depth=8,
            num_decoder_heads=16,
            # num_sequence_patches=2,
            # embedding_dimension=64,
            # encoder_depth=12,
            # num_encoder_heads=8,
            # decoder_embedding_dimension=32,
            # decoder_depth=4,
            # num_decoder_heads=8,
            masking_strategy=MaskingStrategy.INDEX.value,
            # masking_indexes=[0, 1, 6],
            # masking_indexes=[0],
            random_indexes=1,
        ),
    )

    run_eval(config=config)


if __name__ == "__main__":
    do_work()