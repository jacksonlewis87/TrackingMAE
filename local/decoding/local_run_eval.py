import os
import torch

from constants import ROOT_DIR
from data.decoding.data_config import DataConfig
from data.decoding.data_module import setup_data_module
from data.decoding.transforms import Task
from loss.losses import Loss
from model.decoding.model import TrackingMaskedAutoEncoder
from model.decoding.model_config import FullConfig, ModelConfig
from model.utils import patchify, unpatchify
from visualization.tracking import TrackingVisualization, DualTrackingVisualization


def mask_to_masked_player_indexes(config: FullConfig, mask: torch.tensor) -> list[int]:
    indexes = mask.nonzero().squeeze(1).tolist()
    masked_indexes = []
    for i in range(11):
        if i * config.model_config.num_sequence_patches in indexes:
            masked_indexes += [i]
    return masked_indexes


def run_eval(config: FullConfig):
    os.makedirs(config.model_config.experiment_path, exist_ok=True)

    data_module = setup_data_module(config=config.data_config)

    count_y = 0
    count = 0
    for batch in data_module.val_dataloader():
        x, y = batch
        # print(y)
        if y[0][0] == 1 or y[0][1] == 1:
            count_y += 1
        count += 1
    print(count_y)
    print(count)
    # if y[0][0] == 1:
    #     tv = TrackingVisualization.from_tensor(
    #         tracking_tensor=x[0]
    #     )
    #     tv.execute()
    print(asdfasd)

    model = TrackingMaskedAutoEncoder.load_from_checkpoint(config.model_config.checkpoint_path, config=config)
    model.eval()

    with torch.no_grad():
        for batch in data_module.val_dataloader():
            loss, pred, mask = model.forward(batch)
            x = batch.permute(0, 3, 1, 2)
            x_patched = patchify(
                x=x,
                channels=model.channels,
                num_players=model.config.model_config.num_players,
                num_sequence_patches=model.config.model_config.num_sequence_patches,
                patch_length=model.patch_length,
                total_patches=model.total_patches,
            )
            x_unpatched = unpatchify(
                x=x_patched,
                channels=model.channels,
                num_players=model.config.model_config.num_players,
                num_sequence_patches=model.config.model_config.num_sequence_patches,
                patch_length=model.patch_length,
                num_frames=model.config.data_config.num_frames,
            )
            x = x.permute(0, 2, 3, 1)
            pred = unpatchify(
                x=pred,
                channels=model.channels,
                num_players=model.config.model_config.num_players,
                num_sequence_patches=model.config.model_config.num_sequence_patches,
                patch_length=model.patch_length,
                num_frames=model.config.data_config.num_frames,
            ).permute(0, 2, 3, 1)
            for i in range(batch.size(dim=0)):
                masked_indexes = mask_to_masked_player_indexes(config=config, mask=mask[i])
                tv = DualTrackingVisualization.from_tensor(
                    tracking_tensor_0=x[i], tracking_tensor_1=pred[i], masked_indexes=masked_indexes
                )
                tv.execute()


def do_work():
    experiment_name = "decoding_v1"

    checkpoint_version = 1
    checkpoint_epoch = 25
    checkpoint_step = 468

    config = FullConfig(
        data_config=DataConfig(
            tensor_path=f"{ROOT_DIR}/data/tensors/full_events",
            data_split_path=f"{ROOT_DIR}/data/training/{experiment_name}/data_split.json",
            batch_size=1,
            train_size=0.8,
            shuffle_players=True,
            num_frames=50,
            include_z=False,
            task=Task.MADE_BASKET_CLASSIFICATION.value,
            num_event_classification_tasks=2,
            y_frames=15,
            min_frames=35,
            patch_length=10,
        ),
        model_config=ModelConfig(
            experiment_path=f"{ROOT_DIR}/data/training/{experiment_name}",
            learning_rate=0.0001,
            epochs=120,
            checkpoint_path=None,
            # checkpoint_path=f"{ROOT_DIR}/data/training/{experiment_name}/lightning_logs/version_{checkpoint_version}/checkpoints/epoch={checkpoint_epoch}-step={checkpoint_step}.ckpt",
            encoder_checkpoint_path=f"{ROOT_DIR}/data/training/mae_v0/lightning_logs/version_23/checkpoints/epoch=209-step=112980.ckpt",
            encoder_checkpoint_config_path=f"{ROOT_DIR}/data/training/mae_v0/lightning_logs/version_23/hparams.yaml",
            freeze_encoder=True,
            loss=Loss.BCE.value,
            decoder_embedding_dimension=32,
            decoder_depth=4,
            num_decoder_heads=8,
        ),
    )

    run_eval(config=config)


if __name__ == "__main__":
    do_work()
