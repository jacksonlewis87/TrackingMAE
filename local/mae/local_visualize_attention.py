import torch

from constants import ROOT_DIR
from data.mae.data_config import DataConfig
from data.mae.data_module import setup_data_module
from model.mae.masking import MaskingStrategy
from model.mae.model_config import FullConfig, ModelConfig
from model.mae.visualize_attention import AttentionVisualizer


def mask_to_masked_player_indexes(config: FullConfig, mask: torch.tensor) -> list[int]:
    indexes = mask.nonzero().squeeze(1).tolist()
    masked_indexes = []
    for i in range(11):
        if i * config.model_config.num_sequence_patches in indexes:
            masked_indexes += [i]
    return masked_indexes


def run_visualize_attention(config: FullConfig):

    data_module = setup_data_module(config=config.data_config)
    visualizer = AttentionVisualizer(config=config)

    with torch.no_grad():
        for batch in data_module.val_dataloader():
            for sample in batch:
                visualizer.visualize_attention(x=sample)


def do_work():
    experiment_name = "mae_tiny"

    checkpoint_version = 0
    checkpoint_epoch = 50
    checkpoint_step = 3111

    config = FullConfig(
        data_config=DataConfig(
            tensor_path=f"{ROOT_DIR}/data/tensors/split_events",
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
            encoder_depth=1,
            num_encoder_heads=1,
            decoder_embedding_dimension=32,
            decoder_depth=1,
            num_decoder_heads=1,
            masking_strategy=MaskingStrategy.INDEX.value,
            # masking_indexes=[0, 1, 6],
            # masking_indexes=[0],
            random_indexes=1,
        ),
    )

    experiment_name = "mae_v1"

    checkpoint_version = 0
    checkpoint_epoch = 65
    checkpoint_step = 31812

    config = FullConfig(
        data_config=DataConfig(
            tensor_path=f"{ROOT_DIR}/data/tensors/split_events",
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
            embedding_dimension=256,
            encoder_depth=12,
            num_encoder_heads=16,
            decoder_embedding_dimension=128,
            decoder_depth=4,
            num_decoder_heads=8,
            masking_strategy=MaskingStrategy.INDEX.value,
            # masking_indexes=[0, 1, 6],
            # masking_indexes=[0],
            random_indexes=1,
        ),
    )

    run_visualize_attention(config=config)


if __name__ == "__main__":
    do_work()
