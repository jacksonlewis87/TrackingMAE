from constants import ROOT_DIR
from data.mae.data_config import DataConfig
from model.mae.entrypoint import run_training
from model.mae.masking import MaskingStrategy
from model.mae.model_config import FullConfig, ModelConfig


def do_work():
    experiment_name = "mae_v0"

    # checkpoint_version = 3
    # checkpoint_epoch = 59
    # checkpoint_step = 32280

    config = FullConfig(
        data_config=DataConfig(
            tensor_path=f"{ROOT_DIR}/data/tensors",
            data_split_path=f"{ROOT_DIR}/data/training/{experiment_name}/data_split.json",
            batch_size=8,
            train_size=0.8,
            shuffle_players=True,
            num_frames=50,
        ),
        model_config=ModelConfig(
            experiment_path=f"{ROOT_DIR}/data/training/{experiment_name}",
            learning_rate=0.0001,
            epochs=60,
            checkpoint_path=None,
            # checkpoint_path=f"{ROOT_DIR}/data/training/{experiment_name}/lightning_logs/version_{checkpoint_version}/checkpoints/epoch={checkpoint_epoch}-step={checkpoint_step}.ckpt",
            num_players=11,
            num_sequence_patches=5,
            embedding_dimension=64,
            encoder_depth=24,
            num_encoder_heads=4,
            decoder_embedding_dimension=512,
            decoder_depth=8,
            num_decoder_heads=16,
            masking_strategy=MaskingStrategy.INDEX.value,
            masking_indexes=[0],
        ),
    )

    run_training(config=config)


if __name__ == "__main__":
    do_work()
