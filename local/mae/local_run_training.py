from constants import ROOT_DIR
from data.mae.data_config import DataConfig
from model.mae.entrypoint import run_training
from model.mae.masking import MaskingStrategy
from model.mae.model_config import FullConfig, ModelConfig


def do_work():
    experiment_name = "mae_v1"

    checkpoint_version = 25
    checkpoint_epoch = 179
    checkpoint_step = 12240

    config = FullConfig(
        data_config=DataConfig(
            tensor_path=f"{ROOT_DIR}/data/tensors/split_events",
            data_split_path=f"{ROOT_DIR}/data/training/{experiment_name}/data_split.json",
            batch_size=64,
            train_size=0.8,
            shuffle_players=True,
            num_frames=50,
            include_z=False,
        ),
        model_config=ModelConfig(
            experiment_path=f"{ROOT_DIR}/data/training/{experiment_name}",
            learning_rate=0.0001,
            epochs=360,
            checkpoint_path=None,
            # checkpoint_path=f"{ROOT_DIR}/data/training/{experiment_name}/lightning_logs/version_{checkpoint_version}/checkpoints/epoch={checkpoint_epoch}-step={checkpoint_step}.ckpt",
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

    run_training(config=config)


if __name__ == "__main__":
    do_work()
