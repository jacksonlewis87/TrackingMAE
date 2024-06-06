from constants import ROOT_DIR
from data.decoding.data_config import DataConfig
from data.decoding.transforms import Task
from loss.losses import Loss
from model.decoding.entrypoint import run_training
from model.decoding.model_config import FullConfig, ModelConfig


def do_work():
    experiment_name = "decoding_v0"

    checkpoint_version = 2
    checkpoint_epoch = 59
    checkpoint_step = 2040

    config = FullConfig(
        data_config=DataConfig(
            tensor_path=f"{ROOT_DIR}/data/tensors",
            data_split_path=f"{ROOT_DIR}/data/training/{experiment_name}/data_split.json",
            batch_size=128,
            train_size=0.8,
            shuffle_players=True,
            num_frames=50,
            include_z=False,
            task=Task.BALL_HEIGHT_CLASSIFICATION.value,
            min_z=10.0,
        ),
        model_config=ModelConfig(
            experiment_path=f"{ROOT_DIR}/data/training/{experiment_name}",
            learning_rate=0.0001,
            epochs=120,
            # checkpoint_path=None,
            checkpoint_path=f"{ROOT_DIR}/data/training/{experiment_name}/lightning_logs/version_{checkpoint_version}/checkpoints/epoch={checkpoint_epoch}-step={checkpoint_step}.ckpt",
            encoder_checkpoint_path=f"{ROOT_DIR}/data/training/mae_v0/lightning_logs/version_23/checkpoints/epoch=209-step=112980.ckpt",
            encoder_checkpoint_config_path=f"{ROOT_DIR}/data/training/mae_v0/lightning_logs/version_23/hparams.yaml",
            freeze_encoder=True,
            loss=Loss.BCE.value,
            decoder_embedding_dimension=32,
            decoder_depth=4,
            num_decoder_heads=8,
        ),
    )

    run_training(config=config)


if __name__ == "__main__":
    do_work()
