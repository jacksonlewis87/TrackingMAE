from constants import ROOT_DIR
from data.data_config import DataConfig
from data.tracking_preprocessing import preprocess_tracking_data


def do_work():
    config = DataConfig(
        raw_tracking_path=f"{ROOT_DIR}/data/raw_tracking",
        tensor_path=f"{ROOT_DIR}/data/tensors",
        event_duration=10,
        target_frame_rate=25,
        training_frame_rate=5,
    )

    preprocess_tracking_data(config=config)


if __name__ == "__main__":
    do_work()
