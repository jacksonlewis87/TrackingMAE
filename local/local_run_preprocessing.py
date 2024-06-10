from constants import ROOT_DIR
from data.preprocessing_config import PreprocessingConfig
from data.preprocessing import preprocess_tracking_data


def do_work():
    config = PreprocessingConfig(
        raw_tracking_path=f"{ROOT_DIR}/data/raw_tracking",
        tensor_path=f"{ROOT_DIR}/data/tensors/split_events",
        event_duration=10,
        # event_duration=12,
        target_frame_rate=25,
        training_frame_rate=5,
        split_long_events=True,
    )

    preprocess_tracking_data(config=config)


if __name__ == "__main__":
    do_work()
