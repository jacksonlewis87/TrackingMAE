import torch
from enum import Enum

from data.decoding.data_config import DataConfig


class Task(Enum):
    BALL_HEIGHT_CLASSIFICATION = "ball_height_classification"


def get_tracking_to_label_function(task_str: str):
    if task_str == Task.BALL_HEIGHT_CLASSIFICATION.value:
        return tracking_to_ball_height_classification
    else:
        return None


def tracking_to_ball_height_classification(x: torch.tensor, config: DataConfig) -> torch.tensor:
    return (x[0, :, 2] >= config.min_z).long()
