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


def get_made_shot_label(x: torch.tensor, label_start_index: int, max_length: int) -> torch.tensor:
    made_x_min = 4.7
    made_x_max = 6
    made_y_min = 24.3
    made_y_max = 25.7
    made_z_min = 9
    made_z_max = 10.1

    label = torch.zeros(2)

    ball = x[0, label_start_index : label_start_index + max_length]

    label[0] = torch.any(
        (ball[:, 0] >= made_x_min)
        * (ball[:, 0] <= made_x_max)
        * (ball[:, 1] >= made_y_min)
        * (ball[:, 1] <= made_y_max)
        * (ball[:, 2] >= made_z_min)
        * (ball[:, 2] <= made_z_max)
    ).long()
    label[1] = torch.any(
        (ball[:, 0] >= made_x_min)
        * (ball[:, 0] <= made_x_max)
        * (ball[:, 1] >= X_MAX - made_y_max)
        * (ball[:, 1] <= X_MAX - made_y_min)
        * (ball[:, 2] >= made_z_min)
        * (ball[:, 2] <= made_z_max)
    ).long()

    return label
