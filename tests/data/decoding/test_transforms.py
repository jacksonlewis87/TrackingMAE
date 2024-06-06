import pytest
import torch

from constants import X_MAX
from data.decoding.transforms import get_made_shot_label


@pytest.mark.parametrize(
    "row_0, row_2, label_start_index, max_length, expected_result",
    [
        ([11, 11, 11], [11, 11, 11], 0, 4, torch.tensor([0.0, 0.0])),
        ([4.7, 24.3, 9], [11, 11, 11], 0, 4, torch.tensor([1.0, 0.0])),
        ([11, 11, 11], [4.7, X_MAX - 24.3, 9], 0, 4, torch.tensor([0.0, 1.0])),
        ([6, X_MAX - 25.7, 10.1], [6, 25.7, 10.1], 0, 4, torch.tensor([1.0, 1.0])),
        ([6, X_MAX - 25.7, 10.1], [6, 25.7, 10.1], 0, 2, torch.tensor([0.0, 1.0])),
        ([6, X_MAX - 25.7, 10.1], [6, 25.7, 10.1], 1, 3, torch.tensor([1.0, 0.0])),
    ],
)
def test_get_made_shot_label(row_0, row_2, label_start_index, max_length, expected_result):
    x = torch.tensor(
        [
            [row_0, [11, 11, 11], row_2],  # ball
            [[11, 11, 11], [11, 11, 11], [11, 11, 11]],  # player
        ]
    )

    result = get_made_shot_label(x=x, label_start_index=label_start_index, max_length=max_length)

    assert torch.all(torch.eq(result, expected_result))
