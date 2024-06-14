import pytest
import torch

from constants import X_MAX
from data.decoding.transforms import get_made_shot_label


@pytest.mark.parametrize(
    "row_0, row_2, label_start_index, expected_result",
    [
        ([11, 11, 11], [11, 11, 11], 0, torch.tensor([0.0, 0.0])),
        ([4.7, 24.3, 9], [11, 11, 11], 0, torch.tensor([1.0, 0.0])),
        ([11, 11, 11], [X_MAX - 4.7, 24.3, 9], 0, torch.tensor([0.0, 1.0])),
        ([X_MAX - 6, 25.7, 10.1], [6, 25.7, 10.1], 0, torch.tensor([1.0, 1.0])),
        ([X_MAX - 6, 25.7, 10.1], [6, 25.7, 10.1], 1, torch.tensor([1.0, 0.0])),
    ],
)
def test_get_made_shot_label(row_0, row_2, label_start_index, expected_result):
    x = torch.tensor(
        [
            [row_0, [11, 11, 11], row_2],  # ball
            [[11, 11, 11], [11, 11, 11], [11, 11, 11]],  # player
        ]
    )

    result = get_made_shot_label(x=x, label_start_index=label_start_index)

    assert torch.all(torch.eq(result, expected_result))
