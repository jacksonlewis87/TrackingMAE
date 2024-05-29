import pytest
import torch
from unittest.mock import call, patch

from constants import X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX
from data.mae.transforms import (
    random_crop,
    shuffle_players,
    shuffle_tensor_dim,
    normalize_tensor,
    normalize_coordinates,
)


@pytest.mark.parametrize(
    "size, dim, expected_return_size",
    [
        ((11, 6, 3), 1, (11, 4, 3)),
        ((11, 4, 3), 1, (11, 4, 3)),
        ((11, 3, 5), 2, (11, 3, 4)),
        ((11, 3, 5), 0, (4, 3, 5)),
    ],
)
def test_random_crop(size, dim, expected_return_size):
    x = torch.zeros(size)
    length = 4

    result = random_crop(x=x, length=length, dim=dim)

    assert result.size() == expected_return_size


@patch("data.mae.transforms.shuffle_tensor_dim")
def test_shuffle_players(mock_shuffle_tensor_dim):
    ball = [torch.zeros(3)]
    team_0 = [torch.tensor([1, 1, 1]) for _ in range(5)]
    team_1 = [torch.tensor([2, 2, 2]) for _ in range(5)]
    x = torch.stack(ball + team_0 + team_1, dim=0)
    team_0 = torch.stack(team_0, dim=0)
    team_1 = torch.stack(team_1, dim=0)
    mock_shuffle_tensor_dim.return_value = torch.zeros(5, 3)

    result = shuffle_players(x=x, shuffle_players=True)

    # TODO make this work
    # mock_shuffle_tensor_dim.assert_has_calls([
    #     call(x=team_0, dim=0),
    #     call(x=team_1, dim=0),
    # ])
    assert torch.all(torch.eq(result, torch.zeros(11, 3)))


@pytest.mark.parametrize("dim", [0, 1, 2])
def test_shuffle_tensor_dim(dim):
    x_0 = torch.zeros(3, 3, 3)
    x_1 = torch.ones(3, 3, 3)
    x = torch.stack([x_0, x_1], dim=dim)

    for i in range(5):
        result = shuffle_tensor_dim(x, dim=dim)

        assert torch.any(
            torch.tensor(
                [torch.all(torch.eq(result, x)), torch.all(torch.eq(result, torch.stack([x_1, x_0], dim=dim)))]
            )
        )


@pytest.mark.parametrize("_min, _max, expected_return_value", [(0, 2, 0), (1, 2, -1), (0, 1, 1)])
def test_normalize_tensor(_min, _max, expected_return_value):
    x = torch.ones(3)

    result = normalize_tensor(x=x, _min=_min, _max=_max)

    assert torch.all(torch.eq(result, torch.tensor([expected_return_value for _ in range(3)])))


def test_normalize_coordinates():
    x = torch.tensor(
        [
            [
                [X_MIN, Y_MIN, Z_MIN],
                [X_MIN + ((X_MAX - X_MIN) / 2), Y_MIN + ((Y_MAX - Y_MIN) / 2), Z_MIN + ((Z_MAX - Z_MIN) / 2)],
                [X_MAX, Y_MAX, Z_MAX],
            ]
        ]
    )

    result = normalize_coordinates(x=x)

    assert torch.all(torch.eq(result, torch.tensor([[[-1, -1, -1], [0, 0, 0], [1, 1, 1]]])))
