import pytest
import torch
from unittest.mock import call, patch

from data.mae.transforms import random_crop, shuffle_players, shuffle_tensor_dim


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
