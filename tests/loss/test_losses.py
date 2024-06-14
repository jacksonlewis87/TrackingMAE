import pytest
import torch
from unittest.mock import patch, Mock

from loss.losses import Loss, L2Loss, BCELoss, get_loss


def test_l2_loss():
    output = torch.tensor(
        [
            [0.0, 1.0],
            [2.0, 3.0],
        ]
    )  # (B, N)
    labels = torch.tensor(
        [
            [1, 1],
            [0, 0],
        ]
    )  # (B, N)
    loss = L2Loss()

    result = loss.forward(output=output, labels=labels)

    assert torch.round(result, decimals=4) == torch.tensor(7)


@pytest.mark.parametrize(
    "loss_weights, expected_result",
    [
        (None, torch.tensor(0.2452)),
        ([10, 1], torch.tensor(0.3281)),
        ([1, 10], torch.tensor(0.1623)),
    ],
)
def test_bce_loss(loss_weights, expected_result):
    output = torch.tensor([0, 0.5, 0.75, 1])  # (B)
    labels = torch.tensor([0, 0, 1, 1])  # (B)

    if loss_weights:
        loss = BCELoss(loss_weights=loss_weights)
    else:
        loss = BCELoss()

    result = loss.forward(output=output, labels=labels)

    assert torch.round(result, decimals=4) == expected_result


@pytest.mark.parametrize(
    "loss, expected_class",
    [(Loss.L2.value, L2Loss), (Loss.BCE.value, BCELoss), ("test", None)],
)
def test_get_loss(loss, expected_class):
    result = get_loss(loss=loss)

    if expected_class:
        assert isinstance(result, expected_class)
    else:
        assert result is None


@patch("loss.losses.L2Loss")
def test_get_loss_args(mock_l2_loss):
    result = get_loss(Loss.L2.value, some_arg="some_arg")

    mock_l2_loss.assert_called_once_with(some_arg="some_arg")
