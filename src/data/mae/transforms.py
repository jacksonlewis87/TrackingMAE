import torch
from random import randint


def random_crop(x: torch.tensor, length: int, dim: int) -> torch.tensor:
    current_length = x.size(dim=dim)

    if current_length < dim:
        print("Error: current length smaller than crop length")
        raise Exception

    if current_length == dim:
        return x

    rand = randint(0, current_length - length)
    indexes = [i + rand for i in range(length)]

    return torch.index_select(x, dim=dim, index=torch.tensor(indexes, dtype=torch.long))


def shuffle_players(x: torch.tensor, shuffle_players: bool) -> torch.tensor:
    if not shuffle_players:
        return x

    # dont shuffle ball (0), and keep teams together (1-5, 6-10)
    x[1:6] = shuffle_tensor_dim(x=x[1:6], dim=0)
    x[6:] = shuffle_tensor_dim(x=x[6:], dim=0)

    return x


def shuffle_tensor_dim(x: torch.tensor, dim: int):
    idx = torch.randperm(x.shape[dim])
    return torch.index_select(x, dim=dim, index=torch.tensor(idx, dtype=torch.long))
