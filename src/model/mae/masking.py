import torch
from dataclasses import dataclass
from enum import Enum


@dataclass
class MaskingConfig:
    mask_ratio: float = None
    indexes: list[int] = None
    patches_per_index: int = None


class MaskingStrategy(Enum):
    INDEX = "index"
    RANDOM = "random"


def get_masking_strategy(masking_strategy: str, config: MaskingConfig):
    if masking_strategy == MaskingStrategy.INDEX.value:
        return IndexMasking(config=config)
    elif masking_strategy == MaskingStrategy.RANDOM.value:
        return RandomMasking(config=config)
    else:
        return None


class IndexMasking(torch.nn.Module):
    def __init__(self, config: MaskingConfig):
        super(IndexMasking, self).__init__()
        self.config = config

    def forward(self, x):
        N, L, D = x.shape  # batch, length, dim
        len_keep = L - (len(self.config.indexes) * self.config.patches_per_index)

        noise = torch.rand(N, L)  # noise in [0, 1]

        # cherry-pick certain indexes
        for index in self.config.indexes:
            for i in range(self.config.patches_per_index):
                noise[:, (index * self.config.patches_per_index) + i] = 2

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L])
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore


class RandomMasking(torch.nn.Module):
    def __init__(self, config: MaskingConfig):
        super(RandomMasking, self).__init__()
        self.config = config

    def forward(self, x):
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - self.config.mask_ratio))

        noise = torch.rand(N, L)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L])
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore
