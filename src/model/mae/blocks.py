import collections.abc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from itertools import repeat
from functools import partial
from torch.jit import Final
from typing import Optional


class PatchEmbed(nn.Module):
    def __init__(
        self,
        patch_length: int,
        channels: int,
        embedding_dimension: int,
    ):
        super().__init__()
        self.proj = nn.Conv2d(
            channels, embedding_dimension, kernel_size=(1, patch_length), stride=(1, patch_length), bias=True
        )

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class MaskedEncoder2D(nn.Module):
    def __init__(
        self,
        patch_length: int,
        channels: int,
        num_patches_x: int,
        num_patches_y: int,
        embedding_dimension: int,
        depth: int,
        num_heads: int,
        masking_strategy: nn.Module,
    ):
        super().__init__()
        self.patch_length = patch_length
        self.channels = channels
        self.num_patches_x = num_patches_x
        self.num_patches_y = num_patches_y
        self.embedding_dimension = embedding_dimension
        self.depth = depth
        self.num_heads = num_heads
        self.masking_strategy = masking_strategy
        self.patch_embed = self.get_patch_embed()
        self.pos_encoding = self.get_pos_encoding()
        self.class_token = self.get_class_token()
        self.transformer_blocks = self.get_transformer_blocks()
        self.norm = self.get_norm()

    def get_patch_embed(self):
        return PatchEmbed(
            patch_length=self.patch_length,
            channels=self.channels,
            embedding_dimension=self.embedding_dimension,
        )

    def get_pos_encoding(self):
        parameter = nn.Parameter(
            torch.zeros(
                1, (self.num_patches_x * self.num_patches_y) + 1, self.embedding_dimension
            ),  # add 1 for class token
            requires_grad=False,
        )
        encoding = get_2d_pos_encoding(
            embedding_dimension=self.embedding_dimension,
            num_patches_x=self.num_patches_x,
            num_patches_y=self.num_patches_y,
            cls_token=True,
        )
        return parameter.data.copy_(torch.from_numpy(encoding).float().unsqueeze(0))

    def get_class_token(self):
        return nn.Parameter(torch.zeros(1, 1, self.embedding_dimension))

    def get_transformer_blocks(self):
        transformer_blocks = nn.ModuleList()
        for i in range(self.depth):
            transformer_blocks.add_module(
                name=f"transformer_block_{i}",
                module=TransformerBlock(
                    dim=self.embedding_dimension,
                    num_heads=self.num_heads,
                    qkv_bias=True,
                ),
            )
        return transformer_blocks

    def get_norm(self):
        return nn.LayerNorm(self.embedding_dimension)

    def forward(self, x):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_encoding[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.masking_strategy(x)

        # append cls token
        class_token = self.class_token + self.pos_encoding[:, :1, :]
        class_tokens = class_token.expand(x.shape[0], -1, -1)
        x = torch.cat((class_tokens, x), dim=1)

        # apply Transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        x = self.norm(x)

        return x, mask, ids_restore


class Decoder2D(nn.Module):
    def __init__(
        self,
        patch_size: int,
        num_patches_x: int,
        num_patches_y: int,
        embedding_dimension: int,
        decoder_embedding_dimension: int,
        depth: int,
        num_heads: int,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches_x = num_patches_x
        self.num_patches_y = num_patches_y
        self.embedding_dimension = embedding_dimension
        self.decoder_embedding_dimension = decoder_embedding_dimension
        self.depth = depth
        self.num_heads = num_heads

        self.decoder_embed = self.get_decoder_embed()
        self.mask_token = self.get_mask_token()
        self.pos_encoding = self.get_pos_encoding()
        self.transformer_blocks = self.get_transformer_blocks()
        self.norm = self.get_norm()
        self.decoder_pred = self.get_decoder_pred()

    def get_decoder_embed(self):
        return nn.Linear(self.embedding_dimension, self.decoder_embedding_dimension, bias=True)

    def get_mask_token(self):
        return nn.Parameter(torch.zeros(1, 1, self.decoder_embedding_dimension))

    def get_pos_encoding(self):
        parameter = nn.Parameter(
            torch.zeros(
                1, (self.num_patches_x * self.num_patches_y) + 1, self.decoder_embedding_dimension
            ),  # add 1 for class token
            requires_grad=False,
        )
        encoding = get_2d_pos_encoding(
            embedding_dimension=self.decoder_embedding_dimension,
            num_patches_x=self.num_patches_x,
            num_patches_y=self.num_patches_y,
            cls_token=True,
        )
        return parameter.data.copy_(torch.from_numpy(encoding).float().unsqueeze(0))

    def get_transformer_blocks(self):
        transformer_blocks = nn.ModuleList()
        for i in range(self.depth):
            transformer_blocks.add_module(
                name=f"transformer_block_{i}",
                module=TransformerBlock(
                    dim=self.decoder_embedding_dimension,
                    num_heads=self.num_heads,
                    qkv_bias=True,
                ),
            )
        return transformer_blocks

    def get_norm(self):
        return nn.LayerNorm(self.decoder_embedding_dimension)

    def get_decoder_pred(self):
        return nn.Linear(self.decoder_embedding_dimension, self.patch_size, bias=True)

    def forward(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.pos_encoding

        # apply Transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        x = self.norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x


class Attention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = True  # ???

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        init_values: float = 1e-5,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


def drop_path(x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f"drop_prob={round(self.drop_prob,3):0.3f}"


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)


class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        norm_layer=None,
        bias=True,
        drop=0.0,
        use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: Optional[float] = None,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        mlp_layer: nn.Module = Mlp,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


def get_2d_pos_encoding(embedding_dimension: int, num_patches_x: int, num_patches_y: int, cls_token=False):
    grid_x = np.arange(num_patches_x, dtype=np.float32)
    grid_y = np.arange(num_patches_y, dtype=np.float32)
    grid = np.meshgrid(grid_y, grid_x)
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, num_patches_x, num_patches_y])
    pos_embed = get_2d_pos_encoding_from_grid(embedding_dimension=embedding_dimension, grid=grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embedding_dimension]), pos_embed], axis=0)
    return pos_embed


def get_2d_pos_encoding_from_grid(embedding_dimension: int, grid):
    assert embedding_dimension % 2 == 0

    # TODO refine this for non-square grid
    # use half of dimensions to encode grid_x, grid_y
    encoding_x = get_1d_pos_encoding(embedding_dimension=embedding_dimension // 2, pos=grid[0])  # (X*Y, D/2)
    encoding_y = get_1d_pos_encoding(embedding_dimension=embedding_dimension // 2, pos=grid[1])  # (X*Y, D/2)

    encoding = np.concatenate([encoding_x, encoding_y], axis=1)  # (H*W, D)
    return encoding


def get_1d_pos_encoding(embedding_dimension: int, pos):
    assert embedding_dimension % 2 == 0

    omega = np.arange(embedding_dimension // 2, dtype=np.float32)
    omega /= embedding_dimension / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    encoding_sin = np.sin(out)  # (M, D/2)
    encoding_cos = np.cos(out)  # (M, D/2)

    encoding = np.concatenate([encoding_sin, encoding_cos], axis=1)  # (M, D)
    return encoding
