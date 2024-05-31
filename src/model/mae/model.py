import torch
import torch.nn as nn
import torch.utils.checkpoint
from pytorch_lightning import LightningModule
from typing import Tuple

from model.mae.blocks import Decoder2D, MaskedEncoder2D
from model.mae.masking import get_masking_strategy, MaskingConfig
from model.mae.model_config import FullConfig


class TrackingMaskedAutoEncoder(LightningModule):
    def __init__(
        self,
        config: FullConfig,
    ):
        super().__init__()
        self.config = config

        if self.config.data_config.num_frames % self.config.model_config.num_sequence_patches > 0:
            print("Error: num_frames must be divisible by num_sequence_patches")
            raise Exception

        # B = batch size (self.config.data_config.batch_size)
        # C = input channels (3: x, y, z) - set as channels below
        # P = num_players (self.confg.model_config.num_players)
        # F = total number of frames per player (self.config.data_config.num_frames)
        # N = number of patches per player (self.config.model_config.num_sequence_patches)
        # T = total number of patches in a sample (N * P) - set as total_patches below
        # L = patch length along temporal dimension (F / N) - set as patch_length below
        # S = patch size, flattened (L * C) - set as patch_size below

        self.channels = 3

        self.total_patches = self.config.model_config.num_sequence_patches * self.config.model_config.num_players
        self.patch_length = int(self.config.data_config.num_frames / self.config.model_config.num_sequence_patches)
        self.patch_size = self.patch_length * 3

        masking_strategy = get_masking_strategy(
            masking_strategy=self.config.model_config.masking_strategy,
            config=MaskingConfig(
                mask_ratio=self.config.model_config.masking_ratio,
                indexes=self.config.model_config.masking_indexes,
                patches_per_index=self.config.model_config.num_sequence_patches,
            ),
        )

        self.encoder = MaskedEncoder2D(
            patch_length=self.patch_length,
            channels=self.channels,
            num_patches_x=self.config.model_config.num_players,
            num_patches_y=self.config.model_config.num_sequence_patches,
            embedding_dimension=self.config.model_config.embedding_dimension,
            depth=self.config.model_config.encoder_depth,
            num_heads=self.config.model_config.num_encoder_heads,
            masking_strategy=masking_strategy,
        )

        self.decoder = Decoder2D(
            patch_size=self.patch_size,
            num_patches_x=self.config.model_config.num_players,
            num_patches_y=self.config.model_config.num_sequence_patches,
            embedding_dimension=self.config.model_config.embedding_dimension,
            decoder_embedding_dimension=self.config.model_config.decoder_embedding_dimension,
            depth=self.config.model_config.decoder_depth,
            num_heads=self.config.model_config.num_decoder_heads,
        )

        self.initialize_weights()

    def initialize_weights(self):
        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.encoder.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.encoder.class_token, std=0.02)
        torch.nn.init.normal_(self.decoder.mask_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, x):
        # (B, C, P, F) -> (B, T, S)
        x = x.reshape(
            shape=(
                x.shape[0],
                self.channels,
                self.config.model_config.num_players,
                1,
                self.config.model_config.num_sequence_patches,
                self.patch_length,
            )
        )
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape((x.shape[0], self.total_patches, self.patch_length * self.channels))
        return x

    def unpatchify(self, x):
        # (B, T, S) -> (B, C, P, F)
        x = x.reshape(
            shape=(
                x.shape[0],
                self.config.model_config.num_players,
                self.config.model_config.num_sequence_patches,
                1,
                self.patch_length,
                self.channels,
            )
        )
        x = torch.einsum("nhwpqc->nchpwq", x)
        x = x.reshape(
            (x.shape[0], self.channels, self.config.model_config.num_players, self.config.data_config.num_frames)
        )
        return x

    def forward_loss(self, x, pred, mask):
        # x:    (B, C, P, F)
        # pred: (B, T, S)
        # mask: (B, T) - 0=keep, 1=remove
        target = self.patchify(x)

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # (B, T) - mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, x, mask_ratio=0.75):
        x = x.permute(0, 3, 1, 2)
        latent, mask, ids_restore = self.encoder(x, mask_ratio)
        pred = self.decoder(latent, ids_restore)
        loss = self.forward_loss(x, pred, mask)
        return loss, pred, mask

    def training_step(self, batch: Tuple[torch.tensor, torch.tensor], batch_idx: int) -> float:
        x = batch
        loss, pred, mask = self.forward(x)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch: Tuple[torch.tensor, torch.tensor], batch_idx: int) -> None:
        x = batch
        loss, pred, mask = self.forward(x)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.model_config.learning_rate)
        # scheduler = {
        #     "scheduler": optim.lr_scheduler.OneCycleLR(
        #         optimizer,
        #         max_lr=self.config.learning_rate,
        #         total_steps=self.config.epochs,
        #     ),
        # }
        return optimizer  # , [scheduler]
