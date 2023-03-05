from dataclasses import dataclass

import torch
from torch import nn

from .vit.helper.rearrange import unpatchify
from .encoder import Encoder
from .block import BlockList


class Decoder(nn.Module):
    """The decoder part of MAE."""

    @dataclass(kw_only=True)
    class Options:
        """Options to build the model."""

        encoder_options: Encoder.Options
        block_list_option: BlockList.Options

    def __init__(self, opt: Options):
        super().__init__()

        self.options = opt

        self._build()

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:  # type: ignore
        opt = self.options
        eopt = opt.encoder_options
        spatial_dims = eopt.spatial_dims
        patch_sizes = eopt.patch_sizes

        # (B, |D / P|, C_d)
        x = self.embed(encoded)
        x = self.block_list(x)
        x = self.norm(x)
        x = self.proj(x)

        # (B, |D / P|, |P| * C_i)
        x = unpatchify(x, spatial_dims, patch_sizes)

        # (B, C_i, *D)
        return x

    def _embed_layer(self) -> nn.Module:
        opt = self.options
        blopt = opt.block_list_option
        bopt = blopt.block_option
        eopt = self.options.encoder_options
        eblopt = eopt.block_list_option
        ebopt = eblopt.block_option

        return nn.Linear(
            in_features=ebopt.num_channel,
            out_features=bopt.num_channel,
            bias=True,
        )

    def _proj_layer(self) -> nn.Module:
        opt = self.options
        blopt = opt.block_list_option
        bopt = blopt.block_option
        eopt = self.options.encoder_options

        return nn.Linear(
            in_features=bopt.num_channel,
            out_features=eopt.patch_volume * eopt.input_channel,
            bias=True,
        )

    def _build(self):
        opt = self.options
        blopt = opt.block_list_option
        bopt = blopt.block_option

        self.embed = self._embed_layer()
        self.proj = self._proj_layer()

        self.block_list = BlockList(blopt)
        self.norm = nn.LayerNorm(normalized_shape=bopt.num_channel)
