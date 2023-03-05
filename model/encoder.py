import math
from typing import List
from dataclasses import dataclass, field

import torch
from torch import nn
from torch.nn.parameter import Parameter

from .vit.patch_embed import PatchEmbed
from .vit.helper.position_embed import sincos_position_embed

from .block import BlockList


class Encoder(nn.Module):
    """The encoder part of MAE."""

    @dataclass(kw_only=True)
    class Options:
        """Options to build the model."""

        spatial_dims: List[int]
        patch_sizes: List[int]
        input_channel: int
        block_list_option: BlockList.Options

        patchified_dims: List[int] = field(init=False)
        patchified_volume: int = field(init=False)
        patch_volume: int = field(init=False)

        def __post_init__(self):
            assert len(self.patch_sizes) == len(self.spatial_dims)
            for d, p in zip(self.spatial_dims, self.patch_sizes):
                assert d % p == 0

            dims = [d // p for (d, p) in zip(self.spatial_dims, self.patch_sizes)]
            self.patchified_dims = dims
            self.patchified_volume = math.prod(dims)
            self.patch_volume = math.prod(self.patch_sizes)

    def __init__(self, opt: Options):
        super().__init__()
        self.options = opt

        self._build()

    def forward(self, inpt: torch.Tensor) -> torch.Tensor:  # type: ignore
        # (B, C_i, *D)
        x = self.patch_embed(inpt)

        # (B, |D / P|, C_e)
        x += self.pos_embed.unsqueeze(0)
        x = self.block_list(x)
        x = self.norm(x)

        return x

    def _build(self):
        opt = self.options
        blopt = opt.block_list_option
        bopt = blopt.block_option

        self.patch_embed = PatchEmbed(
            PatchEmbed.Options(
                patch_sizes=opt.patch_sizes,
                input_channel=opt.input_channel,
                output_channel=bopt.num_channel,
            )
        )

        pos_embed = sincos_position_embed(
            spatial_dims=opt.patchified_dims,
            embed_dim=bopt.num_channel,
        )
        self.pos_embed = Parameter(pos_embed, requires_grad=False)  # freeze

        self.block_list = BlockList(blopt)
        self.norm = nn.LayerNorm(normalized_shape=bopt.num_channel)
