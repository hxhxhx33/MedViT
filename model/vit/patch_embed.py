from typing import List, Optional
from dataclasses import dataclass

import torch
from torch import nn
from monai.networks.layers.factories import Conv

from .helper.rearrange import flatten_and_swap_spatial_and_channel_dims
from .helper.pad import even_pad


class PatchEmbed(nn.Module):
    """Module for embedding a spatial tensor into patches."""

    @dataclass(kw_only=True)
    class Options:
        """Options to build the PatchEmbed module."""

        patch_sizes: List[int]
        input_channel: int
        output_channel: int

        strides: Optional[List[int]] = None

        # for testing prupose
        layer_norm_elementwise_affine: bool = True
        layer_norm_eps: float = 1e-5

    def __init__(self, opt: Options):
        super().__init__()

        self.options = opt

        num_dim = len(opt.patch_sizes)
        strides = opt.strides or opt.patch_sizes

        # pylint: disable=not-callable
        self.proj = Conv["conv", num_dim](
            in_channels=opt.input_channel,
            out_channels=opt.output_channel,
            kernel_size=opt.patch_sizes,
            stride=strides,
        )
        self.norm = nn.LayerNorm(
            normalized_shape=opt.output_channel,
            elementwise_affine=opt.layer_norm_elementwise_affine,
            eps=opt.layer_norm_eps,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        """Forward.

        Args:
            x (torch.Tensor): The input tensor of shape (B, C_i, *D).

        Returns:
            torch.Tensor: The output tensor of shape (B, |D / P|, C_h).
        """
        opt = self.options
        patch_sizes = opt.patch_sizes

        # (B, C_i, *D)
        [_, _, *dims] = x.shape
        assert len(dims) == len(patch_sizes)

        if opt.strides is None:
            x, _ = even_pad(x, patch_sizes)
        x = self.proj(x)

        # (B, C_h, *D / *P)
        x = x.flatten(2)

        # (B, C_h, |D / P|)
        x = flatten_and_swap_spatial_and_channel_dims(x)

        # (B, |D / P|, C_h)
        x = self.norm(x)

        return x
