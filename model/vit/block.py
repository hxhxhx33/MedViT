from dataclasses import dataclass

import torch
from torch import nn
from monai.networks.blocks.mlp import MLPBlock

from .attention import Attention


class Block(nn.Module):
    """The ViT Block."""

    @dataclass(kw_only=True)
    class Options:
        """Options to build the Block module."""

        num_channel: int
        attention_head: int
        mlp_ratio: int = 4
        attention_use_nonlinear: bool = False

    def __init__(self, opt: Options):
        super().__init__()

        self.norm1 = nn.LayerNorm(
            normalized_shape=opt.num_channel,
        )
        self.attn = Attention(
            Attention.Options(
                num_channel=opt.num_channel,
                num_head=opt.attention_head,
                use_nonlinear=opt.attention_use_nonlinear,
            )
        )
        self.norm2 = nn.LayerNorm(
            normalized_shape=opt.num_channel,
        )
        self.proj = MLPBlock(
            hidden_size=opt.num_channel,
            mlp_dim=opt.num_channel * opt.mlp_ratio,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        """Forward.

        Args:
            x (torch.Tensor): A tensor of shape (B, N, C).

        Returns:
            torch.Tensor: A tensor of the same shape as the input.
        """
        x = x + self.attn(self.norm1(x))
        x = x + self.proj(self.norm2(x))
        return x
