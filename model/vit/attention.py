from dataclasses import dataclass

import torch
from torch import nn
from einops import rearrange  # type: ignore


class Attention(nn.Module):
    """The Attention module."""

    @dataclass(kw_only=True)
    class Options:
        """Options to build the Attention module."""

        num_channel: int
        num_head: int
        use_nonlinear: bool = False

    def __init__(self, opt: Options):
        super().__init__()

        self.options = opt

        assert opt.num_channel % opt.num_head == 0
        self.scale = (opt.num_channel // opt.num_head) ** (-0.5)

        self.qkv = nn.Linear(
            in_features=opt.num_channel,
            out_features=opt.num_channel * 3,
            bias=True,
        )
        self.proj = nn.Linear(
            in_features=opt.num_channel,
            out_features=opt.num_channel,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        """Forward.

        Args:
            x (torch.Tensor): A tensor of shape (B, N, C).

        Returns:
            torch.Tensor: A tensor of the same shape as the input.
        """
        num_head = self.options.num_head

        qkv: torch.Tensor = self.qkv(x)
        qkv = rearrange(qkv, "b n (k h c) -> k b h n c", h=num_head, k=3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q *= self.scale
        p = q @ k.transpose(-2, -1)
        p = p.softmax(dim=-1)
        x = p @ v
        x = rearrange(x, "b h n c -> b n (h c)", h=num_head)
        x = self.proj(x)
        return x
