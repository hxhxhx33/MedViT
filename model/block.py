from typing import Optional, List
from dataclasses import dataclass

import torch
from torch import nn
from monai.networks.blocks.convolutions import Convolution
from monai.networks.layers.factories import Pool

from .vit.block import Block
from .vit.helper.rearrange import patchify, unpatchify


@dataclass(kw_only=True)
class AFAOptions:
    """Options to build the AFA module."""

    num_dim: int
    spatial_size: int
    spatial_attn_hidden_dim: int
    channel_attn_hidden_dim: int
    spatial_size_in_channel: int = 1


class BlockList(nn.Module):
    @dataclass(kw_only=True)
    class Options:
        """Options to build the Block module."""

        block_option: Block.Options
        num_block: int

        # Dense Prediction with Attentive Feature Aggregation
        # https://arxiv.org/abs/2111.00770
        # https://github.com/SysCV/ms-attention-network
        afa_options: Optional[AFAOptions] = None

    def __init__(self, opt: Options):
        super().__init__()
        self.options = opt
        bopt = opt.block_option
        aopt = opt.afa_options

        self.blocks = nn.ModuleList([Block(bopt) for _ in range(opt.num_block)])
        if aopt is not None:
            n = aopt.num_dim
            sc = aopt.spatial_size_in_channel
            c = bopt.num_channel // (sc**n)

            self.spatial_attn = SpatialAttention(
                SpatialAttention.Options(
                    num_dim=n,
                    channel=c,
                    hidden_channel=aopt.spatial_attn_hidden_dim,
                )
            )
            self.channel_attn = ChannelAttention(
                ChannelAttention.Options(
                    num_dim=n,
                    channel=c,
                    hidden_channel=aopt.channel_attn_hidden_dim,
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        opt = self.options
        aopt = opt.afa_options

        if aopt is None:
            for blk in self.blocks:
                x = blk(x)
            return x

        ss = aopt.spatial_size
        sc = aopt.spatial_size_in_channel

        xs: List[torch.Tensor] = []
        qs: List[float] = []
        for blk in self.blocks:
            x = blk(x)

            x_ = unpatchify(
                x=x,
                spatial_dims=[ss * sc] * aopt.num_dim,
                patch_sizes=[sc] * aopt.num_dim,
            )

            xs.append(x_)
            sa = self.spatial_attn(x_)
            ca = self.channel_attn(x_)
            qs.append(sa * ca)

        x = xs[-1] * qs[-1]
        r = 1 - qs[-1]
        for [x_, q] in zip(reversed(xs[:-1]), reversed(qs[:-1])):
            x += x_ * q * r
            r = r * (1 - q)

        x = patchify(x, [sc] * aopt.num_dim)

        return x


class SpatialAttention(nn.Module):
    @dataclass(kw_only=True)
    class Options:
        """Options to build the model."""

        num_dim: int
        channel: int
        hidden_channel: int

    def __init__(self, opt: Options):
        super().__init__()

        self.conv_norm_act1 = Convolution(
            adn_ordering="NDA",
            act=("leakyrelu", {"negative_slope": 0.01, "inplace": True}),
            norm=("instance"),
            spatial_dims=opt.num_dim,
            in_channels=opt.channel,
            out_channels=opt.hidden_channel,
        )
        self.conv_norm_act2 = Convolution(
            adn_ordering="NDA",
            act=("leakyrelu", {"negative_slope": 0.01, "inplace": True}),
            norm=("instance"),
            spatial_dims=opt.num_dim,
            in_channels=opt.hidden_channel,
            out_channels=opt.hidden_channel,
        )
        self.out = Convolution(
            conv_only=True,
            spatial_dims=opt.num_dim,
            in_channels=opt.hidden_channel,
            out_channels=1,
            kernel_size=1,
            strides=1,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        """Calculate the spatial attention.

        Args:
            x (torch.Tensor): A tensor of shape (B, C, *D).

        Returns:
            torch.Tensor: A tensor of shape (B, 1, *D).
        """
        x = self.conv_norm_act1(x)
        x = self.conv_norm_act2(x)
        x = self.out(x)
        x = torch.sigmoid(x)
        return x


class ChannelAttention(nn.Module):
    @dataclass(kw_only=True)
    class Options:
        """Options to build the model."""

        num_dim: int
        channel: int
        hidden_channel: int

    def __init__(self, opt: Options):
        super().__init__()

        # pylint: disable=not-callable
        self.max_pool = Pool["adaptivemax", opt.num_dim](output_size=1)

        # pylint: disable=not-callable
        self.avg_pool = Pool["adaptiveavg", opt.num_dim](output_size=1)

        self.conv_norm_act1 = Convolution(
            adn_ordering="NDA",
            act=("leakyrelu", {"negative_slope": 0.01, "inplace": True}),
            norm=None,
            spatial_dims=opt.num_dim,
            in_channels=opt.channel,
            out_channels=opt.hidden_channel,
            kernel_size=1,
            strides=1,
            padding=0,
        )
        self.conv_norm_act2 = Convolution(
            adn_ordering="NDA",
            act=("leakyrelu", {"negative_slope": 0.01, "inplace": True}),
            norm=None,
            spatial_dims=opt.num_dim,
            in_channels=opt.hidden_channel,
            out_channels=opt.channel,
            kernel_size=1,
            strides=1,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        """Calculate the spatial attention.

        Args:
            x (torch.Tensor): A tensor of shape (B, C, *D).

        Returns:
            torch.Tensor: A tensor of shape (B, C, *1).
        """
        x_avg = self.avg_pool(x)
        x_avg = self.conv_norm_act1(x_avg)
        x_avg = self.conv_norm_act2(x_avg)

        x_max = self.max_pool(x)
        x_max = self.conv_norm_act1(x_max)
        x_max = self.conv_norm_act2(x_max)

        x = x_avg + x_max
        x = torch.sigmoid(x)
        return x
