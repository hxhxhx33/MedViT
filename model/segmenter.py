from dataclasses import dataclass, replace
from typing import List
from einops import rearrange  # type: ignore

import torch
from torch import nn
from monai.networks.blocks.convolutions import Convolution

from .vit.helper.rearrange import patchifier
from .vit.helper.position_embed import sincos_position_embed

from .block import BlockList
from .encoder import Encoder
from .decoder import Decoder as BaseDecoder


class MLP(nn.Module):
    """MLP module."""

    @dataclass(kw_only=True)
    class Options:
        """Options to build the model."""

        input_channel: int
        hidden_channels: List[int]
        output_channel: int

    def __init__(self, opt: Options):
        super().__init__()

        self.options = opt

        dims = [opt.input_channel, *opt.hidden_channels]
        self.projs = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1], bias=True) for i in range(len(dims) - 1)]
        )
        self.norms = nn.ModuleList(
            [nn.LayerNorm(dims[i + 1]) for i in range(len(dims) - 1)]
        )
        self.out = nn.Linear(dims[-1], opt.output_channel, bias=True)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        """Forward.

        Args:
            x (torch.Tensor): A tensor of shape (B, N, C_i)

        Returns:
            torch.Tensor: A tensor of shape (B, N, C_o)
        """
        for proj, norm in zip(self.projs, self.norms):
            x = proj(x)
            x = norm(x)
            x = self.act(x)
        x = self.out(x)
        return x


class DecodeProjector(nn.Module):
    @dataclass(kw_only=True)
    class Options:
        """Options to build the model."""

        num_dim: int
        in_channel: int
        in_patch_size: int
        out_patch_size: int
        out_channel: int
        upsample_channel: int
        hidden_channel: int

    def __init__(self, opt: Options) -> None:
        super().__init__()
        self.options = opt

        assert opt.out_patch_size % opt.in_patch_size == 0
        r = int(opt.out_patch_size / opt.in_patch_size)

        self.upsample = Convolution(
            conv_only=True,
            is_transposed=True,
            spatial_dims=opt.num_dim,
            in_channels=opt.in_channel,
            out_channels=opt.upsample_channel,
            kernel_size=r,
            strides=r,
            padding=0,
            output_padding=0,
        )

        self.residual = (
            Convolution(
                conv_only=True,
                spatial_dims=opt.num_dim,
                in_channels=opt.upsample_channel,
                out_channels=opt.hidden_channel,
            )
            if opt.upsample_channel != opt.hidden_channel
            else nn.Identity()
        )

        self.conv_norm_act = Convolution(
            adn_ordering="NDA",
            act=("leakyrelu", {"negative_slope": 0.01}),
            norm=("instance"),
            spatial_dims=opt.num_dim,
            in_channels=opt.upsample_channel,
            out_channels=opt.hidden_channel,
        )
        self.conv_norm = Convolution(
            adn_ordering="NDA",
            act=None,
            norm=("instance"),
            spatial_dims=opt.num_dim,
            in_channels=opt.hidden_channel,
            out_channels=opt.hidden_channel,
        )
        self.act = nn.LeakyReLU(
            negative_slope=0.01,
            inplace=True,
        )
        self.out = Convolution(
            conv_only=True,
            spatial_dims=opt.num_dim,
            in_channels=opt.hidden_channel,
            out_channels=opt.out_channel,
            kernel_size=1,
            strides=1,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        opt = self.options
        b, n = x.shape[0], x.shape[1]

        ds = " ".join([f"d{i}" for i in range(opt.num_dim)])
        vs1 = {f"d{i}": opt.in_patch_size for i in range(opt.num_dim)}
        vs2 = {f"d{i}": opt.out_patch_size for i in range(opt.num_dim)}

        x = rearrange(x, f"b n ({ds} c) -> (b n) c {ds}", **vs1).contiguous()
        x = self.upsample(x)
        x_ = self.residual(x)

        x = self.conv_norm_act(x)
        x = self.conv_norm(x)
        x += x_
        x = self.act(x)

        x = self.out(x)
        x = rearrange(x, f"(b n) c {ds} -> b n ({ds} c)", b=b, n=n, **vs2)

        return x


class Decoder(BaseDecoder):
    """Decoder."""

    @dataclass(kw_only=True)
    class Options(BaseDecoder.Options):
        """Options to build the model."""

        hidden_channel: int
        output_channel: int
        projector_upsample_channel: int
        projector_hidden_channel: int

    def __init__(self, opt: Options):
        self.options = opt
        eopt = opt.encoder_options
        blopt = opt.block_list_option
        bopt = blopt.block_option

        assert len(set(eopt.patch_sizes)) == 1
        self.encode_patch_size = eopt.patch_sizes[0]

        d = len(eopt.patch_sizes)
        v = bopt.num_channel / opt.hidden_channel
        s = int(round(v ** (1 / d)))
        assert s**d == v
        self.hidden_patch_size = s
        self.num_dim = d

        if (aopt := opt.block_list_option.afa_options) is not None:
            aopt.spatial_size_in_channel = s

        super().__init__(opt)

    def _proj_layer(self) -> nn.Module:
        opt = self.options

        return DecodeProjector(
            DecodeProjector.Options(
                num_dim=self.num_dim,
                in_channel=opt.hidden_channel,
                in_patch_size=self.hidden_patch_size,
                out_channel=opt.output_channel,
                out_patch_size=self.encode_patch_size,
                upsample_channel=opt.projector_upsample_channel,
                hidden_channel=opt.projector_hidden_channel,
            )
        )


class Residual(nn.Module):
    """The Residual module."""

    @dataclass(kw_only=True)
    class Options:
        """Options to build the model."""

        input_channel: int
        output_channel: int

    def __init__(self, opt: Options):
        super().__init__()
        self.options = opt

        hidden_channel = opt.output_channel * 4
        num_block = 2
        self.mlp = MLP(
            MLP.Options(
                input_channel=opt.input_channel,
                hidden_channels=[hidden_channel] * num_block,
                output_channel=opt.output_channel,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        """Forward.

        Args:
            x (torch.Tensor): A tensor of shape (B, C_i, *D).

        Returns:
            torch.Tensor: A tensor of shape (B, C_o, *D).
        """
        x = x.movedim(1, -1)
        x = self.mlp(x)
        x = x.movedim(-1, 1)
        x = x.sigmoid()
        return x


class WindowSegmenter(nn.Module):
    """The WindowSegmenter module."""

    @dataclass(kw_only=True)
    class Options:
        """Options to build the model."""

        decoder_options_list: List[Decoder.Options]
        mxcoder_block_list_option: BlockList.Options

    def __init__(self, opt: Options):
        super().__init__()

        self.options = opt
        dopts = opt.decoder_options_list
        dopts2 = [replace(o, output_channel=o.output_channel + 1) for o in dopts]
        mblopt = opt.mxcoder_block_list_option
        mbopt = opt.mxcoder_block_list_option.block_option

        self.encoders = nn.ModuleList([Encoder(dopt.encoder_options) for dopt in dopts])
        self.decoders = nn.ModuleList([Decoder(dopt) for dopt in dopts2])
        self.mxcode_block_list = BlockList(mblopt)
        self.mxcode_norm = nn.LayerNorm(normalized_shape=mbopt.num_channel)

        mx_pos_embed = sincos_position_embed(
            spatial_dims=[len(dopts)],
            embed_dim=mbopt.num_channel,
        )
        self.mx_pos_embed = nn.Parameter(mx_pos_embed, requires_grad=False)

        dopt = dopts[0]
        eopt = dopt.encoder_options
        self.residual = Residual(
            Residual.Options(
                input_channel=eopt.input_channel,
                output_channel=dopt.output_channel,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        """Forward.

        Args:
            x (torch.Tensor): A tensor of shape (B, C_i, *D).

        Returns:
            torch.Tensor: A tensor of shape (B, C_o, *D).
        """
        residual = self.residual(x)

        xs: List[torch.Tensor] = [encoder(x) for encoder in self.encoders]
        seq_lens = [x_.shape[1] for x_ in xs]

        for i, x_ in enumerate(xs):
            x_ += self.mx_pos_embed[i]

        x = torch.concat(xs, dim=1)
        x = self.mxcode_block_list(x)
        x = self.mxcode_norm(x)

        ps: List[torch.Tensor] = []
        qs: List[torch.Tensor] = []
        start_idx = 0
        for seq_len, decoder in zip(seq_lens, self.decoders):
            seq = x[:, start_idx : start_idx + seq_len]
            start_idx += seq_len

            seq = decoder(seq)
            p = seq[:, 1:]
            a = seq[:, :1]

            q = torch.exp(-torch.abs(a))
            qs.append(q)
            ps.append(p)

        logit = ps[0] * (1 - qs[0])
        r = qs[0]
        for [p, q] in zip(ps[1:], qs[1:]):
            logit += p * (1 - q) * r
            r = r * q

        logit += residual
        prob = torch.sigmoid(logit)

        return prob


class Segmenter(nn.Module):
    """The Segmenter module."""

    @dataclass(kw_only=True)
    class Options:
        """Options to build the model."""

        window_segmenter_options: WindowSegmenter.Options
        window_size: int
        num_dim: int

    def __init__(self, opt: Options):
        super().__init__()

        self.options = opt
        wopt = opt.window_segmenter_options

        self.window_segmenter = WindowSegmenter(wopt)
        self.window_partition = patchifier(pattern=lambda ds, ps: f"(b {ds}) c {ps}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        """Forward.

        Args:
            x (torch.Tensor): A tensor of shape (B, C_i, *D).

        Returns:
            torch.Tensor: A tensor of shape (B, C_o, *D).
        """
        opt = self.options

        [_, _, *spatial_sizes] = x.shape

        # (B, C_i, *D)
        x = self.window_partition(x, [opt.window_size] * opt.num_dim)

        # (B x |*D / *W|, C_i, *W)
        x = self.window_segmenter(x)

        # (B x |*D / *W|, C_o, *W)
        x = self._window_reunion(x, spatial_sizes)

        # (B, C_o, *D)
        return x

    def _window_reunion(
        self, x: torch.Tensor, spatial_sizes: List[int]
    ) -> torch.Tensor:
        opt = self.options
        d = opt.num_dim

        ds = " ".join([f"d{i}" for i in range(d)])
        ws = " ".join([f"w{i}" for i in range(d)])
        dws = " ".join([f"(d{i} w{i})" for i in range(d)])
        dvs = {f"d{i}": d // opt.window_size for i, d in enumerate(spatial_sizes)}
        wvs = {f"w{i}": opt.window_size for i in range(d)}

        x = rearrange(x, f"(b {ds}) c {ws} -> b c {dws}", **dvs, **wvs)

        return x
