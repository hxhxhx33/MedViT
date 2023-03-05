from torch import nn

from model.segmenter import Segmenter, WindowSegmenter, Encoder, Decoder
from model.block import BlockList, Block, AFAOptions

from .argument import ModelArgument


def create_model(args: ModelArgument) -> nn.Module:
    """Create the model.

    Args:
        args (ModelArgument): Arguments specifying model configurations.

    Returns:
        Model: The built model.
    """
    opt = _parse_option(args)
    return Segmenter(opt)


def _parse_option(args: ModelArgument) -> Segmenter.Options:
    ecs = args.encode_afa_spatial_channel_hidden_dim
    dcs = args.decode_afa_spatial_channel_hidden_dim

    wopt = WindowSegmenter.Options(
        mxcoder_block_list_option=BlockList.Options(
            block_option=Block.Options(
                num_channel=args.encode_channel,
                attention_head=args.mxcode_attention_head,
            ),
            num_block=args.mxcode_block,
        ),
        decoder_options_list=[
            Decoder.Options(
                encoder_options=Encoder.Options(
                    spatial_dims=[args.window_size] * args.num_dim,
                    patch_sizes=[patch_size] * args.num_dim,
                    input_channel=args.input_channel,
                    block_list_option=BlockList.Options(
                        block_option=Block.Options(
                            num_channel=args.encode_channel,
                            attention_head=args.encode_attention_head,
                        ),
                        num_block=args.encode_block,
                        afa_options=AFAOptions(
                            num_dim=args.num_dim,
                            spatial_attn_hidden_dim=ecs[0],
                            channel_attn_hidden_dim=ecs[1],
                            spatial_size=args.window_size // patch_size,
                        )
                        if ecs is not None
                        else None,
                    ),
                ),
                block_list_option=BlockList.Options(
                    block_option=Block.Options(
                        num_channel=args.decode_channel,
                        attention_head=args.decode_attention_head,
                    ),
                    num_block=args.decode_block,
                    afa_options=AFAOptions(
                        num_dim=args.num_dim,
                        spatial_attn_hidden_dim=dcs[0],
                        channel_attn_hidden_dim=dcs[1],
                        spatial_size=args.window_size // patch_size,
                    )
                    if dcs is not None
                    else None,
                ),
                hidden_channel=args.decode_hidden_channel,
                output_channel=args.output_channel,
                projector_hidden_channel=args.decode_projector_hidden_channel,
                projector_upsample_channel=args.decode_projector_upsample_channel,
            )
            for patch_size in args.patch_sizes
        ],
    )
    return Segmenter.Options(
        num_dim=args.num_dim,
        window_size=args.window_size,
        window_segmenter_options=wopt,
    )
