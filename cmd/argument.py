from typing import List, Optional, Tuple

from base.argument import (
    BaseArgument,
    CommonArgument,
    DatasetArgument as BaseDatasetArgument,
    TrainArgument as BaseTrainArgument,
    PredictArgument as BasePredictArgument,
)


class DatasetArgument(BaseDatasetArgument):
    """Commend line arguments for data set configurations."""

    no_random_rotate: bool = False
    """if not to randomly rotate the input"""

    no_augment: bool = False
    """if not to augment"""

    no_crop_random_center: bool = False
    """if not to randomly pick center when cropping"""


class ModelArgument(BaseArgument):
    """Commend line arguments for model."""

    num_dim: int
    """number of dimensionality"""

    window_size: int
    """window size"""

    patch_sizes: List[int]
    """size of patches for different scales"""

    input_channel: int
    """number of channel of inputs"""

    output_channel: int
    """number of channel of outputs"""

    encode_channel: int
    """number of channel to embed input patches"""

    encode_attention_head: int
    """number of heads of the multl-head attention in encoder"""

    encode_block: int
    """number of transformer blocks in encoder"""

    mxcode_attention_head: int
    """number of heads of the multl-head attention in mixer"""

    mxcode_block: int
    """number of transformer blocks in mixer"""

    decode_channel: int
    """number of channel for decoder"""

    decode_attention_head: int
    """number of heads of the multl-head attention in decoder"""

    decode_block: int
    """number of transformer blocks in decoder"""

    decode_hidden_channel: int
    """number of hidden channel for decoder"""

    decode_projector_upsample_channel: int
    """number of upsample channel for decoder projector"""

    decode_projector_hidden_channel: int
    """number of hidden channel for decoder projector"""

    encode_afa_spatial_channel_hidden_dim: Optional[Tuple[int, int]] = None
    """spatial and channel hidden channel in spatial AFA for encoder"""

    decode_afa_spatial_channel_hidden_dim: Optional[Tuple[int, int]] = None
    """spatial and channel hidden channel in spatial AFA for decoder"""


class TrainArgument(DatasetArgument, ModelArgument, BaseTrainArgument):
    """Commend line arguments for training."""

    spatial_size: int
    """spatial dimensions to which the input data will be padded thus must be larger
    than all expected input data size"""


class PredictArgument(ModelArgument, BasePredictArgument):
    """Commend line arguments for prediction."""

    spatial_size: int
    """spatial dimensions to which the input data will be padded thus must be larger
    than all expected input data size"""

    sw_batch_size: int = 1
    """batch size for sliding window inference"""

    sw_overlap_ratio: float = 0.6
    """overlap ratio for sliding window inference"""


class EvaluateArgument(CommonArgument):
    """Commend line arguments for evaluate."""

    prediction_dir: str
    """directory in which to-be-evaluated predictions are saved"""
