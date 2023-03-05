from typing import List, Callable
import math

import torch
from einops import rearrange  # type: ignore


def patchifier(
    pattern: Callable[[str, str], str],
) -> Callable[[torch.Tensor, List[int]], torch.Tensor]:
    """Rearrange tensor into patches."""

    def _(x: torch.Tensor, patch_sizes: List[int]) -> torch.Tensor:
        [_, _, *spatial_dims] = x.shape
        assert len(spatial_dims) == len(patch_sizes)
        for d, p in zip(spatial_dims, patch_sizes):
            assert d % p == 0

        ds = " ".join([f"d{i}" for i in range(len(spatial_dims))])
        ps = " ".join([f"p{i}" for i in range(len(patch_sizes))])
        dps = " ".join([f"(d{i} p{i})" for i in range(len(spatial_dims))])

        values = {}
        for i, p in enumerate(patch_sizes):
            values[f"p{i}"] = p

        x = rearrange(x, f"b c {dps} -> {pattern(ds, ps)}", **values)
        return x

    return _


def flatten_and_swap_spatial_and_channel_dims(x: torch.Tensor) -> torch.Tensor:
    """A dimension-agnostic way to flatten and swap spatial dimensions and the channel
    dimension.

    Args:
        x (torch.Tensor): A tensor of shape (B, C, *D) with spatial dimensions at
            the tail.

    Returns:
        torch.Tensor: A tensor of shape (B, |D|, C) with spatial dimensions flattned and
            the channel dimension at the tail.
    """
    [_, _, *dims] = x.shape
    dim_names = " ".join([f"d{i}" for i in range(len(dims))])
    x = rearrange(x, f"n c {dim_names} -> n ({dim_names}) c")
    return x


def patchify(x: torch.Tensor, patch_sizes: List[int]) -> torch.Tensor:
    """Turn a tensor into patches.

    Args:
        x (torch.Tensor): A tensor of shape (B, C, *D,).
        patch_sizes (List[int]): The patch size along each spatial dimensions.

    Returns:
        torch.Tensor: A tensor of shape (B, |D / P|, |P| * C).
    """
    p = patchifier(pattern=lambda ds, ps: f"b ({ds}) ({ps} c)")
    return p(x, patch_sizes)


def unpatchify(
    x: torch.Tensor,
    spatial_dims: List[int],
    patch_sizes: List[int],
) -> torch.Tensor:
    """Recover a patched tensor to its original shape.

    Args:
        x (torch.Tensor): A tensor of shape (B, |D / P|, |P| * C)
        spatial_dims (List[int]): The **original** shape D of the tensor.
        patch_sizes (List[int]): The patch size along each dimensions.

    Returns:
        torch.Tensor: A tensor of shape (B, C, *D).
    """
    assert len(spatial_dims) == len(patch_sizes)

    ds = " ".join([f"d{i}" for i in range(len(spatial_dims))])
    ps = " ".join([f"p{i}" for i in range(len(patch_sizes))])
    dps = " ".join([f"(d{i} p{i})" for i in range(len(spatial_dims))])

    values = {}
    for i, (d, p) in enumerate(zip(spatial_dims, patch_sizes)):
        values[f"d{i}"] = d // p
    for i, p in enumerate(patch_sizes):
        values[f"p{i}"] = p

    x = rearrange(x, f"b ({ds}) ({ps} c) -> b c {dps}", **values)
    return x


def unpatchified_mask(
    mask: torch.Tensor,
    patch_sizes: List[int],
    spatial_dims: List[int],
) -> torch.Tensor:
    """Recover a mask for patches to mask for the original spaces.

    Args:
        mask (torch.Tensor): A tensor of shape (B, |D / P|) as the mask of patches.
        spatial_dims (List[int]): The **original** shape D of the tensor.
        patch_sizes (List[int]): The patch size along each dimensions.

    Returns:
        torch.Tensor: A tensor of shape (B, *D) as the mask of the original tensor.
    """
    patch_volumn = math.prod(patch_sizes)
    patch_mask = mask.unsqueeze(-1).repeat(1, 1, patch_volumn)
    return unpatchify(
        x=patch_mask,
        spatial_dims=spatial_dims,
        patch_sizes=patch_sizes,
    )
