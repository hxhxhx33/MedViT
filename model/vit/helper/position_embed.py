from typing import List

import torch


def sincos_position_embed(
    embed_dim: int,
    spatial_dims: List[int],
) -> torch.Tensor:
    """Generate position embedding for spatial dimensions.

    Args:
        embed_dim (int): The size if embeded dimension C.
        spatial_dims (List[int]): The size of spatial dimensions D = [..., D_1, D_0].

    Returns:
        torch.Tensor: A (|D|, C) tensor embedding each position in D.
    """
    assert embed_dim % len(spatial_dims) == 0

    # a (|D|, *D) tensor where coords[k][d_0][...][d_{|D|-1}] = d_k.
    coords = [torch.arange(d) for d in reversed(spatial_dims)]
    coords = torch.stack(torch.meshgrid(*coords, indexing="ij"))

    # |D| embeddings of shape (|D|, C / |D|) along each dimension
    embed_dim //= len(spatial_dims)
    embs = [_sincos_position_embed_1d(embed_dim, pos) for pos in coords]

    # (|D|, C)
    emb = torch.concat(embs, dim=1)

    return emb


def _sincos_position_embed_1d(
    embed_dim: int,
    pos: torch.Tensor,
) -> torch.Tensor:
    """Generate position embedding along one spatial dimension.

    Args:
        embed_dim (int): The size if embeded dimension C.
        pos (torch.Tensor): A tensor of dimension D = [D_0, D_1, ...] with value
            representing the position of (d_0, d_1, ...) along the concerning dimension.

    Returns:
        torch.Tensor: A (|D|, C) tensor embedding each position in D.
    """
    assert embed_dim % 2 == 0

    # (C/2,)
    omega = torch.arange(1, embed_dim // 2 + 1, dtype=torch.float32)
    omega /= embed_dim // 2
    omega = 10000.0 ** (-omega)

    # (|D|,)
    pos = pos.reshape(-1)

    # (|D|, C/2)
    out = torch.einsum("m,d->md", pos, omega)
    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)

    # (|D|, C)
    # In the original ViT paper position embedding uses alternating sin and cos, which
    # differs only by a linear transformation with concating.
    emb = torch.concat([emb_sin, emb_cos], dim=1)

    return emb
