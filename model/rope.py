import torch
from typing import Tuple


def get_meshgrid_nd(sizes):
    """
    Get n-D meshgrid with given sizes.

    Args:
        sizes: Sizes of the grid along all dimensions.

    Returns:
        grid: The grid.
    """
    dim = len(sizes)
    start = (0,) * dim
    
    axis_grid = []
    for i in range(dim):
        a, n = start[i], sizes[i]
        g = torch.linspace(a, n, n + 1, dtype=torch.float32)[:n]
        axis_grid.append(g)
    grid = torch.meshgrid(*axis_grid, indexing="ij")  # dim x [W, H, D]
    grid = torch.stack(grid, dim=0)  # [dim, W, H, D]

    return grid


def apply_rope(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    Args:
        xq: Query tensor to apply rotary embeddings. [B, S, H, D]
        xk: Key tensor to apply rotary embeddings.   [B, S, H, D]
        freqs_cis: Precomputed frequency tensor for complex exponential.

    Returns:
        xq_out, xk_out: Tuple of modified query tensor and key tensor with rotary embeddings.
    """
    # view_as_complex will pack [..., D/2, 2](real) to [..., D/2](complex)
    xq_ = torch.view_as_complex(
        xq.reshape(*xq.shape[:-1], -1, 2)
    )  # [B, S, H, D//2]
    freqs_cis = freqs_cis[None, :, None, :]  # [1, S, 1, D//2]
    # view_as_real will expand [..., D/2](complex) to [..., D/2, 2](real)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3).type_as(xq)
    xk_ = torch.view_as_complex(
        xk.float().reshape(*xk.shape[:-1], -1, 2)
    )  # [B, S, H, D//2]
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3).type_as(xk)

    return xq_out, xk_out


def get_nd_rope(
    dim_list,
    sizes,
    theta=10000.0,
):
    """
    Get n-D rope as a complex Tensor.

    Args:
        dim_list: Dimension of each rope. len(dim_list) should equal to n.
            sum(dim_list) should equal to head_dim of attention layer.
        sizes: Sizes of the latent along all dimensions.
        theta: Scaling factor for frequency computation.

    Returns:
        emb: Positional embedding [S, D/2]
    """
    grid = get_meshgrid_nd(sizes)  # [n, T, H, W]

    # use 1/ndim of dimensions to encode grid_axis
    embs = []
    for i in range(len(dim_list)):
        emb = get_1d_rope(
            dim_list[i],
            grid[i].reshape(-1),
            theta,
        )  # [THW, D_i/2]
        embs.append(emb)

    emb = torch.cat(embs, dim=1)  # (THW, D/2)
    return emb


def get_1d_rope(
    dim: int,
    pos: torch.Tensor,
    theta: float = 10000.0,
) -> torch.Tensor:
    """
    Get 1D rope as a complex Tensor.

    Args:
        dim: Dimension of the frequency tensor.
        pos: Position indices for the frequency tensor.
        theta: Scaling factor for frequency computation.

    Returns:
        freqs_cis: Precomputed frequency tensor with complex exponential. [S, D/2]
    """
    freqs = 1.0 / (
        theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)
    )  # [D/2]
    freqs = torch.outer(pos, freqs)  # [S, D/2]
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # [S, D/2]
    return freqs_cis
