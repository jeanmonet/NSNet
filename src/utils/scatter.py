import torch


def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
    # From https://github.com/rusty1s/pytorch_scatter/
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand(other.size())
    return src


def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                out: torch.Tensor | None = None,
                dim_size: int | None = None) -> torch.Tensor:
    # From https://github.com/rusty1s/pytorch_scatter/
    index = broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
    return out.scatter_add_(dim, index, src)


def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                 out: torch.Tensor | None = None,
                 dim_size: int | None = None) -> torch.Tensor:

    index = broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
    # Using https://pytorch.org/docs/stable/generated/torch.Tensor.scatter_reduce_.html
    return out.scatter_reduce_(dim=dim, index=index, src=src, reduce="mean", include_self=False)

    # --- Prevous Implem from: ---
    # --- https://github.com/rusty1s/pytorch_scatter/blob/master/torch_scatter/scatter.py ---
    out = scatter_sum(src, index, dim, out, dim_size)
    dim_size = out.size(dim)

    index_dim = dim
    if index_dim < 0:
        index_dim = index_dim + src.dim()
    if index.dim() <= index_dim:
        index_dim = index.dim() - 1

    ones = torch.ones(index.size(), dtype=src.dtype, device=src.device)
    count = scatter_sum(ones, index, index_dim, None, dim_size)
    count[count < 1] = 1
    count = broadcast(count, out, dim)
    if out.is_floating_point():
        out.true_divide_(count)
    else:
        out.div_(count, rounding_mode='floor')
    return out


def scatter_logsumexp(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: torch.Tensor = None,
    dim_size: int | None = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    # Modifying implementation from
    #  https://github.com/rusty1s/pytorch_scatter/
    # to use torch.scatter_reduce and torch.scatter_add
    if not torch.is_floating_point(src):
        raise ValueError('`scatter_logsumexp` can only be computed over '
                         'tensors with floating point data types.')

    index = broadcast(index, src, dim)

    # if out is not None:
    #     dim_size = out.size(dim)
    # else:
    #     if dim_size is None:
    #        dim_size = int(index.max()) + 1
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
    else:
        dim_size = out.size(dim)



    size = list(src.size())
    size[dim] = dim_size
    max_value_per_index = torch.full(size, float('-inf'), dtype=src.dtype,
                                     device=src.device)

    # scatter_max(src, index, dim, max_value_per_index, dim_size=dim_size)[0]
    max_value_per_index.scatter_reduce_(dim=dim, index=index, src=src, reduce="amax", include_self=False)

    max_per_src_element = max_value_per_index.gather(dim, index)
    recentered_score = src - max_per_src_element
    recentered_score.masked_fill_(torch.isnan(recentered_score), float('-inf'))

    if out is not None:
        out = out.sub_(max_value_per_index).exp_()

    # sum_per_index = scatter_sum(recentered_score.exp_(), index, dim, out,
    #                             dim_size)
    sum_per_index = out.scatter_add_(src=recentered_score.exp_(), index=index, dim=dim)

    return sum_per_index.add_(eps).log_().add_(max_value_per_index)
