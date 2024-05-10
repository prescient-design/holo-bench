import math
from typing import Optional

import torch


def motif_search(
    solution: torch.LongTensor,
    motif: torch.LongTensor,
    spacing: Optional[torch.LongTensor] = None,
    mode: str = "present",
    quantization: Optional[int] = None,
):
    """
    Check if a spaced motif is present in a solution.
    The elements of the motif should be spaced according to `spacing`.
    If spacing is not provided, it is assumed to be 1.
    If `mode` is "count", the number of motifs in the solution is returned.
    If `strict` is True, only motifs that are fully satisfied are counted, otherwise
        the fraction of the motif that is satisfied is returned.
    """
    motif_size = motif.size(-1)

    if quantization is None:
        quantization = motif_size

    if spacing is None:
        size = solution.size()[:-1]
        spacing = torch.ones(
            *size,
            motif_size - 1,
            dtype=torch.long,
            device=solution.device,
        )
    else:
        size = solution.size()[:-1]
        spacing = spacing.expand(*size, -1)

    # convert spacing into index tensor for gather
    base_index = torch.cat([torch.zeros_like(spacing[..., 0]).unsqueeze(-1), spacing.cumsum(-1)], dim=-1)

    # slide indices out to maximum length
    max_base = base_index.max()
    num_steps = solution.size(-1) - max_base
    index_delta = torch.arange(num_steps, device=solution.device)
    expand_args = [1] * len(base_index.shape)
    index_delta = index_delta.view(-1, *expand_args)
    index = base_index + index_delta

    # gather solution subsequences
    gathered = torch.stack([torch.gather(solution, -1, step) for step in index])

    # check if motif is present
    is_equal = gathered.eq(motif)
    present_count = is_equal.float().sum(-1)

    # if strict, only count motifs that are fully satisfied
    # if strict:
    #     frac_satisfied = frac_satisfied.eq(1.0).float()
    quant_factor_1 = math.ceil(motif_size / quantization)
    quant_factor_2 = motif_size / quant_factor_1
    quantized_count = present_count // quant_factor_1 / quant_factor_2

    if mode == "count":
        return quantized_count.sum(0)

    return quantized_count.max(0).values
