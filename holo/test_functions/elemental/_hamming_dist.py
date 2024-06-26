import torch


def hamming_dist(x: torch.Tensor, y: torch.Tensor, dim: int = 0, keepdim: bool = False) -> torch.Tensor:
    return (x - y).pow(2).sum(dim=dim, keepdim=keepdim)
