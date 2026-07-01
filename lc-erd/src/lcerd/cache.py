from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class CachedTopK:
    indices: Tensor
    values: Tensor
    vocab_size: int

    def dense(self, fill_value: float | None = None) -> Tensor:
        if fill_value is None:
            fill_value = torch.finfo(self.values.dtype).min
        logits = torch.full(
            (*self.values.shape[:-1], self.vocab_size),
            fill_value,
            dtype=self.values.dtype,
            device=self.values.device,
        )
        return logits.scatter(-1, self.indices, self.values)


class TopKLogitCache:
    """Top-k compression for cached LLE logits, matching the paper's Eq. 16 idea."""

    def __init__(self, k: int):
        if k <= 0:
            raise ValueError("k must be positive")
        self.k = k

    def pack(self, logits: Tensor) -> CachedTopK:
        k = min(self.k, logits.shape[-1])
        values, indices = torch.topk(logits, k=k, dim=-1)
        return CachedTopK(indices=indices, values=values, vocab_size=logits.shape[-1])
