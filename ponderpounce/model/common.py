"""Shared components for PonderPounce: modules used by both S1 and S2."""

import math

import torch
import torch.nn as nn
from torch import Tensor


def require_prompt_ids(adapter: str, ids: torch.Tensor | None) -> torch.Tensor:
    """Narrow ``Tensor | None → Tensor``; raise if a caller skipped the prompt contract.

    Shared by the PaliGemma-prompted S1 adapters (Pi0/Pi05) whose collator/inference
    path must supply ``symbolic_prompt_ids``.
    """
    if ids is None:
        raise ValueError(
            f"{adapter} adapter requires symbolic_prompt_ids (task prompt). Collator/inference must supply it."
        )
    return ids


def sinusoidal_embedding(
    values: torch.Tensor,
    dim: int,
    max_period: float = 10000.0,
    cos_first: bool = False,
    freqs: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sinusoidal positional embedding for arbitrary continuous values.

    Args:
        values: (...) arbitrary shape, continuous input values.
        dim: embedding dimension (must be even).
        max_period: controls frequency range. 100 = high-freq (flow matching timesteps),
                    10000 = low-freq (positional encoding, time-in-ms).
        cos_first: if True, output is [cos, sin] (SimVLA convention);
                   if False, output is [sin, cos] (transformer convention).
        freqs: pre-computed frequency tensor (dim//2,). If None, computed on the fly.
    Returns:
        (..., dim) embedding tensor, same dtype as values.
    """
    if freqs is None:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=values.device) / half
        )
    args = values.unsqueeze(-1).float() * freqs
    if cos_first:
        pe = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    else:
        pe = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    return pe.to(values.dtype)


class TimeTokenizer(nn.Module):
    """Sinusoidal encoding of time values in milliseconds → token embedding.

    Generic over the kind of "time" (cognition age, interval, etc.); learnable
    linear projection adapts the encoding to model needs.
    """

    def __init__(self, d_model: int, max_period: float = 10000.0):
        super().__init__()
        self.d_model = d_model
        self.max_period = max_period
        self.proj = nn.Linear(d_model, d_model)
        half = d_model // 2
        freqs = torch.exp(torch.arange(0, half, dtype=torch.float32) * -(math.log(max_period) / half))
        self.freqs: Tensor
        self.register_buffer("freqs", freqs)

    def forward(self, time_ms: torch.Tensor) -> torch.Tensor:
        """time_ms: (...) → (..., d_model)"""
        freqs: Tensor = self.freqs.to(time_ms.device)
        pe = sinusoidal_embedding(time_ms, self.d_model, freqs=freqs, cos_first=False)
        return self.proj(pe.to(self.proj.weight.dtype))
