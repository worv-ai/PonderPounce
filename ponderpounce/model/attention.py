"""Isolated cognition attention mask for PonderPounce.

Every token attends causally; cog tokens are additionally hidden from
all future tokens (obs or cog) and from cog tokens of other queries.

  1. obs→obs: causal (any past obs).
  2. cog→obs: causal (any past obs — includes same-query past obs and
     past queries' obs/subgoal text).
  3. cog→cog: causal AND same-query only (a cog token sees same-query
     past cog tokens, never cross-query cog; past queries' cog is
     private to the time it was produced).
  4. nothing → cog from any other position (i.e. cog is invisible to
     every future obs and every future / cross-query cog).

       s₁  d₁  s₂  d₂  s₃  d₃
 s₁  [ ✓   ✗   ✗   ✗   ✗   ✗ ]
 d₁  [ ✓   ✓   ✗   ✗   ✗   ✗ ]
 s₂  [ ✓   ✗   ✓   ✗   ✗   ✗ ]
 d₂  [ ✓   ✗   ✓   ✓   ✗   ✗ ]
 s₃  [ ✓   ✗   ✓   ✗   ✓   ✗ ]
 d₃  [ ✓   ✗   ✓   ✗   ✓   ✓ ]

``s_t`` rows show obs cannot attend to any cog (✗ at d-columns). ``d_t``
rows attend to same-query past cog (e.g. d₂ → d₂ via the diagonal) and
all past obs (any q), but not to cog of other queries.

The mask is causal end-to-end — there is no bidirectional
attention anywhere. The earlier "bidirectional within cog chunk"
variant was inconsistent with the linear-attention (GatedDeltaNet)
layers of the hybrid Qwen3.5 backbone (recurrent state cannot give
within-chunk bidirectional info-flow), so we keep the contract uniform
across layer types.
"""

from __future__ import annotations

import torch


def build_4d_attention_mask(
    cognition_mask: torch.Tensor,
    s2_query_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build an explicit 4D attention mask for the isolated-cognition pattern.

    Args:
        cognition_mask: (B, S) bool — True for cognition token positions.
        s2_query_ids:   (B, S) long — S2 query index each token belongs to
                                       (``-1`` for prefix / padding).
        attention_mask: (B, S) — 1 for real tokens, 0 for padding.
        dtype: target dtype.

    Returns:
        (B, 1, S, S) float mask — 0.0 = attend, -inf = masked.
    """
    obs_mask = ~cognition_mask

    q_is_obs = obs_mask.unsqueeze(-1)
    k_is_obs = obs_mask.unsqueeze(-2)
    q_is_cog = cognition_mask.unsqueeze(-1)
    k_is_cog = cognition_mask.unsqueeze(-2)

    q_ids = s2_query_ids.unsqueeze(-1)
    k_ids = s2_query_ids.unsqueeze(-2)

    S = s2_query_ids.shape[-1]
    positions = torch.arange(S, device=s2_query_ids.device)
    q_pos = positions.unsqueeze(-1)
    k_pos = positions.unsqueeze(-2)

    causal = k_pos <= q_pos
    same_query = q_ids == k_ids
    valid_k = attention_mask.bool().unsqueeze(-2)

    obs_to_obs = q_is_obs & k_is_obs & causal
    cog_to_obs = q_is_cog & k_is_obs & causal
    cog_to_cog = q_is_cog & k_is_cog & causal & same_query

    allowed = (obs_to_obs | cog_to_obs | cog_to_cog) & valid_k

    mask = torch.zeros(allowed.shape, dtype=dtype, device=allowed.device)
    mask.masked_fill_(~allowed, float("-inf"))
    return mask.unsqueeze(1)


__all__ = ["build_4d_attention_mask"]
