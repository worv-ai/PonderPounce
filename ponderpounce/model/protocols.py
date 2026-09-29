"""Protocols for PonderPounce VLA subsystem interfaces.

System 1 (reflexive): Handles observation encoding, cognition conditioning,
    and action decoding internally. The orchestrator passes raw pixel values
    + S2 cognitions, never VLM-specific tensors.

System 2 (deliberative): VLM backbone producing per-timestep cognition
    features (cognition). forward() signature varies by VLM architecture
    and is intentionally unconstrained.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class System1(Protocol):
    """System 1: reflexive action generation.

    Owns the full pipeline from raw pixel observations to action predictions.
    Each implementation encapsulates its own VLM encoder and action head.

    Attributes:
        action_dim: Action dimension per embodiment (e.g. 7 for robotics).
        action_chunk: Number of action steps predicted per tick (H).
    """

    action_dim: int
    action_chunk: int

    def to(self, *args, **kwargs) -> System1: ...
    def modules(self, remove_duplicate: bool = True) -> Iterator: ...
    def parameters(self, recurse: bool = True) -> Iterator[torch.nn.Parameter]: ...

    def predict_action(
        self,
        images: torch.Tensor,
        cognition: torch.Tensor,
        age_ms: torch.Tensor,
        proprio: torch.Tensor | None = ...,
    ) -> torch.Tensor:
        """Inference: generate actions.

        Args:
            cognition: (B, K, cognition_dim) S2 cognition tokens.
            age_ms: (B,) cognition age in milliseconds (0 for null/cold).

        Returns:
            (B, H, action_dim) predicted action chunk.
        """
        ...


@runtime_checkable
class System2(Protocol):
    """System 2: deliberative cognition generation.

    VLM backbone producing per-timestep cognition features (cognition).
    Output shape: (B, T, K, H) where T=cognition points, K=tokens, H=hidden.

    forward() signature is unconstrained -- it varies by VLM architecture.
    The orchestrator's _prepare_s2_inputs() must be co-updated with any
    S2 implementation swap.

    Return contract (documented, not type-enforced):
        - Standard: tuple (cognition, last_hidden) where
            cognition is (B, T, K, H) and last_hidden is (B, L, H) for
            downstream subgoal LM loss.
        - With DiT: dict {"cognition": Tensor, "dit_loss": Tensor}

    Attributes:
        hidden_size: Cognition hidden state dimension (H).
        num_cognition_tokens: Soft tokens per timestep (K).
    """

    hidden_size: int
    num_cognition_tokens: int
