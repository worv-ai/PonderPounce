"""In-process implementations of the three system callers.

- ``LocalSystem1`` / ``LocalSystem2`` wrap a ``PonderPounceVLA`` and own the
  ``.to(device)`` / ``.to(cpu)`` shuffle so the Caller contract (CPU
  tensors across the boundary) is preserved. ``LocalSystem2`` constructs
  append-only per-session contexts; each context owns its resident KV cache.
"""

from __future__ import annotations

import torch

from ponderpounce.inference.append_context import SoftS2SessionContext
from ponderpounce.inference.session import Session

__all__ = ["LocalSystem1", "LocalSystem2"]


class LocalSystem1:
    """In-process ``System1Caller`` backed by a ``PonderPounceVLA``.

    The ``PonderPounceVLA.null_cognition`` parameter is a top-level
    attribute (not on ``.system1``); we hold a reference to the whole
    model so we can expose it."""

    def __init__(
        self,
        model,
        device: torch.device,
        dtype: torch.dtype,
        *,
        s1_max_batch_size: int = 16,
        num_inference_steps: int = 10,
    ) -> None:
        """Wrap ``model.system1`` for in-process inference.

        ``num_inference_steps`` is the flow-matching Euler-step count
        forwarded to ``self._s1.predict_action(num_steps=...)``.
        """
        self._model = model
        self._device = torch.device(device)
        self._dtype = dtype
        if s1_max_batch_size < 1:
            raise ValueError(f"s1_max_batch_size must be >= 1, got {s1_max_batch_size}")
        if num_inference_steps < 1:
            raise ValueError(f"num_inference_steps must be >= 1, got {num_inference_steps}")
        self._s1_max_batch_size = int(s1_max_batch_size)
        self._num_inference_steps = int(num_inference_steps)
        # Move the submodule to our configured device + dtype so split-device deployments (``LocalSystem1`` on
        # cuda:1 while the model was loaded on cuda:0) don't crash with a cross-device tensor error. Idempotent if
        # already on the right device.
        self._s1 = model.system1.to(self._device, self._dtype).eval()
        self._null_cognition = model.null_cognition.detach().clone().to("cpu", torch.float32)
        # Resolved once: archs exposing ``noise_shape`` (Pi05) take the
        # service's per-session flow-matching noise; others sample their own.
        self._noise_spec: tuple[int, int] | None = getattr(self._s1, "noise_shape", None)

    @property
    def s1_max_batch_size(self) -> int:
        return self._s1_max_batch_size

    @property
    def null_cognition(self) -> torch.Tensor:
        return self._null_cognition

    @property
    def action_chunk_size(self) -> int:
        return int(self._s1.action_chunk)

    @property
    def action_dim(self) -> int:
        return int(self._s1.action_dim)

    @property
    def noise_spec(self) -> tuple[int, int] | None:
        """``(action_horizon, noise_feature_dim)`` if this arch takes the
        service's per-session flow-matching noise; ``None`` if it samples its own."""
        return self._noise_spec

    def reset_session(self, session_id: str) -> None:
        # No-op: S1 carries no per-session state. Method exists for symmetry
        # with System2Caller so engines can call it without branching.
        del session_id

    def predict_action(
        self,
        *,
        images: torch.Tensor,
        cognition: torch.Tensor,
        age_ms: torch.Tensor,
        proprio: torch.Tensor | None,
        noise: torch.Tensor | None = None,
        symbolic_prompt_ids: torch.Tensor | None = None,
        symbolic_prompt_mask: torch.Tensor | None = None,
        pixel_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B = int(images.shape[0])
        if B == 0:
            return torch.zeros(0, self.action_chunk_size, self.action_dim, dtype=torch.float32)
        cap = self._s1_max_batch_size

        def _slice(t: torch.Tensor | None, lo: int, hi: int) -> torch.Tensor | None:
            return None if t is None else t[lo:hi]

        if B <= cap:
            return self._predict_chunk(
                images=images,
                cognition=cognition,
                age_ms=age_ms,
                proprio=proprio,
                noise=noise,
                symbolic_prompt_ids=symbolic_prompt_ids,
                symbolic_prompt_mask=symbolic_prompt_mask,
                pixel_attention_mask=pixel_attention_mask,
            )
        outs: list[torch.Tensor] = []
        for start in range(0, B, cap):
            end = min(B, start + cap)
            outs.append(
                self._predict_chunk(
                    images=images[start:end],
                    cognition=cognition[start:end],
                    age_ms=age_ms[start:end],
                    proprio=_slice(proprio, start, end),
                    noise=_slice(noise, start, end),
                    symbolic_prompt_ids=_slice(symbolic_prompt_ids, start, end),
                    symbolic_prompt_mask=_slice(symbolic_prompt_mask, start, end),
                    pixel_attention_mask=_slice(pixel_attention_mask, start, end),
                )
            )
        return torch.cat(outs, dim=0)

    def _predict_chunk(
        self,
        *,
        images: torch.Tensor,
        cognition: torch.Tensor,
        age_ms: torch.Tensor,
        proprio: torch.Tensor | None,
        noise: torch.Tensor | None = None,
        symbolic_prompt_ids: torch.Tensor | None = None,
        symbolic_prompt_mask: torch.Tensor | None = None,
        pixel_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        def _to(t: torch.Tensor | None, dtype: torch.dtype | None = None) -> torch.Tensor | None:
            if t is None:
                return None
            return t.to(self._device, dtype) if dtype is not None else t.to(self._device)

        # Noise / prompt / pixel kwargs are arch-specific: the flow-matching ODE
        # noise stays float32 (move device only), and the symbolic-prompt + pixel
        # masks exist only on the hard-channel adapters (pi0/pi05). Pass each only
        # when present, so non-hard-channel adapters (simvla/starvla, whose
        # predict_action has no such params) don't get unexpected kwargs.
        extra: dict[str, torch.Tensor] = {}
        if noise is not None:
            extra["noise"] = noise.to(self._device)
        for name, value in (
            ("symbolic_prompt_ids", symbolic_prompt_ids),
            ("symbolic_prompt_mask", symbolic_prompt_mask),
            ("pixel_attention_mask", pixel_attention_mask),
        ):
            moved = _to(value)
            if moved is not None:
                extra[name] = moved

        with torch.no_grad():
            out = self._s1.predict_action(
                images=images.to(self._device, self._dtype),
                cognition=cognition.to(self._device, self._dtype),
                age_ms=age_ms.to(self._device, self._dtype),
                proprio=proprio.to(self._device, self._dtype) if proprio is not None else None,
                num_steps=self._num_inference_steps,
                **extra,
            )
        return out.detach().to("cpu", torch.float32).contiguous()


class LocalSystem2:
    """In-process ``System2Caller`` over ``PonderPounceVLA.system2``.

    One append-only context per session owns its KV cache directly. This backend
    only constructs those contexts.
    """

    def __init__(
        self,
        model,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self._device = torch.device(device)
        self._dtype = dtype
        # Same submodule-to-device dance as LocalSystem1 so split-device deployments work even if ``load_model``
        # placed the model on a different device.
        self._s2 = model.system2.to(self._device, self._dtype).eval()
        self._hidden_size = int(self._s2.hidden_size)
        self._num_cognition_tokens = int(self._s2.cognition_embed.shape[0])

    @property
    def pad_token_id(self) -> int:
        return int(self._s2.tokenizer.pad_token_id or 0)

    @property
    def subgoal_trigger_id(self) -> int:
        return int(self._s2._subgoal_trigger_id)

    @property
    def num_cognition_tokens(self) -> int:
        return self._num_cognition_tokens

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    @property
    def cognition_token_id(self) -> int:
        return int(self._s2._cognition_token_id)

    @property
    def max_context_tokens(self) -> int:
        """Effective inference cap = min(``max_context_tokens``, ``s2_static_cache_len``).

        Append contexts enforce this before mutating their resident cache.
        """
        return self._s2.effective_context_cap

    def reset_session(self, session_id: str) -> None:
        del session_id

    def create_context(
        self,
        session: Session,
        *,
        max_new_tokens: int,
        convert_s2_coords: bool = False,
    ) -> SoftS2SessionContext:
        del convert_s2_coords
        return SoftS2SessionContext(
            session=session,
            system2=self._s2,
            device=self._device,
            dtype=self._dtype,
            max_new_tokens=max_new_tokens,
        )
