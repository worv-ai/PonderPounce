"""Runtime options for PonderPounce inference.

Inference-only - the ckpt config.json does not carry these. ``load_model`` /
``dispatcher_server`` CLI thread them through to ``System2``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

CompileMode = Literal["max-autotune", "reduce-overhead"]


@dataclass(frozen=True)
class InferenceOpts:
    """Inference-time knobs. ``None`` = no override / library default."""

    # S1 reflexive head compile mode. Reserved; not implemented.
    s1_compile_mode: CompileMode | None = None
    # S2 cached text-append compile mode. None keeps eager execution.
    s2_compile_mode: CompileMode | None = None
    # S2 StaticCache preallocation length. None tracks ``max_context_tokens``.
    s2_static_cache_len: int | None = None
    # Additive pre-softmax attention-logit bias on S1's cognition key positions.
    # None keeps the checkpoint config's value; a float overrides it (0.0 = off).
    s1_cognition_attn_bias: float | None = None

    def __post_init__(self) -> None:
        if self.s1_compile_mode is not None:
            raise NotImplementedError("s1_compile_mode override is reserved but not implemented yet")
        if self.s2_compile_mode not in (None, "max-autotune", "reduce-overhead"):
            raise ValueError(
                "s2_compile_mode must be one of ['max-autotune', 'reduce-overhead'] or None, "
                f"got {self.s2_compile_mode!r}"
            )
        if self.s1_cognition_attn_bias is not None and not isinstance(self.s1_cognition_attn_bias, (int, float)):
            raise TypeError(
                f"s1_cognition_attn_bias must be float or None, got {type(self.s1_cognition_attn_bias).__name__}"
            )
        # ``bool`` is a subclass of ``int``; reject before the ``<= 0`` numeric check.
        if self.s2_static_cache_len is not None:
            if isinstance(self.s2_static_cache_len, bool) or not isinstance(self.s2_static_cache_len, int):
                raise TypeError(
                    f"s2_static_cache_len must be int or None, got {type(self.s2_static_cache_len).__name__}"
                )
            if self.s2_static_cache_len <= 0:
                raise ValueError(f"s2_static_cache_len must be > 0 or None, got {self.s2_static_cache_len}")
