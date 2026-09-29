"""Gemma variant config registry — stub for ``openpi.models.gemma.get_config``.

The vendored ``pi05_pytorch.py`` reads variant names like ``"gemma_2b"`` and
``"gemma_300m"`` and expects a small struct with ``width``, ``depth``,
``mlp_dim``, ``num_heads``, ``num_kv_heads``, ``head_dim``. Mirrors openpi's
:func:`openpi.models.gemma.get_config` (Apache 2.0).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GemmaVariantConfig:
    width: int
    depth: int
    mlp_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int


# Reference: openpi/models/gemma.py:58-109.
_VARIANTS: dict[str, GemmaVariantConfig] = {
    "dummy": GemmaVariantConfig(width=64, depth=4, mlp_dim=128, num_heads=8, num_kv_heads=1, head_dim=16),
    "gemma_300m": GemmaVariantConfig(width=1024, depth=18, mlp_dim=4096, num_heads=8, num_kv_heads=1, head_dim=256),
    "gemma_2b": GemmaVariantConfig(width=2048, depth=18, mlp_dim=16_384, num_heads=8, num_kv_heads=1, head_dim=256),
}


def get_config(variant: str) -> GemmaVariantConfig:
    if variant not in _VARIANTS:
        raise ValueError(f"Unknown gemma variant: {variant!r}. Known: {sorted(_VARIANTS.keys())}")
    return _VARIANTS[variant]
