"""Checkpoint loading for :class:`PonderPounceVLA` — strict, tied-weight aware.

Builds the model from saved config, loads weights with key matching strict
modulo HF tied-weight dedup (so any layout drift fails loudly instead of
silently leaving half the weights at random init), and re-ties shared
parameters that ``safe_serialization`` dropped at save time.

Lives next to the other PonderPounce model code so both training (resume,
sanity-check) and inference (model server, examples) load checkpoints
through the same path. Mirrors the per-arch loading pattern.

Arch-specific re-tie behavior is delegated to the optional
``reflex_retie_embeddings`` hook the S1 adapter may expose; key remapping is
generic — the loader applies the model's own transformers weight-conversion
mapping, so this module names no concrete arch.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, cast

import torch
from accelerate import init_empty_weights
from safetensors.torch import load_file
from transformers import PreTrainedModel

if TYPE_CHECKING:
    from ponderpounce.inference.options import InferenceOpts


__all__ = ["load_model", "load_full_ponderpounce_weights", "collect_tied_keys"]


logger = logging.getLogger(__name__)


def collect_tied_keys(model: torch.nn.Module) -> set[str]:
    """Return every parameter key in the model that is tied to another.

    HF ``PreTrainedModel`` subclasses advertise their local tied keys
    via ``_tied_weights_keys``; walk the module tree and prefix each
    local key with the full dotted path so the set matches state_dict
    keys. Used to filter "missing" keys that were intentionally dropped
    on ``safe_serialization`` save.
    """
    tied: set[str] = set()
    for name, submod in model.named_modules():
        local_keys = getattr(submod, "_tied_weights_keys", None) or []
        for lk in local_keys:
            tied.add(f"{name}.{lk}" if name else lk)
    return tied


def _retie_s1_embeddings(model: torch.nn.Module) -> None:
    """Re-tie shared S1 embed/lm_head storage after a strict full-model load.

    Calls the optional ``reflex_retie_embeddings`` hook on any S1 child that
    exposes it. No-op for arches whose backbone needs no re-tie."""
    found = False
    for submod in model.modules():
        hook = getattr(submod, "reflex_retie_embeddings", None)
        if callable(hook):
            hook()
            found = True
    if not found:
        logger.warning(
            "tie_weights() fallback: no reflex_retie_embeddings hook found; "
            "shared embeddings may remain decoupled from lm_head"
        )


def _remap_state_dict(
    state_dict: dict[str, torch.Tensor],
    model: torch.nn.Module,
) -> dict[str, torch.Tensor]:
    """Apply the model's own transformers weight-conversion renamings.

    ``save_pretrained`` writes checkpoints in the (legacy-)canonical key
    format via ``revert_weight_conversion``; ``from_pretrained`` converts
    back with the same per-submodule mapping. This raw fast-path loader
    bypasses ``from_pretrained``, so it must apply the identical renamings —
    ``get_model_conversion_mapping`` walks the actual module tree and scopes
    each rule, which keeps the load correct regardless of how the installed
    transformers nests sub-modules. A rename is kept only when it lands on a
    real key of the live model (scoped rules make false positives
    implausible, and unknown keys stay untouched to fail loud downstream).
    """
    target_keys = set(model.state_dict().keys())
    unexpected = {k for k in state_dict if k not in target_keys}
    if not unexpected:
        return state_dict

    from transformers.conversion_mapping import WeightConverter, WeightRenaming, get_model_conversion_mapping
    from transformers.core_model_loading import rename_source_key

    # ``model`` is typed as bare nn.Module for the generic signature; the
    # mapping walk only reads module classes, so a cast is safe.
    transforms = get_model_conversion_mapping(cast("PreTrainedModel", model))
    renamings = [t for t in transforms if isinstance(t, WeightRenaming)]
    converters = [t for t in transforms if isinstance(t, WeightConverter)]
    if converters:
        # Many-to-many weight ops cannot be expressed as a key rename; none of
        # our architectures declare them. Fail loud rather than half-convert.
        raise RuntimeError(
            f"model declares {len(converters)} WeightConverter transforms; the raw "
            "fast-path loader only supports pure renamings — load via from_pretrained"
        )

    remapped: dict[str, torch.Tensor] = {}
    resolved = 0
    for k, v in state_dict.items():
        new_key = k
        if k in unexpected:
            candidate, _ = rename_source_key(k, renamings, [])
            if candidate in target_keys:
                new_key = candidate
                resolved += 1
        remapped[new_key] = v
    if resolved:
        logger.info(
            "Key remap: resolved %d/%d unexpected keys via the model's weight-conversion mapping",
            resolved,
            len(unexpected),
        )
    return remapped


def _resolve_weights_path(checkpoint_path: str) -> str:
    """Pick ``model.safetensors`` or ``pytorch_model.bin`` under
    ``checkpoint_path``. Raises if neither exists."""
    safetensors_path = os.path.join(checkpoint_path, "model.safetensors")
    pytorch_bin_path = os.path.join(checkpoint_path, "pytorch_model.bin")
    if os.path.exists(safetensors_path):
        return safetensors_path
    if os.path.exists(pytorch_bin_path):
        return pytorch_bin_path
    raise FileNotFoundError(
        f"No model weights found at {checkpoint_path} (expected model.safetensors or pytorch_model.bin)"
    )


def load_full_ponderpounce_weights(
    model: PreTrainedModel,
    checkpoint_path: str,
    *,
    device: torch.device | str | None = None,
    allow_missing_prefixes: tuple[str, ...] = (),
) -> None:
    """In-place: load a full PonderPounceVLA state_dict + tie weights.

    Does not build the model and does not move dtype/device. Used by
    ``load_model`` (after empty-init build) AND by setup factories when
    chaining stages (stage 3 → 4): each stage builds the model via
    ``build_components`` with its own freeze mask, so the prior stage's
    optimizer state isn't reusable, but the param weights are.

    ``device`` only affects how safetensors are read — pass a CUDA
    device to load directly to GPU (skips a CPU staging copy). For CPU
    or meta builds, leave it None.

    ``allow_missing_prefixes`` names key prefixes that may be absent from the
    checkpoint without failing, for freshly-init modules added after the
    checkpoint was saved (e.g. the read-out aux heads when warm-starting a
    pre-aux checkpoint). Default is empty: fully strict.
    """
    weights_path = _resolve_weights_path(checkpoint_path)
    if weights_path.endswith(".safetensors"):
        target_device = torch.device(device) if device is not None else None
        load_kwargs: dict[str, Any] = (
            {"device": str(target_device)} if target_device is not None and target_device.type == "cuda" else {}
        )
        state_dict = load_file(weights_path, **load_kwargs)
    else:
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)

    if not state_dict:
        raise RuntimeError(
            f"Checkpoint at {checkpoint_path} is empty — cannot load a model "
            f"from zero keys. For a trained checkpoint, point at the real "
            f"training output; for local smoke-testing only, regenerate with "
            f"``scripts/dev/save_scratch_checkpoint.py``."
        )

    # Auto-detect transformers version mismatch: checkpoints saved with an
    # older transformers may have ``language_model.model.layers.X`` while the
    # live model expects ``language_model.layers.X`` (or vice versa). The S1
    # adapter owns the per-key fixup via its remap hook.
    state_dict = _remap_state_dict(state_dict, model)

    tied_keys = collect_tied_keys(model)
    result = model.load_state_dict(state_dict, strict=False, assign=True)
    try:
        model.tie_weights()
    except AttributeError:
        # ``model.tie_weights()`` walks the whole module tree; if any
        # submodule along the way raises (e.g. an action expert that sets
        # ``embed_tokens = None`` to save params, so its
        # ``_tied_weights_keys`` lookup AttributeErrors), the walk aborts
        # before the backbone's tie runs. Trigger the per-submodule tie
        # directly so the shared ``embed_tokens.weight`` ends up shared with
        # ``lm_head.weight`` instead of staying at the random init the base
        # safetensors load left behind.
        logger.warning("tie_weights() failed; running per-submodule tie")
        _retie_s1_embeddings(model)
    # str.startswith(()) is always False, so the empty default stays strict.
    real_missing = [k for k in result.missing_keys if k not in tied_keys and not k.startswith(allow_missing_prefixes)]
    if real_missing or result.unexpected_keys:
        raise RuntimeError(
            "Checkpoint key mismatch against current model layout. "
            f"Missing (non-tied): {real_missing[:10]}"
            f"{' ...' if len(real_missing) > 10 else ''}. "
            f"Unexpected: {result.unexpected_keys[:10]}"
            f"{' ...' if len(result.unexpected_keys) > 10 else ''}. "
            "Regenerate with ``scripts/dev/save_scratch_checkpoint.py`` for "
            "smoke tests, or point at a trained checkpoint saved with the "
            "current model layout."
        )
    logger.info(f"Loaded PonderPounceVLA weights from {checkpoint_path}")


def load_model(
    checkpoint_path: str,
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.bfloat16,
    inference_opts: "InferenceOpts | None" = None,
) -> Any:
    """Load a trained ``PonderPounceVLA`` from ``checkpoint_path``.

    Strict modulo tied-weight dedup: missing keys must correspond to
    entries in the model's ``_tied_weights_keys`` tree; unexpected
    keys are always a failure. Anything else is a layout-drift bug and
    must not silently half-load the model.

    ``inference_opts`` applies inference-time knobs (compile mode, cache
    size) post-load. Training callers omit it; the dispatcher / eval
    scripts thread their CLI flags in here.
    """
    # Lazy imports so module-level loading doesn't pull in heavy deps
    # for the common case where only types are needed.
    from ponderpounce.model.configuration import PonderPounceConfig
    from ponderpounce.model.modeling import PonderPounceVLA

    config = PonderPounceConfig.from_pretrained(checkpoint_path)

    # Resolve weights file before the heavy ``PonderPounceVLA(config)`` init
    # below. transformers 4.45+ ``from_pretrained`` silently returns a
    # default-populated config on missing local ``config.json``
    # (intentional Hub-friendly feature, huggingface/transformers#32356).
    # Without this ordering, a wrong checkpoint path would spend minutes
    # random-initialising a default multi-billion-param model on CPU
    # before the weights check ever fires.
    _resolve_weights_path(checkpoint_path)

    with init_empty_weights():
        model = PonderPounceVLA(config)

    target_device = torch.device(device)
    load_full_ponderpounce_weights(model, checkpoint_path, device=target_device)

    meta_remaining = [n for n, p in model.named_parameters() if p.is_meta]
    if meta_remaining:
        raise RuntimeError(
            f"{len(meta_remaining)} parameter(s) still on the meta device after load + tie: {meta_remaining[:5]}."
        )

    # Move params + buffers to the device — non-persistent buffers (rotary
    # inv_freq) init on CPU under init_empty_weights; .to() self-guards
    # already-placed tensors. dtype is cast on parameters only, so buffers
    # keep their natural precision (float32 rotary inv_freq, as HF loads it).
    model = model.to(device=target_device)
    if any(p.dtype != dtype for p in model.parameters()):
        model = model.to(dtype=dtype)
    if inference_opts is not None:
        _apply_inference_opts(model, inference_opts)
    return model


def _apply_inference_opts(model: Any, opts: "InferenceOpts") -> None:
    """Mutate ``model.system2`` per ``opts``. Validate fully before touching state.

    ``s2_static_cache_len`` controls the resident append-context KV capacity.
    ``s2_compile_mode`` is an opt-in compile mode for cached text appends.
    Validation happens before mutation so bad CLI values cannot leave a
    half-applied model.
    """
    s2 = model.system2
    cache_len: int | None = None
    if opts.s2_static_cache_len is not None:
        cache_len = int(opts.s2_static_cache_len)
        max_ctx = int(s2._max_context_tokens)
        if not 0 < cache_len <= max_ctx:
            raise ValueError(f"s2_static_cache_len must be in (0, max_context_tokens={max_ctx}], got {cache_len}")
    bias = float(opts.s1_cognition_attn_bias) if opts.s1_cognition_attn_bias is not None else None
    if bias is not None and not hasattr(model.system1, "cognition_attn_bias"):
        raise ValueError(
            f"s1_cognition_attn_bias override is only supported for S1 adapters exposing "
            f"`cognition_attn_bias` (got {type(model.system1).__name__})"
        )
    if cache_len is not None:
        s2._s2_static_cache_len = cache_len
    s2.configure_inference_compile(opts.s2_compile_mode)
    if bias is not None:
        model.system1.cognition_attn_bias = bias
        model.config.s1_cognition_attn_bias = bias  # keep recorded config truthful
