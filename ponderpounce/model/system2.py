"""System 2 (Ponder): the Qwen3.5 VLM that turns the episode context into cognition tokens.

S2 sequence: ``[Text, o₁, d₁, o₂, d₂, ..., o_T, d_T]`` — instruction, then per
query one observation and one cognition carrier token (plus subgoal text on
transitions). Inference lives in ``ponderpounce.inference.append_context``:
it uses ``_build_inputs_embeds`` / ``_run_cached_language_model`` here and owns
the append-only StaticCache per session.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any, Literal

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForImageTextToText, AutoTokenizer, PreTrainedTokenizerBase
from transformers.cache_utils import StaticCache

from ponderpounce.constants import resolve_cognition_token_id, resolve_subgoal_trigger_id

logger = logging.getLogger(__name__)

S2CompileMode = Literal["max-autotune", "reduce-overhead"]
_VALID_S2_COMPILE_MODES: tuple[S2CompileMode, ...] = ("max-autotune", "reduce-overhead")


def _validate_s2_compile_mode(mode: S2CompileMode | None) -> S2CompileMode | None:
    if mode is None:
        return None
    if mode not in _VALID_S2_COMPILE_MODES:
        raise ValueError(f"s2_compile_mode must be one of {sorted(_VALID_S2_COMPILE_MODES)} or None, got {mode!r}")
    return mode


def _static_layer_max_cache_len(cache: StaticCache) -> int:
    """The full-attn key-axis width of a ``StaticCache``.

    Linear-attn layers carry no ``max_cache_len``; read off the first
    ``StaticLayer`` instead.
    """
    for layer in cache.layers:
        if hasattr(layer, "max_cache_len"):
            return int(layer.max_cache_len)
    raise RuntimeError("StaticCache has no full-attention StaticLayer to read max_cache_len from")


def _snapshot_cache_state(cache: StaticCache) -> list:
    """Capture each layer's mutable state so a forward can probe the cache and
    roll back to here.

    Full-attn ``StaticLayer``: ``cumulative_length`` (slots past it get
    overwritten). Linear-attn: ``conv_states``/``recurrent_states`` can't be
    rolled back by truncation, so they're cloned (fixed-size, not the 16K KV).
    """
    snap: list = []
    for layer in cache.layers:
        if hasattr(layer, "keys"):
            cum = getattr(layer, "cumulative_length", None)
            snap.append(("full", int(cum.item()) if isinstance(cum, torch.Tensor) else 0))
        else:
            conv = layer.conv_states.clone() if getattr(layer, "is_conv_states_initialized", False) else None  # type: ignore[attr-defined]
            rec = layer.recurrent_states.clone() if getattr(layer, "is_recurrent_states_initialized", False) else None  # type: ignore[attr-defined]
            snap.append(("linear", conv, rec, bool(getattr(layer, "has_previous_state", False))))
    return snap


def _restore_cache_state(cache: StaticCache, snap: list) -> None:
    """Restore a :func:`_snapshot_cache_state` capture in place."""
    for layer, s in zip(cache.layers, snap, strict=True):
        if s[0] == "full":
            cum = getattr(layer, "cumulative_length", None)
            if isinstance(cum, torch.Tensor):
                cum.fill_(s[1])
        else:
            _, conv, rec, has_prev = s
            if conv is not None:
                layer.conv_states.copy_(conv)  # type: ignore[attr-defined]
            if rec is not None:
                layer.recurrent_states.copy_(rec)  # type: ignore[attr-defined]
            layer.has_previous_state = has_prev  # type: ignore[attr-defined]


class ContextOverflowError(RuntimeError):
    """Raised when an append-only inference context would exceed its cap."""

    def __init__(self, *, length: int, cap: int) -> None:
        super().__init__(f"S2 context length {length} exceeds cap {cap}")
        self.length = length
        self.cap = cap


class System2(nn.Module):
    """Qwen3.5 VLM that produces cognition tokens from observations."""

    def __init__(
        self,
        vlm_model_id: str = "Qwen/Qwen3.5-4B",
        num_cognition_tokens: int = 1,
        isolated_cognition_attention: bool = False,
        max_context_tokens: int = 16384,
        current_obs_only: bool = False,
        current_obs_chunk_tokens: int = 4096,
        s2_static_cache_len: int | None = None,
        s2_compile_mode: S2CompileMode | None = None,
    ):
        super().__init__()
        # Install isolation patches before ``from_pretrained`` so the model is built with patched
        # forwards. Patches are no-ops unless ``_cog_mask`` is stashed at runtime.
        if isolated_cognition_attention:
            from .qwen3_5_patches import apply_patches

            apply_patches()
        self._isolate_cog = isolated_cognition_attention
        # Read by the inference append context (rewind-to-prefix per fire); the
        # training side is shaped by the collator and needs no model change.
        self.current_obs_only = bool(current_obs_only)
        self.current_obs_chunk_tokens = int(current_obs_chunk_tokens)
        if max_context_tokens <= 0:
            raise ValueError(f"max_context_tokens must be > 0, got {max_context_tokens}")
        self._max_context_tokens = int(max_context_tokens)
        if s2_static_cache_len is None:
            s2_static_cache_len = max_context_tokens
        if not 0 < s2_static_cache_len <= max_context_tokens:
            raise ValueError(f"s2_static_cache_len must be in (0, max_context_tokens], got {s2_static_cache_len}")
        self._s2_static_cache_len = int(s2_static_cache_len)
        self._s2_compile_mode = _validate_s2_compile_mode(s2_compile_mode)
        # Keep the compiled callable out of nn.Module registration/state_dict.
        self.__dict__["_s2_compiled_cached_language_model"] = None

        # Architecture only: the weights come from the PonderPounce checkpoint
        # (``ponderpounce.model.loading.load_model``), so the base
        # VLM weights are never downloaded. ``vlm.lm_head`` decodes subgoal text;
        # the multimodal path lives on ``vlm.model``.
        self.vlm = AutoModelForImageTextToText.from_config(
            AutoConfig.from_pretrained(vlm_model_id), torch_dtype=torch.bfloat16
        )
        self.hidden_size = self.vlm.config.text_config.hidden_size
        self.num_cognition_tokens = num_cognition_tokens

        # Plain attribute (not a child Module) so it stays out of state_dict.
        tokenizer = AutoTokenizer.from_pretrained(vlm_model_id)
        assert tokenizer is not None, f"AutoTokenizer returned None for {vlm_model_id!r}"
        self.tokenizer: PreTrainedTokenizerBase = tokenizer

        # Fail-fast on a tokenizer missing either control token.
        self._subgoal_trigger_id: int = resolve_subgoal_trigger_id(tokenizer)
        self._cognition_token_id: int = resolve_cognition_token_id(tokenizer)

        # Learnable cognition embeddings (K tokens, shared across S2 queries).
        self.cognition_embed = nn.Parameter(
            torch.randn(num_cognition_tokens, self.hidden_size, dtype=torch.bfloat16) * 0.02
        )

        # Pre-resolve linear-attn mixers as a tuple: avoids per-forward layer
        # walks, and the tuple form sidesteps nn.Module's auto-registration
        # (they're already submodules of self.vlm).
        if isolated_cognition_attention:
            from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet, Qwen3_5TextModel

            lm = self.vlm.model.language_model
            assert isinstance(lm, Qwen3_5TextModel), (
                f"isolated_cognition_attention=True requires a Qwen3.5 backbone "
                f"(Qwen3_5TextModel), got {type(lm).__name__}. Set the flag to false for this config."
            )
            # Keyed on config.layer_types: the DecoderLayer attr was renamed
            # layer_type -> block_type in transformers 5.13.
            layer_types = lm.config.layer_types
            assert layer_types is not None, "Qwen3_5TextConfig always populates layer_types"
            mixers = []
            for layer, layer_type in zip(lm.layers, layer_types, strict=True):
                if layer_type != "linear_attention":
                    continue
                mixer = getattr(layer, "linear_attn", None)
                assert isinstance(mixer, Qwen3_5GatedDeltaNet), (
                    f"expected layer.linear_attn to be Qwen3_5GatedDeltaNet, got {type(mixer).__name__}"
                )
                mixers.append(mixer)
            self._linear_attn_mixers = tuple(mixers)
        else:
            self._linear_attn_mixers = ()

        # One-shot log of the per-batch-row full-attn KV preallocation.
        # Only full-attn ``StaticLayer``s preallocate per-token KV; linear-attn
        # layers hold only fixed conv + recurrent state. Size per row =
        # ``2 * n_full_layers * n_kv_heads * head_dim * cache_len * dtype_bytes``.
        try:
            tc = self.vlm.config.text_config
            n_full = sum(1 for lt in tc.layer_types if lt != "linear_attention")
            head_dim = getattr(tc, "head_dim", tc.hidden_size // tc.num_attention_heads)
            bytes_per_row = 2 * n_full * tc.num_key_value_heads * head_dim * self._s2_static_cache_len * 2  # bf16
            logger.info(
                "S2 StaticCache: %d full-attn layers x %d tokens -> %.2f GiB/batch-row full-attn KV",
                n_full,
                self._s2_static_cache_len,
                bytes_per_row / 2**30,
            )
        except (AttributeError, TypeError):  # pragma: no cover — best-effort log only
            logger.info("S2 StaticCache allocated with max_cache_len=%d", self._s2_static_cache_len)

    def configure_inference_compile(self, mode: S2CompileMode | None) -> None:
        """Configure opt-in compilation for cached text-only inference appends."""
        self._s2_compile_mode = _validate_s2_compile_mode(mode)
        self.__dict__["_s2_compiled_cached_language_model"] = None

    def _run_cached_language_model(
        self,
        *,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        inputs_embeds: torch.Tensor,
        past_key_values: StaticCache,
        compile_ok: bool,
    ) -> Any:
        """Run cached inference LM forward.

        Compilation is opt-in and limited to text-only append steps. Image
        appends have more variable shapes and include vision preprocessing, so
        they stay eager to avoid compile churn.
        """
        runner = self.vlm.model.language_model
        if compile_ok and self._s2_compile_mode is not None:
            compiled = self.__dict__.get("_s2_compiled_cached_language_model")
            if compiled is None:
                logger.info("Compiling S2 cached language-model forward with mode=%s", self._s2_compile_mode)
                compiled = torch.compile(runner, mode=self._s2_compile_mode, fullgraph=False, dynamic=True)
                self.__dict__["_s2_compiled_cached_language_model"] = compiled
            runner = compiled
        return runner(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=True,
        )

    @contextlib.contextmanager
    def _isolation_stash(
        self,
        cognition_mask: torch.Tensor,
        s2_query_ids: torch.Tensor,
        cumulative_cognition_mask: torch.Tensor | None = None,
        cumulative_s2_query_ids: torch.Tensor | None = None,
        static_cache_len: int | None = None,
    ):
        """Thread cog labels to the patched Qwen3.5 forwards for one ``with`` block.

        Per-chunk labels → linear-attn mixers; cumulative labels → TextModel.
        Cumulative defaults to per-chunk when the caller omits it (training /
        prefill, no prior cache). ``static_cache_len`` enables a right-pad on
        the 4D mask to match StaticCache's key-axis width.

        Masks are NOT cleared on exit — grad-checkpointing recompute runs in
        backward, after the ``with`` block. Safe under
        ``gradient_accumulation_steps=1`` because the next forward overwrites.
        Higher accum is unsupported here.
        """
        if not self._isolate_cog:
            yield
            return
        cum_cog = cumulative_cognition_mask if cumulative_cognition_mask is not None else cognition_mask
        cum_qids = cumulative_s2_query_ids if cumulative_s2_query_ids is not None else s2_query_ids
        lm = self.vlm.model.language_model
        # setattr bypasses nn.Module.__setattr__'s strict Module|Tensor overload
        # (pyrefly otherwise rejects these extra attrs).
        setattr(lm, "_cum_cog_mask", cum_cog)
        setattr(lm, "_cum_s2_query_ids", cum_qids)
        setattr(lm, "_static_cache_len", static_cache_len)
        for mixer in self._linear_attn_mixers:
            setattr(mixer, "_cog_mask", cognition_mask)
        yield

    def _build_inputs_embeds(
        self,
        *,
        input_ids: torch.Tensor,
        cognition_mask: torch.Tensor,
        pixel_values: torch.Tensor | None,
        image_grid_thw: torch.Tensor | None,
        s2_total_queries: int,
    ) -> torch.Tensor:
        """Build ``(B, L, H)`` inputs_embeds with image + cognition scatter.

        Stages 1-3 of the VLM pipeline, shared by training ``forward`` and the
        append-only inference contexts:

        1. Vocab embed — cognition slots zeroed first to dodge any cog-id
           vocab collision; the real embedding lands in stage 3.
        2. Image-feature scatter (skipped when ``pixel_values`` is empty).
        3. Learned ``cognition_embed`` scatter into cognition slots.

        ``s2_total_queries`` (Σ T across the batch, fed in from the collator)
        avoids the ``cognition_mask.sum()`` GPU→CPU sync.
        """
        vlm = self.vlm
        safe_ids = input_ids.clone()
        safe_ids[cognition_mask] = 0
        inputs_embeds = vlm.get_input_embeddings()(safe_ids)
        if pixel_values is not None and pixel_values.shape[0] > 0:
            image_outputs = vlm.get_image_features(pixel_values, image_grid_thw, return_dict=True)
            image_embeds = image_outputs.pooler_output
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = vlm.model.get_placeholder_mask(
                safe_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        if s2_total_queries > 0:
            dec_embed = self.cognition_embed.to(inputs_embeds.dtype)
            inputs_embeds[cognition_mask] = dec_embed.repeat(s2_total_queries, 1)
        return inputs_embeds

    @property
    def effective_context_cap(self) -> int:
        """The smaller of ``max_context_tokens`` and ``s2_static_cache_len``.

        Append contexts run against a ``StaticCache`` whose full-attn KV is
        preallocated to ``s2_static_cache_len`` slots — a longer sequence would
        index past it.
        """
        return min(self._max_context_tokens, self._s2_static_cache_len)

    def _new_static_cache(self, max_cache_len: int | None = None) -> StaticCache:
        """Allocate a config-aware HF ``StaticCache``.

        ``StaticCache(config=..., max_cache_len=...)`` walks ``config.layer_types``
        and builds ``LinearAttentionLayer`` for linear-attn layers and
        ``StaticLayer`` (preallocated fixed-shape KV) for full-attn layers. The
        full-attn keys tensor is ``(B, H, max_cache_len, D)``; the batch dim is
        locked at the batch of the cache's first forward.
        """
        cache_len = max_cache_len if max_cache_len is not None else self._s2_static_cache_len
        return StaticCache(config=self.vlm.config, max_cache_len=cache_len)
