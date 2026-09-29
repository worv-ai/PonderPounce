"""Monkey-patches that enforce ``isolated_cognition_attention`` on Qwen3.5.

Two upstream methods are replaced when ``apply_patches()`` runs:

- ``Qwen3_5GatedDeltaNet.forward`` (linear-attention layers): zero β/g at cog
  positions so the recurrent state update is the identity at cog steps (cog
  reads but does not write). A double-pass over the causal conv1d gates the
  cog signal out of subsequent obs tokens' K/V/Q. The saved conv_state is
  also cog-zeroed so the next chunk's prepend carries no cog signal.
- ``Qwen3_5TextModel.forward``: substitute a query-cropped 4D mask (built by
  ``build_4d_attention_mask``) for the default causal mask routed to full-
  attention layers. Linear-attn layers keep the 2D padding mask (their
  isolation is in the GatedDeltaNet patch). Cached decode requires cumulative
  cog labels because queries span only ``new_L`` while keys span
  ``prefix_L + new_L``.

The caller threads labels through ``System2._isolation_stash``; both patches
no-op without a stash.

Validated against the ``transformers`` versions in ``_VALIDATED_TRANSFORMERS_VERSIONS``.
If the installed version is not on that list, ``apply_patches()`` raises — the two
reproduced bodies must be re-diffed against the new upstream source first, guarding
against silent semantic drift.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F
from transformers import __version__ as _transformers_version
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5GatedDeltaNet,
    Qwen3_5ModelOutputWithPast,
    Qwen3_5TextModel,
    apply_mask_to_padding_states,
)

from ponderpounce.model.attention import build_4d_attention_mask

# GatedDeltaNet.forward is byte-identical 5.11.0 -> 5.13.0. TextModel.forward was
# restructured in 5.13.0: per-layer-type mask dict, ``_update_linear_attn_mask`` removed
# in favor of ``masking_utils.create_recurrent_attention_mask`` (same None semantics,
# mask trimmed to the local sequence) — the reproduced body below tracks that, so it is
# 5.13-only (re-validated 2026-07-03 by diffing method sources v5.12.1 vs v5.13.0).
_VALIDATED_TRANSFORMERS_VERSIONS = frozenset({"5.13.0"})

_ORIGINAL_DELTANET_FORWARD: Callable | None = None
_ORIGINAL_TEXTMODEL_FORWARD: Callable | None = None


def _patched_deltanet_forward(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
    # **kwargs mirrors upstream 5.11 (Unpack[TransformersKwargs]); the 5.8->5.11 body is
    # otherwise unchanged (re-validated 2026-06-12) and ignores them, as upstream does.
    cog_mask = getattr(self, "_cog_mask", None)
    if cog_mask is None:
        assert _ORIGINAL_DELTANET_FORWARD is not None, "apply_patches() not run"
        return _ORIGINAL_DELTANET_FORWARD(
            self, hidden_states, cache_params=cache_params, attention_mask=attention_mask, **kwargs
        )

    hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
    batch_size, seq_len, _ = hidden_states.shape
    use_precomputed_states = cache_params is not None and cache_params.has_previous_state(self.layer_idx)

    conv_state = None
    recurrent_state = None
    if use_precomputed_states:
        conv_state = cache_params.layers[self.layer_idx].conv_states
        recurrent_state = cache_params.layers[self.layer_idx].recurrent_states

    mixed_qkv = self.in_proj_qkv(hidden_states)
    mixed_qkv = mixed_qkv.transpose(1, 2)

    z = self.in_proj_z(hidden_states)
    z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)

    b = self.in_proj_b(hidden_states)
    a = self.in_proj_a(hidden_states)

    if use_precomputed_states and seq_len == 1:
        assert conv_state is not None  # use_precomputed_states implies these are set
        mixed_qkv = self.causal_conv1d_update(
            mixed_qkv,
            conv_state,
            self.conv1d.weight.squeeze(1),
            self.conv1d.bias,
            self.activation,
        )
    else:
        if use_precomputed_states:
            assert conv_state is not None
            mixed_qkv = torch.cat([conv_state, mixed_qkv], dim=-1)

        # Pad cog_mask on the left to cover any conv_state prepend. Those
        # slots were cog-zeroed when written by the previous chunk, so False
        # is the correct value there.
        prepend_L = mixed_qkv.shape[-1] - cog_mask.shape[-1]
        full_cog = F.pad(cog_mask, (prepend_L, 0), value=False) if prepend_L > 0 else cog_mask

        if cache_params is not None:
            qkv_for_cache = mixed_qkv * (1.0 - full_cog[:, None, :].to(mixed_qkv.dtype))
            new_conv_state = F.pad(qkv_for_cache, (self.conv_kernel_size - mixed_qkv.shape[-1], 0))
            cache_params.update_conv_state(new_conv_state, self.layer_idx)

        def _apply_conv(x):
            if self.causal_conv1d_fn is not None:
                return self.causal_conv1d_fn(
                    x=x,
                    weight=self.conv1d.weight.squeeze(1),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                    seq_idx=kwargs.get("seq_idx"),  # packed-batch boundaries (upstream 5.11 plumbs this)
                )
            return F.silu(self.conv1d(x)[:, :, : x.shape[-1]])

        # Hook 1: double-pass conv seals the cog→obs leak at the kernel-4
        # boundary. The (1 - cog_bc) gate is correct end-to-end because
        # full_cog is False at the prepended positions.
        cog_bc = full_cog[:, None, :].to(mixed_qkv.dtype)  # (B, 1, prepend_L + new_L)
        x_obs_only = mixed_qkv * (1.0 - cog_bc)
        y_obs = _apply_conv(x_obs_only)
        y_full = _apply_conv(mixed_qkv)
        mixed_qkv = torch.where(full_cog[:, None, :], y_full, y_obs)

        if use_precomputed_states:
            mixed_qkv = mixed_qkv[:, :, -seq_len:]

    mixed_qkv = mixed_qkv.transpose(1, 2)
    query, key, value = torch.split(
        mixed_qkv,
        [self.key_dim, self.key_dim, self.value_dim],
        dim=-1,
    )

    query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
    key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
    value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

    beta = b.sigmoid()
    g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

    # Hook 2: zero β and g at cog positions so the delta-net update reduces to
    # ``S_t = S_{t-1}`` — cog tokens read but do not write the recurrent state.
    not_cog = (~cog_mask).to(beta.dtype).unsqueeze(-1)  # (B, L, 1)
    beta = beta * not_cog
    g = g * not_cog

    if self.num_v_heads // self.num_k_heads > 1:
        query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
        key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

    if use_precomputed_states and seq_len == 1:
        core_attn_out, last_recurrent_state = self.recurrent_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=recurrent_state,
            output_final_state=cache_params is not None,
            use_qk_l2norm_in_kernel=True,
        )
    else:
        core_attn_out, last_recurrent_state = self.chunk_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=recurrent_state if use_precomputed_states else None,
            output_final_state=cache_params is not None,
            use_qk_l2norm_in_kernel=True,
            # Packed/varlen segment lengths (upstream 5.11 plumbs cu_seq_lens_q here).
            cu_seqlens=kwargs.get("cu_seq_lens_q"),
        )

    if cache_params is not None:
        cache_params.update_recurrent_state(last_recurrent_state, self.layer_idx)

    core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
    z = z.reshape(-1, self.head_v_dim)
    core_attn_out = self.norm(core_attn_out, z)
    core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)

    return self.out_proj(core_attn_out)


def _patched_textmodel_forward(
    self,
    input_ids=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    use_cache=None,
    **kwargs,
):
    cum_cog = getattr(self, "_cum_cog_mask", None)
    cum_qids = getattr(self, "_cum_s2_query_ids", None)
    if cum_cog is None or cum_qids is None:
        assert _ORIGINAL_TEXTMODEL_FORWARD is not None, "apply_patches() not run"
        return _ORIGINAL_TEXTMODEL_FORWARD(
            self,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if use_cache and past_key_values is None:
        past_key_values = DynamicCache(config=self.config)

    if position_ids is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
        position_ids = position_ids.view(1, 1, -1).expand(4, inputs_embeds.shape[0], -1)
    elif position_ids.ndim == 2:
        position_ids = position_ids[None, ...].expand(4, position_ids.shape[0], -1)

    if position_ids.ndim == 3 and position_ids.shape[0] == 4:
        text_position_ids = position_ids[0]
        position_ids = position_ids[1:]
    else:
        text_position_ids = None

    # Build (B, 1, S, S) isolation mask from cumulative cog labels and crop the
    # query dim to new_L on cached decode. Training/prefill: cumulative == new
    # chunk, so the crop is a no-op.
    full_attn = attention_mask if attention_mask is not None else torch.ones_like(cum_cog, dtype=torch.long)
    # Inside HF ``generate``'s decode loop the stash holds only the prompt-
    # portion labels while ``attention_mask`` grows +1 per step; auto-extend
    # cog/qid with (False, last_qid) so build_4d_attention_mask shape-matches.
    # No-op for ``_run_vlm`` callers that already pass full cumulative labels.
    cum_L_target = int(full_attn.shape[-1])
    if cum_cog.shape[-1] < cum_L_target:
        pad = cum_L_target - cum_cog.shape[-1]
        cum_cog = F.pad(cum_cog, (0, pad), value=False)
        last_qid = cum_qids[:, -1:].expand(-1, pad)
        cum_qids = torch.cat([cum_qids, last_qid], dim=-1)
    full_mask = build_4d_attention_mask(cum_cog, cum_qids, full_attn, dtype=inputs_embeds.dtype)
    new_L = inputs_embeds.shape[1]
    causal_mask = full_mask[:, :, -new_L:, :] if new_L < full_mask.shape[2] else full_mask

    # StaticCache's full-attn keys are (B, H, max_cache_len, D); right-pad
    # the 4D mask to max_cache_len with -inf so unwritten slots stay
    # unattendable. _static_cache_len is None on DynamicCache / training.
    static_cache_len = getattr(self, "_static_cache_len", None)
    if static_cache_len is not None:
        pad_keys = static_cache_len - causal_mask.shape[-1]
        if pad_keys > 0:
            causal_mask = F.pad(causal_mask, (0, pad_keys), value=float("-inf"))

    # Upstream's create_recurrent_attention_mask returns None once a linear-attn
    # layer has previous state (assumes the new chunk has no padding). Our
    # batched cached forward feeds a left-padded ragged new chunk; force the
    # 2D mask through so apply_mask_to_padding_states zeros the pad rows.
    from transformers.masking_utils import create_recurrent_attention_mask  # 5.13-only symbol

    new_attn = attention_mask[:, -new_L:] if attention_mask is not None else None
    new_chunk_has_pad = new_attn is not None and (new_attn == 0).any()
    if new_chunk_has_pad:
        linear_attn_mask = new_attn
    else:
        linear_attn_mask = create_recurrent_attention_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
        )

    hidden_states = inputs_embeds
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
        layer_mask = linear_attn_mask if self.config.layer_types[i] == "linear_attention" else causal_mask
        hidden_states = decoder_layer(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=layer_mask,
            position_ids=text_position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )

    hidden_states = self.norm(hidden_states)
    return Qwen3_5ModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values,
    )


def apply_patches() -> None:
    """Install the GatedDeltaNet and TextModel forward patches. Idempotent."""
    global _ORIGINAL_DELTANET_FORWARD, _ORIGINAL_TEXTMODEL_FORWARD

    if getattr(Qwen3_5GatedDeltaNet, "_reflex_isolation_patched", False):
        return

    if _transformers_version not in _VALIDATED_TRANSFORMERS_VERSIONS:
        raise RuntimeError(
            f"_qwen3_5_patches.py is validated against transformers in "
            f"{sorted(_VALIDATED_TRANSFORMERS_VERSIONS)}, but {_transformers_version!r} is installed. "
            f"Re-diff the two reproduced method bodies against the new upstream source, then add the "
            f"version to ``_VALIDATED_TRANSFORMERS_VERSIONS``."
        )

    _ORIGINAL_DELTANET_FORWARD = Qwen3_5GatedDeltaNet.forward
    _ORIGINAL_TEXTMODEL_FORWARD = Qwen3_5TextModel.forward
    Qwen3_5GatedDeltaNet.forward = _patched_deltanet_forward
    Qwen3_5TextModel.forward = _patched_textmodel_forward
    Qwen3_5GatedDeltaNet._reflex_isolation_patched = True
