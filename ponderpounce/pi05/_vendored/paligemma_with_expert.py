"""PaliGemma + Gemma action expert with joint attention (π0.5 architecture).

Vendored and adapted from
``robomme_policy_learning/src/openpi/models_pytorch/gemma_pytorch.py``
(originally targeting transformers 4.53.2). Adaptations for transformers 5.5.0:

* PaliGemma side stays **stock** (transformers 5.5.0 ``PaliGemmaForConditionalGeneration``).
  Module hierarchy in 5.5.0 is ``model.model.language_model`` (one extra
  ``.model`` vs 4.53.2). With ``use_adarms[0]=False`` (the only π0.5 setting
  the openpi authors trained), stock GemmaRMSNorm is correct.
* Action expert (``gemma_expert``) uses our patched
  :mod:`ponderpounce.pi05._vendored.modeling_gemma_adarms` with
  ``use_adarms=True`` so its layernorms are AdaRMSNorm and ``adarms_cond``
  is threaded.
* Joint-attention forward branches on ``use_adarms[i]`` to call layernorms
  with the right signature: stock layernorms get one arg (``layer_norm(x)``);
  patched layernorms get the conditioning (``layer_norm(x, cond=adarms_cond)``).
* The action expert's ``model.embed_tokens`` is set to ``None`` because the
  expert receives its tokens already embedded (state + noise + timestep
  via ``embed_action_time`` in :mod:`pi05_pytorch`).
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn
from transformers import PaliGemmaForConditionalGeneration
from transformers.models.auto import CONFIG_MAPPING

from ponderpounce.pi05._vendored import modeling_gemma_adarms as modeling_gemma
from ponderpounce.pi05._vendored.configuration_gemma_adarms import GemmaConfig as PatchedGemmaConfig
from ponderpounce.pi05._vendored.modeling_gemma_adarms import GemmaForCausalLM, _gated_residual


def _layernorm_with_cond(layer_norm, hidden_states, cond):
    """Call a layernorm with optional AdaRMSNorm conditioning.

    Stock ``GemmaRMSNorm`` (transformers 5.5.0) returns a tensor.
    Our patched ``GemmaRMSNorm`` always returns ``(tensor, gate_or_None)``.
    Returns ``(tensor, gate)`` so callers can apply gated residuals uniformly.
    """
    out = layer_norm(hidden_states, cond) if cond is not None else layer_norm(hidden_states)
    if isinstance(out, tuple):
        return out
    return out, None


class PaliGemmaWithExpertModel(nn.Module):
    """π0.5 backbone: stock PaliGemma + AdaRMSNorm-aware Gemma action expert.

    Constructor signature mirrors openpi's. ``vlm_config`` /
    ``action_expert_config`` are openpi-style ``_gemma.Variant`` configs (with
    ``width``, ``mlp_dim``, ``depth``, ``num_heads``, ``num_kv_heads``,
    ``head_dim`` attrs); they're translated to HF transformers configs here.
    """

    def __init__(
        self,
        vlm_config,
        action_expert_config,
        use_adarms=None,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
    ):
        if use_adarms is None:
            use_adarms = [False, False]
        super().__init__()
        self.use_adarms = list(use_adarms)

        # ── PaliGemma (stock) ─────────────────────────────────────────
        vlm_config_hf = CONFIG_MAPPING["paligemma"]()
        vlm_config_hf._vocab_size = 257152  # noqa: SLF001
        vlm_config_hf.image_token_index = 257152
        vlm_config_hf.text_config.hidden_size = vlm_config.width
        vlm_config_hf.text_config.intermediate_size = vlm_config.mlp_dim
        vlm_config_hf.text_config.num_attention_heads = vlm_config.num_heads
        vlm_config_hf.text_config.head_dim = vlm_config.head_dim
        vlm_config_hf.text_config.num_hidden_layers = vlm_config.depth
        vlm_config_hf.text_config.num_key_value_heads = vlm_config.num_kv_heads
        vlm_config_hf.text_config.hidden_activation = "gelu_pytorch_tanh"
        vlm_config_hf.text_config.vocab_size = 257152
        vlm_config_hf.vision_config.intermediate_size = 4304
        # ``projection_dim`` must match the language-model hidden size so that
        # image features (after multi_modal_projector) align with text embeds.
        # openpi hard-coded 2048 for gemma_2b; we make it width-driven to
        # support dummy/smoke variants too.
        vlm_config_hf.vision_config.projection_dim = vlm_config.width
        vlm_config_hf.vision_config.projector_hidden_act = "gelu_fast"

        # PaliGemma side has use_adarms[0]=False in pi05 — no patched config
        # needed; stock layernorms apply. The flag is recorded on this wrapper
        # for the forward branch below.

        # Under ZeRO-3, weight init crashes on sharded 1-D params. Two paths:
        # (1) post_init inside PaliGemma's __init__ → blocked by no_init_weights()
        # (2) initialize_weights_zero3 on outer model after construction →
        #     blocked by making _initialize_weights catch errors silently.
        # Real weights are loaded from Pi05 checkpoint afterward.
        from contextlib import nullcontext

        from transformers import PreTrainedModel
        from transformers.initialization import no_init_weights
        from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled

        if is_deepspeed_zero3_enabled():
            ctx = no_init_weights()
            _orig_initialize = PreTrainedModel._initialize_weights

            def _safe_initialize(*args, **kwargs):
                try:
                    _orig_initialize(*args, **kwargs)
                except (ValueError, IndexError):
                    pass

            PreTrainedModel._initialize_weights = _safe_initialize
        else:
            ctx = nullcontext()
            _orig_initialize = None

        try:
            with ctx:
                self.paligemma = PaliGemmaForConditionalGeneration(config=vlm_config_hf)
        finally:
            if _orig_initialize is not None:
                PreTrainedModel._initialize_weights = _orig_initialize

        # ── Action expert (patched Gemma with AdaRMSNorm) ────────────
        expert_cfg = PatchedGemmaConfig(
            head_dim=action_expert_config.head_dim,
            hidden_size=action_expert_config.width,
            intermediate_size=action_expert_config.mlp_dim,
            num_attention_heads=action_expert_config.num_heads,
            num_hidden_layers=action_expert_config.depth,
            num_key_value_heads=action_expert_config.num_kv_heads,
            vocab_size=257152,
            hidden_act="gelu_pytorch_tanh",
            use_adarms=use_adarms[1],
            adarms_cond_dim=action_expert_config.width if use_adarms[1] else None,
        )
        self.gemma_expert = GemmaForCausalLM(config=expert_cfg)
        # The expert reads pre-embedded action+state+time tokens; no token id
        # lookup. Setting embed_tokens=None saves params and matches the
        # lerobot/pi05_base ckpt layout.
        self.gemma_expert.model.embed_tokens = None

        self.to_bfloat16_for_selected_params(precision)

    def to_bfloat16_for_selected_params(self, precision: Literal["bfloat16", "float32"] = "bfloat16"):
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
        elif precision == "float32":
            self.to(dtype=torch.float32)
            return
        else:
            raise ValueError(f"Invalid precision: {precision}")

        # Keep layernorms + ViT input embeddings in fp32 (openpi
        # gemma_pytorch.py:72-83); bf16 ``mean(x²) → rsqrt`` loses small grads.
        params_to_keep_float32 = [
            "vision_tower.vision_model.embeddings.patch_embedding.weight",
            "vision_tower.vision_model.embeddings.patch_embedding.bias",
            "vision_tower.vision_model.embeddings.position_embedding.weight",
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        ]
        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_keep_float32):
                param.data = param.data.to(dtype=torch.float32)

    # ── Convenience accessors ────────────────────────────────────────

    @property
    def _paligemma_text_model(self):
        """Path to PaliGemma's GemmaModel.

        transformers 5.5.0 nests it as ``model.model.language_model``;
        4.53.2 had ``model.language_model`` directly. We expose it as a
        property so the forward / weight-load paths stay version-agnostic.
        """
        return self.paligemma.model.language_model

    def embed_image(self, image: torch.Tensor):
        # Bypass ``PaliGemmaModel.get_image_features``: stock 5.5.0 divides
        # the projector output by ``sqrt(text_config.hidden_size)``, but
        # the openpi 4.53.2 path that ``lerobot/pi05_base`` was trained
        # with does NOT divide (verified against
        # ``transformers_replace/models/paligemma/modeling_paligemma.py:
        # get_image_features``). Calling vision_tower + projector
        # directly preserves the raw magnitude the action expert was
        # trained to consume — without this, the image branch of
        # ``embed_prefix`` is silently scaled down by 45.25× and the
        # pretrained joint-attention path produces OOD activations.
        vision_out = self.paligemma.model.vision_tower(image)
        return self.paligemma.model.multi_modal_projector(vision_out.last_hidden_state)

    def embed_language_tokens(self, tokens: torch.Tensor):
        # Stock 5.5.0 uses ``GemmaTextScaledWordEmbedding`` which multiplies
        # by ``embed_scale = sqrt(hidden_size)`` inside its forward; openpi
        # 4.53.2 (the path ``lerobot/pi05_base`` was trained against) uses
        # plain ``nn.Embedding`` with no internal scale. Combined with the
        # outer ``* sqrt(hidden_size)`` in ``embed_prefix.lang_embed_func``,
        # the 5.5 path applies the scale TWICE (45.25² = 2048×) and
        # downstream Gemma sees out-of-distribution language embeddings.
        # Undo the internal scale so the joint-attention path matches the
        # openpi training distribution.
        embed_module = self._paligemma_text_model.embed_tokens
        out = embed_module(tokens)
        embed_scale = getattr(embed_module, "embed_scale", None)
        if embed_scale is not None:
            scale = embed_scale if isinstance(embed_scale, torch.Tensor) else torch.tensor(embed_scale)
            out = out / scale.to(out.dtype)
        return out

    # ── Joint-attention forward (compute_layer_complete) ─────────────

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        inputs_embeds: list[torch.FloatTensor | None] | None = None,
        use_cache: bool | None = None,
        adarms_cond: list[torch.Tensor | None] | None = None,
    ):
        """Joint-attention pass over [paligemma_tokens, expert_tokens].

        ``inputs_embeds[0]`` are PaliGemma prefix embeds (image+text);
        ``inputs_embeds[1]`` are action expert embeds (state+noise+time).
        Either may be ``None`` to skip that branch (e.g. cache-only step).

        ``adarms_cond[i]`` is the conditioning vector for branch ``i`` —
        only used when ``self.use_adarms[i]`` is True (action expert side
        for π0.5).
        """
        if adarms_cond is None:
            adarms_cond = [None, None]

        # Single-branch fast paths (used during inference).
        if inputs_embeds[1] is None:
            prefix_output = self._paligemma_text_model(
                inputs_embeds=inputs_embeds[0],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )
            return (prefix_output.last_hidden_state, None), prefix_output.past_key_values
        if inputs_embeds[0] is None:
            suffix_output = self.gemma_expert.model(
                inputs_embeds=inputs_embeds[1],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[1],
            )
            return (None, suffix_output.last_hidden_state), None

        # ── Joint-attention path ─────────────────────────────────────
        models = [self._paligemma_text_model, self.gemma_expert.model]
        num_layers = self.paligemma.config.text_config.num_hidden_layers

        gc_enabled = getattr(self._paligemma_text_model, "gradient_checkpointing", False) and self.training
        block_size = 6  # 18 layers / 3 blocks — balances recompute peak vs saves

        for block_start in range(0, num_layers, block_size):
            block_end = min(block_start + block_size, num_layers)

            def _block_fn(emb0, emb1, _start=block_start, _end=block_end):
                _embeds = [emb0, emb1]
                for layer_idx in range(_start, _end):
                    _embeds = self._compute_layer(
                        layer_idx, _embeds, attention_mask, position_ids, adarms_cond, models
                    )
                return _embeds[0], _embeds[1]

            if gc_enabled:
                inputs_embeds = list(
                    torch.utils.checkpoint.checkpoint(
                        _block_fn,
                        inputs_embeds[0],
                        inputs_embeds[1],
                        use_reentrant=False,
                        preserve_rng_state=False,
                    )
                )
            else:
                inputs_embeds = list(_block_fn(inputs_embeds[0], inputs_embeds[1]))

        # Final norms (per branch).
        outputs_embeds = []
        for i, hidden_states in enumerate(inputs_embeds):
            outputs_embeds.append(_layernorm_with_cond(models[i].norm, hidden_states, adarms_cond[i])[0])

        return (outputs_embeds[0], outputs_embeds[1]), None

    def _compute_layer(self, layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond, models):
        """One joint-attention layer over both branches.

        Steps (mirrors openpi's ``compute_layer_complete``):
            1. Per-branch input-layernorm + Q/K/V projection.
            2. Concatenate Q/K/V across branches; apply rotary; run attention once.
            3. Split output back per branch; per-branch o_proj + gated residual.
            4. Per-branch post-attn layernorm + MLP + gated residual.
        """
        query_states_list = []
        key_states_list = []
        value_states_list = []
        gates_pre = []

        for i, hidden_states in enumerate(inputs_embeds):
            layer = models[i].layers[layer_idx]
            hidden_states_n, gate = _layernorm_with_cond(layer.input_layernorm, hidden_states, adarms_cond[i])
            gates_pre.append(gate)

            input_shape = hidden_states_n.shape[:-1]
            head_dim = layer.self_attn.head_dim
            hidden_shape = (*input_shape, -1, head_dim)
            q = layer.self_attn.q_proj(hidden_states_n).view(hidden_shape).transpose(1, 2)
            k = layer.self_attn.k_proj(hidden_states_n).view(hidden_shape).transpose(1, 2)
            v = layer.self_attn.v_proj(hidden_states_n).view(hidden_shape).transpose(1, 2)
            query_states_list.append(q)
            key_states_list.append(k)
            value_states_list.append(v)

        query_states = torch.cat(query_states_list, dim=2)
        key_states = torch.cat(key_states_list, dim=2)
        value_states = torch.cat(value_states_list, dim=2)

        # Rotary embedding from PaliGemma side.
        dummy = torch.zeros(
            query_states.shape[0],
            query_states.shape[2],
            query_states.shape[-1],
            device=query_states.device,
            dtype=query_states.dtype,
        )
        cos, sin = self._paligemma_text_model.rotary_emb(dummy, position_ids)
        query_states, key_states = modeling_gemma.apply_rotary_pos_emb(
            query_states, key_states, cos, sin, unsqueeze_dim=1
        )

        scaling = self._paligemma_text_model.layers[layer_idx].self_attn.scaling
        # eager_attention_forward: faster than SDPA at seq≤700 (mem_efficient backend
        # overhead exceeds benefit at short sequences, measured +2s/step).
        att_output, _ = modeling_gemma.eager_attention_forward(
            self._paligemma_text_model.layers[layer_idx].self_attn,
            query_states,
            key_states,
            value_states,
            attention_mask,
            scaling,
        )
        batch_size = att_output.shape[0]
        head_dim = self._paligemma_text_model.layers[layer_idx].self_attn.head_dim
        att_output = att_output.reshape(batch_size, -1, query_states.shape[1] * head_dim)

        outputs_embeds = []
        start_pos = 0
        for i, hidden_states in enumerate(inputs_embeds):
            layer = models[i].layers[layer_idx]
            end_pos = start_pos + hidden_states.shape[1]
            chunk = att_output[:, start_pos:end_pos]
            if chunk.dtype != layer.self_attn.o_proj.weight.dtype:
                chunk = chunk.to(layer.self_attn.o_proj.weight.dtype)
            out_emb = layer.self_attn.o_proj(chunk)

            # First (attn) gated residual.
            out_emb = _gated_residual(hidden_states, out_emb, gates_pre[i])
            mid = out_emb

            # Post-attn layernorm + MLP + gated residual.
            out_emb_n, gate2 = _layernorm_with_cond(layer.post_attention_layernorm, out_emb, adarms_cond[i])
            if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
                out_emb_n = out_emb_n.to(dtype=torch.bfloat16)
            out_emb = layer.mlp(out_emb_n)
            out_emb = _gated_residual(mid, out_emb, gate2)
            outputs_embeds.append(out_emb)
            start_pos = end_pos

        return outputs_embeds
