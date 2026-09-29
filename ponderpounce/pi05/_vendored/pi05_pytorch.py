"""π0.5 PyTorch model — vendored and adapted from
``robomme_policy_learning/src/openpi/models_pytorch/pi0_pytorch.py``.

Adaptations for transformers 5.x + PonderPounce:

* Imports rewired to local ``ponderpounce.pi05._vendored`` siblings instead
  of ``openpi.models.*`` and ``openpi.models_pytorch.*``.
* The ``transformers_replace`` install-check (lines 117-124 in the original)
  is dropped — we vendor the patched gemma directly so that side-channel
  doesn't apply.
* ``torch.compile(self.sample_actions, ...)`` is dropped to keep the model
  TorchScript / DeepSpeed-friendly. Re-enable later if needed.
* ``preprocessing_pytorch`` is not vendored; PonderPounce has its own image
  pipeline (``S1ImageProcessor`` with arch="Pi05"). The ``_preprocess_observation``
  helper is dropped.
* PaliGemma module hierarchy in 5.5.0 nests one extra ``.model``: paths like
  ``self.paligemma_with_expert.paligemma.model.language_model`` become
  ``…paligemma.model.language_model`` (vision_tower similarly).
"""

import math

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from ponderpounce.pi05._vendored import gemma_variants as _gemma
from ponderpounce.pi05._vendored.paligemma_with_expert import PaliGemmaWithExpertModel


def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "cpu":
        # CPU doesn't support bfloat16, use float32 instead
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha, beta, bsize, device):
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


def _prepend_prefix(
    prefix_embs: Tensor,
    prefix_pad_masks: Tensor,
    prefix_att_masks: Tensor,
    extra_embs: Tensor | None,
    extra_pad_masks: Tensor | None,
    extra_att_masks: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Prepend ``extra_*`` tokens at the front of PaliGemma's prefix.
    Returns the inputs unchanged when ``extra_embs`` is ``None``."""
    if extra_embs is None:
        return prefix_embs, prefix_pad_masks, prefix_att_masks
    K = extra_embs.shape[1]
    B = extra_embs.shape[0]
    device = extra_embs.device
    if extra_pad_masks is None:
        extra_pad_masks = torch.ones(B, K, dtype=prefix_pad_masks.dtype, device=device)
    if extra_att_masks is None:
        extra_att_masks = torch.zeros(B, K, dtype=prefix_att_masks.dtype, device=device)
    return (
        torch.cat([extra_embs.to(prefix_embs.dtype), prefix_embs], dim=1),
        torch.cat([extra_pad_masks.to(prefix_pad_masks.dtype), prefix_pad_masks], dim=1),
        torch.cat([extra_att_masks.to(prefix_att_masks.dtype), prefix_att_masks], dim=1),
    )


def apply_prepend_logit_bias(att_4d: Tensor, bias: Tensor | None) -> Tensor:
    """Add a per-key logit bias to the first ``bias.shape[-1]`` key columns.

    ``att_4d`` is the additive pre-softmax mask ``(B, 1, Q, S)`` (0 = attend,
    large negative = masked); adding to it biases attention toward those keys
    for every query. Masked entries stay masked (-2.38e38 + b is still an
    effective -inf). ``bias`` is ``(n,)`` or ``(B, n)``. ``None`` = identity.
    """
    if bias is None:
        return att_4d
    n = bias.shape[-1]
    out = att_4d.clone()
    out[..., :n] = out[..., :n] + bias.to(dtype=att_4d.dtype, device=att_4d.device).view(-1, 1, 1, n)
    return out


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


class PI0Pytorch(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pi05 = config.pi05

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True] if self.pi05 else [False, False],
            precision=config.dtype,
        )

        self.action_in_proj = nn.Linear(32, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, 32)

        if self.pi05:
            self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
            self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
        else:
            self.state_proj = nn.Linear(32, action_expert_config.width)
            self.action_time_mlp_in = nn.Linear(2 * action_expert_config.width, action_expert_config.width)
            self.action_time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        torch.set_float32_matmul_precision("high")
        # NOTE: ``torch.compile`` removed for PonderPounce integration —
        # DeepSpeed + HF Trainer don't play nicely with compiled forwards.
        self.gradient_checkpointing_enabled = False

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization.

        Controls:
        - self.gradient_checkpointing_enabled: gates _apply_checkpoint on ViT/projections
        - language_model.gradient_checkpointing: gates per-block GC in paligemma_with_expert
        - vision_tower.gradient_checkpointing: gates HF per-layer GC inside SigLIP (only
          effective when ViT is trainable; frozen ViT uses torch.no_grad + detach instead)
        """
        self.gradient_checkpointing_enabled = True
        # Per-block GC in joint-attention loop reads this flag.
        self.paligemma_with_expert.paligemma.model.language_model.gradient_checkpointing = True
        # HF SigLIP internal per-layer GC (only matters if ViT is trainable).
        self.paligemma_with_expert.paligemma.model.vision_tower.gradient_checkpointing = True
        # Expert layers are covered by per-block GC (same _block_fn wraps both branches).

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.model.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.model.vision_tower.gradient_checkpointing = False

    def is_gradient_checkpointing_enabled(self):
        """Check if gradient checkpointing is enabled."""
        return self.gradient_checkpointing_enabled

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Helper method to prepare 4D attention masks for transformer."""
        # Cast to bool — ``make_att_2d_masks`` may return Long (from
        # bool × bool multiplications); ``torch.where`` requires bool cond.
        att_2d_masks_4d = att_2d_masks[:, None, :, :].to(dtype=torch.bool)
        return torch.where(att_2d_masks_4d, 0.0, -2.3819763e38)

    # NOTE: ``_preprocess_observation`` removed in PonderPounce port — the openpi
    # ``preprocessing_pytorch`` module is not vendored, and PonderPounce provides
    # already-preprocessed tensors via :class:`ponderpounce.pi05.wrapper.Pi05VLA`.
    # ``forward`` / ``sample_actions`` below accept the 5 raw tensors directly.

    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for PaliGemma transformer processing.
        """
        embs = []
        pad_masks = []
        att_masks = []

        # Check ViT freeze state once (not per-image).
        vit_frozen = not any(
            p.requires_grad for p in self.paligemma_with_expert.paligemma.model.vision_tower.parameters()
        )

        # Process images
        for img, img_mask in zip(images, img_masks, strict=True):
            if vit_frozen:
                with torch.no_grad():
                    img_emb = self.paligemma_with_expert.embed_image(img).detach()
            else:

                def image_embed_func(img):
                    return self.paligemma_with_expert.embed_image(img)

                img_emb = self._apply_checkpoint(image_embed_func, img)

            bsize, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))

            # Create attention masks so that image tokens attend to each other
            att_masks += [0] * num_img_embs

        # Process language tokens
        def lang_embed_func(lang_tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, lang_tokens)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        # full attention between image and language inputs
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)

        # Get batch size from the first dimension of the concatenated tensors
        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def embed_suffix(self, state, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        if not self.pi05:
            if self.state_proj.weight.dtype == torch.float32:
                state = state.to(torch.float32)

            # Embed state
            def state_proj_func(state):
                return self.state_proj(state)

            state_emb = self._apply_checkpoint(state_proj_func, state)

            embs.append(state_emb[:, None, :])
            bsize = state_emb.shape[0]
            device = state_emb.device

            state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
            pad_masks.append(state_mask)

            # Set attention masks so that image and language inputs do not attend to state or actions
            att_masks += [1]

        # Cast noisy_actions to action_in_proj weight dtype. Flow-matching x_t
        # is built from float32 noise/time (sample_noise/sample_time pin
        # float32 for numerical accuracy), but action_in_proj is bf16 under
        # DeepSpeed bf16 — without autocast on the inner Linear, the dtype
        # mismatch errors. Same for time_emb feeding time_mlp_in below.
        proj_dtype = self.action_in_proj.weight.dtype
        noisy_actions = noisy_actions.to(dtype=proj_dtype)

        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0, device=timestep.device
        )
        time_emb = time_emb.to(dtype=proj_dtype)

        # Fuse timestep + action information using an MLP
        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        if not self.pi05:
            time_emb = time_emb[:, None, :].expand_as(action_emb)
            action_time_emb = torch.cat([action_emb, time_emb], dim=2)

            # Apply MLP layers
            def mlp_func(action_time_emb):
                x = self.action_time_mlp_in(action_time_emb)
                x = F.silu(x)  # swish == silu
                return self.action_time_mlp_out(x)

            action_time_emb = self._apply_checkpoint(mlp_func, action_time_emb)
            adarms_cond = None
        else:
            # time MLP (for adaRMS)
            def time_mlp_func(time_emb):
                x = self.time_mlp_in(time_emb)
                x = F.silu(x)  # swish == silu
                x = self.time_mlp_out(x)
                return F.silu(x)

            time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
            action_time_emb = action_emb
            adarms_cond = time_emb

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.action_horizon - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    def forward(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        actions,
        noise=None,
        time=None,
        prepend_prefix_embeds: Tensor | None = None,
        prepend_prefix_pad_masks: Tensor | None = None,
        prepend_prefix_att_masks: Tensor | None = None,
        prepend_prefix_logit_bias: Tensor | None = None,
        return_readout_hidden: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Training forward pass.

        Adapted from openpi: caller provides already-preprocessed tensors
        instead of an ``observation`` namespace.

        Args:
            images: list of ``(B, 3, H, W)`` tensors, one per camera slot.
            img_masks: list of ``(B,)`` bool tensors marking real vs padded cameras.
            lang_tokens: ``(B, L)`` text token ids (PaliGemma tokenizer).
            lang_masks: ``(B, L)`` text attention mask.
            state: ``(B, max_state_dim)`` robot state, zero-padded.
            actions: ``(B, action_horizon, max_action_dim)`` ground-truth actions.
            prepend_prefix_embeds: optional ``(B, K, H)`` tokens prepended at the
                front of PaliGemma's prefix (used by PonderPounce to inject S2's
                age + cognition tokens in PaliGemma's hidden-dim space).
            prepend_prefix_pad_masks: optional ``(B, K)`` bool mask — True for
                real prepend tokens, False for padding.
            prepend_prefix_att_masks: optional ``(B, K)`` bool mask used by
                :func:`make_att_2d_masks` (1 = new attention block).
            prepend_prefix_logit_bias: optional ``(K,)`` or ``(B, K)`` additive
                pre-softmax attention-logit bias on the prepended key columns
                for all queries (PonderPounce cognition-attention bias).

        Returns:
            ``(B, action_horizon, max_action_dim)`` per-element flow-matching
            MSE (no reduction). Caller decides reduction (mean / masked mean).
            When ``return_readout_hidden`` is set, also returns the ``(B, width)``
            mean-pooled action-expert last hidden states (the trunk features
            feeding the flow head), for the S1 read-out aux loss.
        """
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_embs, prefix_pad_masks, prefix_att_masks = _prepend_prefix(
            prefix_embs,
            prefix_pad_masks,
            prefix_att_masks,
            prepend_prefix_embeds,
            prepend_prefix_pad_masks,
            prepend_prefix_att_masks,
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, time)
        if (
            self.paligemma_with_expert.paligemma.model.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        # Prepare attention masks
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)
        if prepend_prefix_logit_bias is not None:
            assert prepend_prefix_embeds is not None, "logit bias targets prepended keys; none were prepended"
            att_2d_masks_4d = apply_prepend_logit_bias(att_2d_masks_4d, prepend_prefix_logit_bias)

        # Per-block GC (3 blocks of 6 layers) is applied inside
        # paligemma_with_expert.forward — no outer checkpoint needed.
        (_, suffix_out), _ = self.paligemma_with_expert.forward(
            attention_mask=att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = suffix_out[:, -self.config.action_horizon :]
        # Mean-pool the action-token hidden states before the flow head; this is
        # the trunk read-out the aux loss decodes. Kept in-graph (not detached).
        readout_hidden = suffix_out.mean(dim=1) if return_readout_hidden else None
        # Cast suffix_out to action_out_proj weight dtype before the linear,
        # then bring v_t back to float32 so the flow-matching MSE against
        # u_t (built from float32 noise) is computed in float32.
        suffix_out = suffix_out.to(dtype=self.action_out_proj.weight.dtype)

        # Apply gradient checkpointing to final action projection if enabled
        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out).to(dtype=torch.float32)

        per_elt_mse = F.mse_loss(u_t, v_t, reduction="none")
        if return_readout_hidden:
            assert readout_hidden is not None
            return per_elt_mse, readout_hidden
        return per_elt_mse

    @torch.no_grad()
    def sample_actions(
        self,
        device,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        noise=None,
        num_steps=10,
        prepend_prefix_embeds: Tensor | None = None,
        prepend_prefix_pad_masks: Tensor | None = None,
        prepend_prefix_att_masks: Tensor | None = None,
        prepend_prefix_logit_bias: Tensor | None = None,
    ) -> Tensor:
        """Inference forward — returns predicted action chunk
        ``(B, action_horizon, action_dim)``.

        Same input contract as :meth:`forward` (no ``observation`` wrapper).
        See :meth:`forward` for the ``prepend_prefix_*`` kwargs (PonderPounce).
        """
        if prepend_prefix_logit_bias is not None:
            assert prepend_prefix_embeds is not None, "logit bias targets prepended keys; none were prepended"
        bsize = state.shape[0]
        if noise is None:
            # action_in_proj is hardcoded to 32 (the openpi padded dim); noise
            # must match. The wrapper slices the output back to the real
            # ``config.action_dim`` after sampling.
            actions_shape = (bsize, self.config.action_horizon, self.action_in_proj.in_features)
            noise = self.sample_noise(actions_shape, device)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_embs, prefix_pad_masks, prefix_att_masks = _prepend_prefix(
            prefix_embs,
            prefix_pad_masks,
            prefix_att_masks,
            prepend_prefix_embeds,
            prepend_prefix_pad_masks,
            prepend_prefix_att_masks,
        )

        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        # Cache prefix K/V once, reuse for every Euler step. Prefix-lm
        # attention means prefix is independent of x_t/timestep.
        # ``crop(prefix_len)`` after each step works around the 5.5+
        # mutation of ``past_key_values`` under ``use_cache=False``.
        # Bit-exact vs joint forward — see ``tests/test_pi05_prefix_cache.py``.
        #
        # Mask dtype: the suffix-only cached forward goes through SDPA,
        # which rejects ``attn_mask=fp32`` against bf16 q/k under some
        # backends. Cast the 4D masks to the model's compute dtype here
        # (cached path only) — the joint training forward at ``forward()``
        # uses the eager attention kernel and is unaffected.
        mask_dtype = self.action_in_proj.weight.dtype
        prefix_len = prefix_embs.shape[1]
        prefix_pos_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_att_4d = apply_prepend_logit_bias(
            self._prepare_attention_masks_4d(prefix_att_2d).to(mask_dtype), prepend_prefix_logit_bias
        )
        _, prefix_kv = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_4d,
            position_ids=prefix_pos_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        # Hoist suffix masks + joint position_ids out of the Euler loop
        # — they depend only on shape, not x_t/timestep.
        suffix_pad_masks, suffix_att_masks = self._build_suffix_masks(
            bsize=bsize, device=device, mask_dtype=prefix_att_masks.dtype
        )
        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        suffix_att_2d_masks_4d = apply_prepend_logit_bias(
            self._prepare_attention_masks_4d(att_2d_masks[:, prefix_len:, :]).to(mask_dtype),
            prepend_prefix_logit_bias,
        )
        joint_position_ids = torch.cumsum(pad_masks, dim=1) - 1
        suffix_position_ids = joint_position_ids[:, prefix_len:]

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = self.denoise_step(
                prefix_kv,
                suffix_att_2d_masks_4d,
                suffix_position_ids,
                state,
                x_t,
                expanded_time,
            )
            x_t = x_t + dt * v_t
            time += dt
            prefix_kv.crop(prefix_len)
        return x_t

    def _build_suffix_masks(
        self,
        *,
        bsize: int,
        device: torch.device,
        mask_dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor]:
        """Mirror :meth:`embed_suffix`'s mask logic without running the
        action / state / time projections."""
        if self.pi05:
            suffix_len = self.config.action_horizon
            ar = [1] + [0] * (self.config.action_horizon - 1)
        else:
            suffix_len = 1 + self.config.action_horizon
            ar = [1, 1] + [0] * (self.config.action_horizon - 1)
        pad_masks = torch.ones(bsize, suffix_len, dtype=torch.bool, device=device)
        att_masks = torch.tensor(ar, dtype=mask_dtype, device=device)[None, :].expand(bsize, suffix_len)
        return pad_masks, att_masks

    def denoise_step(
        self,
        prefix_kv,
        suffix_att_2d_masks_4d,
        suffix_position_ids,
        state,
        x_t,
        timestep,
    ):
        """One Euler step. Runs only the suffix branch; prefix K/V come
        from ``prefix_kv``. Mask + position_ids are pre-hoisted in
        :meth:`sample_actions`."""
        suffix_embs, _suffix_pad, _suffix_att, adarms_cond = self.embed_suffix(state, x_t, timestep)

        (_, suffix_out), _ = self.paligemma_with_expert.forward(
            attention_mask=suffix_att_2d_masks_4d,
            position_ids=suffix_position_ids,
            past_key_values=prefix_kv,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = suffix_out[:, -self.config.action_horizon :]
        # Match action_out_proj weight dtype (bf16 under DeepSpeed/inference);
        # cast result back to float32 so the Euler step accumulator stays
        # in float32 across timesteps.
        suffix_out = suffix_out.to(dtype=self.action_out_proj.weight.dtype)
        return self.action_out_proj(suffix_out).to(dtype=torch.float32)
