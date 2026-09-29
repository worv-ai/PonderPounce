"""Pi05VLA: PonderPounce-side wrapper around the vendored π0.5 PyTorch port.

``self.pi05`` is :class:`ponderpounce.pi05._vendored.pi05_pytorch.PI0Pytorch`,
built from a Pi0-style namespace config. SigLIP vision tower frozen by default;
PaliGemma LLM, ``multi_modal_projector``, action expert, and projection/time MLPs
all train. The forward signature accepts PonderPounce-format batches and bridges them
to PI0Pytorch's list-of-cameras shape internally.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel

from ponderpounce.pi05._vendored.gemma_variants import get_config as _get_gemma_variant
from ponderpounce.pi05._vendored.pi05_pytorch import PI0Pytorch
from ponderpounce.pi05.configuration import Pi05Config


def _zero_pad_last(x: torch.Tensor, target: int) -> torch.Tensor:
    return x if x.shape[-1] >= target else F.pad(x, (0, target - x.shape[-1]))


class Pi05VLA(PreTrainedModel):
    """Standalone π0.5 (joint attention + AdaRMSNorm) wrapped for PonderPounce.

    Built from config only; the weights arrive with the PonderPounce checkpoint.
    """

    config_class = Pi05Config

    def __init__(self, config: Pi05Config):
        super().__init__(config)
        self.pi05 = PI0Pytorch(self._inner_config(config))
        if config.freeze_paligemma:
            self.freeze_backbone()

    @staticmethod
    def _inner_config(config: Pi05Config) -> SimpleNamespace:
        """Translate PonderPounce :class:`Pi05Config` to the openpi-style namespace
        :class:`PI0Pytorch` consumes."""
        return SimpleNamespace(
            pi05=True,
            paligemma_variant=config.paligemma_variant,
            action_expert_variant=config.action_expert_variant,
            dtype=config.dtype,
            action_horizon=config.action_horizon,
            action_dim=config.action_dim,
        )

    def _tie_paligemma_embeddings(self) -> None:
        """Point ``embed_tokens.weight`` at ``lm_head.weight``. Stock
        PaliGemma already calls ``tie_weights()`` at init, but loading a
        ckpt that omits ``embed_tokens.weight`` (because it was tied at
        save time) leaves the freshly-loaded ``lm_head.weight`` decoupled
        from the random ``embed_tokens.weight``."""
        text_model = self.pi05.paligemma_with_expert.paligemma.model.language_model
        lm_head = self.pi05.paligemma_with_expert.paligemma.lm_head
        text_model.embed_tokens.weight = lm_head.weight

    # ── Freeze control ────────────────────────────────────────────────────

    def freeze_backbone(self) -> None:
        """Freeze SigLIP vision tower only (matches mme_vla_suite
        ``history_pi0.py:233``: ``PathRegex(".*img.*")``)."""
        pg = self.pi05.paligemma_with_expert.paligemma  # PaliGemmaForConditionalGeneration
        for p in pg.model.vision_tower.parameters():
            p.requires_grad_(False)

    # ── Forward bridges ───────────────────────────────────────────────────

    @property
    def vlm_hidden_size(self) -> int:
        """PaliGemma hidden dim — looked up from the static gemma variant table
        so callers (e.g. PonderPounce adapters) don't reach into vendored
        ``paligemma_with_expert.config`` internals."""
        return _get_gemma_variant(self.config.paligemma_variant).width

    @property
    def _compute_dtype(self) -> torch.dtype:
        """Canonical dtype for state/action/image inputs. action_in_proj is
        always trainable (never frozen), so its dtype matches the action
        expert's compute dtype better than ``self.dtype`` would inside
        autocast."""
        return self.pi05.action_in_proj.weight.dtype

    def _bridge_inputs(
        self,
        state: torch.Tensor,
        pixel_values: torch.Tensor,
        pixel_attention_mask: torch.Tensor | None,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
        """Convert PonderPounce (B, N_cam, 3, H, W) + (B, N_cam) bool mask + (B, P)
        state into the openpi (list of (B, 3, H, W), list of (B,) bool,
        zero-padded state) shape. Casts state and image tensors to the model's
        compute dtype so the bf16 weights don't trip on float32 inputs."""
        target_dtype = self._compute_dtype
        B, N_cam = pixel_values.shape[:2]
        if pixel_attention_mask is None:
            pixel_attention_mask = torch.ones(B, N_cam, dtype=torch.bool, device=pixel_values.device)
        pixel_values = pixel_values.to(dtype=target_dtype)
        images = [pixel_values[:, i] for i in range(N_cam)]
        img_masks = [pixel_attention_mask[:, i] for i in range(N_cam)]
        state = _zero_pad_last(state.to(dtype=target_dtype), 32)
        return images, img_masks, state

    @torch.no_grad()
    def sample_actions(
        self,
        state: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        pixel_attention_mask: torch.Tensor | None = None,
        num_steps: int | None = None,
        prepend_prefix_embeds: torch.Tensor | None = None,
        prepend_prefix_pad_masks: torch.Tensor | None = None,
        prepend_prefix_att_masks: torch.Tensor | None = None,
        prepend_prefix_logit_bias: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Inference: predict ``(B, action_horizon, action_dim)``.

        Returns raw normalized model output — for delta-trained checkpoints
        these are normalized deltas, for absolute checkpoints normalized
        absolutes. The inference server runs ``unnormalize`` and (for delta
        runs) ``apply_absolute_action_mask(raw_state, mask)`` to reconstruct
        env-space actions, matching openpi's ``Unnormalize → AbsoluteActions``
        output transform order.

        ``prepend_prefix_*``: S2's age + K cognition tokens, already projected to
        PaliGemma's hidden dim, prepended at the front of the prefix.
        """
        images, img_masks, state = self._bridge_inputs(state, pixel_values, pixel_attention_mask)
        actions = self.pi05.sample_actions(
            device=state.device,
            images=images,
            img_masks=img_masks,
            lang_tokens=input_ids,
            lang_masks=attention_mask,
            state=state,
            noise=noise,
            num_steps=num_steps if num_steps is not None else 10,
            prepend_prefix_embeds=prepend_prefix_embeds,
            prepend_prefix_pad_masks=prepend_prefix_pad_masks,
            prepend_prefix_att_masks=prepend_prefix_att_masks,
            prepend_prefix_logit_bias=prepend_prefix_logit_bias,
        )
        return actions[..., : self.config.action_dim]
