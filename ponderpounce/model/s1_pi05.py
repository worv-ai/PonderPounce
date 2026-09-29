"""PonderPounce S1 binding for Pi05.

Houses the dual-only surface for the Pi05 arch:
  - :class:`PonderPounceS1Pi05` — the adapter satisfying the
    :class:`ponderpounce.model.protocols.System1` protocol. Same prefix
    layout as the Pi0 adapter, but the soft prefix is threaded in via the
    ``prepend_prefix_*`` kwargs on :meth:`Pi05VLA.forward` rather than
    constructed inline (Pi05's vendored ``PI0Pytorch.embed_prefix`` builds
    image+lang embeds internally, so we can't override ``inputs_embeds`` like
    Pi0). Exposes the optional ``reflex_retie_embeddings`` dual-loader hook.
  - :class:`PonderPounceS1Pi05Config` — a ``Pi05Config`` subclass carrying the dual
    S1 fields/methods (kept off the standalone config).

The concrete ``Pi05VLA`` wrapper is imported lazily so importing this module
pulls only the arch *config*, not the heavy model wrapper.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from ponderpounce.model.common import TimeTokenizer, require_prompt_ids
from ponderpounce.pi05.configuration import Pi05Config

if TYPE_CHECKING:
    from ponderpounce.model.configuration import FlowMatchingConfig, ScheduleConfig
    from ponderpounce.model.processor import S1ImageProcessor
    from ponderpounce.model.protocols import System1
    from ponderpounce.pi05.wrapper import Pi05VLA

logger = logging.getLogger(__name__)


class PonderPounceS1Pi05(nn.Module):
    """Adapts :class:`Pi05VLA` for PonderPounce."""

    def __init__(self, model: "Pi05VLA", cognition_dim: int):
        super().__init__()
        self.model = model
        self.action_dim = model.config.action_dim
        self.action_chunk = model.config.action_horizon
        self.cognition_proj = nn.Linear(cognition_dim, model.vlm_hidden_size).to(dtype=torch.bfloat16)
        self.age_tokenizer = TimeTokenizer(model.vlm_hidden_size).to(dtype=torch.bfloat16)
        # Scalar amplifier on the projected cognition token (set by the dual core
        # from ``config.cognition_gain``); 1.0 = no-op.
        self.cognition_gain: float = 1.0
        # Additive pre-softmax attention-logit bias on cognition KEY positions
        # (set by the dual core from ``config.s1_cognition_attn_bias``); 0.0 = no-op.
        self.cognition_attn_bias: float = 0.0

    def _prefix_logit_bias(self, soft_prefix: torch.Tensor | None) -> torch.Tensor | None:
        """``(1+K,)`` per-key logit bias for the prepended tokens: 0 on the age
        token, ``cognition_attn_bias`` on the K cognition keys. ``None`` at 0.0
        keeps the mask path bit-identical."""
        if soft_prefix is None or self.cognition_attn_bias == 0.0:
            return None
        bias = torch.zeros(soft_prefix.shape[1], device=soft_prefix.device, dtype=torch.float32)
        bias[1:] = self.cognition_attn_bias
        return bias

    def _build_soft_prefix(
        self,
        cognition: torch.Tensor,
        age_ms: torch.Tensor,
    ) -> torch.Tensor | None:
        """Return ``(B, 1+K, H)`` soft tokens, or ``None`` when K=0.

        ``cognition`` arrives in bf16 (S2 hidden dtype) and ``age_tokenizer``
        is bf16-resident, so no per-tick dtype casts are needed."""
        if cognition.shape[1] == 0:
            return None
        d = self.cognition_proj(cognition)  # (B, K, H)
        if self.cognition_gain != 1.0:
            d = d * self.cognition_gain
        dt = self.age_tokenizer(age_ms).unsqueeze(1)  # (B, 1, H)
        return torch.cat([dt, d], dim=1)  # (B, 1+K, H)

    @staticmethod
    def _resolve_lang(
        symbolic_prompt_ids: torch.Tensor,
        symbolic_prompt_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(ids, mask_long)``; synthesise an all-ones mask if the
        caller omitted one. The VLM expects ``attention_mask`` as long, so
        pin the dtype here."""
        mask = (
            symbolic_prompt_mask.to(dtype=torch.long)
            if symbolic_prompt_mask is not None
            else torch.ones_like(symbolic_prompt_ids, dtype=torch.long)
        )
        return symbolic_prompt_ids, mask

    @property
    def noise_shape(self) -> tuple[int, int]:
        """``(action_horizon, noise_feature_dim)`` of the flow-matching noise
        ``sample_actions`` expects — the second dim is the openpi-padded action
        width. Lets a caller pre-draw deterministic per-session noise."""
        return int(self.action_chunk), int(self.model.pi05.action_in_proj.in_features)

    def predict_action(
        self,
        images: torch.Tensor,
        cognition: torch.Tensor,
        age_ms: torch.Tensor,
        proprio: torch.Tensor | None = None,
        *,
        noise: torch.Tensor | None = None,
        symbolic_prompt_ids: torch.Tensor | None = None,
        symbolic_prompt_mask: torch.Tensor | None = None,
        pixel_attention_mask: torch.Tensor | None = None,
        num_steps: int | None = None,
    ) -> torch.Tensor:
        symbolic_prompt_ids = require_prompt_ids("Pi05", symbolic_prompt_ids)
        B = images.shape[0]
        device = images.device
        if proprio is None:
            proprio = torch.zeros(B, self.model.config.state_dim, device=device)
        soft_prefix = self._build_soft_prefix(cognition, age_ms)
        ids, mask = self._resolve_lang(symbolic_prompt_ids, symbolic_prompt_mask)
        return self.model.sample_actions(
            state=proprio,
            input_ids=ids,
            attention_mask=mask,
            pixel_values=images,
            pixel_attention_mask=pixel_attention_mask,
            num_steps=num_steps,
            prepend_prefix_embeds=soft_prefix,
            prepend_prefix_logit_bias=self._prefix_logit_bias(soft_prefix),
            noise=noise,
        )

    # ── Dual-loader hooks (duck-typed by ponderpounce.loading) ──────────

    def reflex_retie_embeddings(self) -> None:
        """Re-tie shared embed/lm_head storage after a strict full-model load.

        ``Pi05VLA.forward`` ties ``embed_tokens.weight`` to ``lm_head.weight``;
        a ``safe_serialization`` save drops the duplicate, so reload leaves them
        decoupled until this runs. Logs whether storage is actually shared so a
        silent regression shows up."""
        self.model._tie_paligemma_embeddings()
        pg = self.model.pi05.paligemma_with_expert.paligemma
        text_model = pg.model.language_model
        shared = text_model.embed_tokens.weight.data_ptr() == pg.lm_head.weight.data_ptr()
        logger.info("Re-tied S1 embed_tokens <-> lm_head (shared_storage=%s)", shared)


class PonderPounceS1Pi05Config(Pi05Config):
    """Pi05 S1 config for the PonderPounce path.

    Subclasses the standalone ``Pi05Config`` and adds the dual-only fields +
    the uniform S1 contract the dual core reads.
    """

    model_type = "ponderpounce_s1_pi05"

    def __init__(
        self,
        # ── PonderPounce S1 fields ──
        # S1 pixel-path image size + dual S1-loop mini-batch cap.
        img_size: int = 224,
        max_s1_batch: int = 64,
        # PaliGemma image-slot count; must be >= dataset cam count.
        num_camera_slots: int = 2,
        # Hard channel: splice the GT subgoal text into PaliGemma's prompt every
        # S1 tick. ``hard_subgoal_enabled=False`` (paper default) keeps the
        # prompt task+obs only; ``True`` enables the "+ subgoal text" ablation.
        hard_subgoal_enabled: bool = False,
        hard_tokenizer_id: str = "google/paligemma-3b-pt-224",
        hard_max_token_len: int = 400,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.img_size = img_size
        self.max_s1_batch = max_s1_batch
        if num_camera_slots < 1:
            raise ValueError(f"num_camera_slots must be >= 1, got {num_camera_slots}.")
        self.num_camera_slots = num_camera_slots
        self.hard_subgoal_enabled = hard_subgoal_enabled
        self.hard_tokenizer_id = hard_tokenizer_id
        self.hard_max_token_len = hard_max_token_len

    # ── PonderPounce S1 contract ────────────────────────────────────────

    @property
    def proprio_dim(self) -> int:
        return self.state_dim

    @property
    def action_chunk(self) -> int:
        return self.action_horizon

    def build_s1_image_transform(self) -> "S1ImageProcessor":
        """PaliGemma's SigLIP tower expects ``[-1, 1]`` normalization."""
        from ponderpounce.model.processor import S1ImageProcessor

        return S1ImageProcessor(img_size=self.img_size, normalization="siglip")

    def s1_hard_prompt_spec(self) -> dict | None:
        """Pi05 routes a PaliGemma text prompt through S1 every tick."""
        return {
            "tokenizer_id": self.hard_tokenizer_id,
            "max_token_len": self.hard_max_token_len,
            "num_camera_slots": self.num_camera_slots,
            "subgoal_enabled": self.hard_subgoal_enabled,
        }

    def build_adapter(
        self,
        cognition_dim: int,
        *,
        flow: "FlowMatchingConfig | None" = None,
        schedule: "ScheduleConfig | None" = None,
    ) -> "System1":
        """Build the Pi05 model from this config and wrap it in the dual adapter.

        Weights (backbone, action expert and the adapter's own projections) are
        loaded afterwards from the PonderPounce checkpoint; nothing is fetched
        from the Hub here. ``flow`` / ``schedule`` are unused (the action expert
        carries openpi's flow-matching schedule internally)."""
        from ponderpounce.pi05.wrapper import Pi05VLA

        del flow, schedule
        return PonderPounceS1Pi05(Pi05VLA(self), cognition_dim)
