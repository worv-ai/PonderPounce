"""PonderPounce VLA (PonderPounce): System 2 (Ponder) + System 1 (Pounce) as one HF model.

System 2 (deliberative): VLM → per-S2-query cognition tokens.
System 1 (reflexive): a pluggable standalone VLA wrapped by an arch-owned
    adapter that injects S2 cognitions (K tokens) into the model's feature
    space together with the cognition's age. The dual core dispatches the
    adapter through the System1 protocol and never names a concrete arch.

Tensor flow:
    cognition    (B, T, K, H)               from System 2
    s1_cognition_ref (B, N_s1)  int64         -1 for null / cold / dropout
    gather       (B, N_s1, K, H)            latest ready cognition per S1 tick
    null substitute  learned (K, H) scatter at cold positions
    → S1 loop consumes (B, N_s1, K, H) and (B, N_s1) age_ms
"""

from __future__ import annotations

import logging

import torch
from transformers import PreTrainedModel

from ponderpounce.model.configuration import PonderPounceConfig

logger = logging.getLogger(__name__)


def _validate_s1_knob_support(config: PonderPounceConfig, system1) -> None:
    """Fail fast when a non-default cognition knob is configured but the active
    S1 adapter cannot apply it. A silent skip would record the knob as active
    in the run config while training without it, invalidating the experiment.
    """
    unsupported = []
    if float(config.cognition_gain) != 1.0 and not hasattr(system1, "cognition_gain"):
        unsupported.append(f"cognition_gain={config.cognition_gain}")
    if float(config.s1_cognition_attn_bias) != 0.0 and not hasattr(system1, "cognition_attn_bias"):
        unsupported.append(f"s1_cognition_attn_bias={config.s1_cognition_attn_bias}")
    if unsupported:
        raise ValueError(
            f"S1 adapter {type(system1).__name__} cannot apply configured knob(s): {', '.join(unsupported)}. "
            "Use the Pi05 S1 binding or reset them to their defaults."
        )


class PonderPounceVLA(PreTrainedModel):
    """PonderPounce: S2 deliberative VLM + S1 reflexive action head."""

    config_class = PonderPounceConfig

    def __init__(self, config: PonderPounceConfig):
        super().__init__(config)

        # System 2: VLM → (B, T, K, H) cognition
        self.system2 = config.s2.build_system2()
        cognition_dim = self.system2.hidden_size

        # System 1: standalone model wrapped by adapter (arch owns its construction)
        self.system1 = config.s1.build_adapter(cognition_dim, flow=config.flow, schedule=config.schedule)
        self.system1.to(dtype=torch.bfloat16)  # uniform dtype for FSDP compat
        # Non-default cognition knobs must not silently no-op on adapters that
        # cannot apply them (raises ValueError).
        _validate_s1_knob_support(config, self.system1)
        # Cognition-gain knob: scale the projected cognition token at injection.
        if hasattr(self.system1, "cognition_gain"):
            self.system1.cognition_gain = float(config.cognition_gain)
        # Cognition attention-logit bias: pre-softmax bias on cognition keys.
        if hasattr(self.system1, "cognition_attn_bias"):
            self.system1.cognition_attn_bias = float(config.s1_cognition_attn_bias)

        # Future slots (PonderPounce-WM): part of the carrier block inside Ponder.
        # When Pounce was trained without them, the S1 adapter drops them from
        # its soft prefix: the leading slots, or the trailing ones under
        # ``s2.future_after_cognition``.
        self.num_future_tokens = int(getattr(config.s2, "num_future_tokens", 0))
        if self.num_future_tokens > 0 and not config.s1_receives_future:
            if not hasattr(self.system1, "cognition_slot_offset"):
                raise ValueError(f"S1 adapter {type(self.system1).__name__} cannot drop future slots")
            if config.s2.future_after_cognition:
                self.system1.cognition_slot_tail = self.num_future_tokens
            else:
                self.system1.cognition_slot_offset = self.num_future_tokens
        # Regression head of the future slots: Ponder hidden state -> predicted
        # change of the pooled SigLIP feature, ``(tokens, D)`` flattened. Not
        # used to act; kept so the checkpoint's predictions can be read out.
        if config.s2_future_weight > 0:
            tokens = config.s2.future_target_tokens(int(getattr(config.s1, "num_camera_slots", 1)))
            self.future_target_shape = (tokens, int(self.system1.future_target_dim))
            self.future_head = torch.nn.Linear(cognition_dim, tokens * self.future_target_shape[1]).to(
                dtype=torch.bfloat16
            )

        if config.s1_readout_aux_weight > 0:
            raise NotImplementedError("checkpoints trained with the S1 read-out aux head are not supported here")
        # Mark S1 as already initialized so HF's ZeRO-3 init walk doesn't
        # call kaiming_uniform_ on sharded 1-D params. S1 weights are loaded
        # from a pretrained checkpoint by the adapter, not random init.
        for m in self.system1.modules():
            m._is_hf_initialized = True

        # Null cognition (K, H): always present. Covers cold start, S2 failure, preemption, and cognition_dropout
        # through a single code path. Zeros init: cold and valid cases are already separated by the torch.where
        # branch below, so symmetry breaking isn't needed.
        self.null_cognition = torch.nn.Parameter(
            torch.zeros(config.s2.num_cognition_tokens, cognition_dim, dtype=torch.bfloat16)
        )

    def _init_weights(self, module):
        # S1's backbone weights are loaded from a pretrained checkpoint — skip
        # random init. Without this, initialize_weights_zero3 on the outer model
        # re-runs _init_weights on the backbone's sharded 1-D params → crash.
        if any(module is m for m in self.system1.modules()):
            return
        super()._init_weights(module)
