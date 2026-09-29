"""Pi05 wrapper configuration.

Thin PonderPounce-side config for the vendored openpi Pi0.5
but with π0.5-specific fields (PaliGemma + AdaRMSNorm action expert via the
vendored joint-attention port).
"""

from typing import Optional

from transformers import PretrainedConfig


class Pi05Config(PretrainedConfig):
    """π0.5 wrapper config.

    The underlying :class:`ponderpounce.pi05._vendored.pi05_pytorch.PI0Pytorch`
    expects an openpi-style ``Pi0Config`` shape (``pi05``, ``paligemma_variant``,
    ``action_expert_variant``, ``dtype``, ``action_horizon``, ``action_dim``).
    We translate to that shape inside :class:`Pi05VLA`.
    """

    model_type = "pi05_reflex"

    def __init__(
        self,
        # Pretrained checkpoint (HF Hub repo id or local path).
        # ``lerobot/pi05_base`` is openpi's published π0.5 base.
        pretrained: Optional[str] = "lerobot/pi05_base",
        # Domain dims (RoboMME = 8/8/20).
        action_dim: int = 8,
        state_dim: int = 8,
        action_horizon: int = 20,
        # Backbone variant. Reference: ``gemma_2b`` for VLM, ``gemma_300m`` for
        # the action expert (mme_vla_suite/training/config.py:556-588).
        paligemma_variant: str = "gemma_2b",
        action_expert_variant: str = "gemma_300m",
        # Param dtype. ``bfloat16`` matches openpi recipe.
        dtype: str = "bfloat16",
        # Subgoal augmentation (pixel jitter, format preserved).
        bbox_noise_pixels: int = 8,
        # Freeze SigLIP vision tower only (see ``Pi05VLA.freeze_backbone``).
        # Flag name kept for back-compat.
        freeze_paligemma: bool = True,
        # Tokenizer truncation cap. Symbolic-grounded yaml overrides to 128.
        max_token_len: int = 200,
        # Delta action encoding mask. ``mask[i] == True`` => target dim i is trained as
        # ``action[i] - state[i]`` (delta from current state) and reconstructed at inference as
        # ``pred[i] + state[i]``. ``False`` leaves the dim absolute (e.g. binary gripper). Length must be in
        # ``[1, action_dim]`` and the subtraction presupposes a shared coordinate frame, so when set we require
        # ``state_dim == action_dim``. The robomme pi05_baseline uses ``[T]*7 + [F]`` (Franka 7-DoF arm,
        # 1-DoF gripper); ``None`` disables and trains on absolute actions end-to-end (the prior behaviour,
        # kept as the safe default).
        delta_action_mask: list[bool] | None = None,
        **kwargs,
    ):
        # ``pi05`` is exposed as a read-only @property below, but HF ``save_pretrained`` serialises @property
        # values into config.json. On reload, transformers 5.5 forwards ``pi05=True`` as a kwarg — which would
        # hit the property's missing setter. Drop it here so the round-trip works without baking the on-disk
        # format change.
        kwargs.pop("pi05", None)
        super().__init__(**kwargs)
        self.pretrained = pretrained
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.action_horizon = action_horizon
        self.paligemma_variant = paligemma_variant
        self.action_expert_variant = action_expert_variant
        self.dtype = dtype
        self.bbox_noise_pixels = bbox_noise_pixels
        self.freeze_paligemma = freeze_paligemma
        self.max_token_len = max_token_len
        if delta_action_mask is not None:
            if state_dim != action_dim:
                raise ValueError(
                    f"delta_action_mask requires state_dim == action_dim "
                    f"(action[i] - state[i] presupposes a shared coordinate frame); "
                    f"got state_dim={state_dim}, action_dim={action_dim}."
                )
            if not 1 <= len(delta_action_mask) <= action_dim:
                raise ValueError(
                    f"delta_action_mask length must be in [1, action_dim={action_dim}], "
                    f"got {len(delta_action_mask)} (pass None to disable)."
                )
        self.delta_action_mask = list(delta_action_mask) if delta_action_mask is not None else None

    @property
    def pi05(self) -> bool:
        """openpi's ``Pi0Config.pi05`` flag — always True for this wrapper."""
        return True
