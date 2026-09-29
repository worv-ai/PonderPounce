"""``Session`` - identity + episode-constant metadata for one session.

What's NOT here:

- streams (owned by the loops that publish them)
- per-system scheduling state (owned by ``system{N}/state.py``)
- per-session model caches like S2 KV/context state (owned by service/backend
  runtime objects)
- lifecycle flags (lifecycle is signalled via ``SessionRegistry.events()``)

What IS here is read by every system on every fire: the task text,
demo frames, the grounded-subgoal flag. These are episode-constant -
set once at ``register_session`` and never mutated afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import torch
from PIL import Image as PILImage

__all__ = ["ObsAdapter", "Session"]


class ObsAdapter(Protocol):
    """Translates harness / robot obs payloads into model-side tensors.

    Implementations are stateless beyond construction-time config so a
    single instance is shared across sessions. The session holds a
    reference so service-side code can call ``adapter.camera_pils(obs)``
    without knowing the concrete harness type.
    """

    def camera_pils(self, obs: Any) -> list[PILImage.Image]: ...

    def proprio_tensor(self, obs: Any) -> torch.Tensor | None: ...

    def raw_proprio_array(self, obs: Any) -> np.ndarray | None:
        """Pre-normalize raw state, sized to the model's proprio dim.

        Used by the inference server to apply the inverse delta
        transform after ``unnormalize`` for delta-trained checkpoints.
        ``None`` for obs with no proprio at all.
        """
        ...

    def demo_pils(self, raw_frames: Any) -> list[PILImage.Image]: ...


@dataclass
class Session:
    """Identity + episode-constant metadata.

    Constructed once per episode by the caller (harness shim, robot
    control program) and passed to ``Dispatcher.register(sid, session)``.
    The dispatcher's lifecycle helper hands it to each service's
    ``init_session(sid, session)``.
    """

    sid: str

    obs_adapter: ObsAdapter
    """How to turn the harness obs payload into PILs / tensors."""

    task_text: str = ""

    demo_images: list[PILImage.Image] = field(default_factory=list)

    subgoal_grounded: bool = False
    """Whether S2's subgoal-generation gate uses the grounded prompt."""

    s2_processor: Any = None
    """Qwen3.5 processor (text + image preprocessing)."""

    s1_transform: Any = None
    """Callable[[PIL.Image], torch.Tensor] applied to each camera PIL
    before stacking for the S1 forward."""

    num_cognition_tokens: int = 0
    """K in the cognition shape ``(K, H)``. Cached here so services can
    build the null-cognition without re-querying the model."""

    s1_prompt_ids: Any = None
    """Pre-tokenized PaliGemma task prompt for Pi0/Pi05 hard channel, shape (L,).
    ``None`` for archs that don't consume it; the Pi05 backend raises if missing."""

    s1_prompt_mask: Any = None
    """Attention mask matching ``s1_prompt_ids`` (same shape, 1=token, 0=pad)."""

    s1_num_camera_slots: int = 0
    """Camera slot count the S1 backbone expects (Pi0.5: 3); 0 disables padding."""
