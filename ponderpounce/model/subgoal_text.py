"""Subgoal text formatting for the S2 sequence (trigger token + grounded-coordinate rewrite)."""

from __future__ import annotations

import re

import numpy as np

from ponderpounce.constants import SUBGOAL_TRIGGER_STR

__all__ = ["format_subgoal"]


# Grounded subgoal coordinate pattern matches robomme's raw format
# (e.g. "pick up the cube at <110, 177>"). Each match's groups are (y, x).
_GROUNDED_COORD_RE = re.compile(r"at <(\d+), (\d+)>")
# Pixel coords in robomme source images are in [0, 255].
_BBOX_MIN, _BBOX_MAX = 0, 255
# Qwen grounding convention: coordinates normalized to [0, 1000] with x, y order.
_NORM_SCALE = 1000


def format_subgoal(
    sg: str,
    *,
    grounded: bool = False,
    rng: np.random.Generator | None = None,
    bbox_noise_pixels: int = 0,
) -> str:
    """``f"{SUBGOAL_TRIGGER_STR}{sg}"`` — the trigger is a single-piece
    special token (see :data:`ponderpounce.constants.SUBGOAL_TRIGGER_STR` for the
    BPE-collision rationale), and the cognition carrier token in the next
    chunk signals end-of-subgoal at inference. No suffix.

    When ``grounded`` and ``sg`` carries ``"at <y, x>"`` coordinates
    (robomme grounded subgoal format), rewrites them to ``at [x, y]`` with
    coordinates normalized to ``[0, 1000]`` — matching Qwen's grounding
    convention (x first, normalized scale). ``bbox_noise_pixels`` jitters
    each coord in pixel space by a truncated Gaussian with std = N / 2.5,
    clamped to ``[-N, N]`` and then clipped to ``[0, 255]`` before
    normalization — matches robomme's training-time
    ``add_grounding_augmentation`` (``noise_range=8``,
    ``_truncated_gaussian_noise``). Non-grounded inputs and grounded-but-
    coordless subgoals (direction strings, ``"no record"``) pass through
    unchanged.
    """
    if grounded and _GROUNDED_COORD_RE.search(sg):
        sg = _rewrite_grounded_coords(sg, rng=rng, noise=bbox_noise_pixels)
    return f"{SUBGOAL_TRIGGER_STR}{sg}"


def _rewrite_grounded_coords(sg: str, *, rng: np.random.Generator | None, noise: int) -> str:
    """Replace every ``at <y, x>`` with ``at [x, y]`` (Qwen grounding convention).

    robomme source stores pixel coords as ``(y, x)`` in ``[0, 255]``. We:
      1. parse the ``(y, x)`` pixel pair from source
      2. jitter in pixel space with a truncated Gaussian (std = noise / 2.5,
         clamped to ``[-noise, noise]``) — mirrors robomme's
         ``add_grounding_augmentation`` + ``_truncated_gaussian_noise``
      3. normalize to ``[0, 1000]`` to match Qwen's grounding scale
      4. emit as ``[x, y]`` (x first) to match Qwen's coordinate order

    Multiple coord groups in a single subgoal are rewritten independently.
    """
    std = noise / 2.5 if noise > 0 else 0.0

    def _sample() -> int:
        assert rng is not None  # guarded by caller
        return int(round(max(-noise, min(noise, rng.normal(0.0, std)))))

    def _sub(m: re.Match[str]) -> str:
        y_px, x_px = int(m.group(1)), int(m.group(2))
        if noise > 0 and rng is not None:
            y_px = int(np.clip(y_px + _sample(), _BBOX_MIN, _BBOX_MAX))
            x_px = int(np.clip(x_px + _sample(), _BBOX_MIN, _BBOX_MAX))
        # Normalize pixel [0, 255] → [0, 1000]; emit (x, y) per Qwen convention.
        x_norm = int(round(x_px * _NORM_SCALE / _BBOX_MAX))
        y_norm = int(round(y_px * _NORM_SCALE / _BBOX_MAX))
        return f"at [{x_norm}, {y_norm}]"

    return _GROUNDED_COORD_RE.sub(_sub, sg)
