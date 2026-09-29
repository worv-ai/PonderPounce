"""Action normalization using q01/q99 quantile scaling to [-1, 1]."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

NORM_STATS_FILENAME = "norm_stats.json"


@dataclass
class NormStats:
    q01: np.ndarray
    q99: np.ndarray
    # State (proprio) normalization
    state_q01: Optional[np.ndarray] = field(default=None)
    state_q99: Optional[np.ndarray] = field(default=None)
    # Delta-action mask the action stats were fitted on. ``None`` ⇒ stats
    # fitted on absolute actions (the legacy default). When set, action
    # quantiles q01/q99 cover ``action - state`` on masked dims, and
    # consumers (dataset, inference server) MUST apply the matching delta
    # transform before normalize / after unnormalize.
    delta_action_mask: Optional[list[bool]] = field(default=None)

    def to_dict(self):
        d = {"q01": self.q01.tolist(), "q99": self.q99.tolist()}
        if self.state_q01 is not None and self.state_q99 is not None:
            d["state_q01"] = self.state_q01.tolist()
            d["state_q99"] = self.state_q99.tolist()
        if self.delta_action_mask is not None:
            d["delta_action_mask"] = list(self.delta_action_mask)
        return d

    @classmethod
    def from_dict(cls, d):
        sq01 = np.array(d["state_q01"], dtype=np.float32) if "state_q01" in d else None
        sq99 = np.array(d["state_q99"], dtype=np.float32) if "state_q99" in d else None
        mask = list(d["delta_action_mask"]) if "delta_action_mask" in d else None
        return cls(
            q01=np.array(d["q01"], dtype=np.float32),
            q99=np.array(d["q99"], dtype=np.float32),
            state_q01=sq01,
            state_q99=sq99,
            delta_action_mask=mask,
        )


def normalize(actions: np.ndarray, stats: NormStats) -> np.ndarray:
    """Scale to [-1, 1] via: 2 * (x - q01) / (q99 - q01) - 1."""
    range_ = stats.q99 - stats.q01
    range_ = np.where(range_ < 1e-8, 1.0, range_)
    return 2.0 * (actions - stats.q01) / range_ - 1.0


def normalize_state(state: np.ndarray, stats: NormStats) -> np.ndarray:
    """Normalize proprio state to [-1, 1] using state_q01/state_q99."""
    if stats.state_q01 is None:
        return state
    range_ = stats.state_q99 - stats.state_q01
    range_ = np.where(range_ < 1e-8, 1.0, range_)
    return 2.0 * (state - stats.state_q01) / range_ - 1.0


def unnormalize(actions: np.ndarray, stats: NormStats) -> np.ndarray:
    """Inverse: 0.5 * (x + 1) * (q99 - q01) + q01."""
    range_ = stats.q99 - stats.q01
    return 0.5 * (actions + 1.0) * range_ + stats.q01


def _apply_state_offset(
    actions: np.ndarray,
    states: np.ndarray,
    mask: list[bool] | tuple[bool, ...] | np.ndarray,
    *,
    sign: int,
) -> np.ndarray:
    """Internal: ``actions[..., :len(mask)] += sign * state`` on masked dims.

    ``sign=-1`` ⇒ delta encoding (training); ``sign=+1`` ⇒ absolute reconstruction
    (inference). Broadcasts state across an optional chunk axis.
    """
    mask_arr = np.asarray(mask, dtype=bool)
    dims = mask_arr.shape[-1]
    if dims == 0:
        return actions.astype(np.float32, copy=True)
    if actions.shape[-1] < dims or states.shape[-1] < dims:
        raise ValueError(
            f"actions ({actions.shape[-1]}) and states ({states.shape[-1]}) must have"
            f" at least {dims} dims to apply mask of length {dims}"
        )
    out = actions.astype(np.float32, copy=True)
    offset = np.where(mask_arr, states[..., :dims], 0.0).astype(np.float32, copy=False)
    # ``actions`` may have an extra chunk axis (..., H, D); ``states`` is per-frame
    # (..., D). Broadcast across the chunk axis if the rank differs by one.
    if out.ndim == states.ndim + 1:
        offset = np.expand_dims(offset, axis=-2)
    if sign < 0:
        out[..., :dims] -= offset
    else:
        out[..., :dims] += offset
    return out


def apply_absolute_action_mask(
    actions: np.ndarray,
    states: np.ndarray,
    mask: list[bool] | tuple[bool, ...] | np.ndarray,
) -> np.ndarray:
    """Inverse of ``apply_delta_action_mask`` — adds state back on masked dims.

    Used at inference: model predicts deltas, env consumes absolute joints.
    """
    return _apply_state_offset(actions, states, mask, sign=+1)


def save_stats(path: str, stats: NormStats):
    with open(path, "w") as f:
        json.dump(stats.to_dict(), f, indent=2)


def load_stats(path: str) -> NormStats:
    with open(path) as f:
        return NormStats.from_dict(json.load(f))
