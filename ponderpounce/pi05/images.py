"""Image helpers shared by the Pi0.5 S1 path (inference)."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image as PILImage


def make_pi0_image_transform(size: int = 224):
    """PIL list → ``(N, 3, size, size)`` float in ``[0, 1]``."""

    def _resize(pils: list[PILImage.Image]) -> torch.Tensor:
        target = (size, size)
        resized = [img if img.size == target else img.resize(target, PILImage.Resampling.BILINEAR) for img in pils]
        arr = np.stack([np.asarray(r, dtype=np.uint8) for r in resized])
        return torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous().float().div_(255.0)

    return _resize


def pad_camera_slots(
    pixels: torch.Tensor,
    n_slots: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero-pad a camera batch up to ``n_slots`` and return the matching
    bool attention mask.

    Pi0 / Pi0.5 base ships with a fixed 3-camera input; RoboMME yields
    1–2 cams, so unused slots are filled with zeros and masked out.

    Args:
        pixels: ``(B, N_real, 3, H, W)`` — leading axis is batch or tick.
        n_slots: target number of camera slots (3 for pi0/pi05 base).

    Returns:
        ``(padded, mask)`` — ``(B, n_slots, 3, H, W)`` and ``(B, n_slots)`` bool.
    """
    B, n_real = pixels.shape[:2]
    if n_real > n_slots:
        raise ValueError(f"{n_real} cameras exceed the {n_slots} slots the S1 backbone expects")
    mask = torch.zeros(B, n_slots, dtype=torch.bool, device=pixels.device)
    mask[:, :n_real] = True
    if n_real == n_slots:
        return pixels, mask
    pad = torch.zeros(B, n_slots - n_real, *pixels.shape[2:], dtype=pixels.dtype, device=pixels.device)
    return torch.cat([pixels, pad], dim=1), mask
