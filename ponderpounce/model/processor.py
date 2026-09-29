"""PonderPounce Processor: resize + normalize for S1 images.

Handles resize and normalize only. No augmentation — that's a training-only
concern handled by the dataset, not the processor.

Used identically in training and inference, analogous to HuggingFace
AutoProcessor. Each S1 arch builds one via ``config.s1.build_s1_image_transform()``
with its own ``img_size``/normalization, so the processor stays arch-agnostic.
"""

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

# SigLIP-trained vision towers expect ``mean=std=[0.5, 0.5, 0.5]`` ⇒ x/255 then
# map [0,1]→[-1,1]. Anything else (EfficientNet-style towers) uses ImageNet stats.
_NORM_STATS = {
    "siglip": ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    "imagenet": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
}


class S1ImageProcessor:
    """Preprocesses S1 images: resize to ``img_size``, normalize, → tensor.

    No augmentation (crop, jitter) — that belongs in the dataset for training.
    At inference, images pass through here directly from the camera.
    """

    def __init__(self, *, img_size: int, normalization: str):
        self.s1_img_size = img_size
        if normalization not in _NORM_STATS:
            raise ValueError(f"normalization must be one of {tuple(_NORM_STATS)}, got {normalization!r}")
        mean, std = _NORM_STATS[normalization]

        self._mean = torch.as_tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
        self._std = torch.as_tensor(std, dtype=torch.float32).view(1, 3, 1, 1)

        self.s1_transform = T.Compose(
            [
                T.Resize((img_size, img_size)),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ]
        )

    def s1_transform_batch(self, imgs: list[Image.Image]) -> torch.Tensor:
        """Resize + normalize a flat list of PIL images as one batch.

        ~3x faster than calling ``s1_transform`` in a Python loop for typical
        S1 schedules (one torch.from_numpy + one elementwise normalize vs per-
        image ``ToTensor`` and ``Normalize`` invocations).
        """
        size = (self.s1_img_size, self.s1_img_size)
        # PIL resize runs at C speed; keep PIL path for bilinear consistency.
        resized = [img if img.size == size else img.resize(size, Image.Resampling.BILINEAR) for img in imgs]
        stacked = np.stack([np.asarray(r, dtype=np.uint8) for r in resized])  # (N, H, W, 3)
        tensor = torch.from_numpy(stacked).permute(0, 3, 1, 2).contiguous().float().mul_(1.0 / 255.0)
        return tensor.sub_(self._mean).div_(self._std)
