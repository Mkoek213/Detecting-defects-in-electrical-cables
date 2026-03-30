from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


_ROOT = Path(__file__).resolve().parent
_PARAMS = np.load(_ROOT / "model.npz")
_INPUT_SIZE = tuple(int(value) for value in _PARAMS["input_size"].tolist())
_PROTOTYPE_IMAGES = _PARAMS["prototype_images"].astype(np.float32)
_PROTOTYPE_MASKS = _PARAMS["prototype_masks"].astype(np.uint8)


def _normalize_image(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    mean = image.mean(axis=(0, 1), keepdims=True)
    std = image.std(axis=(0, 1), keepdims=True) + 1e-3
    return (image - mean) / std


def predict(image: np.ndarray) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected RGB image with shape (H, W, 3).")
    if image.dtype != np.uint8:
        raise ValueError("Expected uint8 image.")

    height, width = image.shape[:2]
    resized = np.asarray(
        Image.fromarray(image, mode="RGB").resize(_INPUT_SIZE, Image.BILINEAR),
        dtype=np.uint8,
    )
    normalized = _normalize_image(resized)
    distances = np.mean(np.abs(_PROTOTYPE_IMAGES - normalized[None, ...]), axis=(1, 2, 3))
    nearest_index = int(np.argmin(distances))
    mask = (_PROTOTYPE_MASKS[nearest_index] > 0).astype(np.uint8) * 255
    resized_mask = Image.fromarray(mask, mode="L").resize((width, height), Image.NEAREST)
    return (np.asarray(resized_mask, dtype=np.uint8) > 0).astype(np.uint8) * 255
