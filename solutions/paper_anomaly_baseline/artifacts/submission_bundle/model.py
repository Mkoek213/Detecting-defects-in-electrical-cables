from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image


_ROOT = Path(__file__).resolve().parent
_PARAMS = np.load(_ROOT / "model_artifact.npz")
_FEATURE_SIZE = int(_PARAMS["feature_size"])
_EPS = float(_PARAMS["eps"])
_THRESHOLD = float(_PARAMS["threshold"])
_MIN_AREA = int(_PARAMS["min_area"])
_MEAN = _PARAMS["mean"].astype(np.float32)
_VAR = _PARAMS["var"].astype(np.float32)


def _box_blur_2d(image: np.ndarray, kernel_size: int) -> np.ndarray:
    if kernel_size <= 1:
        return image.astype(np.float32, copy=False)

    pad = kernel_size // 2
    padded = np.pad(image, ((pad, pad), (pad, pad)), mode="reflect")
    integral = np.cumsum(np.cumsum(padded, axis=0, dtype=np.float64), axis=1, dtype=np.float64)
    integral = np.pad(integral, ((1, 0), (1, 0)), mode="constant", constant_values=0.0)

    height, width = image.shape
    y0 = np.arange(height)[:, None]
    x0 = np.arange(width)[None, :]
    y1 = y0 + kernel_size
    x1 = x0 + kernel_size

    area_sum = integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]
    return (area_sum / float(kernel_size * kernel_size)).astype(np.float32)


def _box_blur(image: np.ndarray, kernel_size: int) -> np.ndarray:
    if image.ndim == 2:
        return _box_blur_2d(image, kernel_size)
    channels = [_box_blur_2d(image[..., channel], kernel_size) for channel in range(image.shape[2])]
    return np.stack(channels, axis=2)


def _extract_features(image: np.ndarray) -> np.ndarray:
    resized = np.asarray(
        Image.fromarray(image, mode="RGB").resize((_FEATURE_SIZE, _FEATURE_SIZE), Image.BILINEAR),
        dtype=np.float32,
    )
    resized /= 255.0

    mean = resized.mean(axis=(0, 1), keepdims=True)
    std = resized.std(axis=(0, 1), keepdims=True) + 1e-5
    normalized_rgb = (resized - mean) / std

    luma = 0.299 * resized[..., 0] + 0.587 * resized[..., 1] + 0.114 * resized[..., 2]
    dx = np.diff(luma, axis=1, append=luma[:, -1:])
    dy = np.diff(luma, axis=0, append=luma[-1:, :])
    gradient = np.sqrt(dx * dx + dy * dy)
    contrast = np.abs(luma - _box_blur_2d(luma, kernel_size=9))

    features = np.concatenate(
        [
            normalized_rgb,
            luma[..., None],
            gradient[..., None],
            contrast[..., None],
        ],
        axis=2,
    )
    return features.astype(np.float32)


def _remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 1:
        return mask.astype(bool, copy=False)

    binary = mask.astype(bool, copy=False)
    visited = np.zeros(binary.shape, dtype=np.uint8)
    output = np.zeros(binary.shape, dtype=bool)
    height, width = binary.shape

    for row in range(height):
        for col in range(width):
            if not binary[row, col] or visited[row, col]:
                continue

            queue: deque[tuple[int, int]] = deque([(row, col)])
            visited[row, col] = 1
            component: list[tuple[int, int]] = []

            while queue:
                y, x = queue.pop()
                component.append((y, x))

                y_min = max(0, y - 1)
                y_max = min(height - 1, y + 1)
                x_min = max(0, x - 1)
                x_max = min(width - 1, x + 1)

                for ny in range(y_min, y_max + 1):
                    for nx in range(x_min, x_max + 1):
                        if ny == y and nx == x:
                            continue
                        if binary[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = 1
                            queue.append((ny, nx))

            if len(component) >= min_area:
                ys, xs = zip(*component)
                output[ys, xs] = True

    return output


def predict(image: np.ndarray) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected RGB image with shape (H, W, 3).")
    if image.dtype != np.uint8:
        raise ValueError("Expected uint8 image.")

    features = _extract_features(image)
    z_squared = np.square(features - _MEAN) / (_VAR + _EPS)
    score_map = np.mean(z_squared, axis=2, dtype=np.float32)
    score_map = _box_blur_2d(score_map, kernel_size=5)

    small_mask = score_map >= _THRESHOLD
    small_mask = _remove_small_components(small_mask, min_area=_MIN_AREA)
    small_mask_uint8 = small_mask.astype(np.uint8) * 255

    full_mask = np.asarray(
        Image.fromarray(small_mask_uint8, mode="L").resize((image.shape[1], image.shape[0]), Image.NEAREST),
        dtype=np.uint8,
    )
    return (full_mask > 0).astype(np.uint8) * 255
