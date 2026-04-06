from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


_ROOT = Path(__file__).resolve().parent
_PARAMS = np.load(_ROOT / "model_artifact.npz")
_EPS = float(_PARAMS["eps"])
_ENSEMBLE_MODE = str(_PARAMS["ensemble_mode"]) if "ensemble_mode" in _PARAMS.files else "union"
_FINAL_DILATE_KERNEL = int(_PARAMS["final_dilate_kernel"]) if "final_dilate_kernel" in _PARAMS.files else 1


def _load_branches() -> list[dict[str, np.ndarray | int | float]]:
    if "num_branches" in _PARAMS.files:
        num_branches = int(_PARAMS["num_branches"])
        return [
            {
                "feature_size": int(_PARAMS[f"feature_size_{index}"]),
                "threshold": float(_PARAMS[f"threshold_{index}"]),
                "threshold_scale": float(_PARAMS[f"threshold_scale_{index}"])
                if f"threshold_scale_{index}" in _PARAMS.files
                else 1.0,
                "min_area": int(_PARAMS[f"min_area_{index}"]),
                "open_kernel": int(_PARAMS[f"open_kernel_{index}"]),
                "close_kernel": int(_PARAMS[f"close_kernel_{index}"]),
                "mean": _PARAMS[f"mean_{index}"].astype(np.float32),
                "var": _PARAMS[f"var_{index}"].astype(np.float32),
            }
            for index in range(num_branches)
        ]

    return [
        {
            "feature_size": int(_PARAMS["feature_size"]),
            "threshold": float(_PARAMS["threshold"]),
            "threshold_scale": 1.0,
            "min_area": int(_PARAMS["min_area"]),
            "open_kernel": int(_PARAMS["open_kernel"]) if "open_kernel" in _PARAMS.files else 1,
            "close_kernel": int(_PARAMS["close_kernel"]) if "close_kernel" in _PARAMS.files else 1,
            "mean": _PARAMS["mean"].astype(np.float32),
            "var": _PARAMS["var"].astype(np.float32),
        }
    ]


_BRANCHES = _load_branches()


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


def _equalize_hist_u8(image: np.ndarray) -> np.ndarray:
    hist = np.bincount(image.reshape(-1), minlength=256).astype(np.int64)
    cdf = np.cumsum(hist)
    nonzero = np.flatnonzero(cdf)
    if nonzero.size == 0:
        return image.copy()

    cdf_min = cdf[nonzero[0]]
    cdf_max = cdf[-1]
    if cdf_max <= cdf_min:
        return image.copy()

    lut = np.round((cdf - cdf_min) / float(cdf_max - cdf_min) * 255.0)
    lut = np.clip(lut, 0, 255).astype(np.uint8)
    return lut[image]


def _clahe_like_local_normalization(image: np.ndarray, kernel_size: int = 31) -> np.ndarray:
    local_mean = _box_blur_2d(image, kernel_size=kernel_size)
    local_sq_mean = _box_blur_2d(image * image, kernel_size=kernel_size)
    local_var = np.maximum(local_sq_mean - local_mean * local_mean, 1e-4)
    local_std = np.sqrt(local_var)

    normalized = (image - local_mean) / (local_std + 1e-3)
    normalized = np.clip(normalized, -3.0, 3.0)
    return ((normalized + 3.0) / 6.0).astype(np.float32)


def _compute_sobel_laplacian(luma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    padded = np.pad(luma, ((1, 1), (1, 1)), mode="reflect")

    gx = (
        (padded[:-2, 2:] + 2.0 * padded[1:-1, 2:] + padded[2:, 2:])
        - (padded[:-2, :-2] + 2.0 * padded[1:-1, :-2] + padded[2:, :-2])
    )
    gy = (
        (padded[2:, :-2] + 2.0 * padded[2:, 1:-1] + padded[2:, 2:])
        - (padded[:-2, :-2] + 2.0 * padded[:-2, 1:-1] + padded[:-2, 2:])
    )

    grad_mag = np.sqrt(gx * gx + gy * gy)
    lap_abs = np.abs(
        -4.0 * padded[1:-1, 1:-1]
        + padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
    )

    grad_scale = np.percentile(grad_mag, 99.0) + 1e-6
    lap_scale = np.percentile(lap_abs, 99.0) + 1e-6
    grad_norm = np.clip(grad_mag / grad_scale, 0.0, 1.0)
    lap_norm = np.clip(lap_abs / lap_scale, 0.0, 1.0)
    return grad_norm.astype(np.float32), lap_norm.astype(np.float32)


def _extract_features(image: np.ndarray, feature_size: int) -> np.ndarray:
    resized_u8 = np.asarray(
        Image.fromarray(image, mode="RGB").resize((feature_size, feature_size), Image.BILINEAR),
        dtype=np.uint8,
    )
    resized = resized_u8.astype(np.float32) / 255.0

    mean = resized.mean(axis=(0, 1), keepdims=True)
    std = resized.std(axis=(0, 1), keepdims=True) + 1e-5
    normalized_rgb = (resized - mean) / std

    luma = 0.299 * resized[..., 0] + 0.587 * resized[..., 1] + 0.114 * resized[..., 2]
    luma_eq = _equalize_hist_u8((luma * 255.0).astype(np.uint8)).astype(np.float32) / 255.0
    luma_local = _clahe_like_local_normalization(luma_eq)

    grad_mag, lap_abs = _compute_sobel_laplacian(luma_local)
    local_contrast = np.abs(luma_local - _box_blur_2d(luma_local, 9))

    max_rgb = resized.max(axis=2)
    min_rgb = resized.min(axis=2)
    saturation = (max_rgb - min_rgb) / (max_rgb + 1e-4)

    yy, xx = np.indices(luma.shape, dtype=np.float32)
    center_y = (luma.shape[0] - 1) * 0.5
    center_x = (luma.shape[1] - 1) * 0.5
    radial_distance = np.sqrt((yy - center_y) ** 2 + (xx - center_x) ** 2)
    radial_distance /= radial_distance.max() + 1e-6

    return np.concatenate(
        [
            normalized_rgb,
            luma_eq[..., None],
            luma_local[..., None],
            grad_mag[..., None],
            lap_abs[..., None],
            local_contrast[..., None],
            saturation[..., None],
            radial_distance[..., None],
        ],
        axis=2,
    ).astype(np.float32)


def _normalize_kernel_size(kernel_size: int) -> int:
    if kernel_size <= 1:
        return 1
    return kernel_size if kernel_size % 2 == 1 else kernel_size + 1


def _apply_morphology(mask: np.ndarray, open_kernel: int, close_kernel: int) -> np.ndarray:
    mask_u8 = mask.astype(np.uint8) * 255
    pil_mask = Image.fromarray(mask_u8, mode="L")

    if open_kernel > 1:
        size = _normalize_kernel_size(open_kernel)
        pil_mask = pil_mask.filter(ImageFilter.MinFilter(size))
        pil_mask = pil_mask.filter(ImageFilter.MaxFilter(size))

    if close_kernel > 1:
        size = _normalize_kernel_size(close_kernel)
        pil_mask = pil_mask.filter(ImageFilter.MaxFilter(size))
        pil_mask = pil_mask.filter(ImageFilter.MinFilter(size))

    return np.asarray(pil_mask, dtype=np.uint8) > 0


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


def _predict_branch_mask(image: np.ndarray, branch: dict[str, np.ndarray | int | float]) -> np.ndarray:
    feature_size = int(branch["feature_size"])
    features = _extract_features(image, feature_size=feature_size)
    mean = branch["mean"]
    var = branch["var"]
    z_squared = np.square(features - mean) / (var + _EPS)
    score_map = 0.85 * np.mean(z_squared, axis=2, dtype=np.float32) + 0.15 * np.max(z_squared, axis=2)
    score_map = _box_blur_2d(score_map, kernel_size=7)

    small_mask = score_map >= (float(branch["threshold"]) * float(branch["threshold_scale"]))
    small_mask = _apply_morphology(
        small_mask,
        open_kernel=int(branch["open_kernel"]),
        close_kernel=int(branch["close_kernel"]),
    )
    small_mask = _remove_small_components(small_mask, min_area=int(branch["min_area"]))
    small_mask_u8 = small_mask.astype(np.uint8) * 255
    full_mask = np.asarray(
        Image.fromarray(small_mask_u8, mode="L").resize((image.shape[1], image.shape[0]), Image.NEAREST),
        dtype=np.uint8,
    )
    return (full_mask > 0).astype(np.uint8) * 255


def _apply_final_dilation(mask: np.ndarray, dilate_kernel: int) -> np.ndarray:
    if dilate_kernel <= 1:
        return mask.astype(np.uint8, copy=False)

    size = _normalize_kernel_size(dilate_kernel)
    pil_mask = Image.fromarray(mask.astype(np.uint8), mode="L").filter(ImageFilter.MaxFilter(size))
    return (np.asarray(pil_mask, dtype=np.uint8) > 0).astype(np.uint8) * 255


def predict(image: np.ndarray) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected RGB image with shape (H, W, 3).")
    if image.dtype != np.uint8:
        raise ValueError("Expected uint8 image.")
    if _ENSEMBLE_MODE != "union":
        raise ValueError(f"Unsupported ensemble mode: {_ENSEMBLE_MODE}")

    merged = np.zeros(image.shape[:2], dtype=bool)
    for branch in _BRANCHES:
        merged |= _predict_branch_mask(image, branch) > 0
    return _apply_final_dilation(merged.astype(np.uint8) * 255, _FINAL_DILATE_KERNEL)
