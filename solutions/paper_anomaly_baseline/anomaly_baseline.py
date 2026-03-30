from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import json
import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FEATURE_SIZE = 256
FEATURE_EPS = 1e-6
CONTRAST_KERNEL = 9
SCORE_SMOOTH_KERNEL = 5


@dataclass(frozen=True)
class SplitSample:
    split: str
    class_name: str
    is_good: bool
    source_split: str
    image_path: Path
    mask_path: Path | None


@dataclass(frozen=True)
class ModelArtifact:
    feature_size: int
    eps: float
    threshold: float
    min_area: int
    mean: np.ndarray
    var: np.ndarray


def load_split_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_split_samples(manifest: dict[str, Any], split: str) -> list[SplitSample]:
    rows = manifest["splits"].get(split, [])
    samples: list[SplitSample] = []
    for row in rows:
        image_path = (REPO_ROOT / row["image_path"]).resolve()
        mask_path = None
        if row["mask_path"] is not None:
            mask_path = (REPO_ROOT / row["mask_path"]).resolve()
        samples.append(
            SplitSample(
                split=row["split"],
                class_name=row["class_name"],
                is_good=bool(row["is_good"]),
                source_split=row["source_split"],
                image_path=image_path,
                mask_path=mask_path,
            )
        )
    return samples


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def load_binary_mask(path: Path | None, shape: tuple[int, int]) -> np.ndarray:
    if path is None:
        return np.zeros(shape, dtype=np.uint8)
    mask = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    if mask.shape != shape:
        mask = np.asarray(
            Image.fromarray(mask, mode="L").resize((shape[1], shape[0]), Image.NEAREST),
            dtype=np.uint8,
        )
    return (mask > 0).astype(np.uint8) * 255


def _box_blur_2d(image: np.ndarray, kernel_size: int) -> np.ndarray:
    if kernel_size <= 1:
        return image.astype(np.float32, copy=False)
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd.")

    padded = np.pad(image, ((kernel_size // 2, kernel_size // 2), (kernel_size // 2, kernel_size // 2)), mode="reflect")
    integral = np.cumsum(np.cumsum(padded, axis=0, dtype=np.float64), axis=1, dtype=np.float64)
    integral = np.pad(integral, ((1, 0), (1, 0)), mode="constant", constant_values=0.0)

    height, width = image.shape
    y0 = np.arange(height)[:, None]
    x0 = np.arange(width)[None, :]
    y1 = y0 + kernel_size
    x1 = x0 + kernel_size

    sums = integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]
    return (sums / float(kernel_size * kernel_size)).astype(np.float32)


def box_blur(image: np.ndarray, kernel_size: int) -> np.ndarray:
    if image.ndim == 2:
        return _box_blur_2d(image, kernel_size)
    if image.ndim == 3:
        channels = [_box_blur_2d(image[..., index], kernel_size) for index in range(image.shape[2])]
        return np.stack(channels, axis=2)
    raise ValueError(f"Unsupported shape for box blur: {image.shape}")


def extract_features(image: np.ndarray, feature_size: int = DEFAULT_FEATURE_SIZE) -> np.ndarray:
    resized = np.asarray(
        Image.fromarray(image, mode="RGB").resize((feature_size, feature_size), Image.BILINEAR),
        dtype=np.float32,
    )
    resized /= 255.0

    channel_mean = resized.mean(axis=(0, 1), keepdims=True)
    channel_std = resized.std(axis=(0, 1), keepdims=True) + 1e-5
    normalized_rgb = (resized - channel_mean) / channel_std

    luma = 0.299 * resized[..., 0] + 0.587 * resized[..., 1] + 0.114 * resized[..., 2]
    dx = np.diff(luma, axis=1, append=luma[:, -1:])
    dy = np.diff(luma, axis=0, append=luma[-1:, :])
    gradient_magnitude = np.sqrt(dx * dx + dy * dy)
    local_contrast = np.abs(luma - box_blur(luma, CONTRAST_KERNEL))

    return np.concatenate(
        [
            normalized_rgb,
            luma[..., None],
            gradient_magnitude[..., None],
            local_contrast[..., None],
        ],
        axis=2,
    ).astype(np.float32)


def compute_anomaly_map(features: np.ndarray, mean: np.ndarray, var: np.ndarray, eps: float = FEATURE_EPS) -> np.ndarray:
    if features.shape != mean.shape or features.shape != var.shape:
        raise ValueError(
            f"Mismatched shapes: features={features.shape} mean={mean.shape} var={var.shape}"
        )
    z_squared = np.square(features - mean) / (var + eps)
    score = np.mean(z_squared, axis=2, dtype=np.float32)
    return box_blur(score, SCORE_SMOOTH_KERNEL).astype(np.float32)


def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 1:
        return mask.astype(bool, copy=False)

    binary = mask.astype(bool, copy=False)
    visited = np.zeros(binary.shape, dtype=np.uint8)
    kept = np.zeros(binary.shape, dtype=bool)
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
                kept[ys, xs] = True

    return kept


def build_binary_mask(
    score_map: np.ndarray,
    threshold: float,
    min_area: int,
    output_shape: tuple[int, int],
) -> np.ndarray:
    small_mask = score_map >= threshold
    small_mask = remove_small_components(small_mask, min_area=min_area)
    mask_uint8 = small_mask.astype(np.uint8) * 255
    resized = np.asarray(
        Image.fromarray(mask_uint8, mode="L").resize((output_shape[1], output_shape[0]), Image.NEAREST),
        dtype=np.uint8,
    )
    return (resized > 0).astype(np.uint8) * 255


def mean_iou(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction_bin = prediction > 0
    target_bin = target > 0
    intersection = np.logical_and(prediction_bin, target_bin).sum()
    union = np.logical_or(prediction_bin, target_bin).sum()
    return 1.0 if union == 0 else float(intersection / union)


def save_model_artifact(path: Path, artifact: ModelArtifact) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        feature_size=np.array(artifact.feature_size, dtype=np.int32),
        eps=np.array(artifact.eps, dtype=np.float32),
        threshold=np.array(artifact.threshold, dtype=np.float32),
        min_area=np.array(artifact.min_area, dtype=np.int32),
        mean=artifact.mean.astype(np.float32),
        var=artifact.var.astype(np.float32),
    )


def load_model_artifact(path: Path) -> ModelArtifact:
    params = np.load(path)
    return ModelArtifact(
        feature_size=int(params["feature_size"]),
        eps=float(params["eps"]),
        threshold=float(params["threshold"]),
        min_area=int(params["min_area"]),
        mean=params["mean"].astype(np.float32),
        var=params["var"].astype(np.float32),
    )


def predict_with_artifact(image: np.ndarray, artifact: ModelArtifact) -> np.ndarray:
    features = extract_features(image=image, feature_size=artifact.feature_size)
    score_map = compute_anomaly_map(features=features, mean=artifact.mean, var=artifact.var, eps=artifact.eps)
    return build_binary_mask(
        score_map=score_map,
        threshold=artifact.threshold,
        min_area=artifact.min_area,
        output_shape=image.shape[:2],
    )
