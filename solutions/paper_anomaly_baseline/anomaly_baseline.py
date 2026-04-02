from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import json
import numpy as np
from PIL import Image, ImageFilter


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FEATURE_SIZE = 256
DEFAULT_ENSEMBLE_FEATURE_SIZES = (192, 256)
FEATURE_EPS = 1e-6
LOCAL_NORMALIZATION_KERNEL = 31
LOCAL_CONTRAST_KERNEL = 9
SCORE_SMOOTH_KERNEL = 7


@dataclass(frozen=True)
class SplitSample:
    split: str
    class_name: str
    is_good: bool
    source_split: str
    image_path: Path
    mask_path: Path | None


@dataclass(frozen=True)
class BranchArtifact:
    feature_size: int
    threshold: float
    threshold_scale: float
    min_area: int
    open_kernel: int
    close_kernel: int
    mean: np.ndarray
    var: np.ndarray


@dataclass(frozen=True)
class ModelArtifact:
    eps: float
    branches: tuple[BranchArtifact, ...]
    ensemble_mode: str = "union"
    final_dilate_kernel: int = 1


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

    pad = kernel_size // 2
    padded = np.pad(image, ((pad, pad), (pad, pad)), mode="reflect")
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


def equalize_hist_u8(image: np.ndarray) -> np.ndarray:
    if image.dtype != np.uint8:
        raise ValueError("equalize_hist_u8 expects uint8 input.")

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


def clahe_like_local_normalization(image: np.ndarray, kernel_size: int = LOCAL_NORMALIZATION_KERNEL) -> np.ndarray:
    local_mean = _box_blur_2d(image, kernel_size=kernel_size)
    local_sq_mean = _box_blur_2d(image * image, kernel_size=kernel_size)
    local_var = np.maximum(local_sq_mean - local_mean * local_mean, 1e-4)
    local_std = np.sqrt(local_var)

    normalized = (image - local_mean) / (local_std + 1e-3)
    normalized = np.clip(normalized, -3.0, 3.0)
    return ((normalized + 3.0) / 6.0).astype(np.float32)


def compute_sobel_laplacian(luma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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

    laplacian_abs = np.abs(
        -4.0 * padded[1:-1, 1:-1]
        + padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
    )

    grad_scale = np.percentile(grad_mag, 99.0) + 1e-6
    lap_scale = np.percentile(laplacian_abs, 99.0) + 1e-6
    grad_norm = np.clip(grad_mag / grad_scale, 0.0, 1.0)
    lap_norm = np.clip(laplacian_abs / lap_scale, 0.0, 1.0)
    return grad_norm.astype(np.float32), lap_norm.astype(np.float32)


def extract_features(image: np.ndarray, feature_size: int = DEFAULT_FEATURE_SIZE) -> np.ndarray:
    resized_u8 = np.asarray(
        Image.fromarray(image, mode="RGB").resize((feature_size, feature_size), Image.BILINEAR),
        dtype=np.uint8,
    )
    resized = resized_u8.astype(np.float32) / 255.0

    # Per-image normalization reduces exposure/color drift between samples.
    channel_mean = resized.mean(axis=(0, 1), keepdims=True)
    channel_std = resized.std(axis=(0, 1), keepdims=True) + 1e-5
    normalized_rgb = (resized - channel_mean) / channel_std

    luma = 0.299 * resized[..., 0] + 0.587 * resized[..., 1] + 0.114 * resized[..., 2]
    luma_eq = equalize_hist_u8((luma * 255.0).astype(np.uint8)).astype(np.float32) / 255.0
    luma_local = clahe_like_local_normalization(luma_eq)

    gradient_magnitude, laplacian_abs = compute_sobel_laplacian(luma_local)
    local_contrast = np.abs(luma_local - _box_blur_2d(luma_local, LOCAL_CONTRAST_KERNEL))

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
            gradient_magnitude[..., None],
            laplacian_abs[..., None],
            local_contrast[..., None],
            saturation[..., None],
            radial_distance[..., None],
        ],
        axis=2,
    ).astype(np.float32)


def compute_anomaly_map(features: np.ndarray, mean: np.ndarray, var: np.ndarray, eps: float = FEATURE_EPS) -> np.ndarray:
    if features.shape != mean.shape or features.shape != var.shape:
        raise ValueError(f"Mismatched shapes: features={features.shape} mean={mean.shape} var={var.shape}")

    z_squared = np.square(features - mean) / (var + eps)
    score = 0.85 * np.mean(z_squared, axis=2, dtype=np.float32) + 0.15 * np.max(z_squared, axis=2)
    return box_blur(score, SCORE_SMOOTH_KERNEL).astype(np.float32)


def _normalize_kernel_size(kernel_size: int) -> int:
    if kernel_size <= 1:
        return 1
    return kernel_size if kernel_size % 2 == 1 else kernel_size + 1


def apply_morphology(mask: np.ndarray, open_kernel: int, close_kernel: int) -> np.ndarray:
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
    open_kernel: int = 1,
    close_kernel: int = 1,
) -> np.ndarray:
    small_mask = score_map >= threshold
    small_mask = apply_morphology(small_mask, open_kernel=open_kernel, close_kernel=close_kernel)
    small_mask = remove_small_components(small_mask, min_area=min_area)
    mask_uint8 = small_mask.astype(np.uint8) * 255
    resized = np.asarray(
        Image.fromarray(mask_uint8, mode="L").resize((output_shape[1], output_shape[0]), Image.NEAREST),
        dtype=np.uint8,
    )
    return (resized > 0).astype(np.uint8) * 255


def apply_final_dilation(mask: np.ndarray, dilate_kernel: int) -> np.ndarray:
    if dilate_kernel <= 1:
        return mask.astype(np.uint8, copy=False)

    size = _normalize_kernel_size(dilate_kernel)
    dilated = Image.fromarray(mask.astype(np.uint8), mode="L").filter(ImageFilter.MaxFilter(size))
    return (np.asarray(dilated, dtype=np.uint8) > 0).astype(np.uint8) * 255


def mean_iou(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction_bin = prediction > 0
    target_bin = target > 0
    intersection = np.logical_and(prediction_bin, target_bin).sum()
    union = np.logical_or(prediction_bin, target_bin).sum()
    return 1.0 if union == 0 else float(intersection / union)


def save_model_artifact(path: Path, artifact: ModelArtifact) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "artifact_version": np.array(2, dtype=np.int32),
        "eps": np.array(artifact.eps, dtype=np.float32),
        "ensemble_mode": np.array(artifact.ensemble_mode),
        "num_branches": np.array(len(artifact.branches), dtype=np.int32),
        "final_dilate_kernel": np.array(artifact.final_dilate_kernel, dtype=np.int32),
    }
    for index, branch in enumerate(artifact.branches):
        arrays[f"feature_size_{index}"] = np.array(branch.feature_size, dtype=np.int32)
        arrays[f"threshold_{index}"] = np.array(branch.threshold, dtype=np.float32)
        arrays[f"threshold_scale_{index}"] = np.array(branch.threshold_scale, dtype=np.float32)
        arrays[f"min_area_{index}"] = np.array(branch.min_area, dtype=np.int32)
        arrays[f"open_kernel_{index}"] = np.array(branch.open_kernel, dtype=np.int32)
        arrays[f"close_kernel_{index}"] = np.array(branch.close_kernel, dtype=np.int32)
        arrays[f"mean_{index}"] = branch.mean.astype(np.float32)
        arrays[f"var_{index}"] = branch.var.astype(np.float32)
    np.savez_compressed(path, **arrays)


def load_model_artifact(path: Path) -> ModelArtifact:
    params = np.load(path)
    if "num_branches" not in params.files:
        open_kernel = int(params["open_kernel"]) if "open_kernel" in params.files else 1
        close_kernel = int(params["close_kernel"]) if "close_kernel" in params.files else 1
        branch = BranchArtifact(
            feature_size=int(params["feature_size"]),
            threshold=float(params["threshold"]),
            threshold_scale=1.0,
            min_area=int(params["min_area"]),
            open_kernel=open_kernel,
            close_kernel=close_kernel,
            mean=params["mean"].astype(np.float32),
            var=params["var"].astype(np.float32),
        )
        return ModelArtifact(
            eps=float(params["eps"]),
            branches=(branch,),
            ensemble_mode="union",
            final_dilate_kernel=1,
        )

    num_branches = int(params["num_branches"])
    branches: list[BranchArtifact] = []
    for index in range(num_branches):
        branches.append(
            BranchArtifact(
                feature_size=int(params[f"feature_size_{index}"]),
                threshold=float(params[f"threshold_{index}"]),
                threshold_scale=float(params[f"threshold_scale_{index}"])
                if f"threshold_scale_{index}" in params.files
                else 1.0,
                min_area=int(params[f"min_area_{index}"]),
                open_kernel=int(params[f"open_kernel_{index}"]),
                close_kernel=int(params[f"close_kernel_{index}"]),
                mean=params[f"mean_{index}"].astype(np.float32),
                var=params[f"var_{index}"].astype(np.float32),
            )
        )
    ensemble_mode = str(params["ensemble_mode"]) if "ensemble_mode" in params.files else "union"
    return ModelArtifact(
        eps=float(params["eps"]),
        branches=tuple(branches),
        ensemble_mode=ensemble_mode,
        final_dilate_kernel=int(params["final_dilate_kernel"]) if "final_dilate_kernel" in params.files else 1,
    )


def predict_branch_mask_from_score_map(
    score_map: np.ndarray,
    output_shape: tuple[int, int],
    branch: BranchArtifact,
) -> np.ndarray:
    return build_binary_mask(
        score_map=score_map,
        threshold=branch.threshold * branch.threshold_scale,
        min_area=branch.min_area,
        output_shape=output_shape,
        open_kernel=branch.open_kernel,
        close_kernel=branch.close_kernel,
    )


def predict_branch_mask(image: np.ndarray, branch: BranchArtifact, eps: float = FEATURE_EPS) -> np.ndarray:
    features = extract_features(image=image, feature_size=branch.feature_size)
    score_map = compute_anomaly_map(features=features, mean=branch.mean, var=branch.var, eps=eps)
    return predict_branch_mask_from_score_map(
        score_map=score_map,
        output_shape=image.shape[:2],
        branch=branch,
    )


def predict_with_artifact(image: np.ndarray, artifact: ModelArtifact) -> np.ndarray:
    branch_masks = [
        predict_branch_mask(image=image, branch=branch, eps=artifact.eps)
        for branch in artifact.branches
    ]
    if not branch_masks:
        raise RuntimeError("Model artifact does not contain any branches.")
    if len(branch_masks) == 1:
        return branch_masks[0]
    if artifact.ensemble_mode != "union":
        raise ValueError(f"Unsupported ensemble mode: {artifact.ensemble_mode}")

    merged = np.zeros_like(branch_masks[0], dtype=np.uint8)
    for branch_mask in branch_masks:
        merged = np.maximum(merged, branch_mask.astype(np.uint8))
    return apply_final_dilation(merged, artifact.final_dilate_kernel)
