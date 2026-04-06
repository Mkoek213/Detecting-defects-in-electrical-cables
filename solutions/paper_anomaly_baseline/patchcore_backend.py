from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from padim_backend import (
    DEFAULT_AXIS_ALIGN,
    DEFAULT_EPS,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_ROI_MARGIN_RATIO,
    DEFAULT_SELECTED_DIMS,
    axis_align,
    build_model,
    extract_embeddings,
    roi_normalize,
    select_feature_indices,
)


DEFAULT_PATCHES_PER_IMAGE = 200
DEFAULT_MEMORY_SIZE = 40000
DEFAULT_DISTANCE_CHUNK = 2048
DEFAULT_K = 3
DEFAULT_L2_NORM = True


@dataclass(frozen=True)
class PatchCoreArtifact:
    image_size: int
    selected_indices: np.ndarray
    memory_bank: np.ndarray
    threshold: float
    threshold_scale: float
    min_area: int
    open_kernel: int
    close_kernel: int
    final_dilate_kernel: int
    eps: float
    backbone_state_dict: dict[str, torch.Tensor]
    backbone: str = "resnet18"
    use_roi: bool = False
    roi_margin_ratio: float = DEFAULT_ROI_MARGIN_RATIO
    use_axis_align: bool = DEFAULT_AXIS_ALIGN
    patches_per_image: int = DEFAULT_PATCHES_PER_IMAGE
    memory_size: int = DEFAULT_MEMORY_SIZE
    distance_chunk: int = DEFAULT_DISTANCE_CHUNK
    k_neighbors: int = DEFAULT_K
    l2_normalize: bool = DEFAULT_L2_NORM


def preprocess_for_patchcore(
    image: np.ndarray,
    image_size: int,
    use_roi: bool,
    roi_margin_ratio: float,
    use_axis_align: bool,
) -> np.ndarray:
    working = roi_normalize(image, roi_margin_ratio) if use_roi else image
    if use_axis_align:
        working = axis_align(working, use_roi=False, roi_margin_ratio=roi_margin_ratio)
    resized = np.asarray(
        Image.fromarray(working, mode="RGB").resize((image_size, image_size), Image.BILINEAR),
        dtype=np.float32,
    )
    return torch.from_numpy(resized.transpose(2, 0, 1)).float() / 255.0


def compute_patchcore_scores(
    embeddings: torch.Tensor,
    memory_bank: np.ndarray,
    distance_chunk: int,
    k_neighbors: int,
    l2_normalize: bool,
    device: torch.device,
) -> torch.Tensor:
    if embeddings.ndim == 4:
        embeddings = embeddings[0]
    channels, height, width = embeddings.shape
    flat = embeddings.reshape(channels, height * width).transpose(0, 1).to(device=device, dtype=torch.float32)
    memory = torch.from_numpy(memory_bank).to(device=device, dtype=torch.float32)
    if l2_normalize:
        flat = F.normalize(flat, dim=1, eps=1e-6)
        memory = F.normalize(memory, dim=1, eps=1e-6)
    k = max(1, int(k_neighbors))
    best_dists = torch.full((flat.shape[0], k), float("inf"), device=flat.device)
    for start in range(0, memory.shape[0], distance_chunk):
        chunk = memory[start : start + distance_chunk]
        dists = torch.cdist(flat, chunk)
        chunk_topk = torch.topk(dists, k=min(k, dists.shape[1]), largest=False).values
        merged = torch.cat([best_dists, chunk_topk], dim=1)
        best_dists = torch.topk(merged, k=k, largest=False).values
    if best_dists.numel() == 0:
        raise RuntimeError("Empty memory bank for PatchCore.")
    scores = torch.mean(best_dists, dim=1).reshape(height, width)
    scores = F.avg_pool2d(scores.unsqueeze(0).unsqueeze(0), kernel_size=7, stride=1, padding=3)[0, 0]
    return scores


def save_artifact(path: Path, artifact: PatchCoreArtifact) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "backend": "patchcore_resnet18",
        "image_size": int(artifact.image_size),
        "selected_indices": artifact.selected_indices.astype(np.int64),
        "memory_bank": artifact.memory_bank.astype(np.float32),
        "threshold": float(artifact.threshold),
        "threshold_scale": float(artifact.threshold_scale),
        "min_area": int(artifact.min_area),
        "open_kernel": int(artifact.open_kernel),
        "close_kernel": int(artifact.close_kernel),
        "final_dilate_kernel": int(artifact.final_dilate_kernel),
        "eps": float(artifact.eps),
        "backbone": str(artifact.backbone),
        "use_roi": bool(artifact.use_roi),
        "roi_margin_ratio": float(artifact.roi_margin_ratio),
        "use_axis_align": bool(artifact.use_axis_align),
        "patches_per_image": int(artifact.patches_per_image),
        "memory_size": int(artifact.memory_size),
        "distance_chunk": int(artifact.distance_chunk),
        "k_neighbors": int(artifact.k_neighbors),
        "l2_normalize": bool(artifact.l2_normalize),
        "backbone_state_dict": {
            key: value.detach().cpu() for key, value in artifact.backbone_state_dict.items()
        },
    }
    torch.save(payload, path)


def load_artifact(path: Path) -> PatchCoreArtifact:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return PatchCoreArtifact(
        image_size=int(payload["image_size"]),
        selected_indices=np.asarray(payload["selected_indices"], dtype=np.int64),
        memory_bank=np.asarray(payload["memory_bank"], dtype=np.float32),
        threshold=float(payload["threshold"]),
        threshold_scale=float(payload.get("threshold_scale", 1.0)),
        min_area=int(payload["min_area"]),
        open_kernel=int(payload["open_kernel"]),
        close_kernel=int(payload["close_kernel"]),
        final_dilate_kernel=int(payload.get("final_dilate_kernel", 1)),
        eps=float(payload.get("eps", DEFAULT_EPS)),
        backbone=str(payload.get("backbone", "resnet18")),
        use_roi=bool(payload.get("use_roi", False)),
        roi_margin_ratio=float(payload.get("roi_margin_ratio", DEFAULT_ROI_MARGIN_RATIO)),
        use_axis_align=bool(payload.get("use_axis_align", DEFAULT_AXIS_ALIGN)),
        patches_per_image=int(payload.get("patches_per_image", DEFAULT_PATCHES_PER_IMAGE)),
        memory_size=int(payload.get("memory_size", DEFAULT_MEMORY_SIZE)),
        distance_chunk=int(payload.get("distance_chunk", DEFAULT_DISTANCE_CHUNK)),
        k_neighbors=int(payload.get("k_neighbors", DEFAULT_K)),
        l2_normalize=bool(payload.get("l2_normalize", DEFAULT_L2_NORM)),
        backbone_state_dict=payload["backbone_state_dict"],
    )


def load_model_from_artifact(artifact: PatchCoreArtifact) -> torch.nn.Module:
    model = build_model(weights_path=None, backbone=artifact.backbone)
    model.load_state_dict(artifact.backbone_state_dict)
    model.eval()
    return model


__all__ = [
    "DEFAULT_IMAGE_SIZE",
    "DEFAULT_SELECTED_DIMS",
    "DEFAULT_EPS",
    "DEFAULT_PATCHES_PER_IMAGE",
    "DEFAULT_MEMORY_SIZE",
    "DEFAULT_DISTANCE_CHUNK",
    "DEFAULT_K",
    "DEFAULT_L2_NORM",
    "PatchCoreArtifact",
    "preprocess_for_patchcore",
    "compute_patchcore_scores",
    "build_model",
    "extract_embeddings",
    "select_feature_indices",
    "save_artifact",
    "load_artifact",
    "load_model_from_artifact",
]
