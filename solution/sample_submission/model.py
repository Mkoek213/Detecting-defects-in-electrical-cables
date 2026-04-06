"""
sample_submission/model.py
Self-contained inference module. No training at runtime.
Required files (same directory): coreset.npy, projection_components.npy, threshold.npy
"""

import os

import numpy as np
from PIL import Image
import scipy.ndimage
import torch
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import Wide_ResNet50_2_Weights


_DIR = os.path.dirname(os.path.abspath(__file__))
CORESET = np.load(os.path.join(_DIR, "coreset.npy")).astype(np.float32)
PROJECTION = np.load(os.path.join(_DIR, "projection_components.npy")).astype(np.float32)
THRESHOLD = float(np.load(os.path.join(_DIR, "threshold.npy")))

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_BACKBONE = models.wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)
_BACKBONE.eval().to(_DEVICE)
for parameter in _BACKBONE.parameters():
    parameter.requires_grad_(False)

_FEATURES: dict[str, torch.Tensor] = {}


def _hook(name: str):
    def fn(_, __, output: torch.Tensor) -> None:
        _FEATURES[name] = output

    return fn


_BACKBONE.layer2.register_forward_hook(_hook("layer2"))
_BACKBONE.layer3.register_forward_hook(_hook("layer3"))

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_ANOMALY_K = 3
_GAUSSIAN_SIGMA = 0.5
_MIN_COMPONENT_AREA = 128


def _extract_patches(image_rgb_uint8: np.ndarray) -> np.ndarray:
    img = Image.fromarray(image_rgb_uint8, mode="RGB").resize((256, 256), Image.BILINEAR)
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = (x - _MEAN) / _STD
    t = torch.from_numpy(x.transpose(2, 0, 1)[None]).to(_DEVICE)

    _FEATURES.clear()
    with torch.no_grad():
        _BACKBONE(t)

    l2 = _FEATURES["layer2"]
    l3 = F.interpolate(_FEATURES["layer3"], size=(32, 32), mode="bilinear", align_corners=False)
    combined = torch.cat([l2, l3], dim=1)
    aggregated = F.avg_pool2d(combined, kernel_size=3, stride=1, padding=1)

    patches = aggregated.squeeze(0).permute(1, 2, 0).reshape(1024, 1536).detach().cpu().numpy().astype(np.float32)
    reduced = patches @ PROJECTION
    return reduced.astype(np.float32)


def _anomaly_map(patches: np.ndarray) -> np.ndarray:
    a2 = np.sum(patches**2, axis=1, keepdims=True)
    b2 = np.sum(CORESET**2, axis=1, keepdims=True).T
    ab = patches @ CORESET.T
    dists = np.sqrt(np.clip(a2 + b2 - 2.0 * ab, 0.0, None))

    k = max(1, min(_ANOMALY_K, int(CORESET.shape[0])))
    scores = np.partition(dists, kth=k - 1, axis=1)[:, :k].mean(axis=1)
    return scores.reshape(32, 32).astype(np.float32)


def _postprocess_binary_mask(mask01: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 0:
        return mask01.astype(np.uint8)

    labeled, num_labels = scipy.ndimage.label(mask01.astype(bool))
    if num_labels == 0:
        return np.zeros_like(mask01, dtype=np.uint8)

    counts = np.bincount(labeled.ravel())
    keep_labels = np.where(counts >= int(min_area))[0]
    keep_labels = keep_labels[keep_labels != 0]
    if keep_labels.size == 0:
        return np.zeros_like(mask01, dtype=np.uint8)

    filtered = np.isin(labeled, keep_labels)
    return filtered.astype(np.uint8)


def predict(image: np.ndarray) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected RGB image with shape (H, W, 3).")
    if image.dtype != np.uint8:
        raise ValueError("Expected uint8 image.")

    height, width = image.shape[:2]

    patches = _extract_patches(image)
    score_32 = _anomaly_map(patches)

    zoom_h = height / 32.0
    zoom_w = width / 32.0
    score_full = scipy.ndimage.zoom(score_32, (zoom_h, zoom_w), order=1)

    if score_full.shape != (height, width):
        score_full = np.asarray(
            Image.fromarray(score_full.astype(np.float32), mode="F").resize((width, height), Image.BILINEAR),
            dtype=np.float32,
        )

    score_full = score_full.astype(np.float32)
    if _GAUSSIAN_SIGMA > 0:
        score_full = scipy.ndimage.gaussian_filter(score_full, sigma=_GAUSSIAN_SIGMA).astype(np.float32)

    mask01 = (score_full > THRESHOLD).astype(np.uint8)
    mask01 = _postprocess_binary_mask(mask01, min_area=_MIN_COMPONENT_AREA)
    return (mask01 * 255).astype(np.uint8).reshape(height, width)