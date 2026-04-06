from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFilter


_ROOT = Path(__file__).resolve().parent
_PAYLOAD = torch.load(_ROOT / "model_artifact_patchcore.pt", map_location="cpu", weights_only=False)
_IMAGE_SIZE = int(_PAYLOAD["image_size"])
_SELECTED_INDICES = torch.as_tensor(_PAYLOAD["selected_indices"], dtype=torch.long)
_MEMORY_BANK = torch.from_numpy(np.asarray(_PAYLOAD["memory_bank"], dtype=np.float32))
_BACKBONE = str(_PAYLOAD.get("backbone", "resnet18"))
_THRESHOLD = float(_PAYLOAD["threshold"])
_THRESHOLD_SCALE = float(_PAYLOAD.get("threshold_scale", 1.0))
_MIN_AREA = int(_PAYLOAD["min_area"])
_OPEN_KERNEL = int(_PAYLOAD["open_kernel"])
_CLOSE_KERNEL = int(_PAYLOAD["close_kernel"])
_FINAL_DILATE_KERNEL = int(_PAYLOAD.get("final_dilate_kernel", 1))
_EPS = float(_PAYLOAD.get("eps", 1e-6))
_USE_ROI = bool(_PAYLOAD.get("use_roi", False))
_ROI_MARGIN_RATIO = float(_PAYLOAD.get("roi_margin_ratio", 0.08))
_USE_AXIS_ALIGN = bool(_PAYLOAD.get("use_axis_align", True))
_DISTANCE_CHUNK = int(_PAYLOAD.get("distance_chunk", 2048))
_K_NEIGHBORS = int(_PAYLOAD.get("k_neighbors", 3))
_L2_NORMALIZE = bool(_PAYLOAD.get("l2_normalize", True))


def _conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def _conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class _BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: nn.Module | None = None) -> None:
        super().__init__()
        self.conv1 = _conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        identity = inputs
        outputs = self.conv1(inputs)
        outputs = self.bn1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.bn2(outputs)
        if self.downsample is not None:
            identity = self.downsample(inputs)
        outputs += identity
        outputs = self.relu(outputs)
        return outputs


class _ResNetFeatureExtractor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 2)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 1000)

    def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * _BasicBlock.expansion:
            downsample = nn.Sequential(
                _conv1x1(self.inplanes, planes * _BasicBlock.expansion, stride),
                nn.BatchNorm2d(planes * _BasicBlock.expansion),
            )

        layers = [_BasicBlock(self.inplanes, planes, stride=stride, downsample=downsample)]
        self.inplanes = planes * _BasicBlock.expansion
        for _ in range(1, blocks):
            layers.append(_BasicBlock(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        outputs = self.conv1(inputs)
        outputs = self.bn1(outputs)
        outputs = self.relu(outputs)
        outputs = self.maxpool(outputs)
        feature1 = self.layer1(outputs)
        feature2 = self.layer2(feature1)
        feature3 = self.layer3(feature2)
        return feature1, feature2, feature3


class _Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: nn.Module | None = None) -> None:
        super().__init__()
        self.conv1 = _conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = _conv3x3(planes, planes, stride=stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = _conv1x1(planes, planes * _Bottleneck.expansion)
        self.bn3 = nn.BatchNorm2d(planes * _Bottleneck.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        identity = inputs
        outputs = self.conv1(inputs)
        outputs = self.bn1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.bn2(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv3(outputs)
        outputs = self.bn3(outputs)
        if self.downsample is not None:
            identity = self.downsample(inputs)
        outputs += identity
        outputs = self.relu(outputs)
        return outputs


class _ResNetFeatureExtractor50(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 3)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * _Bottleneck.expansion, 1000)

    def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * _Bottleneck.expansion:
            downsample = nn.Sequential(
                _conv1x1(self.inplanes, planes * _Bottleneck.expansion, stride),
                nn.BatchNorm2d(planes * _Bottleneck.expansion),
            )

        layers = [_Bottleneck(self.inplanes, planes, stride=stride, downsample=downsample)]
        self.inplanes = planes * _Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(_Bottleneck(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        outputs = self.conv1(inputs)
        outputs = self.bn1(outputs)
        outputs = self.relu(outputs)
        outputs = self.maxpool(outputs)
        feature1 = self.layer1(outputs)
        feature2 = self.layer2(feature1)
        feature3 = self.layer3(feature2)
        return feature1, feature2, feature3


def _load_model() -> nn.Module:
    if _BACKBONE == "resnet50":
        model: nn.Module = _ResNetFeatureExtractor50()
    else:
        model = _ResNetFeatureExtractor()
    model.load_state_dict(_PAYLOAD["backbone_state_dict"])
    model.eval()
    return model


_MODEL = _load_model()
_MEMORY_BANK = _MEMORY_BANK.float()


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


def _apply_final_dilation(mask: np.ndarray, dilate_kernel: int) -> np.ndarray:
    if dilate_kernel <= 1:
        return mask.astype(np.uint8, copy=False)

    size = _normalize_kernel_size(dilate_kernel)
    pil_mask = Image.fromarray(mask.astype(np.uint8), mode="L").filter(ImageFilter.MaxFilter(size))
    return (np.asarray(pil_mask, dtype=np.uint8) > 0).astype(np.uint8) * 255


def _normalize_batch(batch: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=batch.dtype).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=batch.dtype).view(1, 3, 1, 1)
    return (batch - mean) / std


def _estimate_cable_region(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    small_side = 160
    resized = np.asarray(
        Image.fromarray(image, mode="RGB").resize((small_side, small_side), Image.BILINEAR),
        dtype=np.float32,
    ) / 255.0
    luma = 0.299 * resized[..., 0] + 0.587 * resized[..., 1] + 0.114 * resized[..., 2]
    threshold = float(np.quantile(luma, 0.82))
    dark_mask = luma < threshold

    yy, xx = np.indices((small_side, small_side), dtype=np.float32)
    cy = (small_side - 1) * 0.5
    cx = (small_side - 1) * 0.5
    ry = small_side * 0.42
    rx = small_side * 0.42
    central_ellipse = (((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2) <= 1.0

    mask_u8 = (np.logical_or(dark_mask, central_ellipse).astype(np.uint8) * 255)
    mask_u8 = np.asarray(Image.fromarray(mask_u8, mode="L").filter(ImageFilter.MaxFilter(11)), dtype=np.uint8)
    mask_u8 = np.asarray(
        Image.fromarray(mask_u8, mode="L").resize((width, height), Image.NEAREST),
        dtype=np.uint8,
    )
    return mask_u8 > 0


def _roi_normalize(image: np.ndarray, margin_ratio: float) -> np.ndarray:
    mask = _estimate_cable_region(image)
    if not np.any(mask):
        return image
    ys, xs = np.where(mask)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    height, width = image.shape[:2]
    y_margin = int((y1 - y0 + 1) * margin_ratio)
    x_margin = int((x1 - x0 + 1) * margin_ratio)
    y0 = max(0, y0 - y_margin)
    y1 = min(height - 1, y1 + y_margin)
    x0 = max(0, x0 - x_margin)
    x1 = min(width - 1, x1 + x_margin)
    return image[y0 : y1 + 1, x0 : x1 + 1]


def _estimate_axis_angle(mask: np.ndarray) -> float:
    ys, xs = np.where(mask)
    if ys.size < 32:
        return 0.0
    coords = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    coords -= coords.mean(axis=0, keepdims=True)
    cov = np.cov(coords.T)
    values, vectors = np.linalg.eigh(cov)
    principal = vectors[:, int(np.argmax(values))]
    angle = float(np.degrees(np.arctan2(principal[1], principal[0])))
    return angle - 90.0


def _axis_align(image: np.ndarray) -> np.ndarray:
    mask = _estimate_cable_region(image)
    angle = _estimate_axis_angle(mask)
    if abs(angle) < 1.0:
        return image
    return np.asarray(Image.fromarray(image, mode="RGB").rotate(angle, resample=Image.BILINEAR), dtype=np.uint8)


def _preprocess_image(image: np.ndarray, size: int) -> np.ndarray:
    working = _roi_normalize(image, _ROI_MARGIN_RATIO) if _USE_ROI else image
    if _USE_AXIS_ALIGN:
        working = _axis_align(working)
    return np.asarray(
        Image.fromarray(working, mode="RGB").resize((size, size), Image.BILINEAR),
        dtype=np.float32,
    )


def _extract_embeddings(batch: torch.Tensor) -> torch.Tensor:
    feature1, feature2, feature3 = _MODEL(_normalize_batch(batch))
    target_size = feature1.shape[-2:]
    feature2 = F.interpolate(feature2, size=target_size, mode="bilinear", align_corners=False)
    feature3 = F.interpolate(feature3, size=target_size, mode="bilinear", align_corners=False)
    combined = torch.cat([feature1, feature2, feature3], dim=1)
    return torch.index_select(combined, dim=1, index=_SELECTED_INDICES)


def _compute_score_map(embeddings: torch.Tensor) -> torch.Tensor:
    if embeddings.ndim == 4:
        embeddings = embeddings[0]
    channels, height, width = embeddings.shape
    flat = embeddings.reshape(channels, height * width).transpose(0, 1)
    if _L2_NORMALIZE:
        flat = F.normalize(flat, dim=1, eps=1e-6)
        memory = F.normalize(_MEMORY_BANK, dim=1, eps=1e-6)
    else:
        memory = _MEMORY_BANK
    k = max(1, int(_K_NEIGHBORS))
    best_dists = torch.full((flat.shape[0], k), float("inf"), device=flat.device)
    for start in range(0, _MEMORY_BANK.shape[0], _DISTANCE_CHUNK):
        chunk = memory[start : start + _DISTANCE_CHUNK]
        dists = torch.cdist(flat, chunk)
        chunk_topk = torch.topk(dists, k=min(k, dists.shape[1]), largest=False).values
        merged = torch.cat([best_dists, chunk_topk], dim=1)
        best_dists = torch.topk(merged, k=k, largest=False).values
    if best_dists.numel() == 0:
        raise RuntimeError("Empty memory bank.")
    scores = torch.mean(best_dists, dim=1).reshape(height, width)
    scores = F.avg_pool2d(scores.unsqueeze(0).unsqueeze(0), kernel_size=7, stride=1, padding=3)[0, 0]
    return scores


def predict(image: np.ndarray) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected RGB image with shape (H, W, 3).")
    if image.dtype != np.uint8:
        raise ValueError("Expected uint8 image.")

    resized = _preprocess_image(image, _IMAGE_SIZE)
    tensor = torch.from_numpy(resized.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    with torch.no_grad():
        embeddings = _extract_embeddings(tensor)
        score_map = _compute_score_map(embeddings).cpu().numpy().astype(np.float32)

    small_mask = score_map >= (_THRESHOLD * _THRESHOLD_SCALE)
    small_mask = _apply_morphology(small_mask, open_kernel=_OPEN_KERNEL, close_kernel=_CLOSE_KERNEL)
    small_mask = _remove_small_components(small_mask, min_area=_MIN_AREA)
    small_mask_u8 = small_mask.astype(np.uint8) * 255

    full_mask = np.asarray(
        Image.fromarray(small_mask_u8, mode="L").resize((image.shape[1], image.shape[0]), Image.NEAREST),
        dtype=np.uint8,
    )
    full_mask = (full_mask > 0).astype(np.uint8) * 255
    return _apply_final_dilation(full_mask, _FINAL_DILATE_KERNEL)
