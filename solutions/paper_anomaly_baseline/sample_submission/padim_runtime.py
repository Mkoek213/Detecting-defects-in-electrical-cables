from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFilter


_ROOT = Path(__file__).resolve().parent
_PAYLOAD = torch.load(_ROOT / "model_artifact_padim.pt", map_location="cpu", weights_only=False)
_IMAGE_SIZE = int(_PAYLOAD["image_size"])
_SELECTED_INDICES = torch.as_tensor(_PAYLOAD["selected_indices"], dtype=torch.long)
_MEAN = torch.from_numpy(np.transpose(np.asarray(_PAYLOAD["mean"], dtype=np.float32), (2, 0, 1))).unsqueeze(0)
_VAR = torch.from_numpy(np.transpose(np.asarray(_PAYLOAD["var"], dtype=np.float32), (2, 0, 1))).unsqueeze(0)
_THRESHOLD = float(_PAYLOAD["threshold"])
_THRESHOLD_SCALE = float(_PAYLOAD.get("threshold_scale", 1.0))
_MIN_AREA = int(_PAYLOAD["min_area"])
_OPEN_KERNEL = int(_PAYLOAD["open_kernel"])
_CLOSE_KERNEL = int(_PAYLOAD["close_kernel"])
_FINAL_DILATE_KERNEL = int(_PAYLOAD.get("final_dilate_kernel", 1))
_EPS = float(_PAYLOAD.get("eps", 1e-6))
_USE_ROI = bool(_PAYLOAD.get("use_roi", False))
_ROI_MARGIN_RATIO = float(_PAYLOAD.get("roi_margin_ratio", 0.08))
_USE_PROFILE = bool(_PAYLOAD.get("use_profile", False))
_PROFILE_SIZE = int(_PAYLOAD.get("profile_size", 256))
_PROFILE_WEIGHT = float(_PAYLOAD.get("profile_weight", 0.35))
_PROFILE_MEAN = None if _PAYLOAD.get("profile_mean") is None else np.asarray(_PAYLOAD.get("profile_mean"), dtype=np.float32)
_PROFILE_STD = None if _PAYLOAD.get("profile_std") is None else np.asarray(_PAYLOAD.get("profile_std"), dtype=np.float32)
_USE_AXIS_ALIGN = bool(_PAYLOAD.get("use_axis_align", True))
_USE_PEAK_MASK = bool(_PAYLOAD.get("use_peak_mask", False))
_PEAK_COUNT = int(_PAYLOAD.get("peak_count", 6))
_PEAK_TOLERANCE_RATIO = float(_PAYLOAD.get("peak_tolerance_ratio", 0.03))
_PEAK_WEIGHT = float(_PAYLOAD.get("peak_weight", 1.0))
_EXPECTED_PEAKS = None if _PAYLOAD.get("expected_peaks") is None else np.asarray(_PAYLOAD.get("expected_peaks"), dtype=np.float32)
_USE_TEMPLATE = bool(_PAYLOAD.get("use_template", False))
_TEMPLATE_WEIGHT = float(_PAYLOAD.get("template_weight", 0.5))
_TEMPLATE_MODE = str(_PAYLOAD.get("template_mode", "edge"))
_TEMPLATE_MEAN = None if _PAYLOAD.get("template_mean") is None else np.asarray(_PAYLOAD.get("template_mean"), dtype=np.float32)
_TEMPLATE_STD = None if _PAYLOAD.get("template_std") is None else np.asarray(_PAYLOAD.get("template_std"), dtype=np.float32)


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


def _load_model() -> _ResNetFeatureExtractor:
    model = _ResNetFeatureExtractor()
    model.load_state_dict(_PAYLOAD["backbone_state_dict"])
    model.eval()
    return model


_MODEL = _load_model()
_MEAN = _MEAN.float()
_VAR = _VAR.float()


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


def _compute_profile_map(image: np.ndarray) -> np.ndarray | None:
    if not _USE_PROFILE or _PROFILE_MEAN is None or _PROFILE_STD is None:
        return None
    resized = _preprocess_image(image, _PROFILE_SIZE) / 255.0
    luma = 0.299 * resized[..., 0] + 0.587 * resized[..., 1] + 0.114 * resized[..., 2]
    grad = np.abs(np.gradient(luma, axis=1))
    profile = grad.mean(axis=0)
    std = np.maximum(_PROFILE_STD, 1e-4)
    z = (profile - _PROFILE_MEAN) / std
    z = np.clip(-z, 0.0, 3.0) / 3.0
    return np.repeat(z[None, :], _PROFILE_SIZE, axis=0).astype(np.float32)


def _compute_template_map(image: np.ndarray) -> np.ndarray | None:
    if not _USE_TEMPLATE or _TEMPLATE_MEAN is None or _TEMPLATE_STD is None:
        return None
    resized = _preprocess_image(image, _IMAGE_SIZE) / 255.0
    std = np.maximum(_TEMPLATE_STD, 1e-4)
    if _TEMPLATE_MODE == "edge":
        luma = 0.299 * resized[..., 0] + 0.587 * resized[..., 1] + 0.114 * resized[..., 2]
        grad_y, grad_x = np.gradient(luma)
        mag = np.sqrt(grad_x * grad_x + grad_y * grad_y)
        z = (_TEMPLATE_MEAN - mag) / std
        z = np.clip(z, 0.0, 3.0) / 3.0
        return z.astype(np.float32)
    z = np.abs((resized - _TEMPLATE_MEAN) / std)
    z = np.clip(z, 0.0, 3.0) / 3.0
    return np.max(z, axis=2).astype(np.float32)


def _detect_peaks(profile: np.ndarray, peak_count: int, min_distance: int) -> np.ndarray:
    peaks = []
    values = profile.copy()
    for _ in range(max(1, peak_count)):
        idx = int(np.argmax(values))
        if values[idx] <= 0:
            break
        peaks.append(idx)
        left = max(0, idx - min_distance)
        right = min(values.size, idx + min_distance + 1)
        values[left:right] = -np.inf
    return np.sort(np.asarray(peaks, dtype=np.int32))


def _build_peak_mask(image: np.ndarray) -> np.ndarray | None:
    if not _USE_PEAK_MASK or _EXPECTED_PEAKS is None:
        return None
    resized = _preprocess_image(image, _PROFILE_SIZE) / 255.0
    luma = 0.299 * resized[..., 0] + 0.587 * resized[..., 1] + 0.114 * resized[..., 2]
    grad = np.abs(np.gradient(luma, axis=1))
    profile = grad.mean(axis=0)
    tolerance = max(2, int(_PROFILE_SIZE * _PEAK_TOLERANCE_RATIO))
    detected = _detect_peaks(profile, peak_count=int(_PEAK_COUNT), min_distance=tolerance)
    missing = []
    for peak in _EXPECTED_PEAKS.astype(np.int32):
        if detected.size == 0 or np.min(np.abs(detected - peak)) > tolerance:
            missing.append(int(peak))
    if not missing:
        return None
    mask = np.zeros((_PROFILE_SIZE, _PROFILE_SIZE), dtype=np.float32)
    for peak in missing:
        x0 = max(0, peak - tolerance)
        x1 = min(_PROFILE_SIZE - 1, peak + tolerance)
        mask[:, x0 : x1 + 1] = 1.0
    return mask


def _extract_embeddings(batch: torch.Tensor) -> torch.Tensor:
    feature1, feature2, feature3 = _MODEL(_normalize_batch(batch))
    target_size = feature1.shape[-2:]
    feature2 = F.interpolate(feature2, size=target_size, mode="bilinear", align_corners=False)
    feature3 = F.interpolate(feature3, size=target_size, mode="bilinear", align_corners=False)
    combined = torch.cat([feature1, feature2, feature3], dim=1)
    return torch.index_select(combined, dim=1, index=_SELECTED_INDICES)


def _compute_score_map(embeddings: torch.Tensor) -> torch.Tensor:
    z_squared = torch.square(embeddings - _MEAN) / (_VAR + _EPS)
    score = 0.85 * torch.mean(z_squared, dim=1) + 0.15 * torch.amax(z_squared, dim=1)
    return F.avg_pool2d(score.unsqueeze(1), kernel_size=7, stride=1, padding=3)[0, 0]


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
    profile_map = _compute_profile_map(image)
    if profile_map is not None:
        profile_map = np.asarray(
            Image.fromarray(profile_map, mode="F").resize((score_map.shape[1], score_map.shape[0]), Image.BILINEAR),
            dtype=np.float32,
        )
        score_map = score_map + float(_PROFILE_WEIGHT) * profile_map
    template_map = _compute_template_map(image)
    if template_map is not None:
        template_map = np.asarray(
            Image.fromarray(template_map, mode="F").resize((score_map.shape[1], score_map.shape[0]), Image.BILINEAR),
            dtype=np.float32,
        )
        score_map = np.maximum(score_map, float(_TEMPLATE_WEIGHT) * template_map)
    peak_mask = _build_peak_mask(image)
    if peak_mask is not None:
        peak_mask = np.asarray(
            Image.fromarray(peak_mask, mode="F").resize((score_map.shape[1], score_map.shape[0]), Image.BILINEAR),
            dtype=np.float32,
        )
        score_map = np.maximum(score_map, float(_PEAK_WEIGHT) * peak_mask)

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
