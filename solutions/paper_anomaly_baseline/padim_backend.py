from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFilter


DEFAULT_IMAGE_SIZE = 256
DEFAULT_SELECTED_DIMS = 160
DEFAULT_EPS = 1e-6
DEFAULT_PROFILE_SIZE = 256
DEFAULT_PROFILE_WEIGHT = 0.35
DEFAULT_ROI_MARGIN_RATIO = 0.08
DEFAULT_AXIS_ALIGN = True
DEFAULT_PEAK_COUNT = 6
DEFAULT_PEAK_TOLERANCE_RATIO = 0.03
DEFAULT_PEAK_WEIGHT = 1.0
DEFAULT_TEMPLATE_WEIGHT = 0.5
DEFAULT_TEMPLATE_MODE = "edge"
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)


@dataclass(frozen=True)
class PadimArtifact:
    image_size: int
    selected_indices: np.ndarray
    mean: np.ndarray
    var: np.ndarray
    threshold: float
    threshold_scale: float
    min_area: int
    open_kernel: int
    close_kernel: int
    final_dilate_kernel: int
    eps: float
    backbone_state_dict: dict[str, torch.Tensor]
    use_roi: bool = False
    roi_margin_ratio: float = DEFAULT_ROI_MARGIN_RATIO
    use_profile: bool = False
    profile_size: int = DEFAULT_PROFILE_SIZE
    profile_weight: float = DEFAULT_PROFILE_WEIGHT
    profile_mean: np.ndarray | None = None
    profile_std: np.ndarray | None = None
    use_axis_align: bool = DEFAULT_AXIS_ALIGN
    use_peak_mask: bool = False
    peak_count: int = DEFAULT_PEAK_COUNT
    peak_tolerance_ratio: float = DEFAULT_PEAK_TOLERANCE_RATIO
    peak_weight: float = DEFAULT_PEAK_WEIGHT
    expected_peaks: np.ndarray | None = None
    use_template: bool = False
    template_weight: float = DEFAULT_TEMPLATE_WEIGHT
    template_mode: str = DEFAULT_TEMPLATE_MODE
    template_mean: np.ndarray | None = None
    template_std: np.ndarray | None = None


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride=stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * Bottleneck.expansion)
        self.bn3 = nn.BatchNorm2d(planes * Bottleneck.expansion)
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


class ResNetFeatureExtractor(nn.Module):
    def __init__(self, layers: list[int] = [2, 2, 2, 2]) -> None:
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 1000)

    def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * BasicBlock.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * BasicBlock.expansion, stride),
                nn.BatchNorm2d(planes * BasicBlock.expansion),
            )

        layers = [BasicBlock(self.inplanes, planes, stride=stride, downsample=downsample)]
        self.inplanes = planes * BasicBlock.expansion
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes))
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


class ResNetFeatureExtractor50(nn.Module):
    def __init__(self, layers: list[int] = [3, 4, 6, 3]) -> None:
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * Bottleneck.expansion, 1000)

    def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * Bottleneck.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * Bottleneck.expansion, stride),
                nn.BatchNorm2d(planes * Bottleneck.expansion),
            )

        layers = [Bottleneck(self.inplanes, planes, stride=stride, downsample=downsample)]
        self.inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes))
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


def estimate_cable_region(image: np.ndarray) -> np.ndarray:
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


def roi_normalize(image: np.ndarray, margin_ratio: float) -> np.ndarray:
    mask = estimate_cable_region(image)
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


def preprocess_image(image: np.ndarray, image_size: int, use_roi: bool, roi_margin_ratio: float) -> np.ndarray:
    working = roi_normalize(image, roi_margin_ratio) if use_roi else image
    return np.asarray(
        Image.fromarray(working, mode="RGB").resize((image_size, image_size), Image.BILINEAR),
        dtype=np.float32,
    )


def preprocess_template_image(
    image: np.ndarray,
    image_size: int,
    use_roi: bool,
    roi_margin_ratio: float,
    use_axis_align: bool,
) -> np.ndarray:
    working = roi_normalize(image, roi_margin_ratio) if use_roi else image
    if use_axis_align:
        working = axis_align(working, use_roi=use_roi, roi_margin_ratio=roi_margin_ratio)
    resized = np.asarray(
        Image.fromarray(working, mode="RGB").resize((image_size, image_size), Image.BILINEAR),
        dtype=np.float32,
    )
    return resized / 255.0


def compute_template_source(
    image: np.ndarray,
    image_size: int,
    use_roi: bool,
    roi_margin_ratio: float,
    use_axis_align: bool,
    mode: str,
) -> np.ndarray:
    resized = preprocess_template_image(
        image,
        image_size=image_size,
        use_roi=use_roi,
        roi_margin_ratio=roi_margin_ratio,
        use_axis_align=use_axis_align,
    )
    if mode == "edge":
        luma = 0.299 * resized[..., 0] + 0.587 * resized[..., 1] + 0.114 * resized[..., 2]
        grad_y, grad_x = np.gradient(luma)
        mag = np.sqrt(grad_x * grad_x + grad_y * grad_y)
        return mag.astype(np.float32)
    if mode == "rgb":
        return resized.astype(np.float32)
    raise ValueError(f"Unknown template mode: {mode}")


def estimate_cable_axis_angle(mask: np.ndarray) -> float:
    ys, xs = np.where(mask)
    if ys.size < 32:
        return 0.0
    coords = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    coords -= coords.mean(axis=0, keepdims=True)
    cov = np.cov(coords.T)
    values, vectors = np.linalg.eigh(cov)
    principal = vectors[:, int(np.argmax(values))]
    angle = float(np.degrees(np.arctan2(principal[1], principal[0])))
    # rotate to vertical axis
    return angle - 90.0


def axis_align(image: np.ndarray, use_roi: bool, roi_margin_ratio: float) -> np.ndarray:
    if not use_roi:
        mask = estimate_cable_region(image)
    else:
        mask = estimate_cable_region(image)
    angle = estimate_cable_axis_angle(mask)
    if abs(angle) < 1.0:
        return image
    return np.asarray(Image.fromarray(image, mode="RGB").rotate(angle, resample=Image.BILINEAR), dtype=np.uint8)


def image_to_tensor(image: np.ndarray, image_size: int, use_roi: bool = False, roi_margin_ratio: float = DEFAULT_ROI_MARGIN_RATIO) -> torch.Tensor:
    resized = preprocess_image(image=image, image_size=image_size, use_roi=use_roi, roi_margin_ratio=roi_margin_ratio)
    tensor = torch.from_numpy(resized.transpose(2, 0, 1)).float() / 255.0
    return tensor


def compute_profile_vector(
    image: np.ndarray,
    profile_size: int,
    use_roi: bool,
    roi_margin_ratio: float,
    use_axis_align: bool,
) -> np.ndarray:
    working = roi_normalize(image, roi_margin_ratio) if use_roi else image
    if use_axis_align:
        working = axis_align(working, use_roi=use_roi, roi_margin_ratio=roi_margin_ratio)
    resized = np.asarray(
        Image.fromarray(working, mode="RGB").resize((profile_size, profile_size), Image.BILINEAR),
        dtype=np.float32,
    ) / 255.0
    luma = 0.299 * resized[..., 0] + 0.587 * resized[..., 1] + 0.114 * resized[..., 2]
    grad = np.abs(np.gradient(luma, axis=1))
    profile = grad.mean(axis=0)
    return profile.astype(np.float32)


def compute_profile_map(
    image: np.ndarray,
    profile_mean: np.ndarray,
    profile_std: np.ndarray,
    profile_size: int,
    use_roi: bool,
    roi_margin_ratio: float,
    use_axis_align: bool,
) -> np.ndarray:
    profile = compute_profile_vector(
        image,
        profile_size=profile_size,
        use_roi=use_roi,
        roi_margin_ratio=roi_margin_ratio,
        use_axis_align=use_axis_align,
    )
    std = np.maximum(profile_std, 1e-4)
    z = (profile - profile_mean) / std
    # Missing wires => lower profile response -> negative z.
    z = np.clip(-z, 0.0, 3.0) / 3.0
    return np.repeat(z[None, :], profile_size, axis=0).astype(np.float32)


def compute_template_map(
    image: np.ndarray,
    template_mean: np.ndarray,
    template_std: np.ndarray,
    image_size: int,
    use_roi: bool,
    roi_margin_ratio: float,
    use_axis_align: bool,
    template_mode: str,
) -> np.ndarray:
    source = compute_template_source(
        image,
        image_size=image_size,
        use_roi=use_roi,
        roi_margin_ratio=roi_margin_ratio,
        use_axis_align=use_axis_align,
        mode=template_mode,
    )
    std = np.maximum(template_std, 1e-4)
    if template_mode == "edge":
        z = (template_mean - source) / std
        z = np.clip(z, 0.0, 3.0) / 3.0
        return z.astype(np.float32)
    z = np.abs((source - template_mean) / std)
    z = np.clip(z, 0.0, 3.0) / 3.0
    return np.max(z, axis=2).astype(np.float32)


def detect_profile_peaks(profile: np.ndarray, peak_count: int, min_distance: int) -> np.ndarray:
    if profile.size == 0:
        return np.empty((0,), dtype=np.int32)
    peaks: list[int] = []
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


def build_peak_mask(
    profile: np.ndarray,
    expected_peaks: np.ndarray,
    tolerance: int,
    profile_size: int,
) -> np.ndarray:
    if expected_peaks.size == 0:
        return np.zeros((profile_size, profile_size), dtype=np.float32)
    detected = detect_profile_peaks(profile, peak_count=int(expected_peaks.size), min_distance=max(2, tolerance))
    missing = []
    for peak in expected_peaks:
        if detected.size == 0 or np.min(np.abs(detected - peak)) > tolerance:
            missing.append(int(peak))
    if not missing:
        return np.zeros((profile_size, profile_size), dtype=np.float32)
    mask = np.zeros((profile_size, profile_size), dtype=np.float32)
    for peak in missing:
        x0 = max(0, peak - tolerance)
        x1 = min(profile_size - 1, peak + tolerance)
        mask[:, x0 : x1 + 1] = 1.0
    return mask


def normalize_batch(batch: torch.Tensor) -> torch.Tensor:
    mean = IMAGENET_MEAN.to(device=batch.device, dtype=batch.dtype)
    std = IMAGENET_STD.to(device=batch.device, dtype=batch.dtype)
    return (batch - mean) / std


def combine_feature_maps(features: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
    feature1, feature2, feature3 = features
    target_size = feature1.shape[-2:]
    upsampled = [feature1]
    for feature_map in (feature2, feature3):
        upsampled.append(
            F.interpolate(feature_map, size=target_size, mode="bilinear", align_corners=False)
        )
    return torch.cat(upsampled, dim=1)


def select_feature_indices(total_dims: int, selected_dims: int, seed: int = 20260402) -> np.ndarray:
    if selected_dims <= 0 or selected_dims > total_dims:
        raise ValueError(f"selected_dims must be in [1, {total_dims}], got {selected_dims}")
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(total_dims, size=selected_dims, replace=False)).astype(np.int64)
    return indices


def extract_embeddings(
    model: ResNetFeatureExtractor,
    batch: torch.Tensor,
    selected_indices: np.ndarray | None = None,
) -> torch.Tensor:
    with torch.no_grad():
        features = combine_feature_maps(model(normalize_batch(batch)))
    if selected_indices is None:
        return features
    index_tensor = torch.from_numpy(selected_indices).to(device=features.device, dtype=torch.long)
    return torch.index_select(features, dim=1, index=index_tensor)


def compute_anomaly_map(
    embeddings: torch.Tensor,
    mean: torch.Tensor,
    var: torch.Tensor,
    eps: float = DEFAULT_EPS,
) -> torch.Tensor:
    z_squared = torch.square(embeddings - mean) / (var + eps)
    score = 0.85 * torch.mean(z_squared, dim=1) + 0.15 * torch.amax(z_squared, dim=1)
    return F.avg_pool2d(score.unsqueeze(1), kernel_size=7, stride=1, padding=3)[:, 0]


def build_model(weights_path: Path | None = None, backbone: str = "resnet18") -> nn.Module:
    if backbone == "resnet18":
        model: nn.Module = ResNetFeatureExtractor()
    elif backbone == "resnet50":
        model = ResNetFeatureExtractor50()
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")
    if weights_path is not None:
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state_dict)
    model.eval()
    return model


def save_artifact(path: Path, artifact: PadimArtifact) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "backend": "padim_resnet18_diag",
        "image_size": int(artifact.image_size),
        "selected_indices": artifact.selected_indices.astype(np.int64),
        "mean": artifact.mean.astype(np.float32),
        "var": artifact.var.astype(np.float32),
        "threshold": float(artifact.threshold),
        "threshold_scale": float(artifact.threshold_scale),
        "min_area": int(artifact.min_area),
        "open_kernel": int(artifact.open_kernel),
        "close_kernel": int(artifact.close_kernel),
        "final_dilate_kernel": int(artifact.final_dilate_kernel),
        "eps": float(artifact.eps),
        "use_roi": bool(artifact.use_roi),
        "roi_margin_ratio": float(artifact.roi_margin_ratio),
        "use_profile": bool(artifact.use_profile),
        "profile_size": int(artifact.profile_size),
        "profile_weight": float(artifact.profile_weight),
        "profile_mean": None if artifact.profile_mean is None else artifact.profile_mean.astype(np.float32),
        "profile_std": None if artifact.profile_std is None else artifact.profile_std.astype(np.float32),
        "use_axis_align": bool(artifact.use_axis_align),
        "use_peak_mask": bool(artifact.use_peak_mask),
        "peak_count": int(artifact.peak_count),
        "peak_tolerance_ratio": float(artifact.peak_tolerance_ratio),
        "peak_weight": float(artifact.peak_weight),
        "expected_peaks": None if artifact.expected_peaks is None else artifact.expected_peaks.astype(np.float32),
        "use_template": bool(artifact.use_template),
        "template_weight": float(artifact.template_weight),
        "template_mode": str(artifact.template_mode),
        "template_mean": None if artifact.template_mean is None else artifact.template_mean.astype(np.float32),
        "template_std": None if artifact.template_std is None else artifact.template_std.astype(np.float32),
        "backbone_state_dict": {
            key: value.detach().cpu()
            for key, value in artifact.backbone_state_dict.items()
        },
    }
    torch.save(payload, path)


def load_artifact(path: Path) -> PadimArtifact:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return PadimArtifact(
        image_size=int(payload["image_size"]),
        selected_indices=np.asarray(payload["selected_indices"], dtype=np.int64),
        mean=np.asarray(payload["mean"], dtype=np.float32),
        var=np.asarray(payload["var"], dtype=np.float32),
        threshold=float(payload["threshold"]),
        threshold_scale=float(payload.get("threshold_scale", 1.0)),
        min_area=int(payload["min_area"]),
        open_kernel=int(payload["open_kernel"]),
        close_kernel=int(payload["close_kernel"]),
        final_dilate_kernel=int(payload.get("final_dilate_kernel", 1)),
        eps=float(payload.get("eps", DEFAULT_EPS)),
        use_roi=bool(payload.get("use_roi", False)),
        roi_margin_ratio=float(payload.get("roi_margin_ratio", DEFAULT_ROI_MARGIN_RATIO)),
        use_profile=bool(payload.get("use_profile", False)),
        profile_size=int(payload.get("profile_size", DEFAULT_PROFILE_SIZE)),
        profile_weight=float(payload.get("profile_weight", DEFAULT_PROFILE_WEIGHT)),
        profile_mean=None if payload.get("profile_mean") is None else np.asarray(payload.get("profile_mean"), dtype=np.float32),
        profile_std=None if payload.get("profile_std") is None else np.asarray(payload.get("profile_std"), dtype=np.float32),
        use_axis_align=bool(payload.get("use_axis_align", DEFAULT_AXIS_ALIGN)),
        use_peak_mask=bool(payload.get("use_peak_mask", False)),
        peak_count=int(payload.get("peak_count", DEFAULT_PEAK_COUNT)),
        peak_tolerance_ratio=float(payload.get("peak_tolerance_ratio", DEFAULT_PEAK_TOLERANCE_RATIO)),
        peak_weight=float(payload.get("peak_weight", DEFAULT_PEAK_WEIGHT)),
        expected_peaks=None if payload.get("expected_peaks") is None else np.asarray(payload.get("expected_peaks"), dtype=np.float32),
        use_template=bool(payload.get("use_template", False)),
        template_weight=float(payload.get("template_weight", DEFAULT_TEMPLATE_WEIGHT)),
        template_mode=str(payload.get("template_mode", DEFAULT_TEMPLATE_MODE)),
        template_mean=None if payload.get("template_mean") is None else np.asarray(payload.get("template_mean"), dtype=np.float32),
        template_std=None if payload.get("template_std") is None else np.asarray(payload.get("template_std"), dtype=np.float32),
        backbone_state_dict=payload["backbone_state_dict"],
    )


def load_model_from_artifact(artifact: PadimArtifact) -> ResNetFeatureExtractor:
    model = build_model(weights_path=None)
    model.load_state_dict(artifact.backbone_state_dict)
    model.eval()
    return model
