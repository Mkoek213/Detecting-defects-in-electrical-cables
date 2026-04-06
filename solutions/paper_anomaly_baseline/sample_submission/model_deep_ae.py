from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFilter


_ROOT = Path(__file__).resolve().parent
_PAYLOAD = torch.load(_ROOT / "model_artifact.pt", map_location="cpu", weights_only=False)
_IMAGE_SIZE = int(_PAYLOAD["image_size"])
_THRESHOLD = float(_PAYLOAD["threshold"])
_THRESHOLD_SCALE = float(_PAYLOAD.get("threshold_scale", 1.0))
_MIN_AREA = int(_PAYLOAD["min_area"])
_OPEN_KERNEL = int(_PAYLOAD["open_kernel"])
_CLOSE_KERNEL = int(_PAYLOAD["close_kernel"])
_FINAL_DILATE_KERNEL = int(_PAYLOAD.get("final_dilate_kernel", 1))
_SCORE_BLUR_KERNEL = int(_PAYLOAD.get("score_blur_kernel", 7))
_RGB_WEIGHT = float(_PAYLOAD.get("rgb_weight", 0.7))
_GRAD_WEIGHT = float(_PAYLOAD.get("grad_weight", 0.3))
_CHANNELS = tuple(int(value) for value in _PAYLOAD.get("channels", [16, 32, 64, 96, 128]))


def _num_groups(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class _ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.norm = nn.GroupNorm(_num_groups(out_channels), out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(inputs)))


class _DeconvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.deconv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
        )
        self.norm = nn.GroupNorm(_num_groups(out_channels), out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.deconv(inputs)))


class _DenoisingAutoencoder(nn.Module):
    def __init__(self, channels: tuple[int, ...]) -> None:
        super().__init__()
        encoder_layers: list[nn.Module] = [_ConvBlock(3, channels[0], stride=1)]
        in_channels = channels[0]
        for out_channels in channels[1:]:
            encoder_layers.append(_ConvBlock(in_channels, out_channels, stride=2))
            encoder_layers.append(_ConvBlock(out_channels, out_channels, stride=1))
            in_channels = out_channels
        encoder_layers.append(_ConvBlock(in_channels, in_channels, stride=1))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers: list[nn.Module] = []
        reversed_channels = list(reversed(channels))
        in_channels = reversed_channels[0]
        for out_channels in reversed_channels[1:]:
            decoder_layers.append(_DeconvBlock(in_channels, out_channels))
            decoder_layers.append(_ConvBlock(out_channels, out_channels, stride=1))
            in_channels = out_channels
        decoder_layers.append(nn.Conv2d(in_channels, 3, kernel_size=3, padding=1))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(inputs)
        outputs = self.decoder(latent)
        return torch.sigmoid(outputs)


def _load_model() -> _DenoisingAutoencoder:
    model = _DenoisingAutoencoder(_CHANNELS)
    model.load_state_dict(_PAYLOAD["state_dict"])
    model.eval()
    return model


_MODEL = _load_model()


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


def _luma_channel(images: torch.Tensor) -> torch.Tensor:
    return (
        0.299 * images[:, 0:1]
        + 0.587 * images[:, 1:2]
        + 0.114 * images[:, 2:3]
    )


def _sobel_magnitude(images: torch.Tensor) -> torch.Tensor:
    kernel_x = torch.tensor(
        [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]],
        dtype=images.dtype,
    ).unsqueeze(0)
    kernel_y = torch.tensor(
        [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]],
        dtype=images.dtype,
    ).unsqueeze(0)
    grad_x = F.conv2d(images, kernel_x, padding=1)
    grad_y = F.conv2d(images, kernel_y, padding=1)
    return torch.sqrt(grad_x * grad_x + grad_y * grad_y + 1e-6)


def _score_map_from_tensors(inputs: torch.Tensor, reconstructions: torch.Tensor) -> torch.Tensor:
    rgb_error = torch.mean(torch.abs(inputs - reconstructions), dim=1, keepdim=True)
    grad_error = torch.abs(_sobel_magnitude(_luma_channel(inputs)) - _sobel_magnitude(_luma_channel(reconstructions)))
    score = _RGB_WEIGHT * rgb_error + _GRAD_WEIGHT * grad_error
    if _SCORE_BLUR_KERNEL > 1:
        padding = _SCORE_BLUR_KERNEL // 2
        score = F.avg_pool2d(score, kernel_size=_SCORE_BLUR_KERNEL, stride=1, padding=padding)
    return score[0, 0]


def _predict_small_mask(image: np.ndarray) -> np.ndarray:
    resized = np.asarray(
        Image.fromarray(image, mode="RGB").resize((_IMAGE_SIZE, _IMAGE_SIZE), Image.BILINEAR),
        dtype=np.float32,
    )
    tensor = torch.from_numpy(resized.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    with torch.no_grad():
        reconstruction = _MODEL(tensor)
        score_map = _score_map_from_tensors(tensor, reconstruction).cpu().numpy().astype(np.float32)

    mask = score_map >= (_THRESHOLD * _THRESHOLD_SCALE)
    mask = _apply_morphology(mask, open_kernel=_OPEN_KERNEL, close_kernel=_CLOSE_KERNEL)
    mask = _remove_small_components(mask, min_area=_MIN_AREA)
    return mask.astype(np.uint8) * 255


def predict(image: np.ndarray) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected RGB image with shape (H, W, 3).")
    if image.dtype != np.uint8:
        raise ValueError("Expected uint8 image.")

    small_mask = _predict_small_mask(image)
    full_mask = np.asarray(
        Image.fromarray(small_mask, mode="L").resize((image.shape[1], image.shape[0]), Image.NEAREST),
        dtype=np.uint8,
    )
    full_mask = (full_mask > 0).astype(np.uint8) * 255
    return _apply_final_dilation(full_mask, _FINAL_DILATE_KERNEL)
