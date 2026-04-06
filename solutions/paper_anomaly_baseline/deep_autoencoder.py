from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


DEFAULT_IMAGE_SIZE = 256
DEFAULT_SCORE_BLUR_KERNEL = 7
DEFAULT_RGB_WEIGHT = 0.7
DEFAULT_GRAD_WEIGHT = 0.3


@dataclass(frozen=True)
class DeepArtifact:
    image_size: int
    threshold: float
    threshold_scale: float
    min_area: int
    open_kernel: int
    close_kernel: int
    final_dilate_kernel: int
    score_blur_kernel: int
    rgb_weight: float
    grad_weight: float
    channels: tuple[int, ...]
    state_dict: dict[str, torch.Tensor]


def _num_groups(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ConvBlock(nn.Module):
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


class DeconvBlock(nn.Module):
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


class DenoisingAutoencoder(nn.Module):
    def __init__(self, channels: tuple[int, ...] = (16, 32, 64, 96, 128)) -> None:
        super().__init__()
        if len(channels) < 2:
            raise ValueError("channels must contain at least two stages.")
        self.channels = tuple(int(value) for value in channels)

        encoder_layers: list[nn.Module] = [ConvBlock(3, self.channels[0], stride=1)]
        in_channels = self.channels[0]
        for out_channels in self.channels[1:]:
            encoder_layers.append(ConvBlock(in_channels, out_channels, stride=2))
            encoder_layers.append(ConvBlock(out_channels, out_channels, stride=1))
            in_channels = out_channels
        encoder_layers.append(ConvBlock(in_channels, in_channels, stride=1))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers: list[nn.Module] = []
        reversed_channels = list(reversed(self.channels))
        in_channels = reversed_channels[0]
        for out_channels in reversed_channels[1:]:
            decoder_layers.append(DeconvBlock(in_channels, out_channels))
            decoder_layers.append(ConvBlock(out_channels, out_channels, stride=1))
            in_channels = out_channels
        decoder_layers.append(nn.Conv2d(in_channels, 3, kernel_size=3, padding=1))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(inputs)
        outputs = self.decoder(latent)
        return torch.sigmoid(outputs)


def image_to_tensor(image: np.ndarray, image_size: int) -> torch.Tensor:
    resized = np.asarray(
        Image.fromarray(image, mode="RGB").resize((image_size, image_size), Image.BILINEAR),
        dtype=np.float32,
    )
    return torch.from_numpy(resized.transpose(2, 0, 1)).float() / 255.0


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.detach().cpu().clamp(0.0, 1.0).numpy()
    return np.transpose(array, (1, 2, 0))


def apply_training_corruption(batch: torch.Tensor) -> torch.Tensor:
    corrupted = batch.clone()
    corrupted = torch.clamp(corrupted + torch.randn_like(corrupted) * 0.03, 0.0, 1.0)

    batch_size, _, height, width = corrupted.shape
    for index in range(batch_size):
        if torch.rand(1).item() < 0.7:
            rect_height = int(torch.randint(height // 10, max(height // 4, height // 10) + 1, (1,)).item())
            rect_width = int(torch.randint(width // 10, max(width // 4, width // 10) + 1, (1,)).item())
            top = int(torch.randint(0, max(1, height - rect_height + 1), (1,)).item())
            left = int(torch.randint(0, max(1, width - rect_width + 1), (1,)).item())
            fill = corrupted[index : index + 1].mean(dim=(2, 3), keepdim=True)
            corrupted[index, :, top : top + rect_height, left : left + rect_width] = fill[0]

        color_scale = 1.0 + torch.empty(3, device=corrupted.device).uniform_(-0.08, 0.08)
        corrupted[index] = torch.clamp(corrupted[index] * color_scale[:, None, None], 0.0, 1.0)

    return corrupted


def luma_channel(images: torch.Tensor) -> torch.Tensor:
    return (
        0.299 * images[:, 0:1]
        + 0.587 * images[:, 1:2]
        + 0.114 * images[:, 2:3]
    )


def sobel_magnitude(images: torch.Tensor) -> torch.Tensor:
    kernel_x = torch.tensor(
        [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]],
        device=images.device,
        dtype=images.dtype,
    ).unsqueeze(0)
    kernel_y = torch.tensor(
        [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]],
        device=images.device,
        dtype=images.dtype,
    ).unsqueeze(0)
    grad_x = F.conv2d(images, kernel_x, padding=1)
    grad_y = F.conv2d(images, kernel_y, padding=1)
    return torch.sqrt(grad_x * grad_x + grad_y * grad_y + 1e-6)


def reconstruction_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    rgb_loss = F.l1_loss(predictions, targets)
    grad_loss = F.l1_loss(sobel_magnitude(luma_channel(predictions)), sobel_magnitude(luma_channel(targets)))
    return 0.8 * rgb_loss + 0.2 * grad_loss


def score_map_from_tensors(
    inputs: torch.Tensor,
    reconstructions: torch.Tensor,
    rgb_weight: float = DEFAULT_RGB_WEIGHT,
    grad_weight: float = DEFAULT_GRAD_WEIGHT,
    blur_kernel: int = DEFAULT_SCORE_BLUR_KERNEL,
) -> torch.Tensor:
    rgb_error = torch.mean(torch.abs(inputs - reconstructions), dim=1, keepdim=True)
    grad_error = torch.abs(sobel_magnitude(luma_channel(inputs)) - sobel_magnitude(luma_channel(reconstructions)))
    score = rgb_weight * rgb_error + grad_weight * grad_error
    if blur_kernel > 1:
        padding = blur_kernel // 2
        score = F.avg_pool2d(score, kernel_size=blur_kernel, stride=1, padding=padding)
    return score[:, 0]


def save_artifact(path: Path, artifact: DeepArtifact) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "backend": "deep_denoising_autoencoder",
        "image_size": int(artifact.image_size),
        "threshold": float(artifact.threshold),
        "threshold_scale": float(artifact.threshold_scale),
        "min_area": int(artifact.min_area),
        "open_kernel": int(artifact.open_kernel),
        "close_kernel": int(artifact.close_kernel),
        "final_dilate_kernel": int(artifact.final_dilate_kernel),
        "score_blur_kernel": int(artifact.score_blur_kernel),
        "rgb_weight": float(artifact.rgb_weight),
        "grad_weight": float(artifact.grad_weight),
        "channels": list(int(value) for value in artifact.channels),
        "state_dict": {key: value.detach().cpu() for key, value in artifact.state_dict.items()},
    }
    torch.save(payload, path)


def load_artifact(path: Path) -> DeepArtifact:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return DeepArtifact(
        image_size=int(payload["image_size"]),
        threshold=float(payload["threshold"]),
        threshold_scale=float(payload.get("threshold_scale", 1.0)),
        min_area=int(payload["min_area"]),
        open_kernel=int(payload["open_kernel"]),
        close_kernel=int(payload["close_kernel"]),
        final_dilate_kernel=int(payload.get("final_dilate_kernel", 1)),
        score_blur_kernel=int(payload.get("score_blur_kernel", DEFAULT_SCORE_BLUR_KERNEL)),
        rgb_weight=float(payload.get("rgb_weight", DEFAULT_RGB_WEIGHT)),
        grad_weight=float(payload.get("grad_weight", DEFAULT_GRAD_WEIGHT)),
        channels=tuple(int(value) for value in payload.get("channels", [16, 32, 64, 96, 128])),
        state_dict=payload["state_dict"],
    )


def build_model_from_artifact(artifact: DeepArtifact) -> DenoisingAutoencoder:
    model = DenoisingAutoencoder(channels=artifact.channels)
    model.load_state_dict(artifact.state_dict)
    model.eval()
    return model
