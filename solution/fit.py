from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image
from sklearn.random_projection import SparseRandomProjection
import torch
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import Wide_ResNet50_2_Weights
from tqdm import tqdm


DEFECT_CLASSES = [
    "bent_wire",
    "cable_swap",
    "combined",
    "cut_inner_insulation",
    "cut_outer_insulation",
    "missing_cable",
    "missing_wire",
    "poke_insulation",
]

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass(frozen=True)
class Sample:
    image_path: Path
    is_defect: bool
    class_name: str
    mask_path: Path | None


def _sorted_pngs(folder: Path) -> list[Path]:
    return sorted(folder.glob("*.png"), key=lambda path: str(path.resolve()))


def collect_samples(data_dir: Path) -> list[Sample]:
    train_good_dir = data_dir / "train" / "good"
    test_good_dir = data_dir / "test" / "good"
    test_dir = data_dir / "test"
    gt_dir = data_dir / "ground_truth"

    samples: list[Sample] = []
    good_paths = _sorted_pngs(train_good_dir) + _sorted_pngs(test_good_dir)
    good_paths = sorted(good_paths, key=lambda path: str(path.resolve()))
    for image_path in good_paths:
        samples.append(Sample(image_path=image_path, is_defect=False, class_name="good", mask_path=None))

    defect_samples: list[Sample] = []
    for class_name in DEFECT_CLASSES:
        class_test_dir = test_dir / class_name
        class_gt_dir = gt_dir / class_name
        for image_path in _sorted_pngs(class_test_dir):
            mask_path = class_gt_dir / f"{image_path.stem}_mask.png"
            if not mask_path.exists():
                raise FileNotFoundError(f"Missing mask for {image_path}: {mask_path}")
            defect_samples.append(
                Sample(
                    image_path=image_path,
                    is_defect=True,
                    class_name=class_name,
                    mask_path=mask_path,
                )
            )

    defect_samples = sorted(defect_samples, key=lambda sample: str(sample.image_path.resolve()))
    samples.extend(defect_samples)

    return samples


def build_train_val_split(
    data_dir: Path,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[list[Sample], list[Sample]]:
    samples = collect_samples(data_dir)
    rng = random.Random(seed)
    rng.shuffle(samples)

    split_index = int(len(samples) * train_ratio)
    train_set = samples[:split_index]
    val_set = samples[split_index:]
    return train_set, val_set


def load_rgb_image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def preprocess_image(image_rgb_uint8: np.ndarray) -> torch.Tensor:
    resized = Image.fromarray(image_rgb_uint8, mode="RGB").resize((256, 256), Image.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    normalized = (array - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(normalized.transpose(2, 0, 1)[None])
    return tensor


class PatchCoreFeatureExtractor:
    def __init__(self, device: torch.device | None = None) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone = models.wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        self.backbone.eval().to(self.device)
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)

        self._features: dict[str, torch.Tensor] = {}
        self.backbone.layer2.register_forward_hook(self._hook("layer2"))
        self.backbone.layer3.register_forward_hook(self._hook("layer3"))

    def _hook(self, name: str):
        def fn(_, __, output: torch.Tensor) -> None:
            self._features[name] = output

        return fn

    def extract_raw_patches(self, image_rgb_uint8: np.ndarray) -> np.ndarray:
        tensor = preprocess_image(image_rgb_uint8).to(self.device)

        self._features.clear()
        with torch.no_grad():
            self.backbone(tensor)

        layer2 = self._features["layer2"]
        layer3 = F.interpolate(
            self._features["layer3"],
            size=(32, 32),
            mode="bilinear",
            align_corners=False,
        )
        combined = torch.cat([layer2, layer3], dim=1)
        aggregated = F.avg_pool2d(combined, kernel_size=3, stride=1, padding=1)
        patches = aggregated.squeeze(0).permute(1, 2, 0).reshape(32 * 32, 1536)
        return patches.detach().cpu().numpy().astype(np.float32)


def reduce_patches(patches: np.ndarray, projection: np.ndarray) -> np.ndarray:
    return (patches @ projection).astype(np.float32)


def compute_anomaly_map(patch_feats: np.ndarray, coreset: np.ndarray, k: int = 3) -> np.ndarray:
    if coreset.shape[0] == 0:
        raise ValueError("Coreset is empty.")

    effective_k = max(1, min(k, coreset.shape[0]))
    a2 = np.sum(patch_feats**2, axis=1, keepdims=True)
    b2 = np.sum(coreset**2, axis=1, keepdims=True).T
    ab = patch_feats @ coreset.T
    dists = np.sqrt(np.clip(a2 + b2 - 2.0 * ab, 0.0, None))

    topk = np.partition(dists, kth=effective_k - 1, axis=1)[:, :effective_k]
    scores = topk.mean(axis=1)
    return scores.reshape(32, 32).astype(np.float32)


def greedy_coreset(
    features: np.ndarray,
    ratio: float = 0.1,
    seed: int = 42,
    max_points: int = 20_000,
) -> np.ndarray:
    rng = np.random.default_rng(seed)

    num_points = int(features.shape[0])
    if num_points == 0:
        raise ValueError("No features were provided for coreset construction.")

    target = max(int(num_points * ratio), 1)
    target = min(target, max_points)
    if target >= num_points:
        return features.astype(np.float32, copy=True)

    selected_indices = [int(rng.integers(num_points))]
    min_dists = np.full(num_points, np.inf, dtype=np.float32)
    feat_norms = np.sum(features * features, axis=1)

    for _ in tqdm(range(target - 1), desc="Greedy coreset", leave=False):
        last_idx = selected_indices[-1]
        last_vec = features[last_idx]
        last_norm = float(np.dot(last_vec, last_vec))
        dists = np.clip(feat_norms + last_norm - 2.0 * (features @ last_vec), 0.0, None)
        min_dists = np.minimum(min_dists, dists.astype(np.float32, copy=False))
        selected_indices.append(int(np.argmax(min_dists)))

    return features[np.asarray(selected_indices, dtype=np.int64)].astype(np.float32)


def _extract_raw_train_patches(train_set: Sequence[Sample], extractor: PatchCoreFeatureExtractor) -> np.ndarray:
    total = len(train_set) * 1024
    stacked = np.empty((total, 1536), dtype=np.float32)

    cursor = 0
    for sample in tqdm(train_set, desc="Extracting train patches"):
        image = load_rgb_image(sample.image_path)
        patches = extractor.extract_raw_patches(image)
        stacked[cursor : cursor + patches.shape[0], :] = patches
        cursor += patches.shape[0]

    return stacked


def fit_patchcore(
    train_set: Sequence[Sample],
    output_dir: Path,
    ratio: float = 0.1,
    seed: int = 42,
    coreset_keep: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    output_dir.mkdir(parents=True, exist_ok=True)
    extractor = PatchCoreFeatureExtractor()

    train_raw = _extract_raw_train_patches(train_set, extractor)

    projector = SparseRandomProjection(n_components=128, random_state=seed)
    projector.fit(train_raw)

    projection = projector.components_.toarray().T.astype(np.float32)
    np.save(output_dir / "projection_components.npy", projection)

    reduced_train = reduce_patches(train_raw, projection)
    coreset = greedy_coreset(reduced_train, ratio=ratio, seed=seed, max_points=20_000)
    if coreset_keep is not None:
        keep = max(1, min(int(coreset_keep), int(coreset.shape[0])))
        coreset = coreset[:keep]
    np.save(output_dir / "coreset.npy", coreset.astype(np.float32))

    return coreset.astype(np.float32), projection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit PatchCore coreset for cable defect segmentation.")
    parser.add_argument("--data_dir", type=Path, required=True, help="Path to data/cable directory.")
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory where coreset.npy and projection_components.npy are saved.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split/projection/coreset.")
    parser.add_argument("--ratio", type=float, default=0.1, help="Coreset ratio before max cap.")
    parser.add_argument(
        "--coreset_size",
        type=int,
        default=10_000,
        help="Number of coreset vectors to keep after greedy sampling.",
    )
    parser.add_argument(
        "--include_defect_train",
        action="store_true",
        help="Include defect samples from the mixed split in coreset fitting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_set, val_set = build_train_val_split(args.data_dir, train_ratio=0.8, seed=args.seed)
    fit_train_set = train_set if args.include_defect_train else [sample for sample in train_set if not sample.is_defect]

    if len(fit_train_set) == 0:
        raise RuntimeError("No training samples available for fitting.")

    coreset, projection = fit_patchcore(
        train_set=fit_train_set,
        output_dir=args.output_dir,
        ratio=args.ratio,
        seed=args.seed,
        coreset_keep=args.coreset_size,
    )

    print("PatchCore fitting complete")
    print(f"Train images: {len(train_set)}")
    print(f"Fit images:   {len(fit_train_set)}")
    print(f"Val images:   {len(val_set)}")
    print(f"Projection:   {projection.shape}")
    print(f"Coreset:      {coreset.shape}")
    print(f"Saved:        {args.output_dir / 'projection_components.npy'}")
    print(f"Saved:        {args.output_dir / 'coreset.npy'}")


if __name__ == "__main__":
    main()