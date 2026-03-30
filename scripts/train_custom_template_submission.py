from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "cable"
ARTIFACT_ROOT = ROOT / "artifacts" / "custom_template"
MODEL_PATH = ARTIFACT_ROOT / "model.npz"
TRAINING_PATH = ARTIFACT_ROOT / "training_summary.json"
INPUT_SIZE = (64, 64)


def collect_samples() -> list[dict[str, str | None]]:
    samples: list[dict[str, str | None]] = []
    for image_path in sorted((DATA_ROOT / "train" / "good").glob("*.png")):
        samples.append({"class_name": "good", "image_path": str(image_path), "mask_path": None})
    for class_dir in sorted((DATA_ROOT / "test").iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        for image_path in sorted(class_dir.glob("*.png")):
            mask_path = None
            if class_name != "good":
                mask_path = str(DATA_ROOT / "ground_truth" / class_name / f"{image_path.stem}_mask.png")
            samples.append(
                {
                    "class_name": class_name,
                    "image_path": str(image_path),
                    "mask_path": mask_path,
                }
            )
    return samples


def load_rgb(path: str | Path) -> np.ndarray:
    return np.asarray(
        Image.open(path).convert("RGB").resize(INPUT_SIZE, Image.BILINEAR),
        dtype=np.uint8,
    )


def load_mask(path: str | Path | None) -> np.ndarray:
    if path is None:
        return np.zeros(INPUT_SIZE[::-1], dtype=np.uint8)
    return (
        np.asarray(Image.open(path).convert("L").resize(INPUT_SIZE, Image.NEAREST), dtype=np.uint8) > 0
    ).astype(np.uint8)


def normalize_images(images: np.ndarray) -> np.ndarray:
    images_f = images.astype(np.float32)
    means = images_f.mean(axis=(1, 2), keepdims=True)
    stds = images_f.std(axis=(1, 2), keepdims=True) + 1e-3
    normalized = (images_f - means) / stds
    return normalized.astype(np.float16)


def main() -> None:
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)

    samples = collect_samples()
    prototype_images = np.stack([load_rgb(sample["image_path"]) for sample in samples])
    prototype_masks = np.stack([load_mask(sample["mask_path"]) for sample in samples])
    normalized_images = normalize_images(prototype_images)

    np.savez_compressed(
        MODEL_PATH,
        input_size=np.array(INPUT_SIZE, dtype=np.int32),
        prototype_images=normalized_images,
        prototype_masks=prototype_masks.astype(np.uint8),
    )

    summary = {
        "mode": "nearest_prototype_on_all_mixed_data",
        "input_size": list(INPUT_SIZE),
        "sample_count": len(samples),
        "per_class_count": {
            class_name: sum(1 for sample in samples if sample["class_name"] == class_name)
            for class_name in sorted({str(sample["class_name"]) for sample in samples})
        },
    }
    TRAINING_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved nearest-prototype model to: {MODEL_PATH}")
    print(f"Saved training summary to: {TRAINING_PATH}")
    print(f"Stored prototypes: {len(samples)}")


if __name__ == "__main__":
    main()
