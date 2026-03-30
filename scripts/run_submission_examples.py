from __future__ import annotations

import argparse
import importlib.util
import json
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "cable"
DEFAULT_MODEL_PATH = ROOT / "submissions" / "cable_submission" / "model.py"
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "submission_examples"


@dataclass(frozen=True)
class Sample:
    class_name: str
    image_path: Path
    mask_path: Path | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a submission model on all cable examples.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--examples-per-class",
        type=int,
        default=3,
        help="How many example panels to save per class.",
    )
    return parser.parse_args()


def load_submission_model(model_path: Path):
    spec = importlib.util.spec_from_file_location("submission_model", model_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load model from {model_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "predict"):
        raise RuntimeError(f"Model file {model_path} does not define predict(image)")
    return module


def collect_samples() -> list[Sample]:
    samples: list[Sample] = []
    for class_dir in sorted((DATA_ROOT / "test").iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        for image_path in sorted(class_dir.glob("*.png")):
            mask_path = None
            if class_name != "good":
                mask_path = DATA_ROOT / "ground_truth" / class_name / f"{image_path.stem}_mask.png"
            samples.append(Sample(class_name=class_name, image_path=image_path, mask_path=mask_path))
    return samples


def load_image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def load_mask(path: Path | None, shape: tuple[int, int]) -> np.ndarray:
    if path is None:
        return np.zeros(shape, dtype=np.uint8)
    return (np.asarray(Image.open(path).convert("L")) > 0).astype(np.uint8) * 255


def mean_iou(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction_bin = prediction > 0
    target_bin = target > 0
    intersection = np.logical_and(prediction_bin, target_bin).sum()
    union = np.logical_or(prediction_bin, target_bin).sum()
    return 1.0 if union == 0 else float(intersection / union)


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    rgb = np.zeros(mask.shape + (3,), dtype=np.uint8)
    rgb[mask > 0] = np.array([255, 255, 255], dtype=np.uint8)
    return rgb


def error_overlay(image: np.ndarray, target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    target_bin = target > 0
    pred_bin = prediction > 0

    true_positive = target_bin & pred_bin
    false_positive = (~target_bin) & pred_bin
    false_negative = target_bin & (~pred_bin)

    overlay = np.clip(image.astype(np.float32) * 0.45, 0, 255).astype(np.uint8)
    overlay[true_positive] = np.array([0, 220, 0], dtype=np.uint8)
    overlay[false_positive] = np.array([255, 0, 0], dtype=np.uint8)
    overlay[false_negative] = np.array([0, 120, 255], dtype=np.uint8)
    return overlay


def tile_with_caption(image: np.ndarray, caption: str, width: int, font: ImageFont.ImageFont) -> Image.Image:
    panel = Image.fromarray(image, mode="RGB").resize((width, width), Image.BILINEAR)
    canvas = Image.new("RGB", (width, width + 22), color=(18, 18, 18))
    canvas.paste(panel, (0, 22))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 5), caption, fill=(235, 235, 235), font=font)
    return canvas


def save_panel(
    sample: Sample,
    image: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    iou: float,
    elapsed_ms: float,
    output_dir: Path,
) -> None:
    font = ImageFont.load_default()
    tile_width = 224
    margin = 8

    panels = [
        tile_with_caption(image, "image", tile_width, font),
        tile_with_caption(mask_to_rgb(target), "ground truth", tile_width, font),
        tile_with_caption(mask_to_rgb(prediction), "prediction", tile_width, font),
        tile_with_caption(error_overlay(image, target, prediction), "errors", tile_width, font),
    ]

    panel_width = sum(panel.width for panel in panels) + margin * (len(panels) - 1)
    panel_height = max(panel.height for panel in panels) + 32
    canvas = Image.new("RGB", (panel_width, panel_height), color=(28, 28, 28))
    draw = ImageDraw.Draw(canvas)
    header = (
        f"{sample.class_name} / {sample.image_path.name}  "
        f"IoU={iou:.4f}  time={elapsed_ms:.1f} ms"
    )
    draw.text((8, 8), header, fill=(255, 220, 170), font=font)

    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 32))
        x += panel.width + margin

    file_name = f"{sample.class_name}_{sample.image_path.stem}.png"
    canvas.save(output_dir / file_name)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_submission_model(args.model_path)
    samples = collect_samples()

    saved_per_class: dict[str, int] = defaultdict(int)
    rows: list[dict[str, object]] = []

    for index, sample in enumerate(samples, start=1):
        image = load_image(sample.image_path)
        target = load_mask(sample.mask_path, image.shape[:2])

        started = time.perf_counter()
        prediction = model.predict(image)
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        score = mean_iou(prediction, target)
        rows.append(
            {
                "class_name": sample.class_name,
                "image_path": str(sample.image_path),
                "mask_path": None if sample.mask_path is None else str(sample.mask_path),
                "iou": score,
                "elapsed_ms": elapsed_ms,
            }
        )

        if saved_per_class[sample.class_name] < args.examples_per_class:
            save_panel(sample, image, target, prediction, score, elapsed_ms, output_dir)
            saved_per_class[sample.class_name] += 1

        print(
            f"[{index:03d}/{len(samples):03d}] {sample.class_name:>20}  "
            f"IoU={score:.4f}  time={elapsed_ms:.1f} ms  file={sample.image_path.name}"
        )

    summary = {
        "model_path": str(args.model_path),
        "output_dir": str(output_dir),
        "examples_per_class": args.examples_per_class,
        "num_samples": len(rows),
        "saved_examples_per_class": dict(sorted(saved_per_class.items())),
    }
    (output_dir / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nSaved example panels to: {output_dir}")
    print(f"Saved per-image results to: {output_dir / 'results.json'}")
    print(f"Saved summary to: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
