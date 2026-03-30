from __future__ import annotations

import argparse
import importlib.util
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "cable"
MODEL_PATH = ROOT / "sample_submission" / "model.py"
DEFAULT_VIS_DIR = ROOT / "artifacts" / "eval_vis"


@dataclass(frozen=True)
class Sample:
    class_name: str
    image_path: Path
    mask_path: Path | None


@dataclass(frozen=True)
class EvalResult:
    sample: Sample
    iou: float
    elapsed_ms: float
    image: np.ndarray
    target: np.ndarray
    prediction: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a cable-defect submission.")
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--class-name", type=str, default=None, help="Evaluate only one class.")
    parser.add_argument("--save-vis", action="store_true", help="Save visual diagnostics for the worst cases.")
    parser.add_argument("--vis-dir", type=Path, default=DEFAULT_VIS_DIR)
    parser.add_argument("--top-k", type=int, default=24, help="How many lowest-IoU examples to visualize.")
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
    test_root = DATA_ROOT / "test"

    for class_dir in sorted(test_root.iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        for image_path in sorted(class_dir.glob("*.png")):
            if class_name == "good":
                mask_path = None
            else:
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


def _mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    rgb = np.zeros(mask.shape + (3,), dtype=np.uint8)
    rgb[mask > 0] = np.array([255, 255, 255], dtype=np.uint8)
    return rgb


def _error_overlay(image: np.ndarray, target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
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


def _tile_with_caption(image: np.ndarray, caption: str, width: int, font: ImageFont.ImageFont) -> Image.Image:
    pil_image = Image.fromarray(image, mode="RGB").resize((width, width), Image.BILINEAR)
    canvas = Image.new("RGB", (width, width + 22), color=(18, 18, 18))
    canvas.paste(pil_image, (0, 22))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 5), caption, fill=(235, 235, 235), font=font)
    return canvas


def save_visualizations(results: list[EvalResult], output_dir: Path, top_k: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()

    ranked = sorted(
        results,
        key=lambda item: (item.iou, item.sample.class_name, item.sample.image_path.name),
    )[:top_k]

    summary_tiles: list[Image.Image] = []
    tile_width = 192
    margin = 8

    for index, result in enumerate(ranked, start=1):
        image = result.image
        target_rgb = _mask_to_rgb(result.target)
        pred_rgb = _mask_to_rgb(result.prediction)
        error_rgb = _error_overlay(image, result.target, result.prediction)

        panels = [
            _tile_with_caption(image, "image", tile_width, font),
            _tile_with_caption(target_rgb, "ground truth", tile_width, font),
            _tile_with_caption(pred_rgb, "prediction", tile_width, font),
            _tile_with_caption(error_rgb, "errors", tile_width, font),
        ]

        panel_width = sum(panel.width for panel in panels) + margin * (len(panels) - 1)
        panel_height = max(panel.height for panel in panels) + 32
        canvas = Image.new("RGB", (panel_width, panel_height), color=(28, 28, 28))
        draw = ImageDraw.Draw(canvas)
        header = (
            f"{index:02d}. {result.sample.class_name} / {result.sample.image_path.name}  "
            f"IoU={result.iou:.4f}  time={result.elapsed_ms:.1f} ms"
        )
        draw.text((8, 8), header, fill=(255, 220, 170), font=font)

        x = 0
        for panel in panels:
            canvas.paste(panel, (x, 32))
            x += panel.width + margin

        file_name = (
            f"{index:02d}_{result.sample.class_name}_{result.sample.image_path.stem}_"
            f"iou_{result.iou:.4f}.png"
        )
        canvas.save(output_dir / file_name)

        thumb = canvas.resize((panel_width // 3, panel_height // 3), Image.BILINEAR)
        summary_tiles.append(thumb)

    if not summary_tiles:
        return

    cols = 3
    rows = (len(summary_tiles) + cols - 1) // cols
    summary_width = max(tile.width for tile in summary_tiles)
    summary_height = max(tile.height for tile in summary_tiles)
    summary = Image.new(
        "RGB",
        (cols * summary_width + margin * (cols - 1), rows * summary_height + margin * (rows - 1)),
        color=(12, 12, 12),
    )

    for idx, tile in enumerate(summary_tiles):
        row = idx // cols
        col = idx % cols
        x = col * (summary_width + margin)
        y = row * (summary_height + margin)
        summary.paste(tile, (x, y))

    summary.save(output_dir / "summary.png")


def main() -> None:
    args = parse_args()
    model = load_submission_model(args.model_path)
    samples = collect_samples()
    if args.class_name is not None:
        samples = [sample for sample in samples if sample.class_name == args.class_name]
    if not samples:
        raise RuntimeError("No samples matched the requested filter.")

    total_scores: list[float] = []
    per_class_scores: dict[str, list[float]] = {}
    runtimes_ms: list[float] = []
    results: list[EvalResult] = []

    for index, sample in enumerate(samples, start=1):
        image = load_image(sample.image_path)
        target = load_mask(sample.mask_path, image.shape[:2])

        started = time.perf_counter()
        prediction = model.predict(image)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        runtimes_ms.append(elapsed_ms)

        if prediction.shape != target.shape:
            raise RuntimeError(
                f"Invalid mask shape for {sample.image_path}: got {prediction.shape}, expected {target.shape}"
            )
        if prediction.dtype != np.uint8:
            raise RuntimeError(
                f"Invalid mask dtype for {sample.image_path}: got {prediction.dtype}, expected uint8"
            )

        unique_values = set(np.unique(prediction).tolist())
        if not unique_values.issubset({0, 255}):
            raise RuntimeError(
                f"Invalid mask values for {sample.image_path}: got {sorted(unique_values)}, expected only 0/255"
            )

        score = mean_iou(prediction, target)
        total_scores.append(score)
        per_class_scores.setdefault(sample.class_name, []).append(score)
        results.append(
            EvalResult(
                sample=sample,
                iou=score,
                elapsed_ms=elapsed_ms,
                image=image,
                target=target,
                prediction=prediction,
            )
        )

        print(
            f"[{index:03d}/{len(samples):03d}] {sample.class_name:>20}  "
            f"IoU={score:.4f}  time={elapsed_ms:.1f} ms  file={sample.image_path.name}"
        )

    print("\nOverall mean IoU:", f"{np.mean(total_scores):.6f}")
    print("Average inference time per image:", f"{np.mean(runtimes_ms):.2f} ms")
    print("Total inference time:", f"{np.sum(runtimes_ms) / 1000.0:.2f} s")
    print("\nPer-class mean IoU:")
    for class_name, scores in sorted(per_class_scores.items()):
        print(f"  {class_name:>20}: {np.mean(scores):.6f}  (n={len(scores)})")

    if args.save_vis:
        save_visualizations(results, args.vis_dir, args.top_k)
        print(f"\nSaved visual diagnostics to: {args.vis_dir}")


if __name__ == "__main__":
    main()
