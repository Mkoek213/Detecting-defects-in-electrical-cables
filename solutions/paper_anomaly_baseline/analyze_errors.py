from __future__ import annotations

import argparse
import importlib.util
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "cable"
DEFAULT_MODEL_PATH = THIS_DIR / "sample_submission" / "model.py"
DEFAULT_OUTPUT_DIR = THIS_DIR / "artifacts" / "error_analysis"


@dataclass(frozen=True)
class Sample:
    class_name: str
    image_path: Path
    mask_path: Path | None


@dataclass
class Scored:
    sample: Sample
    iou: float
    fp_rate: float
    fn_rate: float
    gt_area: int
    pred_area: int
    elapsed_ms: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze model errors and save worst-case panels.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument(
        "--classes",
        type=str,
        default=None,
        help="Comma-separated class list (default: all defect classes in data/cable/test).",
    )
    return parser.parse_args()


def load_model(model_path: Path):
    spec = importlib.util.spec_from_file_location("paper_baseline_model", model_path.resolve())
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import model from {model_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "predict"):
        raise RuntimeError(f"Missing predict(image) in {model_path}")
    return module


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def load_mask(path: Path | None, shape: tuple[int, int]) -> np.ndarray:
    if path is None:
        return np.zeros(shape, dtype=np.uint8)
    mask = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    if mask.shape != shape:
        mask = np.asarray(
            Image.fromarray(mask, mode="L").resize((shape[1], shape[0]), Image.NEAREST),
            dtype=np.uint8,
        )
    return (mask > 0).astype(np.uint8) * 255


def iou(prediction: np.ndarray, target: np.ndarray) -> float:
    pred_bin = prediction > 0
    target_bin = target > 0
    intersection = np.logical_and(pred_bin, target_bin).sum()
    union = np.logical_or(pred_bin, target_bin).sum()
    return 1.0 if union == 0 else float(intersection / union)


def error_rates(prediction: np.ndarray, target: np.ndarray) -> tuple[float, float, int, int]:
    pred_bin = prediction > 0
    target_bin = target > 0
    fp = np.logical_and(pred_bin, ~target_bin).sum()
    fn = np.logical_and(~pred_bin, target_bin).sum()
    pred_area = int(pred_bin.sum())
    gt_area = int(target_bin.sum())
    fp_rate = 0.0 if pred_area == 0 else float(fp / pred_area)
    fn_rate = 0.0 if gt_area == 0 else float(fn / gt_area)
    return fp_rate, fn_rate, gt_area, pred_area


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


def tile(image: np.ndarray, caption: str, width: int, font: ImageFont.ImageFont) -> Image.Image:
    panel = Image.fromarray(image, mode="RGB").resize((width, width), Image.BILINEAR)
    canvas = Image.new("RGB", (width, width + 22), color=(20, 20, 20))
    canvas.paste(panel, (0, 22))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 5), caption, fill=(235, 235, 235), font=font)
    return canvas


def save_panel(sample: Sample, image: np.ndarray, target: np.ndarray, prediction: np.ndarray, scored: Scored, output_dir: Path) -> None:
    font = ImageFont.load_default()
    tile_width = 220
    margin = 8

    panels = [
        tile(image, "image", tile_width, font),
        tile(mask_to_rgb(target), "ground truth", tile_width, font),
        tile(mask_to_rgb(prediction), "prediction", tile_width, font),
        tile(error_overlay(image, target, prediction), "errors", tile_width, font),
    ]

    panel_width = sum(panel.width for panel in panels) + margin * (len(panels) - 1)
    panel_height = max(panel.height for panel in panels) + 32
    canvas = Image.new("RGB", (panel_width, panel_height), color=(28, 28, 28))

    draw = ImageDraw.Draw(canvas)
    header = (
        f"{sample.class_name} / {sample.image_path.name}  IoU={scored.iou:.4f}  "
        f"FP={scored.fp_rate:.2f} FN={scored.fn_rate:.2f}  time={scored.elapsed_ms:.1f} ms"
    )
    draw.text((8, 8), header, fill=(255, 220, 170), font=font)

    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 32))
        x += panel.width + margin

    out_name = f"{sample.class_name}_{sample.image_path.stem}.png"
    canvas.save(output_dir / out_name)


def collect_samples(data_root: Path, classes: set[str] | None) -> list[Sample]:
    rows: list[Sample] = []
    test_root = data_root / "test"
    for class_dir in sorted(test_root.iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        if class_name == "good":
            continue
        if classes is not None and class_name not in classes:
            continue

        for image_path in sorted(class_dir.glob("*.png")):
            mask_path = data_root / "ground_truth" / class_name / f"{image_path.stem}_mask.png"
            rows.append(Sample(class_name=class_name, image_path=image_path, mask_path=mask_path))
    return rows


def main() -> None:
    args = parse_args()
    model = load_model(args.model_path)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_classes = None
    if args.classes:
        selected_classes = {token.strip() for token in args.classes.split(",") if token.strip()}

    samples = collect_samples(data_root=args.data_root.resolve(), classes=selected_classes)
    if not samples:
        raise RuntimeError("No defect samples selected. Check --classes or dataset path.")

    per_class: dict[str, list[Scored]] = {}
    for sample in samples:
        image = load_rgb(sample.image_path)
        target = load_mask(sample.mask_path, image.shape[:2])

        started = time.perf_counter()
        prediction = model.predict(image)
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        score = iou(prediction, target)
        fp_rate, fn_rate, gt_area, pred_area = error_rates(prediction, target)
        scored = Scored(
            sample=sample,
            iou=score,
            fp_rate=fp_rate,
            fn_rate=fn_rate,
            gt_area=gt_area,
            pred_area=pred_area,
            elapsed_ms=elapsed_ms,
        )
        per_class.setdefault(sample.class_name, []).append(scored)

    summary = {}
    for class_name, rows in per_class.items():
        rows_sorted = sorted(rows, key=lambda r: (r.iou, -r.fn_rate, -r.fp_rate))
        top = rows_sorted[: args.top_k]

        out_dir = output_dir / class_name
        out_dir.mkdir(parents=True, exist_ok=True)
        for scored in top:
            image = load_rgb(scored.sample.image_path)
            target = load_mask(scored.sample.mask_path, image.shape[:2])
            prediction = model.predict(image)
            save_panel(scored.sample, image, target, prediction, scored, out_dir)

        summary[class_name] = {
            "num_samples": len(rows),
            "mean_iou": float(np.mean([r.iou for r in rows])),
            "mean_fp_rate": float(np.mean([r.fp_rate for r in rows])),
            "mean_fn_rate": float(np.mean([r.fn_rate for r in rows])),
            "mean_gt_area": float(np.mean([r.gt_area for r in rows])),
            "mean_pred_area": float(np.mean([r.pred_area for r in rows])),
            "worst_samples": [
                {
                    "image": r.sample.image_path.name,
                    "iou": r.iou,
                    "fp_rate": r.fp_rate,
                    "fn_rate": r.fn_rate,
                    "gt_area": r.gt_area,
                    "pred_area": r.pred_area,
                }
                for r in top
            ],
        }

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved summary to: {output_dir / 'summary.json'}")
    print(f"Saved worst-case panels under: {output_dir}")


if __name__ == "__main__":
    main()
