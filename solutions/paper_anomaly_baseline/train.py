from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from anomaly_baseline import (
    DEFAULT_FEATURE_SIZE,
    FEATURE_EPS,
    ModelArtifact,
    build_binary_mask,
    compute_anomaly_map,
    extract_features,
    iter_split_samples,
    load_binary_mask,
    load_rgb,
    load_split_manifest,
    mean_iou,
    predict_with_artifact,
    save_model_artifact,
)


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_SPLIT_PATH = THIS_DIR / "artifacts" / "split.json"
DEFAULT_OUTPUT_DIR = THIS_DIR / "artifacts" / "model"


@dataclass(frozen=True)
class ScoredSample:
    class_name: str
    image_name: str
    output_shape: tuple[int, int]
    target_full: np.ndarray
    target_small: np.ndarray
    score_map: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train paper-aligned one-class anomaly baseline.")
    parser.add_argument("--split-path", type=Path, default=DEFAULT_SPLIT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--feature-size", type=int, default=DEFAULT_FEATURE_SIZE)
    parser.add_argument("--eps", type=float, default=FEATURE_EPS)
    parser.add_argument("--threshold-quantile-low", type=float, default=0.93)
    parser.add_argument("--threshold-quantile-high", type=float, default=0.999)
    parser.add_argument("--threshold-steps", type=int, default=24)
    parser.add_argument("--min-area-candidates", type=str, default="0,8,16,32,64,128")
    return parser.parse_args()


def parse_int_list(value: str) -> list[int]:
    return [int(token.strip()) for token in value.split(",") if token.strip()]


def fit_one_class_model(train_samples: list[Any], feature_size: int, eps: float) -> tuple[np.ndarray, np.ndarray]:
    sum_features: np.ndarray | None = None
    sum_squares: np.ndarray | None = None
    count = 0

    for index, sample in enumerate(train_samples, start=1):
        image = load_rgb(sample.image_path)
        features = extract_features(image=image, feature_size=feature_size)
        features64 = features.astype(np.float64)

        if sum_features is None:
            sum_features = np.zeros_like(features64, dtype=np.float64)
            sum_squares = np.zeros_like(features64, dtype=np.float64)

        sum_features += features64
        sum_squares += features64 * features64
        count += 1

        if index % 25 == 0 or index == len(train_samples):
            print(f"[fit] processed {index:03d}/{len(train_samples):03d} good samples")

    if sum_features is None or sum_squares is None or count == 0:
        raise RuntimeError("No training samples available for model fitting.")

    mean64 = sum_features / count
    var64 = sum_squares / count - mean64 * mean64
    var64 = np.maximum(var64, eps)
    mean = mean64.astype(np.float32)
    var = var64.astype(np.float32)
    return mean, var


def build_small_mask(score_map: np.ndarray, threshold: float, min_area: int) -> np.ndarray:
    output_shape = score_map.shape
    return build_binary_mask(
        score_map=score_map,
        threshold=threshold,
        min_area=min_area,
        output_shape=output_shape,
    )


def score_samples(
    samples: list[Any],
    mean: np.ndarray,
    var: np.ndarray,
    eps: float,
    feature_size: int,
) -> list[ScoredSample]:
    scored: list[ScoredSample] = []
    for index, sample in enumerate(samples, start=1):
        image = load_rgb(sample.image_path)
        target_full = load_binary_mask(sample.mask_path, image.shape[:2])
        features = extract_features(image=image, feature_size=feature_size)
        score_map = compute_anomaly_map(features=features, mean=mean, var=var, eps=eps)
        target_small = np.asarray(
            Image.fromarray(target_full, mode="L").resize((feature_size, feature_size), Image.NEAREST),
            dtype=np.uint8,
        )
        target_small = (target_small > 0).astype(np.uint8) * 255

        scored.append(
            ScoredSample(
                class_name=sample.class_name,
                image_name=sample.image_path.name,
                output_shape=target_full.shape,
                target_full=target_full,
                target_small=target_small,
                score_map=score_map,
            )
        )
        if index % 20 == 0 or index == len(samples):
            print(f"[score] processed {index:03d}/{len(samples):03d} samples")

    return scored


def evaluate_grid(
    samples: list[ScoredSample],
    threshold: float,
    min_area: int,
) -> tuple[float, dict[str, float]]:
    per_class: dict[str, list[float]] = defaultdict(list)
    for sample in samples:
        prediction = build_small_mask(
            score_map=sample.score_map,
            threshold=threshold,
            min_area=min_area,
        )
        score = mean_iou(prediction, sample.target_small)
        per_class[sample.class_name].append(score)

    class_means = {
        class_name: float(np.mean(values))
        for class_name, values in sorted(per_class.items())
    }
    overall = float(np.mean(list(class_means.values()))) if class_means else 0.0
    return overall, class_means


def calibrate_threshold(
    val_samples: list[ScoredSample],
    quantile_low: float,
    quantile_high: float,
    threshold_steps: int,
    min_area_candidates: list[int],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not val_samples:
        raise RuntimeError("No validation samples available for threshold calibration.")

    all_scores = np.concatenate([sample.score_map.reshape(-1) for sample in val_samples])
    quantiles = np.linspace(quantile_low, quantile_high, threshold_steps)
    thresholds = np.unique(np.quantile(all_scores, quantiles).astype(np.float32))

    rows: list[dict[str, Any]] = []
    for threshold in thresholds.tolist():
        for min_area in min_area_candidates:
            overall, per_class = evaluate_grid(val_samples, threshold=threshold, min_area=min_area)
            rows.append(
                {
                    "threshold": float(threshold),
                    "min_area": int(min_area),
                    "val_balanced_mean_iou_small": float(overall),
                    "val_per_class_iou_small": per_class,
                }
            )

    rows_sorted = sorted(
        rows,
        key=lambda row: (row["val_balanced_mean_iou_small"], -row["threshold"], -row["min_area"]),
        reverse=True,
    )
    return rows_sorted[0], rows_sorted


def summarize_scores(iou_rows: list[dict[str, Any]]) -> dict[str, Any]:
    per_class: dict[str, list[float]] = defaultdict(list)
    for row in iou_rows:
        per_class[str(row["class_name"])].append(float(row["iou"]))

    mean_iou_score = float(np.mean([row["iou"] for row in iou_rows])) if iou_rows else 0.0
    balanced_iou = float(np.mean([np.mean(values) for values in per_class.values()])) if per_class else 0.0
    return {
        "num_samples": len(iou_rows),
        "mean_iou": mean_iou_score,
        "balanced_mean_iou": balanced_iou,
        "per_class_iou": {
            class_name: float(np.mean(values))
            for class_name, values in sorted(per_class.items())
        },
        "mean_runtime_ms": float(np.mean([row["elapsed_ms"] for row in iou_rows])) if iou_rows else 0.0,
        "p95_runtime_ms": float(np.percentile([row["elapsed_ms"] for row in iou_rows], 95.0)) if iou_rows else 0.0,
    }


def evaluate_full_pipeline(samples: list[Any], artifact: ModelArtifact) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for index, sample in enumerate(samples, start=1):
        image = load_rgb(sample.image_path)
        target = load_binary_mask(sample.mask_path, image.shape[:2])

        started = time.perf_counter()
        prediction = predict_with_artifact(image=image, artifact=artifact)
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        rows.append(
            {
                "class_name": sample.class_name,
                "image_name": sample.image_path.name,
                "iou": float(mean_iou(prediction, target)),
                "elapsed_ms": float(elapsed_ms),
            }
        )

        if index % 20 == 0 or index == len(samples):
            print(f"[eval] processed {index:03d}/{len(samples):03d} samples")

    return summarize_scores(rows), rows


def main() -> None:
    args = parse_args()
    split_path = args.split_path.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_split_manifest(split_path)
    train_samples = iter_split_samples(manifest, "train")
    val_samples = iter_split_samples(manifest, "val")
    test_samples = iter_split_samples(manifest, "test")

    if not train_samples:
        raise RuntimeError("Empty train split in manifest.")
    if any(not sample.is_good for sample in train_samples):
        raise RuntimeError("Train split contains defects. This baseline requires one-class training on good only.")

    min_area_candidates = parse_int_list(args.min_area_candidates)
    if not min_area_candidates:
        raise ValueError("No min-area candidates provided.")
    if any(value < 0 for value in min_area_candidates):
        raise ValueError("min-area candidates must be non-negative.")

    print("Fitting one-class Gaussian model on train/good only...")
    mean, var = fit_one_class_model(
        train_samples=train_samples,
        feature_size=args.feature_size,
        eps=args.eps,
    )

    print("Scoring validation split for threshold calibration...")
    val_scored = score_samples(
        samples=val_samples,
        mean=mean,
        var=var,
        eps=args.eps,
        feature_size=args.feature_size,
    )

    best_config, ranking = calibrate_threshold(
        val_samples=val_scored,
        quantile_low=args.threshold_quantile_low,
        quantile_high=args.threshold_quantile_high,
        threshold_steps=args.threshold_steps,
        min_area_candidates=min_area_candidates,
    )
    print(
        "Selected validation config:",
        f"threshold={best_config['threshold']:.6f}",
        f"min_area={best_config['min_area']}",
        f"val_balanced_mIoU_small={best_config['val_balanced_mean_iou_small']:.4f}",
    )

    artifact = ModelArtifact(
        feature_size=args.feature_size,
        eps=args.eps,
        threshold=float(best_config["threshold"]),
        min_area=int(best_config["min_area"]),
        mean=mean,
        var=var,
    )

    model_path = output_dir / "model_artifact.npz"
    save_model_artifact(model_path, artifact)
    print(f"Saved model artifact: {model_path}")

    print("Running full-resolution evaluation on val split...")
    val_metrics, val_rows = evaluate_full_pipeline(val_samples, artifact)
    print("Running full-resolution evaluation on test split...")
    test_metrics, test_rows = evaluate_full_pipeline(test_samples, artifact)

    summary = {
        "split_manifest": str(split_path),
        "model_artifact": str(model_path),
        "training_setup": {
            "feature_size": int(args.feature_size),
            "eps": float(args.eps),
            "anomaly_model_fitting_data": {
                "split": "train",
                "class_filter": "good",
                "num_samples": len(train_samples),
            },
            "threshold_calibration_data": {
                "split": "val",
                "class_filter": "all",
                "num_samples": len(val_samples),
            },
        },
        "selected_thresholding": {
            "threshold": float(best_config["threshold"]),
            "min_area": int(best_config["min_area"]),
            "selection_objective": "balanced mean IoU on validation (small-map proxy)",
            "validation_small_proxy_score": float(best_config["val_balanced_mean_iou_small"]),
        },
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }

    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "threshold_search.json").write_text(json.dumps(ranking[:50], indent=2), encoding="utf-8")
    (output_dir / "val_predictions.json").write_text(json.dumps(val_rows, indent=2), encoding="utf-8")
    (output_dir / "test_predictions.json").write_text(json.dumps(test_rows, indent=2), encoding="utf-8")
    print(f"Saved training summary: {output_dir / 'training_summary.json'}")
    print(f"Saved threshold ranking: {output_dir / 'threshold_search.json'}")
    print(f"Validation mean IoU: {val_metrics['mean_iou']:.4f}")
    print(f"Test mean IoU: {test_metrics['mean_iou']:.4f}")


if __name__ == "__main__":
    main()
