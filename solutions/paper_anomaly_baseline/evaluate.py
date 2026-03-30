from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from anomaly_baseline import (
    iter_split_samples,
    load_binary_mask,
    load_model_artifact,
    load_rgb,
    load_split_manifest,
    mean_iou,
    predict_with_artifact,
)


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_SPLIT_PATH = THIS_DIR / "artifacts" / "split.json"
DEFAULT_MODEL_PATH = THIS_DIR / "artifacts" / "model" / "model_artifact.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate paper anomaly baseline on a selected split.")
    parser.add_argument("--split-path", type=Path, default=DEFAULT_SPLIT_PATH)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--output-path", type=Path, default=None)
    return parser.parse_args()


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    class_scores: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        class_scores[str(row["class_name"])].append(float(row["iou"]))

    mean_iou_score = float(np.mean([row["iou"] for row in rows])) if rows else 0.0
    balanced_iou = float(np.mean([np.mean(values) for values in class_scores.values()])) if class_scores else 0.0
    mean_runtime = float(np.mean([row["elapsed_ms"] for row in rows])) if rows else 0.0
    p95_runtime = float(np.percentile([row["elapsed_ms"] for row in rows], 95.0)) if rows else 0.0

    return {
        "num_samples": len(rows),
        "mean_iou": mean_iou_score,
        "balanced_mean_iou": balanced_iou,
        "per_class_iou": {
            class_name: float(np.mean(values))
            for class_name, values in sorted(class_scores.items())
        },
        "mean_runtime_ms": mean_runtime,
        "p95_runtime_ms": p95_runtime,
    }


def main() -> None:
    args = parse_args()
    split_manifest = load_split_manifest(args.split_path.resolve())
    samples = iter_split_samples(split_manifest, args.split)
    artifact = load_model_artifact(args.model_path.resolve())

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
            print(f"[{index:03d}/{len(samples):03d}] {sample.class_name:>20} IoU={rows[-1]['iou']:.4f}")

    summary = {
        "split": args.split,
        "split_path": str(args.split_path.resolve()),
        "model_path": str(args.model_path.resolve()),
        "metrics": summarize(rows),
    }
    print(json.dumps(summary, indent=2))

    if args.output_path is not None:
        output_path = args.output_path.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"summary": summary, "rows": rows}
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved detailed evaluation to: {output_path}")


if __name__ == "__main__":
    main()
