from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from anomaly_baseline import REPO_ROOT


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "cable"
DEFAULT_OUTPUT_PATH = THIS_DIR / "artifacts" / "split.json"
DEFAULT_SEED = 20260330


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build deterministic split manifest for paper anomaly baseline.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--good-train-ratio", type=float, default=0.70)
    parser.add_argument("--good-val-ratio", type=float, default=0.15)
    parser.add_argument("--defect-val-ratio", type=float, default=0.40)
    return parser.parse_args()


def to_repo_relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT.resolve()))


def load_good_samples(data_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source_split in ("train", "test"):
        class_dir = data_root / source_split / "good"
        for image_path in sorted(class_dir.glob("*.png")):
            rows.append(
                {
                    "class_name": "good",
                    "is_good": True,
                    "source_split": source_split,
                    "image_path": to_repo_relative(image_path),
                    "mask_path": None,
                }
            )
    return rows


def load_defect_samples(data_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    test_root = data_root / "test"
    for class_dir in sorted(test_root.iterdir()):
        if not class_dir.is_dir() or class_dir.name == "good":
            continue
        class_name = class_dir.name
        for image_path in sorted(class_dir.glob("*.png")):
            mask_path = data_root / "ground_truth" / class_name / f"{image_path.stem}_mask.png"
            if not mask_path.exists():
                raise FileNotFoundError(f"Missing mask for {image_path}: {mask_path}")
            rows.append(
                {
                    "class_name": class_name,
                    "is_good": False,
                    "source_split": "test",
                    "image_path": to_repo_relative(image_path),
                    "mask_path": to_repo_relative(mask_path),
                }
            )
    return rows


def assign_split_fields(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for row in rows:
        enriched.append(
            {
                "split": split,
                "class_name": row["class_name"],
                "is_good": row["is_good"],
                "source_split": row["source_split"],
                "image_path": row["image_path"],
                "mask_path": row["mask_path"],
            }
        )
    return enriched


def split_good_samples(
    rows: list[dict[str, Any]],
    rng: np.random.Generator,
    train_ratio: float,
    val_ratio: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("good train ratio must be in (0, 1)")
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("good val ratio must be in (0, 1)")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("good train ratio + val ratio must be < 1")

    order = np.arange(len(rows))
    rng.shuffle(order)
    shuffled = [rows[index] for index in order.tolist()]

    total = len(shuffled)
    if total < 3:
        raise RuntimeError("Need at least 3 good samples to build train/val/test split.")

    train_count = int(round(total * train_ratio))
    val_count = int(round(total * val_ratio))
    train_count = min(max(train_count, 1), total - 2)
    val_count = min(max(val_count, 1), total - train_count - 1)

    train_rows = shuffled[:train_count]
    val_rows = shuffled[train_count : train_count + val_count]
    test_rows = shuffled[train_count + val_count :]
    return train_rows, val_rows, test_rows


def split_defect_samples(
    rows: list[dict[str, Any]],
    seed: int,
    val_ratio: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("defect val ratio must be in (0, 1)")

    rows_by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_class[str(row["class_name"])].append(row)

    val_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []

    for class_index, class_name in enumerate(sorted(rows_by_class)):
        class_rows = sorted(rows_by_class[class_name], key=lambda item: str(item["image_path"]))
        rng = np.random.default_rng(seed + 1000 + class_index)
        order = np.arange(len(class_rows))
        rng.shuffle(order)
        shuffled = [class_rows[index] for index in order.tolist()]

        if len(shuffled) < 2:
            raise RuntimeError(f"Need at least 2 defective samples in class {class_name}.")

        val_count = int(round(len(shuffled) * val_ratio))
        val_count = min(max(val_count, 1), len(shuffled) - 1)
        val_rows.extend(shuffled[:val_count])
        test_rows.extend(shuffled[val_count:])

    return val_rows, test_rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(row["class_name"])] += 1
    return {
        "num_samples": len(rows),
        "per_class": dict(sorted(counts.items())),
    }


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    output_path = args.output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    good_rows = load_good_samples(data_root=data_root)
    defect_rows = load_defect_samples(data_root=data_root)

    rng = np.random.default_rng(args.seed)
    good_train, good_val, good_test = split_good_samples(
        rows=good_rows,
        rng=rng,
        train_ratio=args.good_train_ratio,
        val_ratio=args.good_val_ratio,
    )
    defect_val, defect_test = split_defect_samples(
        rows=defect_rows,
        seed=args.seed,
        val_ratio=args.defect_val_ratio,
    )

    split_rows = {
        "train": assign_split_fields(good_train, split="train"),
        "val": assign_split_fields(good_val + defect_val, split="val"),
        "test": assign_split_fields(good_test + defect_test, split="test"),
    }
    for split_name in split_rows:
        split_rows[split_name] = sorted(
            split_rows[split_name],
            key=lambda row: (str(row["class_name"]), str(row["image_path"])),
        )

    manifest = {
        "description": "Deterministic split manifest for paper anomaly baseline.",
        "seed": int(args.seed),
        "data_root": to_repo_relative(data_root),
        "split_policy": {
            "good": {
                "train_ratio": float(args.good_train_ratio),
                "val_ratio": float(args.good_val_ratio),
                "test_ratio": float(1.0 - args.good_train_ratio - args.good_val_ratio),
            },
            "defect": {
                "train_ratio": 0.0,
                "val_ratio": float(args.defect_val_ratio),
                "test_ratio": float(1.0 - args.defect_val_ratio),
                "stratified_by_class": True,
            },
        },
        "usage_constraints": {
            "anomaly_model_fitting": "train split, good class only",
            "threshold_calibration": "val split, good + defects",
            "final_evaluation": "test split, good + defects",
        },
        "counts": {
            split_name: summarize(rows)
            for split_name, rows in split_rows.items()
        },
        "splits": split_rows,
    }

    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved split manifest: {output_path}")
    print(json.dumps(manifest["counts"], indent=2))


if __name__ == "__main__":
    main()
