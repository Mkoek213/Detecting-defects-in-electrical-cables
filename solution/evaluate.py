from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
import scipy.ndimage
from tqdm import tqdm

from fit import (
    DEFECT_CLASSES,
    PatchCoreFeatureExtractor,
    Sample,
    build_train_val_split,
    compute_anomaly_map,
    fit_patchcore,
    load_rgb_image,
    reduce_patches,
)


def iou(pred_binary: np.ndarray, gt_binary: np.ndarray) -> float:
    inter = np.logical_and(pred_binary, gt_binary).sum()
    union = np.logical_or(pred_binary, gt_binary).sum()
    if union == 0:
        return 1.0
    return float(inter / union)


def postprocess_binary_mask(mask01: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 0:
        return mask01.astype(np.uint8)

    mask_bool = mask01.astype(bool)
    labeled, num_labels = scipy.ndimage.label(mask_bool)
    if num_labels == 0:
        return np.zeros_like(mask01, dtype=np.uint8)

    counts = np.bincount(labeled.ravel())
    keep_labels = np.where(counts >= int(min_area))[0]
    keep_labels = keep_labels[keep_labels != 0]
    if keep_labels.size == 0:
        return np.zeros_like(mask01, dtype=np.uint8)

    filtered = np.isin(labeled, keep_labels)
    return filtered.astype(np.uint8)


def load_gt_mask(sample: Sample, image_shape: tuple[int, int]) -> np.ndarray:
    height, width = image_shape
    if not sample.is_defect or sample.mask_path is None:
        return np.zeros((height, width), dtype=np.uint8)

    gt_image = Image.open(sample.mask_path).convert("L")
    if gt_image.size != (width, height):
        gt_image = gt_image.resize((width, height), Image.NEAREST)

    gt = np.asarray(gt_image, dtype=np.uint8)
    return (gt > 127).astype(np.uint8)


def image_anomaly_map(
    image_rgb_uint8: np.ndarray,
    extractor: PatchCoreFeatureExtractor,
    projection: np.ndarray,
    coreset: np.ndarray,
    k: int,
    gaussian_sigma: float,
) -> np.ndarray:
    patches = extractor.extract_raw_patches(image_rgb_uint8)
    reduced = reduce_patches(patches, projection)

    score_32 = compute_anomaly_map(reduced, coreset, k=k)
    score_256 = scipy.ndimage.zoom(score_32, 8.0, order=1)

    height, width = image_rgb_uint8.shape[:2]
    zoom_h = height / float(score_256.shape[0])
    zoom_w = width / float(score_256.shape[1])
    score_full = scipy.ndimage.zoom(score_256, (zoom_h, zoom_w), order=1)

    if score_full.shape != (height, width):
        score_full = np.asarray(
            Image.fromarray(score_full.astype(np.float32), mode="F").resize((width, height), Image.BILINEAR),
            dtype=np.float32,
        )

    score_full = score_full.astype(np.float32)
    if gaussian_sigma > 0:
        score_full = scipy.ndimage.gaussian_filter(score_full, sigma=gaussian_sigma).astype(np.float32)

    return score_full


def evaluate_thresholds(
    val_samples: list[Sample],
    anomaly_maps: list[np.ndarray],
    gt_masks: list[np.ndarray],
    min_area: int,
) -> tuple[float, float, float, float, dict[str, float]]:
    global_min = float(min(amap.min() for amap in anomaly_maps))
    global_max = float(max(amap.max() for amap in anomaly_maps))
    if global_max == global_min:
        thresholds = np.array([global_min], dtype=np.float32)
    else:
        thresholds = np.linspace(global_min, global_max, 300, dtype=np.float32)

    best_threshold = float(thresholds[0])
    best_mean_iou = -1.0
    best_ious: list[float] = []

    for threshold in thresholds:
        current_ious: list[float] = []
        for amap, gt in zip(anomaly_maps, gt_masks):
            pred = (amap > threshold).astype(np.uint8)
            pred = postprocess_binary_mask(pred, min_area=min_area)
            current_ious.append(iou(pred, gt))

        mean_iou = float(np.mean(current_ious))
        if mean_iou > best_mean_iou:
            best_mean_iou = mean_iou
            best_threshold = float(threshold)
            best_ious = current_ious

    class_to_ious: dict[str, list[float]] = defaultdict(list)
    good_ious: list[float] = []
    defect_ious: list[float] = []

    for sample, score in zip(val_samples, best_ious):
        if sample.is_defect:
            class_to_ious[sample.class_name].append(score)
            defect_ious.append(score)
        else:
            good_ious.append(score)

    class_means = {
        class_name: float(np.mean(class_to_ious[class_name])) if class_to_ious[class_name] else 0.0
        for class_name in DEFECT_CLASSES
    }

    good_mean = float(np.mean(good_ious)) if good_ious else 0.0
    defect_mean = float(np.mean(defect_ious)) if defect_ious else 0.0
    return best_mean_iou, best_threshold, good_mean, defect_mean, class_means


def print_report(
    val_mean_iou: float,
    best_threshold: float,
    good_iou: float,
    defect_iou: float,
    class_scores: dict[str, float],
    val_samples: list[Sample],
) -> None:
    good_count = sum(1 for sample in val_samples if not sample.is_defect)
    defect_count = sum(1 for sample in val_samples if sample.is_defect)

    print("-- Validation Report --------------------------")
    print(f"Val mean IoU        : {val_mean_iou:.3f}")
    print(f"Best threshold      : {best_threshold:.4f}")
    print(f"Good images IoU     : {good_iou:.3f}  (N={good_count})")
    print(f"Defect images IoU   : {defect_iou:.3f}  (N={defect_count})")
    print("Per-class IoU:")
    for class_name in DEFECT_CLASSES:
        print(f"  {class_name:<22}: {class_scores[class_name]:.3f}")
    print("-----------------------------------------------")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PatchCore and calibrate threshold.")
    parser.add_argument("--data_dir", type=Path, required=True, help="Path to data/cable directory.")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(__file__).resolve().parent / "sample_submission",
        help="Directory where coreset/projection/threshold artifacts are stored.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic split/fitting.")
    parser.add_argument("--ratio", type=float, default=0.1, help="Coreset ratio before max cap.")
    parser.add_argument("--coreset_size", type=int, default=10_000, help="Number of coreset vectors to keep.")
    parser.add_argument("--k", type=int, default=3, help="Number of nearest neighbors for anomaly scoring.")
    parser.add_argument(
        "--gaussian_sigma",
        type=float,
        default=0.5,
        help="Gaussian smoothing sigma applied to anomaly maps before thresholding.",
    )
    parser.add_argument(
        "--min_area",
        type=int,
        default=128,
        help="Minimum connected component area in final binary masks.",
    )
    parser.add_argument(
        "--include_defect_train",
        action="store_true",
        help="Include defect samples from the mixed split in coreset fitting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    train_set, val_set = build_train_val_split(args.data_dir, train_ratio=0.8, seed=args.seed)
    fit_train_set = train_set if args.include_defect_train else [sample for sample in train_set if not sample.is_defect]

    if len(fit_train_set) == 0:
        raise RuntimeError("No training samples available for fitting.")

    coreset, projection = fit_patchcore(
        train_set=fit_train_set,
        output_dir=output_dir,
        ratio=args.ratio,
        seed=args.seed,
        coreset_keep=args.coreset_size,
    )

    extractor = PatchCoreFeatureExtractor()
    anomaly_maps: list[np.ndarray] = []
    gt_masks: list[np.ndarray] = []

    for sample in tqdm(val_set, desc="Scoring val set"):
        image = load_rgb_image(sample.image_path)
        anomaly_maps.append(
            image_anomaly_map(
                image,
                extractor,
                projection,
                coreset,
                k=args.k,
                gaussian_sigma=args.gaussian_sigma,
            )
        )
        gt_masks.append(load_gt_mask(sample, image.shape[:2]))

    val_mean_iou, best_threshold, good_iou, defect_iou, class_scores = evaluate_thresholds(
        val_samples=val_set,
        anomaly_maps=anomaly_maps,
        gt_masks=gt_masks,
        min_area=args.min_area,
    )

    np.save(output_dir / "threshold.npy", np.float32(best_threshold))
    print_report(val_mean_iou, best_threshold, good_iou, defect_iou, class_scores, val_set)
    print(f"Saved: {output_dir / 'threshold.npy'}")


if __name__ == "__main__":
    main()