from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from anomaly_baseline import (
    apply_final_dilation,
    build_binary_mask,
    iter_split_samples,
    load_binary_mask,
    load_rgb,
    load_split_manifest,
    mean_iou,
)
from padim_backend import (
    DEFAULT_EPS,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_PROFILE_SIZE,
    DEFAULT_PROFILE_WEIGHT,
    DEFAULT_ROI_MARGIN_RATIO,
    DEFAULT_PEAK_COUNT,
    DEFAULT_PEAK_TOLERANCE_RATIO,
    DEFAULT_PEAK_WEIGHT,
    DEFAULT_TEMPLATE_WEIGHT,
    DEFAULT_TEMPLATE_MODE,
    DEFAULT_AXIS_ALIGN,
    DEFAULT_SELECTED_DIMS,
    PadimArtifact,
    build_model,
    compute_anomaly_map,
    compute_profile_vector,
    compute_profile_map,
    compute_template_map,
    compute_template_source,
    build_peak_mask,
    detect_profile_peaks,
    extract_embeddings,
    image_to_tensor,
    load_model_from_artifact,
    preprocess_image,
    save_artifact,
    select_feature_indices,
)
from train import ScoredSample, calibrate_threshold, parse_float_list, parse_int_list, summarize_scores


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_SPLIT_PATH = THIS_DIR / "artifacts" / "split.json"
DEFAULT_OUTPUT_DIR = THIS_DIR / "artifacts" / "padim"
DEFAULT_SAMPLE_ARTIFACT_PATH = THIS_DIR / "sample_submission" / "model_artifact_padim.pt"
DEFAULT_WEIGHTS_PATH = THIS_DIR.parents[1] / ".torch_cache" / "hub" / "checkpoints" / "resnet18-f37072fd.pth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ResNet18 PaDiM-style anomaly backend on good-only samples.")
    parser.add_argument("--split-path", type=Path, default=DEFAULT_SPLIT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-artifact-path", type=Path, default=DEFAULT_SAMPLE_ARTIFACT_PATH)
    parser.add_argument("--weights-path", type=Path, default=DEFAULT_WEIGHTS_PATH)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--selected-dims", type=int, default=DEFAULT_SELECTED_DIMS)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eps", type=float, default=DEFAULT_EPS)
    parser.add_argument("--threshold-quantile-low", type=float, default=0.75)
    parser.add_argument("--threshold-quantile-high", type=float, default=0.998)
    parser.add_argument("--threshold-steps", type=int, default=18)
    parser.add_argument("--stage1-top-k", type=int, default=6)
    parser.add_argument("--min-area-candidates", type=str, default="0,16,32,64,128")
    parser.add_argument("--open-kernel-candidates", type=str, default="1,3,5")
    parser.add_argument("--close-kernel-candidates", type=str, default="1,3")
    parser.add_argument("--threshold-scale-candidates", type=str, default="0.75,0.8,0.85,0.9,0.95,1.0")
    parser.add_argument("--final-dilate-candidates", type=str, default="1,3,5,7")
    parser.add_argument("--use-roi", action="store_true", default=False)
    parser.add_argument("--roi-margin-ratio", type=float, default=DEFAULT_ROI_MARGIN_RATIO)
    parser.add_argument("--use-profile", action="store_true", default=False)
    parser.add_argument("--profile-size", type=int, default=DEFAULT_PROFILE_SIZE)
    parser.add_argument("--profile-weight", type=float, default=DEFAULT_PROFILE_WEIGHT)
    parser.add_argument("--axis-align", action="store_true", default=DEFAULT_AXIS_ALIGN)
    parser.add_argument("--use-peak-mask", action="store_true", default=False)
    parser.add_argument("--peak-count", type=int, default=DEFAULT_PEAK_COUNT)
    parser.add_argument("--peak-tolerance-ratio", type=float, default=DEFAULT_PEAK_TOLERANCE_RATIO)
    parser.add_argument("--peak-weight", type=float, default=DEFAULT_PEAK_WEIGHT)
    parser.add_argument("--use-template", action="store_true", default=False)
    parser.add_argument("--template-weight", type=float, default=DEFAULT_TEMPLATE_WEIGHT)
    parser.add_argument("--template-mode", type=str, default=DEFAULT_TEMPLATE_MODE, choices=["rgb", "edge"])
    return parser.parse_args()


def select_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def iter_sample_batches(samples: list[Any], batch_size: int) -> list[list[Any]]:
    return [samples[index : index + batch_size] for index in range(0, len(samples), batch_size)]


def load_batch_tensor(samples: list[Any], image_size: int, use_roi: bool, roi_margin_ratio: float) -> torch.Tensor:
    tensors = [
        image_to_tensor(
            load_rgb(sample.image_path),
            image_size=image_size,
            use_roi=use_roi,
            roi_margin_ratio=roi_margin_ratio,
        )
        for sample in samples
    ]
    return torch.stack(tensors)


def fit_gaussian_stats(
    model: torch.nn.Module,
    samples: list[Any],
    image_size: int,
    selected_dims: int,
    batch_size: int,
    eps: float,
    device: torch.device,
    use_roi: bool,
    roi_margin_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    total_dims = 64 + 128 + 256
    selected_indices = select_feature_indices(total_dims=total_dims, selected_dims=selected_dims)

    sum_features: np.ndarray | None = None
    sum_squares: np.ndarray | None = None
    count = 0

    for batch_samples in iter_sample_batches(samples, batch_size=batch_size):
        batch_tensor = load_batch_tensor(
            batch_samples,
            image_size=image_size,
            use_roi=use_roi,
            roi_margin_ratio=roi_margin_ratio,
        ).to(device=device, dtype=torch.float32)
        embeddings = extract_embeddings(model, batch_tensor, selected_indices=selected_indices).cpu().numpy()
        embeddings = np.transpose(embeddings, (0, 2, 3, 1)).astype(np.float64)

        if sum_features is None:
            sum_features = np.zeros_like(embeddings[0], dtype=np.float64)
            sum_squares = np.zeros_like(embeddings[0], dtype=np.float64)

        sum_features += np.sum(embeddings, axis=0)
        sum_squares += np.sum(np.square(embeddings), axis=0)
        count += embeddings.shape[0]

        print(f"[fit] processed {count:03d}/{len(samples):03d} good samples")

    if sum_features is None or sum_squares is None or count == 0:
        raise RuntimeError("No training samples were processed for Gaussian fitting.")

    mean = (sum_features / count).astype(np.float32)
    var = np.maximum(sum_squares / count - np.square(mean.astype(np.float64)), eps).astype(np.float32)
    return selected_indices, mean, var


def score_samples(
    model: torch.nn.Module,
    samples: list[Any],
    image_size: int,
    selected_indices: np.ndarray,
    mean: np.ndarray,
    var: np.ndarray,
    eps: float,
    batch_size: int,
    device: torch.device,
    use_roi: bool,
    roi_margin_ratio: float,
    use_profile: bool,
    profile_mean: np.ndarray | None,
    profile_std: np.ndarray | None,
    profile_size: int,
    profile_weight: float,
    use_axis_align: bool,
    use_peak_mask: bool,
    expected_peaks: np.ndarray | None,
    peak_tolerance_ratio: float,
    peak_weight: float,
    use_template: bool,
    template_mean: np.ndarray | None,
    template_std: np.ndarray | None,
    template_weight: float,
    template_mode: str,
) -> list[ScoredSample]:
    rows: list[ScoredSample] = []
    mean_tensor = torch.from_numpy(np.transpose(mean, (2, 0, 1))).unsqueeze(0).to(device=device, dtype=torch.float32)
    var_tensor = torch.from_numpy(np.transpose(var, (2, 0, 1))).unsqueeze(0).to(device=device, dtype=torch.float32)

    processed = 0
    for batch_samples in iter_sample_batches(samples, batch_size=batch_size):
        batch_tensor = load_batch_tensor(
            batch_samples,
            image_size=image_size,
            use_roi=use_roi,
            roi_margin_ratio=roi_margin_ratio,
        ).to(device=device, dtype=torch.float32)
        embeddings = extract_embeddings(model, batch_tensor, selected_indices=selected_indices)
        score_maps = compute_anomaly_map(embeddings, mean_tensor, var_tensor, eps=eps).cpu().numpy().astype(np.float32)

        for sample, score_map in zip(batch_samples, score_maps, strict=True):
            image = load_rgb(sample.image_path)
            target_full = load_binary_mask(sample.mask_path, image.shape[:2])
            if use_profile and profile_mean is not None and profile_std is not None:
                profile_map = compute_profile_map(
                    image.astype(np.uint8),
                    profile_mean,
                    profile_std,
                    profile_size=profile_size,
                    use_roi=use_roi,
                    roi_margin_ratio=roi_margin_ratio,
                    use_axis_align=use_axis_align,
                )
                profile_map = np.asarray(
                    Image.fromarray(profile_map, mode="F").resize((score_map.shape[1], score_map.shape[0]), Image.BILINEAR),
                    dtype=np.float32,
                )
                score_map = score_map + float(profile_weight) * profile_map
            if use_template and template_mean is not None and template_std is not None:
                template_map = compute_template_map(
                    image.astype(np.uint8),
                    template_mean,
                    template_std,
                    image_size=image_size,
                    use_roi=use_roi,
                    roi_margin_ratio=roi_margin_ratio,
                    use_axis_align=use_axis_align,
                    template_mode=template_mode,
                )
                template_map = np.asarray(
                    Image.fromarray(template_map, mode="F").resize((score_map.shape[1], score_map.shape[0]), Image.BILINEAR),
                    dtype=np.float32,
                )
                score_map = np.maximum(score_map, float(template_weight) * template_map)
            if use_peak_mask and expected_peaks is not None:
                profile_vec = compute_profile_vector(
                    image.astype(np.uint8),
                    profile_size=profile_size,
                    use_roi=use_roi,
                    roi_margin_ratio=roi_margin_ratio,
                    use_axis_align=use_axis_align,
                )
                tolerance = max(2, int(profile_size * peak_tolerance_ratio))
                peak_mask = build_peak_mask(profile_vec, expected_peaks.astype(np.int32), tolerance, profile_size=profile_size)
                peak_mask = np.asarray(
                    Image.fromarray(peak_mask, mode="F").resize((score_map.shape[1], score_map.shape[0]), Image.BILINEAR),
                    dtype=np.float32,
                )
                score_map = np.maximum(score_map, float(peak_weight) * peak_mask)
            target_small = np.asarray(
                Image.fromarray(target_full, mode="L").resize((score_map.shape[1], score_map.shape[0]), Image.NEAREST),
                dtype=np.uint8,
            )
            target_small = (target_small > 0).astype(np.uint8) * 255
            rows.append(
                ScoredSample(
                    class_name=sample.class_name,
                    image_name=sample.image_path.name,
                    target_small=target_small,
                    target_full=target_full,
                    score_map=score_map,
                )
            )

        processed += len(batch_samples)
        print(f"[score] processed {processed:03d}/{len(samples):03d} samples")

    return rows


def calibrate_postprocess(
    scored_samples: list[ScoredSample],
    threshold: float,
    min_area: int,
    open_kernel: int,
    close_kernel: int,
    threshold_scale_candidates: list[float],
    final_dilate_candidates: list[int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for threshold_scale in threshold_scale_candidates:
        for final_dilate_kernel in final_dilate_candidates:
            iou_rows: list[dict[str, Any]] = []
            for sample in scored_samples:
                prediction = build_binary_mask(
                    score_map=sample.score_map,
                    threshold=float(threshold * threshold_scale),
                    min_area=min_area,
                    open_kernel=open_kernel,
                    close_kernel=close_kernel,
                    output_shape=sample.target_full.shape,
                )
                prediction = apply_final_dilation(prediction, final_dilate_kernel)
                iou_rows.append(
                    {
                        "class_name": sample.class_name,
                        "image_name": sample.image_name,
                        "iou": float(mean_iou(prediction, sample.target_full)),
                        "elapsed_ms": 0.0,
                    }
                )
            summary = summarize_scores(iou_rows)
            rows.append(
                {
                    "threshold_scale": float(threshold_scale),
                    "final_dilate_kernel": int(final_dilate_kernel),
                    "val_mean_iou": float(summary["mean_iou"]),
                    "val_balanced_mean_iou": float(summary["balanced_mean_iou"]),
                    "val_defect_balanced_mean_iou": float(summary["defect_balanced_mean_iou"]),
                    "val_per_class_iou": summary["per_class_iou"],
                }
            )

    rows_sorted = sorted(
        rows,
        key=lambda row: (
            row["val_balanced_mean_iou"],
            row["val_defect_balanced_mean_iou"],
            row["val_mean_iou"],
            -row["final_dilate_kernel"],
        ),
        reverse=True,
    )
    return rows_sorted[0], {"top": rows_sorted[:120]}


def predict_sample(
    model: torch.nn.Module,
    sample: Any,
    artifact: PadimArtifact,
    device: torch.device,
) -> tuple[np.ndarray, float]:
    image = load_rgb(sample.image_path)
    started = time.perf_counter()
    batch_tensor = image_to_tensor(
        image,
        image_size=artifact.image_size,
        use_roi=artifact.use_roi,
        roi_margin_ratio=artifact.roi_margin_ratio,
    ).unsqueeze(0).to(device=device, dtype=torch.float32)
    embeddings = extract_embeddings(model, batch_tensor, selected_indices=artifact.selected_indices)
    mean_tensor = torch.from_numpy(np.transpose(artifact.mean, (2, 0, 1))).unsqueeze(0).to(device=device, dtype=torch.float32)
    var_tensor = torch.from_numpy(np.transpose(artifact.var, (2, 0, 1))).unsqueeze(0).to(device=device, dtype=torch.float32)
    score_map = compute_anomaly_map(embeddings, mean_tensor, var_tensor, eps=artifact.eps)[0].cpu().numpy().astype(np.float32)
    if artifact.use_profile and artifact.profile_mean is not None and artifact.profile_std is not None:
        working = image.astype(np.uint8)
        profile_map = compute_profile_map(
            working,
            artifact.profile_mean,
            artifact.profile_std,
            profile_size=artifact.profile_size,
            use_roi=artifact.use_roi,
            roi_margin_ratio=artifact.roi_margin_ratio,
            use_axis_align=artifact.use_axis_align,
        )
        profile_map = np.asarray(
            Image.fromarray(profile_map, mode="F").resize((score_map.shape[1], score_map.shape[0]), Image.BILINEAR),
            dtype=np.float32,
        )
        score_map = score_map + float(artifact.profile_weight) * profile_map
    if artifact.use_template and artifact.template_mean is not None and artifact.template_std is not None:
        template_map = compute_template_map(
            image.astype(np.uint8),
            artifact.template_mean,
            artifact.template_std,
            image_size=artifact.image_size,
            use_roi=artifact.use_roi,
            roi_margin_ratio=artifact.roi_margin_ratio,
            use_axis_align=artifact.use_axis_align,
            template_mode=artifact.template_mode,
        )
        template_map = np.asarray(
            Image.fromarray(template_map, mode="F").resize((score_map.shape[1], score_map.shape[0]), Image.BILINEAR),
            dtype=np.float32,
        )
        score_map = np.maximum(score_map, float(artifact.template_weight) * template_map)
    if artifact.use_peak_mask and artifact.expected_peaks is not None:
        profile_vec = compute_profile_vector(
            image.astype(np.uint8),
            profile_size=artifact.profile_size,
            use_roi=artifact.use_roi,
            roi_margin_ratio=artifact.roi_margin_ratio,
            use_axis_align=artifact.use_axis_align,
        )
        tolerance = max(2, int(artifact.profile_size * artifact.peak_tolerance_ratio))
        peak_mask = build_peak_mask(profile_vec, artifact.expected_peaks.astype(np.int32), tolerance, profile_size=artifact.profile_size)
        peak_mask = np.asarray(
            Image.fromarray(peak_mask, mode="F").resize((score_map.shape[1], score_map.shape[0]), Image.BILINEAR),
            dtype=np.float32,
        )
        score_map = np.maximum(score_map, float(artifact.peak_weight) * peak_mask)
    prediction = build_binary_mask(
        score_map=score_map,
        threshold=float(artifact.threshold * artifact.threshold_scale),
        min_area=artifact.min_area,
        open_kernel=artifact.open_kernel,
        close_kernel=artifact.close_kernel,
        output_shape=image.shape[:2],
    )
    prediction = apply_final_dilation(prediction, artifact.final_dilate_kernel)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return prediction, float(elapsed_ms)


def evaluate_split(
    model: torch.nn.Module,
    samples: list[Any],
    artifact: PadimArtifact,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for index, sample in enumerate(samples, start=1):
        image = load_rgb(sample.image_path)
        target = load_binary_mask(sample.mask_path, image.shape[:2])
        prediction, elapsed_ms = predict_sample(model, sample, artifact, device)
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
    if not args.weights_path.exists():
        raise FileNotFoundError(
            f"Backbone weights not found: {args.weights_path}. "
            "Download resnet18 weights into .torch_cache first."
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_artifact_path = args.sample_artifact_path.resolve()
    sample_artifact_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = load_split_manifest(args.split_path.resolve())
    train_samples = iter_split_samples(manifest, "train")
    val_samples = iter_split_samples(manifest, "val")
    test_samples = iter_split_samples(manifest, "test")

    if not train_samples:
        raise RuntimeError("Empty train split in manifest.")
    if any(not sample.is_good for sample in train_samples):
        raise RuntimeError("Train split contains defects. PaDiM fitting must use good only.")

    min_area_candidates = parse_int_list(args.min_area_candidates)
    open_kernel_candidates = parse_int_list(args.open_kernel_candidates)
    close_kernel_candidates = parse_int_list(args.close_kernel_candidates)
    threshold_scale_candidates = parse_float_list(args.threshold_scale_candidates)
    final_dilate_candidates = parse_int_list(args.final_dilate_candidates)

    device = select_device()
    print(f"Using device: {device}")

    model = build_model(weights_path=args.weights_path).to(device)
    model.eval()

    print("Fitting per-location Gaussian statistics on train/good only...")
    selected_indices, mean, var = fit_gaussian_stats(
        model=model,
        samples=train_samples,
        image_size=args.image_size,
        selected_dims=args.selected_dims,
        batch_size=args.batch_size,
        eps=args.eps,
        device=device,
        use_roi=args.use_roi,
        roi_margin_ratio=args.roi_margin_ratio,
    )
    profile_mean = None
    profile_std = None
    if args.use_profile:
        profiles: list[np.ndarray] = []
        for sample in train_samples:
            image = load_rgb(sample.image_path)
            working = preprocess_image(
                image,
                image_size=args.profile_size,
                use_roi=args.use_roi,
                roi_margin_ratio=args.roi_margin_ratio,
            ).astype(np.uint8)
            profile_vec = compute_profile_vector(
                working,
                profile_size=args.profile_size,
                use_roi=args.use_roi,
                roi_margin_ratio=args.roi_margin_ratio,
                use_axis_align=args.axis_align,
            )
            profiles.append(profile_vec)
        stacked = np.stack(profiles, axis=0)
        profile_mean = stacked.mean(axis=0).astype(np.float32)
        profile_std = np.maximum(stacked.std(axis=0), 1e-4).astype(np.float32)
    template_mean = None
    template_std = None
    if args.use_template:
        sum_template: np.ndarray | None = None
        sumsq_template: np.ndarray | None = None
        count_template = 0
        for sample in train_samples:
            image = load_rgb(sample.image_path)
            resized = compute_template_source(
                image,
                image_size=args.image_size,
                use_roi=args.use_roi,
                roi_margin_ratio=args.roi_margin_ratio,
                use_axis_align=args.axis_align,
                mode=args.template_mode,
            ).astype(np.float64)
            if sum_template is None:
                sum_template = np.zeros_like(resized, dtype=np.float64)
                sumsq_template = np.zeros_like(resized, dtype=np.float64)
            sum_template += resized
            sumsq_template += np.square(resized)
            count_template += 1
        if sum_template is not None and sumsq_template is not None and count_template > 0:
            template_mean = (sum_template / count_template).astype(np.float32)
            template_std = np.maximum(
                sumsq_template / count_template - np.square(template_mean.astype(np.float64)),
                1e-6,
            )
            template_std = np.sqrt(template_std).astype(np.float32)
    expected_peaks = None
    if args.use_peak_mask:
        peaks: list[np.ndarray] = []
        min_dist = max(2, int(args.profile_size * 0.05))
        for sample in train_samples:
            image = load_rgb(sample.image_path)
            profile_vec = compute_profile_vector(
                image.astype(np.uint8),
                profile_size=args.profile_size,
                use_roi=args.use_roi,
                roi_margin_ratio=args.roi_margin_ratio,
                use_axis_align=args.axis_align,
            )
            detected = detect_profile_peaks(profile_vec, peak_count=args.peak_count, min_distance=min_dist)
            if detected.size == args.peak_count:
                peaks.append(detected.astype(np.float32))
        if peaks:
            expected_peaks = np.mean(np.stack(peaks, axis=0), axis=0)

    print("Scoring validation split for threshold calibration...")
    val_scored = score_samples(
        model=model,
        samples=val_samples,
        image_size=args.image_size,
        selected_indices=selected_indices,
        mean=mean,
        var=var,
        eps=args.eps,
        batch_size=args.batch_size,
        device=device,
        use_roi=args.use_roi,
        roi_margin_ratio=args.roi_margin_ratio,
        use_profile=args.use_profile,
        profile_mean=profile_mean,
        profile_std=profile_std,
        profile_size=args.profile_size,
        profile_weight=args.profile_weight,
        use_axis_align=args.axis_align,
        use_peak_mask=args.use_peak_mask,
        expected_peaks=expected_peaks,
        peak_tolerance_ratio=args.peak_tolerance_ratio,
        peak_weight=args.peak_weight,
        use_template=args.use_template,
        template_mean=template_mean,
        template_std=template_std,
        template_weight=args.template_weight,
        template_mode=args.template_mode,
    )

    print("Scoring test split for reporting...")
    test_scored = score_samples(
        model=model,
        samples=test_samples,
        image_size=args.image_size,
        selected_indices=selected_indices,
        mean=mean,
        var=var,
        eps=args.eps,
        batch_size=args.batch_size,
        device=device,
        use_roi=args.use_roi,
        roi_margin_ratio=args.roi_margin_ratio,
        use_profile=args.use_profile,
        profile_mean=profile_mean,
        profile_std=profile_std,
        profile_size=args.profile_size,
        profile_weight=args.profile_weight,
        use_axis_align=args.axis_align,
        use_peak_mask=args.use_peak_mask,
        expected_peaks=expected_peaks,
        peak_tolerance_ratio=args.peak_tolerance_ratio,
        peak_weight=args.peak_weight,
        use_template=args.use_template,
        template_mean=template_mean,
        template_std=template_std,
        template_weight=args.template_weight,
        template_mode=args.template_mode,
    )
    _ = test_scored

    best_threshold, threshold_ranking = calibrate_threshold(
        val_samples=val_scored,
        quantile_low=args.threshold_quantile_low,
        quantile_high=args.threshold_quantile_high,
        threshold_steps=args.threshold_steps,
        stage1_top_k=args.stage1_top_k,
        min_area_candidates=min_area_candidates,
        open_kernel_candidates=open_kernel_candidates,
        close_kernel_candidates=close_kernel_candidates,
    )
    print(
        "Selected validation config:",
        f"threshold={best_threshold['threshold']:.6f}",
        f"open_kernel={best_threshold['open_kernel']}",
        f"close_kernel={best_threshold['close_kernel']}",
        f"min_area={best_threshold['min_area']}",
        f"val_balanced_mIoU_small={best_threshold['val_balanced_mean_iou_small']:.4f}",
    )

    postprocess_best, postprocess_ranking = calibrate_postprocess(
        scored_samples=val_scored,
        threshold=float(best_threshold["threshold"]),
        min_area=int(best_threshold["min_area"]),
        open_kernel=int(best_threshold["open_kernel"]),
        close_kernel=int(best_threshold["close_kernel"]),
        threshold_scale_candidates=threshold_scale_candidates,
        final_dilate_candidates=final_dilate_candidates,
    )
    print(
        "Selected full-resolution config:",
        f"threshold_scale={postprocess_best['threshold_scale']:.3f}",
        f"final_dilate_kernel={postprocess_best['final_dilate_kernel']}",
        f"val_balanced_mIoU_full={postprocess_best['val_balanced_mean_iou']:.4f}",
        f"val_defect_balanced_mIoU_full={postprocess_best['val_defect_balanced_mean_iou']:.4f}",
    )

    artifact = PadimArtifact(
        image_size=args.image_size,
        selected_indices=selected_indices,
        mean=mean,
        var=var,
        threshold=float(best_threshold["threshold"]),
        threshold_scale=float(postprocess_best["threshold_scale"]),
        min_area=int(best_threshold["min_area"]),
        open_kernel=int(best_threshold["open_kernel"]),
        close_kernel=int(best_threshold["close_kernel"]),
        final_dilate_kernel=int(postprocess_best["final_dilate_kernel"]),
        eps=float(args.eps),
        backbone_state_dict={key: value.detach().cpu() for key, value in model.state_dict().items()},
        use_roi=bool(args.use_roi),
        roi_margin_ratio=float(args.roi_margin_ratio),
        use_profile=bool(args.use_profile),
        profile_size=int(args.profile_size),
        profile_weight=float(args.profile_weight),
        profile_mean=profile_mean,
        profile_std=profile_std,
        use_axis_align=bool(args.axis_align),
        use_peak_mask=bool(args.use_peak_mask),
        peak_count=int(args.peak_count),
        peak_tolerance_ratio=float(args.peak_tolerance_ratio),
        peak_weight=float(args.peak_weight),
        expected_peaks=expected_peaks,
        use_template=bool(args.use_template),
        template_weight=float(args.template_weight),
        template_mode=str(args.template_mode),
        template_mean=template_mean,
        template_std=template_std,
    )

    artifact_path = output_dir / "model_artifact_padim.pt"
    save_artifact(artifact_path, artifact)
    save_artifact(sample_artifact_path, artifact)
    print(f"Saved PaDiM artifact: {artifact_path}")
    print(f"Copied PaDiM artifact for sample submission: {sample_artifact_path}")

    runtime_model = load_model_from_artifact(artifact).to(device)
    runtime_model.eval()

    print("Running full-resolution evaluation on val split...")
    val_metrics, val_rows = evaluate_split(runtime_model, val_samples, artifact, device)
    print("Running full-resolution evaluation on test split...")
    test_metrics, test_rows = evaluate_split(runtime_model, test_samples, artifact, device)

    summary = {
        "split_manifest": str(args.split_path.resolve()),
        "model_artifact": str(artifact_path),
        "training_setup": {
            "backend": "padim_resnet18_diag",
            "weights_path": str(args.weights_path.resolve()),
            "device": str(device),
            "image_size": int(args.image_size),
            "selected_dims": int(args.selected_dims),
            "batch_size": int(args.batch_size),
            "eps": float(args.eps),
            "use_roi": bool(args.use_roi),
            "roi_margin_ratio": float(args.roi_margin_ratio),
            "use_profile": bool(args.use_profile),
            "profile_size": int(args.profile_size),
            "profile_weight": float(args.profile_weight),
            "use_peak_mask": bool(args.use_peak_mask),
            "peak_count": int(args.peak_count),
            "peak_tolerance_ratio": float(args.peak_tolerance_ratio),
            "peak_weight": float(args.peak_weight),
            "use_template": bool(args.use_template),
            "template_weight": float(args.template_weight),
            "template_mode": str(args.template_mode),
            "threshold_scale_candidates": [float(value) for value in threshold_scale_candidates],
            "final_dilate_candidates": [int(value) for value in final_dilate_candidates],
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
        "selected_threshold_config": {
            "threshold": float(best_threshold["threshold"]),
            "open_kernel": int(best_threshold["open_kernel"]),
            "close_kernel": int(best_threshold["close_kernel"]),
            "min_area": int(best_threshold["min_area"]),
            "threshold_scale": float(postprocess_best["threshold_scale"]),
            "final_dilate_kernel": int(postprocess_best["final_dilate_kernel"]),
            "validation_small_proxy_score": float(best_threshold["val_balanced_mean_iou_small"]),
            "validation_full_balanced_mean_iou": float(postprocess_best["val_balanced_mean_iou"]),
            "validation_full_defect_balanced_mean_iou": float(postprocess_best["val_defect_balanced_mean_iou"]),
        },
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }

    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "threshold_search.json").write_text(
        json.dumps(
            {
                "threshold_calibration": threshold_ranking,
                "postprocess_calibration": postprocess_ranking,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "val_predictions.json").write_text(json.dumps(val_rows, indent=2), encoding="utf-8")
    (output_dir / "test_predictions.json").write_text(json.dumps(test_rows, indent=2), encoding="utf-8")

    print(f"Validation mean IoU: {val_metrics['mean_iou']:.4f}")
    print(f"Test mean IoU: {test_metrics['mean_iou']:.4f}")


if __name__ == "__main__":
    main()
