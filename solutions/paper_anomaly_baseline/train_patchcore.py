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
from padim_backend import DEFAULT_SELECTED_DIMS
from patchcore_backend import (
    DEFAULT_DISTANCE_CHUNK,
    DEFAULT_EPS,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_K,
    DEFAULT_L2_NORM,
    DEFAULT_MEMORY_SIZE,
    DEFAULT_PATCHES_PER_IMAGE,
    DEFAULT_ROI_MARGIN_RATIO,
    PatchCoreArtifact,
    build_model,
    compute_patchcore_scores,
    extract_embeddings,
    preprocess_for_patchcore,
    save_artifact,
    select_feature_indices,
)
from train import ScoredSample, calibrate_threshold, parse_float_list, parse_int_list, summarize_scores


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_SPLIT_PATH = THIS_DIR / "artifacts" / "split.json"
DEFAULT_OUTPUT_DIR = THIS_DIR / "artifacts" / "patchcore"
DEFAULT_SAMPLE_ARTIFACT_PATH = THIS_DIR / "sample_submission" / "model_artifact_patchcore.pt"
DEFAULT_WEIGHTS_PATH = THIS_DIR.parents[1] / ".torch_cache" / "hub" / "checkpoints" / "resnet18-f37072fd.pth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PatchCore-style anomaly backend on good-only samples.")
    parser.add_argument("--split-path", type=Path, default=DEFAULT_SPLIT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-artifact-path", type=Path, default=DEFAULT_SAMPLE_ARTIFACT_PATH)
    parser.add_argument("--weights-path", type=Path, default=DEFAULT_WEIGHTS_PATH)
    parser.add_argument("--backbone", type=str, default="resnet18", choices=["resnet18", "resnet50"])
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
    parser.add_argument("--axis-align", action="store_true", default=True)
    parser.add_argument("--patches-per-image", type=int, default=DEFAULT_PATCHES_PER_IMAGE)
    parser.add_argument("--memory-size", type=int, default=DEFAULT_MEMORY_SIZE)
    parser.add_argument("--distance-chunk", type=int, default=DEFAULT_DISTANCE_CHUNK)
    parser.add_argument("--k-neighbors", type=int, default=DEFAULT_K)
    parser.add_argument("--l2-normalize", action="store_true", default=DEFAULT_L2_NORM)
    return parser.parse_args()


def select_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def iter_sample_batches(samples: list[Any], batch_size: int) -> list[list[Any]]:
    return [samples[index : index + batch_size] for index in range(0, len(samples), batch_size)]


def collect_memory_bank(
    model: torch.nn.Module,
    samples: list[Any],
    image_size: int,
    selected_indices: np.ndarray,
    device: torch.device,
    use_roi: bool,
    roi_margin_ratio: float,
    use_axis_align: bool,
    patches_per_image: int,
    memory_size: int,
) -> np.ndarray:
    rng = np.random.default_rng(20260402)
    memory_list: list[np.ndarray] = []

    for idx, sample in enumerate(samples, start=1):
        image = load_rgb(sample.image_path)
        tensor = preprocess_for_patchcore(
            image,
            image_size=image_size,
            use_roi=use_roi,
            roi_margin_ratio=roi_margin_ratio,
            use_axis_align=use_axis_align,
        ).unsqueeze(0).to(device=device, dtype=torch.float32)
        embeddings = extract_embeddings(model, tensor, selected_indices=selected_indices)[0].cpu().numpy()
        channels, height, width = embeddings.shape
        patches = embeddings.reshape(channels, height * width).transpose(1, 0)
        if patches_per_image < patches.shape[0]:
            choice = rng.choice(patches.shape[0], size=patches_per_image, replace=False)
            patches = patches[choice]
        memory_list.append(patches.astype(np.float32))
        if idx % 25 == 0 or idx == len(samples):
            print(f"[memory] processed {idx:03d}/{len(samples):03d} good samples")

    memory = np.concatenate(memory_list, axis=0) if memory_list else np.empty((0, selected_indices.size), dtype=np.float32)
    if memory_size > 0 and memory.shape[0] > memory_size:
        choice = rng.choice(memory.shape[0], size=memory_size, replace=False)
        memory = memory[choice]
    return memory


def score_samples(
    model: torch.nn.Module,
    samples: list[Any],
    image_size: int,
    selected_indices: np.ndarray,
    memory_bank: np.ndarray,
    distance_chunk: int,
    k_neighbors: int,
    l2_normalize: bool,
    device: torch.device,
    use_roi: bool,
    roi_margin_ratio: float,
    use_axis_align: bool,
) -> list[ScoredSample]:
    rows: list[ScoredSample] = []
    processed = 0
    for batch_samples in iter_sample_batches(samples, batch_size=1):
        sample = batch_samples[0]
        image = load_rgb(sample.image_path)
        tensor = preprocess_for_patchcore(
            image,
            image_size=image_size,
            use_roi=use_roi,
            roi_margin_ratio=roi_margin_ratio,
            use_axis_align=use_axis_align,
        ).unsqueeze(0).to(device=device, dtype=torch.float32)
        embeddings = extract_embeddings(model, tensor, selected_indices=selected_indices)
        score_map = compute_patchcore_scores(
            embeddings,
            memory_bank,
            distance_chunk=distance_chunk,
            k_neighbors=k_neighbors,
            l2_normalize=l2_normalize,
            device=device,
        )
        score_map_np = score_map.cpu().numpy().astype(np.float32)
        target_full = load_binary_mask(sample.mask_path, image.shape[:2])
        target_small = np.asarray(
            Image.fromarray(target_full, mode="L").resize((score_map_np.shape[1], score_map_np.shape[0]), Image.NEAREST),
            dtype=np.uint8,
        )
        target_small = (target_small > 0).astype(np.uint8) * 255
        rows.append(
            ScoredSample(
                class_name=sample.class_name,
                image_name=sample.image_path.name,
                target_small=target_small,
                target_full=target_full,
                score_map=score_map_np,
            )
        )
        processed += 1
        if processed % 20 == 0 or processed == len(samples):
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
    artifact: PatchCoreArtifact,
    device: torch.device,
) -> tuple[np.ndarray, float]:
    image = load_rgb(sample.image_path)
    started = time.perf_counter()
    tensor = preprocess_for_patchcore(
        image,
        image_size=artifact.image_size,
        use_roi=artifact.use_roi,
        roi_margin_ratio=artifact.roi_margin_ratio,
        use_axis_align=artifact.use_axis_align,
    ).unsqueeze(0).to(device=device, dtype=torch.float32)
    embeddings = extract_embeddings(model, tensor, selected_indices=artifact.selected_indices)
    score_map = compute_patchcore_scores(
        embeddings,
        artifact.memory_bank,
        distance_chunk=artifact.distance_chunk,
        k_neighbors=artifact.k_neighbors,
        l2_normalize=artifact.l2_normalize,
        device=device,
    ).cpu().numpy().astype(np.float32)
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
    artifact: PatchCoreArtifact,
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
        raise RuntimeError("Train split contains defects. PatchCore fitting must use good only.")

    min_area_candidates = parse_int_list(args.min_area_candidates)
    open_kernel_candidates = parse_int_list(args.open_kernel_candidates)
    close_kernel_candidates = parse_int_list(args.close_kernel_candidates)
    threshold_scale_candidates = parse_float_list(args.threshold_scale_candidates)
    final_dilate_candidates = parse_int_list(args.final_dilate_candidates)

    device = select_device()
    print(f"Using device: {device}")

    model = build_model(weights_path=args.weights_path, backbone=args.backbone).to(device)
    model.eval()

    if args.backbone == "resnet50":
        total_dims = 256 + 512 + 1024
    else:
        total_dims = 64 + 128 + 256
    selected_indices = select_feature_indices(total_dims=total_dims, selected_dims=args.selected_dims)

    print("Building memory bank from train/good only...")
    memory_bank = collect_memory_bank(
        model=model,
        samples=train_samples,
        image_size=args.image_size,
        selected_indices=selected_indices,
        device=device,
        use_roi=args.use_roi,
        roi_margin_ratio=args.roi_margin_ratio,
        use_axis_align=args.axis_align,
        patches_per_image=args.patches_per_image,
        memory_size=args.memory_size,
    )

    print("Scoring validation split for threshold calibration...")
    val_scored = score_samples(
        model=model,
        samples=val_samples,
        image_size=args.image_size,
        selected_indices=selected_indices,
        memory_bank=memory_bank,
        distance_chunk=args.distance_chunk,
        k_neighbors=args.k_neighbors,
        l2_normalize=args.l2_normalize,
        device=device,
        use_roi=args.use_roi,
        roi_margin_ratio=args.roi_margin_ratio,
        use_axis_align=args.axis_align,
    )

    print("Scoring test split for reporting...")
    test_scored = score_samples(
        model=model,
        samples=test_samples,
        image_size=args.image_size,
        selected_indices=selected_indices,
        memory_bank=memory_bank,
        distance_chunk=args.distance_chunk,
        k_neighbors=args.k_neighbors,
        l2_normalize=args.l2_normalize,
        device=device,
        use_roi=args.use_roi,
        roi_margin_ratio=args.roi_margin_ratio,
        use_axis_align=args.axis_align,
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

    artifact = PatchCoreArtifact(
        image_size=args.image_size,
        selected_indices=selected_indices,
        memory_bank=memory_bank,
        threshold=float(best_threshold["threshold"]),
        threshold_scale=float(postprocess_best["threshold_scale"]),
        min_area=int(best_threshold["min_area"]),
        open_kernel=int(best_threshold["open_kernel"]),
        close_kernel=int(best_threshold["close_kernel"]),
        final_dilate_kernel=int(postprocess_best["final_dilate_kernel"]),
        eps=float(args.eps),
        backbone_state_dict={key: value.detach().cpu() for key, value in model.state_dict().items()},
        backbone=str(args.backbone),
        use_roi=bool(args.use_roi),
        roi_margin_ratio=float(args.roi_margin_ratio),
        use_axis_align=bool(args.axis_align),
        patches_per_image=int(args.patches_per_image),
        memory_size=int(args.memory_size),
        distance_chunk=int(args.distance_chunk),
        k_neighbors=int(args.k_neighbors),
        l2_normalize=bool(args.l2_normalize),
    )

    artifact_path = output_dir / "model_artifact_patchcore.pt"
    save_artifact(artifact_path, artifact)
    save_artifact(sample_artifact_path, artifact)
    print(f"Saved PatchCore artifact: {artifact_path}")
    print(f"Copied PatchCore artifact for sample submission: {sample_artifact_path}")

    runtime_model = build_model(weights_path=None, backbone=args.backbone).to(device)
    runtime_model.load_state_dict(artifact.backbone_state_dict)
    runtime_model.eval()

    print("Running full-resolution evaluation on val split...")
    val_metrics, val_rows = evaluate_split(runtime_model, val_samples, artifact, device)
    print("Running full-resolution evaluation on test split...")
    test_metrics, test_rows = evaluate_split(runtime_model, test_samples, artifact, device)

    summary = {
        "split_manifest": str(args.split_path.resolve()),
        "model_artifact": str(artifact_path),
        "training_setup": {
            "backend": "patchcore_resnet18",
            "weights_path": str(args.weights_path.resolve()),
            "device": str(device),
            "image_size": int(args.image_size),
            "selected_dims": int(args.selected_dims),
            "batch_size": int(args.batch_size),
            "eps": float(args.eps),
            "backbone": str(args.backbone),
            "use_roi": bool(args.use_roi),
            "roi_margin_ratio": float(args.roi_margin_ratio),
            "use_axis_align": bool(args.axis_align),
            "patches_per_image": int(args.patches_per_image),
            "memory_size": int(args.memory_size),
            "distance_chunk": int(args.distance_chunk),
            "k_neighbors": int(args.k_neighbors),
            "l2_normalize": bool(args.l2_normalize),
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
