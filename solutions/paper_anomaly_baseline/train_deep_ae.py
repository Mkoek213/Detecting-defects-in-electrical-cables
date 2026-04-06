from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

from anomaly_baseline import (
    apply_final_dilation,
    build_binary_mask,
    iter_split_samples,
    load_binary_mask,
    load_rgb,
    load_split_manifest,
    mean_iou,
)
from deep_autoencoder import (
    DEFAULT_GRAD_WEIGHT,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_RGB_WEIGHT,
    DEFAULT_SCORE_BLUR_KERNEL,
    DeepArtifact,
    DenoisingAutoencoder,
    apply_training_corruption,
    image_to_tensor,
    reconstruction_loss,
    save_artifact,
    score_map_from_tensors,
)
from train import ScoredSample, calibrate_threshold, parse_float_list, parse_int_list, summarize_scores


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_SPLIT_PATH = THIS_DIR / "artifacts" / "split.json"
DEFAULT_OUTPUT_DIR = THIS_DIR / "artifacts" / "deep_ae"
DEFAULT_SAMPLE_ARTIFACT_PATH = THIS_DIR / "sample_submission" / "model_artifact.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train denoising autoencoder backend on good-only samples.")
    parser.add_argument("--split-path", type=Path, default=DEFAULT_SPLIT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-artifact-path", type=Path, default=DEFAULT_SAMPLE_ARTIFACT_PATH)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--channels", type=str, default="16,32,64,96,128")
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--threshold-quantile-low", type=float, default=0.75)
    parser.add_argument("--threshold-quantile-high", type=float, default=0.998)
    parser.add_argument("--threshold-steps", type=int, default=18)
    parser.add_argument("--stage1-top-k", type=int, default=6)
    parser.add_argument("--min-area-candidates", type=str, default="0,16,32,64,128")
    parser.add_argument("--open-kernel-candidates", type=str, default="1,3,5")
    parser.add_argument("--close-kernel-candidates", type=str, default="1,3")
    parser.add_argument("--threshold-scale-candidates", type=str, default="0.75,0.8,0.85,0.9,0.95,1.0")
    parser.add_argument("--final-dilate-candidates", type=str, default="1,3,5,7")
    parser.add_argument("--rgb-weight", type=float, default=DEFAULT_RGB_WEIGHT)
    parser.add_argument("--grad-weight", type=float, default=DEFAULT_GRAD_WEIGHT)
    parser.add_argument("--score-blur-kernel", type=int, default=DEFAULT_SCORE_BLUR_KERNEL)
    return parser.parse_args()


def parse_channels(value: str) -> tuple[int, ...]:
    channels = tuple(int(token.strip()) for token in value.split(",") if token.strip())
    if len(channels) < 2:
        raise ValueError("channels must contain at least two values.")
    return channels


def select_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preload_good_tensors(samples: list[Any], image_size: int) -> torch.Tensor:
    tensors = [image_to_tensor(load_rgb(sample.image_path), image_size=image_size) for sample in samples]
    return torch.stack(tensors)


def evaluate_reconstruction_loss(
    model: DenoisingAutoencoder,
    images: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> float:
    model.eval()
    losses: list[float] = []
    loader = DataLoader(TensorDataset(images), batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for (targets,) in loader:
            targets = targets.to(device=device, dtype=torch.float32)
            reconstructions = model(targets)
            losses.append(float(reconstruction_loss(reconstructions, targets).item()))
    return float(np.mean(losses)) if losses else 0.0


def train_autoencoder(
    model: DenoisingAutoencoder,
    train_images: torch.Tensor,
    monitor_images: torch.Tensor,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
) -> tuple[dict[str, torch.Tensor], list[dict[str, float]]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    loader = DataLoader(TensorDataset(train_images), batch_size=batch_size, shuffle=True, drop_last=False)

    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses: list[float] = []
        started = time.perf_counter()

        for (targets,) in loader:
            targets = targets.to(device=device, dtype=torch.float32)
            inputs = apply_training_corruption(targets)
            predictions = model(inputs)
            loss = reconstruction_loss(predictions, targets)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        scheduler.step()
        train_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        monitor_loss = evaluate_reconstruction_loss(
            model=model,
            images=monitor_images,
            device=device,
            batch_size=batch_size,
        )
        elapsed = time.perf_counter() - started
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "monitor_loss": monitor_loss,
                "elapsed_sec": float(elapsed),
            }
        )
        print(
            f"[epoch {epoch:02d}/{epochs:02d}] train_loss={train_loss:.5f} "
            f"monitor_loss={monitor_loss:.5f} elapsed={elapsed:.1f}s"
        )

        if monitor_loss < best_loss:
            best_loss = monitor_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Training did not produce a model state.")
    return best_state, history


def score_samples(
    samples: list[Any],
    model: DenoisingAutoencoder,
    device: torch.device,
    image_size: int,
    rgb_weight: float,
    grad_weight: float,
    score_blur_kernel: int,
) -> list[ScoredSample]:
    model.eval()
    rows: list[ScoredSample] = []

    with torch.no_grad():
        for index, sample in enumerate(samples, start=1):
            image = load_rgb(sample.image_path)
            target_full = load_binary_mask(sample.mask_path, image.shape[:2])
            image_tensor = image_to_tensor(image, image_size=image_size).unsqueeze(0).to(device=device, dtype=torch.float32)
            reconstruction = model(image_tensor)
            score_map = score_map_from_tensors(
                image_tensor,
                reconstruction,
                rgb_weight=rgb_weight,
                grad_weight=grad_weight,
                blur_kernel=score_blur_kernel,
            )[0].cpu().numpy().astype(np.float32)
            target_small = np.asarray(
                Image.fromarray(target_full, mode="L").resize((image_size, image_size), Image.NEAREST),
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
            if index % 20 == 0 or index == len(samples):
                print(f"[score] processed {index:03d}/{len(samples):03d} samples")

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
    sample: Any,
    model: DenoisingAutoencoder,
    device: torch.device,
    artifact: DeepArtifact,
) -> tuple[np.ndarray, float]:
    model.eval()
    image = load_rgb(sample.image_path)
    started = time.perf_counter()
    with torch.no_grad():
        image_tensor = image_to_tensor(image, image_size=artifact.image_size).unsqueeze(0).to(device=device, dtype=torch.float32)
        reconstruction = model(image_tensor)
        score_map = score_map_from_tensors(
            image_tensor,
            reconstruction,
            rgb_weight=artifact.rgb_weight,
            grad_weight=artifact.grad_weight,
            blur_kernel=artifact.score_blur_kernel,
        )[0].cpu().numpy().astype(np.float32)
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
    samples: list[Any],
    model: DenoisingAutoencoder,
    device: torch.device,
    artifact: DeepArtifact,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for index, sample in enumerate(samples, start=1):
        target = load_binary_mask(sample.mask_path, load_rgb(sample.image_path).shape[:2])
        prediction, elapsed_ms = predict_sample(sample, model, device, artifact)
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
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_split_manifest(args.split_path.resolve())
    train_samples = iter_split_samples(manifest, "train")
    val_samples = iter_split_samples(manifest, "val")
    test_samples = iter_split_samples(manifest, "test")
    val_good_samples = [sample for sample in val_samples if sample.is_good]

    if not train_samples:
        raise RuntimeError("Empty train split in manifest.")
    if any(not sample.is_good for sample in train_samples):
        raise RuntimeError("Train split contains defects. Deep autoencoder must fit on good only.")
    if not val_good_samples:
        raise RuntimeError("Validation split must contain at least one good sample for monitoring.")

    channels = parse_channels(args.channels)
    min_area_candidates = parse_int_list(args.min_area_candidates)
    open_kernel_candidates = parse_int_list(args.open_kernel_candidates)
    close_kernel_candidates = parse_int_list(args.close_kernel_candidates)
    threshold_scale_candidates = parse_float_list(args.threshold_scale_candidates)
    final_dilate_candidates = parse_int_list(args.final_dilate_candidates)

    device = select_device()
    print(f"Using device: {device}")

    train_images = preload_good_tensors(train_samples, image_size=args.image_size)
    monitor_images = preload_good_tensors(val_good_samples, image_size=args.image_size)

    model = DenoisingAutoencoder(channels=channels).to(device)
    best_state, history = train_autoencoder(
        model=model,
        train_images=train_images,
        monitor_images=monitor_images,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    model.load_state_dict(best_state)
    model.eval()

    print("Scoring validation split for threshold calibration...")
    val_scored = score_samples(
        samples=val_samples,
        model=model,
        device=device,
        image_size=args.image_size,
        rgb_weight=args.rgb_weight,
        grad_weight=args.grad_weight,
        score_blur_kernel=args.score_blur_kernel,
    )
    print("Scoring test split for reporting...")
    test_scored = score_samples(
        samples=test_samples,
        model=model,
        device=device,
        image_size=args.image_size,
        rgb_weight=args.rgb_weight,
        grad_weight=args.grad_weight,
        score_blur_kernel=args.score_blur_kernel,
    )

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

    artifact = DeepArtifact(
        image_size=args.image_size,
        threshold=float(best_threshold["threshold"]),
        threshold_scale=float(postprocess_best["threshold_scale"]),
        min_area=int(best_threshold["min_area"]),
        open_kernel=int(best_threshold["open_kernel"]),
        close_kernel=int(best_threshold["close_kernel"]),
        final_dilate_kernel=int(postprocess_best["final_dilate_kernel"]),
        score_blur_kernel=int(args.score_blur_kernel),
        rgb_weight=float(args.rgb_weight),
        grad_weight=float(args.grad_weight),
        channels=channels,
        state_dict=best_state,
    )

    artifact_path = output_dir / "model_artifact.pt"
    save_artifact(artifact_path, artifact)
    args.sample_artifact_path.resolve().parent.mkdir(parents=True, exist_ok=True)
    save_artifact(args.sample_artifact_path.resolve(), artifact)
    print(f"Saved deep artifact: {artifact_path}")
    print(f"Copied deep artifact for sample submission: {args.sample_artifact_path.resolve()}")

    print("Running full-resolution evaluation on val split...")
    val_metrics, val_rows = evaluate_split(val_samples, model, device, artifact)
    print("Running full-resolution evaluation on test split...")
    test_metrics, test_rows = evaluate_split(test_samples, model, device, artifact)

    summary = {
        "split_manifest": str(args.split_path.resolve()),
        "model_artifact": str(artifact_path),
        "training_setup": {
            "backend": "deep_denoising_autoencoder",
            "device": str(device),
            "image_size": int(args.image_size),
            "channels": [int(value) for value in channels],
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "rgb_weight": float(args.rgb_weight),
            "grad_weight": float(args.grad_weight),
            "score_blur_kernel": int(args.score_blur_kernel),
            "threshold_scale_candidates": [float(value) for value in threshold_scale_candidates],
            "final_dilate_candidates": [int(value) for value in final_dilate_candidates],
            "anomaly_model_fitting_data": {
                "split": "train",
                "class_filter": "good",
                "num_samples": len(train_samples),
            },
            "monitoring_data": {
                "split": "val",
                "class_filter": "good",
                "num_samples": len(val_good_samples),
            },
        },
        "training_history": history,
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
