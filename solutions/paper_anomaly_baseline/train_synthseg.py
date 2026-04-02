from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from anomaly_baseline import (
    apply_final_dilation,
    build_binary_mask,
    iter_split_samples,
    load_binary_mask,
    load_rgb,
    load_split_manifest,
    mean_iou,
)
from padim_backend import build_model
from synthetic_seg_backend import (
    DEFAULT_IMAGE_SIZE,
    SyntheticSegArtifact,
    SyntheticSegNet,
    apply_synthetic_anomaly,
    bce_dice_loss,
    build_model_from_artifact,
    estimate_cable_region,
    generate_synthetic_mask,
    image_to_tensor,
    save_artifact,
)
from train import ScoredSample, calibrate_threshold, parse_int_list, summarize_scores


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_SPLIT_PATH = THIS_DIR / "artifacts" / "split.json"
DEFAULT_OUTPUT_DIR = THIS_DIR / "artifacts" / "synthseg"
DEFAULT_SAMPLE_ARTIFACT_PATH = THIS_DIR / "sample_submission" / "model_artifact_synthseg.pt"
DEFAULT_WEIGHTS_PATH = THIS_DIR.parents[1] / ".torch_cache" / "hub" / "checkpoints" / "resnet18-f37072fd.pth"


@dataclass(frozen=True)
class PreloadedImage:
    name: str
    image: np.ndarray
    candidate_mask: np.ndarray


class SyntheticAnomalyDataset(Dataset):
    def __init__(self, images: list[PreloadedImage], image_size: int, length_multiplier: int = 6) -> None:
        self.images = images
        self.image_size = image_size
        self.length_multiplier = max(1, int(length_multiplier))

    def __len__(self) -> int:
        return len(self.images) * self.length_multiplier

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        base = self.images[index % len(self.images)]
        rng = np.random.default_rng(seed=(index * 7919 + 20260402) % (2**32))
        source = self.images[int(rng.integers(0, len(self.images)))]

        if rng.random() < 0.18:
            synth_image = base.image
            synth_mask = np.zeros(base.candidate_mask.shape, dtype=np.uint8)
        else:
            anomaly_mask = generate_synthetic_mask(rng, base.candidate_mask)
            synth_image = apply_synthetic_anomaly(base.image, source.image, anomaly_mask, rng)
            synth_mask = anomaly_mask.astype(np.uint8) * 255

        image_tensor = image_to_tensor(synth_image, image_size=self.image_size)
        mask_small = np.asarray(
            Image.fromarray(synth_mask, mode="L").resize((self.image_size, self.image_size), Image.NEAREST),
            dtype=np.uint8,
        )
        mask_tensor = torch.from_numpy((mask_small > 0).astype(np.float32))[None, ...]
        return image_tensor, mask_tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train synthetic-anomaly segmentation backend on good-only images.")
    parser.add_argument("--split-path", type=Path, default=DEFAULT_SPLIT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-artifact-path", type=Path, default=DEFAULT_SAMPLE_ARTIFACT_PATH)
    parser.add_argument("--weights-path", type=Path, default=DEFAULT_WEIGHTS_PATH)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--threshold-steps", type=int, default=18)
    parser.add_argument("--stage1-top-k", type=int, default=6)
    parser.add_argument("--min-area-candidates", type=str, default="0,16,32,64,128")
    parser.add_argument("--open-kernel-candidates", type=str, default="1,3,5")
    parser.add_argument("--close-kernel-candidates", type=str, default="1,3")
    parser.add_argument("--final-dilate-candidates", type=str, default="1,3,5,7")
    parser.add_argument("--freeze-encoder", action="store_true", default=False)
    parser.add_argument("--length-multiplier", type=int, default=6)
    return parser.parse_args()


def select_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preload_images(samples: list[Any]) -> list[PreloadedImage]:
    rows: list[PreloadedImage] = []
    for sample in samples:
        image = load_rgb(sample.image_path)
        rows.append(
            PreloadedImage(
                name=sample.image_path.name,
                image=image,
                candidate_mask=estimate_cable_region(image),
            )
        )
    return rows


def train_model(
    model: SyntheticSegNet,
    dataset: SyntheticAnomalyDataset,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
) -> list[dict[str, float]]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses: list[float] = []
        started = time.perf_counter()

        for batch_images, batch_masks in loader:
            batch_images = batch_images.to(device=device, dtype=torch.float32)
            batch_masks = batch_masks.to(device=device, dtype=torch.float32)
            logits = model(batch_images)
            loss = bce_dice_loss(logits, batch_masks)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        scheduler.step()
        elapsed = time.perf_counter() - started
        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        history.append({"epoch": float(epoch), "train_loss": mean_loss, "elapsed_sec": float(elapsed)})
        print(f"[epoch {epoch:02d}/{epochs:02d}] train_loss={mean_loss:.5f} elapsed={elapsed:.1f}s")

    return history


def score_samples(
    model: SyntheticSegNet,
    samples: list[Any],
    image_size: int,
    device: torch.device,
) -> list[ScoredSample]:
    rows: list[ScoredSample] = []
    model.eval()
    with torch.no_grad():
        for index, sample in enumerate(samples, start=1):
            image = load_rgb(sample.image_path)
            target_full = load_binary_mask(sample.mask_path, image.shape[:2])
            batch = image_to_tensor(image, image_size=image_size).unsqueeze(0).to(device=device, dtype=torch.float32)
            score_map = torch.sigmoid(model(batch))[0, 0].cpu().numpy().astype(np.float32)
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


def calibrate_final_dilation(
    scored_samples: list[ScoredSample],
    threshold: float,
    min_area: int,
    open_kernel: int,
    close_kernel: int,
    final_dilate_candidates: list[int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for final_dilate_kernel in final_dilate_candidates:
        iou_rows: list[dict[str, Any]] = []
        for sample in scored_samples:
            prediction = build_binary_mask(
                score_map=sample.score_map,
                threshold=threshold,
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
    model: SyntheticSegNet,
    sample: Any,
    artifact: SyntheticSegArtifact,
    device: torch.device,
) -> tuple[np.ndarray, float]:
    image = load_rgb(sample.image_path)
    started = time.perf_counter()
    batch = image_to_tensor(image, image_size=artifact.image_size).unsqueeze(0).to(device=device, dtype=torch.float32)
    score_map = torch.sigmoid(model(batch))[0, 0].cpu().numpy().astype(np.float32)
    prediction = build_binary_mask(
        score_map=score_map,
        threshold=artifact.threshold,
        min_area=artifact.min_area,
        open_kernel=artifact.open_kernel,
        close_kernel=artifact.close_kernel,
        output_shape=image.shape[:2],
    )
    prediction = apply_final_dilation(prediction, artifact.final_dilate_kernel)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return prediction, float(elapsed_ms)


def evaluate_split(
    model: SyntheticSegNet,
    samples: list[Any],
    artifact: SyntheticSegArtifact,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
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
        raise FileNotFoundError(f"Backbone weights not found: {args.weights_path}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_artifact_path = args.sample_artifact_path.resolve()
    sample_artifact_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = load_split_manifest(args.split_path.resolve())
    train_samples = iter_split_samples(manifest, "train")
    val_samples = iter_split_samples(manifest, "val")
    test_samples = iter_split_samples(manifest, "test")

    train_images = preload_images(train_samples)
    dataset = SyntheticAnomalyDataset(
        images=train_images,
        image_size=args.image_size,
        length_multiplier=args.length_multiplier,
    )

    device = select_device()
    print(f"Using device: {device}")

    encoder = build_model(weights_path=args.weights_path)
    model = SyntheticSegNet(encoder=encoder)
    if args.freeze_encoder:
        for parameter in model.encoder.parameters():
            parameter.requires_grad = False
    model = model.to(device)

    history = train_model(
        model=model,
        dataset=dataset,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    print("Scoring validation split for threshold calibration...")
    val_scored = score_samples(model, val_samples, image_size=args.image_size, device=device)
    print("Scoring test split for reporting...")
    test_scored = score_samples(model, test_samples, image_size=args.image_size, device=device)
    _ = test_scored

    best_threshold, threshold_ranking = calibrate_threshold(
        val_samples=val_scored,
        quantile_low=0.45,
        quantile_high=0.995,
        threshold_steps=args.threshold_steps,
        stage1_top_k=args.stage1_top_k,
        min_area_candidates=parse_int_list(args.min_area_candidates),
        open_kernel_candidates=parse_int_list(args.open_kernel_candidates),
        close_kernel_candidates=parse_int_list(args.close_kernel_candidates),
    )
    print(
        "Selected validation config:",
        f"threshold={best_threshold['threshold']:.6f}",
        f"open_kernel={best_threshold['open_kernel']}",
        f"close_kernel={best_threshold['close_kernel']}",
        f"min_area={best_threshold['min_area']}",
        f"val_balanced_mIoU_small={best_threshold['val_balanced_mean_iou_small']:.4f}",
    )

    postprocess_best, postprocess_ranking = calibrate_final_dilation(
        scored_samples=val_scored,
        threshold=float(best_threshold["threshold"]),
        min_area=int(best_threshold["min_area"]),
        open_kernel=int(best_threshold["open_kernel"]),
        close_kernel=int(best_threshold["close_kernel"]),
        final_dilate_candidates=parse_int_list(args.final_dilate_candidates),
    )
    print(
        "Selected full-resolution config:",
        f"final_dilate_kernel={postprocess_best['final_dilate_kernel']}",
        f"val_balanced_mIoU_full={postprocess_best['val_balanced_mean_iou']:.4f}",
        f"val_defect_balanced_mIoU_full={postprocess_best['val_defect_balanced_mean_iou']:.4f}",
    )

    artifact = SyntheticSegArtifact(
        image_size=args.image_size,
        threshold=float(best_threshold["threshold"]),
        min_area=int(best_threshold["min_area"]),
        open_kernel=int(best_threshold["open_kernel"]),
        close_kernel=int(best_threshold["close_kernel"]),
        final_dilate_kernel=int(postprocess_best["final_dilate_kernel"]),
        freeze_encoder=bool(args.freeze_encoder),
        encoder_state_dict={key: value.detach().cpu() for key, value in model.encoder.state_dict().items()},
        decoder_state_dict={
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
            if not key.startswith("encoder.")
        },
    )

    artifact_path = output_dir / "model_artifact_synthseg.pt"
    save_artifact(artifact_path, artifact)
    save_artifact(sample_artifact_path, artifact)
    print(f"Saved synthseg artifact: {artifact_path}")
    print(f"Copied synthseg artifact for sample submission: {sample_artifact_path}")

    runtime_model = build_model_from_artifact(artifact).to(device)
    runtime_model.eval()
    print("Running full-resolution evaluation on val split...")
    val_metrics, val_rows = evaluate_split(runtime_model, val_samples, artifact, device)
    print("Running full-resolution evaluation on test split...")
    test_metrics, test_rows = evaluate_split(runtime_model, test_samples, artifact, device)

    summary = {
        "split_manifest": str(args.split_path.resolve()),
        "model_artifact": str(artifact_path),
        "training_setup": {
            "backend": "synthetic_seg_resnet18",
            "weights_path": str(args.weights_path.resolve()),
            "device": str(device),
            "image_size": int(args.image_size),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "freeze_encoder": bool(args.freeze_encoder),
            "length_multiplier": int(args.length_multiplier),
            "anomaly_model_fitting_data": {
                "split": "train",
                "class_filter": "good",
                "num_samples": len(train_samples),
                "method": "synthetic anomaly generation with synthetic masks only",
            },
        },
        "training_history": history,
        "selected_threshold_config": {
            "threshold": float(best_threshold["threshold"]),
            "open_kernel": int(best_threshold["open_kernel"]),
            "close_kernel": int(best_threshold["close_kernel"]),
            "min_area": int(best_threshold["min_area"]),
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
