# Paper-Aligned Anomaly Baseline (Second Solution)

This folder contains a **separate**, literature-aligned anomaly-detection baseline for cable defect segmentation.
It does not modify the existing high-scoring submission in the repository root.

## Alignment with MVTec AD / Bergmann et al. (CVPR 2019)

The workflow follows the one-class anomaly philosophy used for MVTec AD:

- anomaly model fitting uses **only defect-free (`good`) images**
- defective images and masks are used only for:
  - validation/model-selection
  - threshold calibration
  - final evaluation
- no nearest-neighbor memorization over defective masks
- no supervised defect-mask segmentation training as the primary method

## Method Summary

Model type: **PaDiM-style per-location Gaussian anomaly model (diagonal covariance)** on deterministic hand-crafted features:

- resized image features at fixed spatial grid (`feature_size`, default 256)
- normalized RGB + histogram-equalized luminance + CLAHE-like local normalization
- Sobel gradient magnitude + Laplacian response + local contrast + saturation + radial-position prior
- per-location mean/variance estimated from normal training data
- anomaly score = per-location Mahalanobis-like z-score (diagonal)
- binary mask via validation-calibrated threshold + morphology (opening/closing) + component filtering

Inference artifact is lightweight (`model_artifact.npz`) and runtime code only depends on `numpy` and `Pillow`.

## Data Split Logic

Use `prepare_data.py` to build a deterministic manifest from `data/cable/`:

- `train`: only `good`
- `val`: `good` + all defect classes (stratified defect split)
- `test`: `good` + all defect classes (remaining samples)

This explicitly enforces:

- anomaly-model fitting: `train/good` only
- threshold calibration: `val` (good + defective)
- evaluation: `test` (good + defective)

## Files

- `prepare_data.py`: deterministic split manifest creation
- `train.py`: one-class fit, threshold calibration, metric export, artifact save
- `evaluate.py`: standalone evaluation on `val` or `test`
- `export_submission.py`: copies trained artifact into local `sample_submission/` and creates zip bundle
- `anomaly_baseline.py`: shared core functions for training/evaluation
- `sample_submission/model.py`: final `predict(image)` implementation
- `sample_submission/requirements.txt`: inference dependencies

## Run

From repository root:

```bash
./.venv/bin/python solutions/paper_anomaly_baseline/prepare_data.py
./.venv/bin/python solutions/paper_anomaly_baseline/train.py --threshold-steps 16 --stage1-top-k 6 --min-area-candidates 0,32,64,128 --open-kernel-candidates 1,3,5 --close-kernel-candidates 1,3
./.venv/bin/python solutions/paper_anomaly_baseline/evaluate.py --split test --output-path solutions/paper_anomaly_baseline/artifacts/model/test_eval.json
./.venv/bin/python solutions/paper_anomaly_baseline/export_submission.py
```

## Expected Artifacts

- `artifacts/split.json`
- `artifacts/model/model_artifact.npz`
- `artifacts/model/training_summary.json`
- `artifacts/model/threshold_search.json`
- `artifacts/model/val_predictions.json`
- `artifacts/model/test_predictions.json`
- `sample_submission/model_artifact.npz` (after export)
- `artifacts/submission_bundle/` and zipped archive

## Local Results (2026-03-30)

Deterministic split used (`seed=20260330`):

- train: 189 (`good` only)
- val: 74 (`good` + defects)
- test: 91 (`good` + defects)

Selected thresholding:

- score threshold: `20.0107497`
- opening kernel: `3`
- closing kernel: `3`
- min component area (feature grid): `32`

Metrics:

- validation mean IoU: `0.5484`
- validation balanced per-class mean IoU: `0.1758`
- test mean IoU: `0.4164`
- test balanced per-class mean IoU: `0.1559`
- full local evaluator (`scripts/evaluate_submission.py`) mean IoU: `0.3959`
- inference time: `~39 ms / image` on local run

Detailed values are stored in:

- `metrics.json`
- `artifacts/model/training_summary.json`
- `artifacts/model/test_eval.json`

## Limitations

- compact, hand-crafted feature space (not deep backbone features)
- global threshold may underfit class-specific anomaly scales
- no domain adaptation/registration beyond resize and per-image normalization
- designed as a robust, explainable baseline, not necessarily leaderboard-optimal
