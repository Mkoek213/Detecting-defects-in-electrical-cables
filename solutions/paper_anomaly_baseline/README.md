# Paper-Aligned Anomaly Backends

This folder contains a separate, literature-aligned anomaly track for cable defect segmentation.
It stays isolated from the existing root submission and keeps the one-class constraint from `AGENT_BRIEF.md`.

## Current Best Backend

Best verified local submission: **`padim_resnet18_diag`**

- pretrained `ResNet18` feature extractor
- PaDiM-style per-location diagonal Gaussian anomaly model
- fit on `train/good` only
- thresholding and postprocessing calibrated on `val`
- packaged submission uses:
  - `model.py`
  - `padim_runtime.py`
  - `model_artifact_padim.pt`
  - `requirements.txt`

Best full local evaluator result:

- mean IoU: `0.420021`
- average inference time: `260.17 ms / image`

Previous handcrafted union ensemble is still kept as a lighter fallback, but it is no longer the best local result.

## Alignment With MVTec AD / Bergmann et al.

- anomaly model fitting uses only defect-free `good` images
- defective images and masks are used only for validation, calibration, and evaluation
- no supervised defect-mask training is used as the main method in this track
- no defective masks are used during anomaly-model fitting

## Data Split Logic

Use `prepare_data.py` to create the deterministic manifest from `data/cable/`:

- `train`: only `good`
- `val`: `good` + defects
- `test`: `good` + defects

This enforces:

- fitting: `train/good` only
- threshold calibration: `val`
- reporting: `test`

## Main Files

- `prepare_data.py`: deterministic split manifest
- `anomaly_baseline.py`: handcrafted anomaly backend
- `train.py`: handcrafted backend training/calibration
- `padim_backend.py`: deep PaDiM-style backend utilities
- `train_padim.py`: pretrained `ResNet18` PaDiM backend training/calibration
- `export_submission.py`: backend-aware bundle export
- `sample_submission/model_padim.py`: local entrypoint for PaDiM submission runtime
- `sample_submission/padim_runtime.py`: packaged PaDiM inference helper

## Recommended Run

From repository root:

```bash
./.venv/bin/python solutions/paper_anomaly_baseline/prepare_data.py
./.venv/bin/python solutions/paper_anomaly_baseline/train_padim.py --image-size 256 --selected-dims 160 --batch-size 8
./.venv/bin/python scripts/evaluate_submission.py --model-path solutions/paper_anomaly_baseline/sample_submission/model_padim.py
./.venv/bin/python solutions/paper_anomaly_baseline/export_submission.py --backend padim
```

The training script expects pretrained `ResNet18` weights at:

```bash
.torch_cache/hub/checkpoints/resnet18-f37072fd.pth
```

## Verified Local Results

### `padim_resnet18_diag`

Configuration:

- image size: `256`
- selected feature dimensions: `160`
- threshold: `102.0678849`
- threshold scale: `0.85`
- opening: `5`
- closing: `3`
- min area: `64`
- final dilation: `7`

Split metrics:

- validation mean IoU: `0.4618`
- validation balanced mean IoU: `0.1919`
- validation defect-balanced mean IoU: `0.1221`
- test mean IoU: `0.4003`
- test balanced mean IoU: `0.1823`

Full local evaluator:

- mean IoU: `0.4200`
- mean runtime: `260.2 ms / image`

Selected per-class local evaluator means:

- `good`: `0.9783`
- `bent_wire`: `0.1724`
- `cable_swap`: `0.0700`
- `combined`: `0.1376`
- `cut_inner_insulation`: `0.1065`
- `missing_cable`: `0.1008`
- `missing_wire`: `0.0320`
- `poke_insulation`: `0.1666`

### Handcrafted Union Ensemble

Kept for comparison:

- full local evaluator mean IoU: `0.3990`
- mean runtime: `91.8 ms / image`

## Submission Bundle

Current exported ZIP:

- `artifacts/paper_anomaly_baseline_submission.zip`

Contents for the best backend:

- `model.py`
- `requirements.txt`
- `model_artifact_padim.pt`
- `padim_runtime.py`

## Tradeoffs

`padim_resnet18_diag`

- clear local accuracy gain over handcrafted baseline
- much better preservation of `good`
- better coverage on `cable_swap`, `missing_cable`, and `cut_inner_insulation`
- materially slower on CPU
- `missing_wire` is still weak

Handcrafted ensemble

- faster and simpler
- lighter artifact
- substantially worse on several structural anomaly classes

## Artifacts

Best deep backend artifacts:

- `artifacts/padim/model_artifact_padim.pt`
- `artifacts/padim/training_summary.json`
- `artifacts/padim/threshold_search.json`
- `artifacts/padim/val_predictions.json`
- `artifacts/padim/test_predictions.json`

Summary metrics:

- `metrics.json`
