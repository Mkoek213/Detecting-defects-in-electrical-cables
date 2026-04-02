# Agent Brief: Cable Defect Segmentation Improvement

Date: 2026-04-02  
Repository: `/home/mikolaj/work/Detecting-defects-in-electrical-cables`

## 1) Goal

Improve defect segmentation quality for the cable dataset while preserving the required cloud-submission interface:

```python
def predict(image: np.ndarray) -> np.ndarray
```

Output mask must be:
- shape `(H, W)`
- dtype `uint8`
- values only `{0, 255}`
- `255 = defect`, `0 = background`

## 2) What Has Already Been Done

We built a **separate** paper-aligned solution in:

- `solutions/paper_anomaly_baseline/`

This does **not** overwrite the existing top-scoring solution in other folders.

Current baseline uses one-class anomaly detection:
- one-class fitting on `good` only
- validation-driven thresholding/postprocessing
- handcrafted features + per-location Gaussian model

Key files:
- `solutions/paper_anomaly_baseline/prepare_data.py`
- `solutions/paper_anomaly_baseline/train.py`
- `solutions/paper_anomaly_baseline/anomaly_baseline.py`
- `solutions/paper_anomaly_baseline/evaluate.py`
- `solutions/paper_anomaly_baseline/export_submission.py`
- `solutions/paper_anomaly_baseline/sample_submission/model.py`

## 3) Current Metrics (Reference Point)

From current artifacts:
- split-test mean IoU: `0.4163718488`
- full local evaluator mean IoU: `0.395946`
- mean runtime: about `38-39 ms/image`

See:
- `solutions/paper_anomaly_baseline/metrics.json`
- `solutions/paper_anomaly_baseline/artifacts/model/training_summary.json`
- `solutions/paper_anomaly_baseline/artifacts/model/test_eval.json`

## 4) Dataset Structure

Raw dataset path:
- `data/cable/`

Observed folders:
- `data/cable/train/good`
- `data/cable/test/good`
- `data/cable/test/<defect_class>`
- `data/cable/ground_truth/<defect_class>/*_mask.png`

Defect classes:
- `bent_wire`
- `cable_swap`
- `combined`
- `cut_inner_insulation`
- `cut_outer_insulation`
- `missing_cable`
- `missing_wire`
- `poke_insulation`

Raw counts:
- `train/good = 224`
- `test/good = 46`
- defects in `test/* = 84`

Deterministic project split currently used (`seed=20260330`):
- train: `189` (`good` only)
- val: `74` (`40 good + 34 defective`)
- test: `91` (`41 good + 50 defective`)

Split manifest:
- `solutions/paper_anomaly_baseline/artifacts/split.json`

## 5) Assignment/Methodology Constraints

Keep alignment with MVTec AD / Bergmann et al.:
- anomaly-model fitting uses only normal/defect-free samples
- defective masks are used only for validation/calibration/evaluation
- no mask memorization from visible test data
- no supervised mask-training as primary method inside this anomaly track

## 6) Cloud Submission Format

Final ZIP should contain:
- `model.py`
- `requirements.txt`
- model artifact file(s), currently `model_artifact.npz`

Current ZIP path:
- `solutions/paper_anomaly_baseline/artifacts/paper_anomaly_baseline_submission.zip`

Server environment assumptions:
- Python `3.12`
- no training during inference
- inference should be robust and reasonably fast

## 7) Commands (Current Workflow)

Prepare split:
```bash
./.venv/bin/python solutions/paper_anomaly_baseline/prepare_data.py
```

Train/calibrate:
```bash
./.venv/bin/python solutions/paper_anomaly_baseline/train.py --threshold-steps 16 --stage1-top-k 6 --min-area-candidates 0,32,64,128 --open-kernel-candidates 1,3,5 --close-kernel-candidates 1,3
```

Evaluate split-test:
```bash
./.venv/bin/python solutions/paper_anomaly_baseline/evaluate.py --split test --output-path solutions/paper_anomaly_baseline/artifacts/model/test_eval.json
```

Evaluate with repo evaluator:
```bash
./.venv/bin/python scripts/evaluate_submission.py --model-path solutions/paper_anomaly_baseline/sample_submission/model.py
```

Export submission ZIP:
```bash
./.venv/bin/python solutions/paper_anomaly_baseline/export_submission.py
```

## 8) Explicit Permission: Propose New Approaches

You are **explicitly encouraged** to propose and implement better approaches than the current one, as long as interface and project constraints are preserved.

You may propose:
- stronger anomaly backbones (PaDiM/PatchCore with deep pretrained features)
- better anomaly-map calibration strategies
- smarter postprocessing (class-agnostic but validation-driven)
- region/ROI normalization for cable geometry
- multi-scale or ensemble anomaly scoring
- architecture changes that improve generalization

If you propose a new approach, include:
- rationale
- expected tradeoffs (accuracy vs speed/size/complexity)
- exact implementation plan
- validation protocol

## 9) Mission for the Next Agent

Primary target:
- improve over current full local evaluator mean IoU `0.395946`

Hard requirements:
- keep work isolated in `solutions/paper_anomaly_baseline/`
- preserve `predict(image)` API and output contract
- keep one-class fitting (good-only) for anomaly model training
- do not leak hidden/visible test masks into fitting

Required deliverables:
- updated code
- updated metrics artifacts
- updated submission ZIP
- concise report of changes and results

## 10) Validation Checklist (Must Pass)

- `py_compile` on modified python files
- `predict(image)` returns:
  - correct shape `(H, W)`
  - dtype `uint8`
  - values only `{0,255}`
- `scripts/evaluate_submission.py` run succeeds
- `export_submission.py` produces a valid ZIP
- README/metrics updated with truthful numbers

## 11) Suggested Improvement Direction (Technical)

Recommended first step:
- implement a deep-feature anomaly backend (PaDiM/PatchCore style) in parallel to the current handcrafted backend
- compare both on the same split and selection protocol
- keep the better backend for export

Do not claim hidden-set performance. Report only local measured metrics.
