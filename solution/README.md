## Setup
```bash
pip install -r requirements.txt
```

## Fit (run once — takes ~5 min on CPU, ~1 min on GPU)
```bash
python fit.py --data_dir data/cable --output_dir sample_submission/
```
This produces `coreset.npy` and `projection_components.npy` inside `sample_submission/`.

## Evaluate + calibrate threshold
```bash
python evaluate.py --data_dir data/cable
```
This produces `threshold.npy` inside `sample_submission/` and prints the validation report.

## Inference (example)
```python
from PIL import Image
import numpy as np
from sample_submission.model import predict

img = np.array(Image.open("data/cable/test/bent_wire/000.png").convert("RGB"))
mask = predict(img)   # (H, W) uint8
```

## Notes
- The scripts use a deterministic mixed split over all good + defect samples with seed 42.
- If you run from repository root, use:
  - `python solution/fit.py --data_dir data/cable --output_dir solution/sample_submission/`
  - `python solution/evaluate.py --data_dir data/cable --output_dir solution/sample_submission/`