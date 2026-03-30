from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = THIS_DIR / "artifacts" / "model" / "model_artifact.npz"
DEFAULT_SAMPLE_DIR = THIS_DIR / "sample_submission"
DEFAULT_BUNDLE_DIR = THIS_DIR / "artifacts" / "submission_bundle"
DEFAULT_ZIP_PATH = THIS_DIR / "artifacts" / "paper_anomaly_baseline_submission.zip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export isolated submission bundle for paper anomaly baseline.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--sample-dir", type=Path, default=DEFAULT_SAMPLE_DIR)
    parser.add_argument("--bundle-dir", type=Path, default=DEFAULT_BUNDLE_DIR)
    parser.add_argument("--zip-path", type=Path, default=DEFAULT_ZIP_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = args.model_path.resolve()
    sample_dir = args.sample_dir.resolve()
    bundle_dir = args.bundle_dir.resolve()
    zip_path = args.zip_path.resolve()

    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")

    model_py = sample_dir / "model.py"
    requirements_txt = sample_dir / "requirements.txt"
    if not model_py.exists():
        raise FileNotFoundError(f"Missing submission model file: {model_py}")
    if not requirements_txt.exists():
        raise FileNotFoundError(f"Missing submission requirements file: {requirements_txt}")

    sample_dir.mkdir(parents=True, exist_ok=True)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    target_model_npz = sample_dir / "model_artifact.npz"
    shutil.copy2(model_path, target_model_npz)
    print(f"Copied trained artifact to sample submission: {target_model_npz}")

    for file_name in ("model.py", "requirements.txt", "model_artifact.npz"):
        source_path = sample_dir / file_name
        target_path = bundle_dir / file_name
        shutil.copy2(source_path, target_path)

    manifest = {
        "files": ["model.py", "requirements.txt", "model_artifact.npz"],
        "description": "Paper-aligned one-class anomaly baseline submission bundle.",
    }
    (bundle_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_name in manifest["files"]:
            archive.write(bundle_dir / file_name, arcname=file_name)

    print(f"Prepared submission bundle: {bundle_dir}")
    print(f"Created zip archive: {zip_path}")


if __name__ == "__main__":
    main()
