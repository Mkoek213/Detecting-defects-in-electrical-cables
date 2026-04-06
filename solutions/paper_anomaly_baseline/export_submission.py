from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = THIS_DIR / "artifacts" / "model" / "model_artifact.npz"
DEFAULT_DEEP_MODEL_PATH = THIS_DIR / "artifacts" / "deep_ae" / "model_artifact.pt"
DEFAULT_PADIM_MODEL_PATH = THIS_DIR / "artifacts" / "padim" / "model_artifact_padim.pt"
DEFAULT_SAMPLE_DIR = THIS_DIR / "sample_submission"
DEFAULT_BUNDLE_DIR = THIS_DIR / "artifacts" / "submission_bundle"
DEFAULT_ZIP_PATH = THIS_DIR / "artifacts" / "paper_anomaly_baseline_submission.zip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export isolated submission bundle for paper anomaly baseline.")
    parser.add_argument("--backend", type=str, default="handcrafted", choices=["handcrafted", "deep_ae", "padim", "patchcore"])
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--sample-dir", type=Path, default=DEFAULT_SAMPLE_DIR)
    parser.add_argument("--bundle-dir", type=Path, default=DEFAULT_BUNDLE_DIR)
    parser.add_argument("--zip-path", type=Path, default=DEFAULT_ZIP_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample_dir = args.sample_dir.resolve()
    bundle_dir = args.bundle_dir.resolve()
    zip_path = args.zip_path.resolve()
    if args.model_path is None:
        default_model_map = {
            "handcrafted": DEFAULT_MODEL_PATH,
            "deep_ae": DEFAULT_DEEP_MODEL_PATH,
            "padim": DEFAULT_PADIM_MODEL_PATH,
            "patchcore": THIS_DIR / "artifacts" / "patchcore" / "model_artifact_patchcore.pt",
        }
        model_path = default_model_map[args.backend].resolve()
    else:
        model_path = args.model_path.resolve()

    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")

    backend_files = {
        "handcrafted": {
            "model": "model.py",
            "requirements": "requirements.txt",
            "artifact": "model_artifact.npz",
            "extras": [],
        },
        "deep_ae": {
            "model": "model_deep_ae.py",
            "requirements": "requirements_deep_ae.txt",
            "artifact": "model_artifact.pt",
            "extras": [],
        },
        "padim": {
            "model": "model_padim.py",
            "requirements": "requirements_padim.txt",
            "artifact": "model_artifact_padim.pt",
            "extras": ["padim_runtime.py"],
        },
        "patchcore": {
            "model": "model_patchcore.py",
            "requirements": "requirements_patchcore.txt",
            "artifact": "model_artifact_patchcore.pt",
            "extras": ["patchcore_runtime.py"],
        },
    }
    backend_config = backend_files[args.backend]

    model_file_name = backend_config["model"]
    requirements_file_name = backend_config["requirements"]
    artifact_file_name = backend_config["artifact"]

    model_py = sample_dir / model_file_name
    requirements_txt = sample_dir / requirements_file_name
    if not model_py.exists():
        raise FileNotFoundError(f"Missing submission model file: {model_py}")
    if not requirements_txt.exists():
        raise FileNotFoundError(f"Missing submission requirements file: {requirements_txt}")

    sample_dir.mkdir(parents=True, exist_ok=True)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    for existing_path in bundle_dir.iterdir():
        if existing_path.is_file():
            existing_path.unlink()

    target_artifact = sample_dir / artifact_file_name
    shutil.copy2(model_path, target_artifact)
    print(f"Copied trained artifact to sample submission: {target_artifact}")

    shutil.copy2(model_py, bundle_dir / "model.py")
    shutil.copy2(requirements_txt, bundle_dir / "requirements.txt")
    shutil.copy2(target_artifact, bundle_dir / artifact_file_name)
    for extra_file_name in backend_config["extras"]:
        shutil.copy2(sample_dir / extra_file_name, bundle_dir / extra_file_name)

    manifest = {
        "files": ["model.py", "requirements.txt", artifact_file_name, *backend_config["extras"]],
        "description": f"Paper-aligned {args.backend} submission bundle.",
    }
    (bundle_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_name in manifest["files"]:
            archive.write(bundle_dir / file_name, arcname=file_name)

    print(f"Prepared submission bundle: {bundle_dir}")
    print(f"Created zip archive: {zip_path}")


if __name__ == "__main__":
    main()
