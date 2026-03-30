from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "artifacts" / "custom_template"
SAMPLE_ROOT = ROOT / "sample_submission"
SUBMISSION_ROOT = ROOT / "submissions" / "cable_submission"
ZIP_PATH = ROOT / "submissions" / "cable_submission.zip"
RUNTIME_REQUIREMENTS = SAMPLE_ROOT / "requirements.txt"


def _find_first(pattern: str) -> Path:
    matches = sorted(MODEL_ROOT.rglob(pattern))
    if not matches:
        raise FileNotFoundError(f"Could not find {pattern} under {MODEL_ROOT}")
    return matches[0]


def main() -> None:
    model_npz = _find_first("model.npz")
    if not RUNTIME_REQUIREMENTS.exists():
        raise FileNotFoundError(f"Could not find submission requirements file: {RUNTIME_REQUIREMENTS}")

    SAMPLE_ROOT.mkdir(parents=True, exist_ok=True)
    SUBMISSION_ROOT.mkdir(parents=True, exist_ok=True)
    ZIP_PATH.parent.mkdir(parents=True, exist_ok=True)

    for stale_path in (
        SAMPLE_ROOT / "model.npz",
        SUBMISSION_ROOT / "model.npz",
    ):
        stale_path.unlink(missing_ok=True)

    shutil.copy2(model_npz, SAMPLE_ROOT / "model.npz")

    shutil.copy2(SAMPLE_ROOT / "model.py", SUBMISSION_ROOT / "model.py")
    shutil.copy2(RUNTIME_REQUIREMENTS, SUBMISSION_ROOT / "requirements.txt")
    shutil.copy2(SAMPLE_ROOT / "model.npz", SUBMISSION_ROOT / "model.npz")

    manifest = {
        "files": [
            "model.py",
            "requirements.txt",
            "model.npz",
        ]
    }
    (SUBMISSION_ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_name in manifest["files"]:
            archive.write(SUBMISSION_ROOT / file_name, arcname=file_name)

    print(f"Prepared sample submission in: {SAMPLE_ROOT}")
    print(f"Prepared submission folder in: {SUBMISSION_ROOT}")
    print(f"Created zip archive: {ZIP_PATH}")


if __name__ == "__main__":
    main()
