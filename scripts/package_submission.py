from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUBMISSION_ROOT = ROOT / "submissions" / "cable_submission"
ZIP_PATH = ROOT / "submissions" / "cable_submission.zip"
LEGACY_MODEL_ROOT = ROOT / "artifacts" / "custom_template"
SOLUTION_SAMPLE_ROOT = ROOT / "solution" / "sample_submission"
ROOT_SAMPLE_ROOT = ROOT / "sample_submission"
PATCHCORE_FILES = ["coreset.npy", "projection_components.npy", "threshold.npy"]


def _find_first(pattern: str) -> Path:
    matches = sorted(LEGACY_MODEL_ROOT.rglob(pattern))
    if not matches:
        raise FileNotFoundError(f"Could not find {pattern} under {LEGACY_MODEL_ROOT}")
    return matches[0]


def _resolve_sample_root() -> Path:
    if (SOLUTION_SAMPLE_ROOT / "model.py").exists():
        return SOLUTION_SAMPLE_ROOT
    if (ROOT_SAMPLE_ROOT / "model.py").exists():
        return ROOT_SAMPLE_ROOT
    raise FileNotFoundError("Could not find a sample_submission/model.py to package.")


def _resolve_requirements(sample_root: Path) -> Path:
    local_requirements = sample_root / "requirements.txt"
    if local_requirements.exists():
        return local_requirements

    solution_requirements = ROOT / "solution" / "requirements.txt"
    if sample_root == SOLUTION_SAMPLE_ROOT and solution_requirements.exists():
        return solution_requirements

    root_requirements = ROOT_SAMPLE_ROOT / "requirements.txt"
    if root_requirements.exists():
        return root_requirements

    raise FileNotFoundError("Could not find submission requirements.txt")


def _resolve_model_files(sample_root: Path) -> list[str]:
    if all((sample_root / name).exists() for name in PATCHCORE_FILES):
        return ["model.py", "requirements.txt", *PATCHCORE_FILES]

    if (sample_root / "model.npz").exists():
        return ["model.py", "requirements.txt", "model.npz"]

    if LEGACY_MODEL_ROOT.exists():
        model_npz = _find_first("model.npz")
        shutil.copy2(model_npz, sample_root / "model.npz")
        return ["model.py", "requirements.txt", "model.npz"]

    raise FileNotFoundError(
        "Could not find PatchCore artifacts (coreset/projection/threshold) "
        "or legacy model.npz for packaging."
    )


def main() -> None:
    sample_root = _resolve_sample_root()
    runtime_requirements = _resolve_requirements(sample_root)
    model_files = _resolve_model_files(sample_root)

    sample_root.mkdir(parents=True, exist_ok=True)
    SUBMISSION_ROOT.mkdir(parents=True, exist_ok=True)
    ZIP_PATH.parent.mkdir(parents=True, exist_ok=True)

    for stale_path in SUBMISSION_ROOT.glob("*"):
        if stale_path.is_file():
            stale_path.unlink()

    shutil.copy2(sample_root / "model.py", SUBMISSION_ROOT / "model.py")
    shutil.copy2(runtime_requirements, SUBMISSION_ROOT / "requirements.txt")
    for artifact_name in model_files:
        if artifact_name in {"model.py", "requirements.txt"}:
            continue
        shutil.copy2(sample_root / artifact_name, SUBMISSION_ROOT / artifact_name)

    manifest = {
        "files": model_files
    }
    (SUBMISSION_ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_name in manifest["files"]:
            archive.write(SUBMISSION_ROOT / file_name, arcname=file_name)

    print(f"Prepared sample submission in: {sample_root}")
    print(f"Prepared submission folder in: {SUBMISSION_ROOT}")
    print(f"Created zip archive: {ZIP_PATH}")


if __name__ == "__main__":
    main()
