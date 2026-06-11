"""Repo-relative path constants. No directories are created at import time."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
MIMII_DIR = DATA_DIR / "mimii"
CHECKPOINT_DIR = DATA_DIR / "checkpoints"
CACHE_DIR = DATA_DIR / "cache"
ARTIFACT_DIR = DATA_DIR / "artifacts"


def ensure_dirs() -> None:
    """Create all data directories (parents=True, exist_ok=True)."""
    for directory in (
        DATA_DIR,
        MIMII_DIR,
        CHECKPOINT_DIR,
        CACHE_DIR,
        ARTIFACT_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
