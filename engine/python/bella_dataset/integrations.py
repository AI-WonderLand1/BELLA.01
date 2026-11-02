from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

DEFAULT_HF_DATASET_REPO = "ai-wonderland-innovation/Bella-Training-Datasets"
DATASET_SLUG = re.compile(r"^[A-Za-z0-9_-]+/[A-Za-z0-9_-]+$")


def download_kaggle_dataset(dataset: str, destination: Path) -> dict[str, Any]:
    """Download and extract a Kaggle dataset into destination."""
    if not DATASET_SLUG.fullmatch(dataset):
        raise ValueError("Kaggle dataset must use owner/dataset-slug format")
    if not os.environ.get("KAGGLE_USERNAME") or not os.environ.get("KAGGLE_KEY"):
        raise RuntimeError("Set KAGGLE_USERNAME and KAGGLE_KEY in the environment")
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:
        raise RuntimeError("Kaggle integration requires: pip install -r python/requirements.txt") from exc

    destination.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(dataset, path=str(destination), unzip=True, quiet=False)
    files = sum(1 for item in destination.rglob("*") if item.is_file())
    return {"provider": "kaggle", "dataset": dataset, "destination": str(destination), "files_downloaded": files}


def mirror_to_huggingface(source: Path, repo_id: str | None = None) -> dict[str, Any]:
    """Upload prepared output to the configured Hugging Face dataset repo."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("Set HF_TOKEN in the environment; never commit it to the repository")
    target = repo_id or os.environ.get("HF_DATASET_REPO", DEFAULT_HF_DATASET_REPO)
    if not DATASET_SLUG.fullmatch(target):
        raise ValueError("HF_DATASET_REPO must use owner/repository format")
    if not source.is_dir():
        raise FileNotFoundError(f"Prepared dataset directory does not exist: {source}")
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("Hugging Face integration requires: pip install -r python/requirements.txt") from exc

    api = HfApi(token=token)
    if not api.repo_exists(repo_id=target, repo_type="dataset"):
        raise RuntimeError(f"Hugging Face dataset repository does not exist or is not accessible: {target}")
    result = api.upload_folder(
        repo_id=target,
        repo_type="dataset",
        folder_path=str(source),
        commit_message="Mirror prepared dataset from Kaggle-backed BELLA.01 pipeline",
        ignore_patterns=["*.tmp", ".DS_Store"],
    )
    return {"provider": "huggingface", "repo_id": target, "commit_url": str(result), "source": str(source)}
