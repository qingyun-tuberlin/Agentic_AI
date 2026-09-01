"""Distribute the large tasks/ data folder via a Hugging Face dataset repo using per-task archives.

The per-task datasets (train.csv/test.csv eval splits + raw/ snapshots) are too
large to keep in git, so each task folder inside ``tasks/`` is packed into its own
``{task_name}.tar.gz`` and stored in a Hugging Face *dataset* repo.

Usage
-----
Download all tasks::

    python machine_learning_engineering/download_tasks.py

Download a specific task::

    python machine_learning_engineering/download_tasks.py --task credit-card-fraud

Upload all tasks (maintainer only; needs write access)::

    python machine_learning_engineering/download_tasks.py --upload

Upload a specific task (maintainer only; needs write access)::

    python machine_learning_engineering/download_tasks.py --upload --task credit-card-fraud

Configuration
-------------
MLE_TASKS_HF_REPO   HF dataset repo id, e.g. "your-username/mle-star-tasks".
                    Falls back to DEFAULT_REPO_ID below.
HF_TOKEN            Auth token for uploading (or run `huggingface-cli login`).
"""
from __future__ import annotations

import argparse
import os
import tarfile
from pathlib import Path
import yaml

from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download

# Load repo-root .env so MLE_TASKS_HF_REPO / HF_TOKEN can be set there
# (same .env used by download_datasets.py).
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# Hardcoded so downloads work out of the box; override with MLE_TASKS_HF_REPO.
DEFAULT_REPO_ID = "tschuster/mle-star-tasks"
TASKS_DIR = Path(__file__).resolve().parent / "tasks"


def repo_id() -> str:
    return os.environ.get("MLE_TASKS_HF_REPO", "").strip() or DEFAULT_REPO_ID


def get_all_configured_tasks() -> list[str]:
    """Get all task names defined in datasets.yaml."""
    config_path = Path(__file__).resolve().parent / "datasets.yaml"
    if not config_path.exists():
        return []
    with open(config_path) as f:
        config = yaml.safe_load(f)
    datasets = list(config.get("datasets", {}).keys())
    competitions = list(config.get("competitions", {}).keys())
    return sorted(list(set(datasets + competitions)))


def upload(task_name: str | None = None) -> None:
    """Pack task(s) into tarball(s) and push to the HF dataset repo."""
    rid = repo_id()
    api = HfApi()
    api.create_repo(rid, repo_type="dataset", exist_ok=True)

    if task_name:
        tasks = [task_name]
    else:
        try:
            tasks = get_all_configured_tasks()
        except Exception as e:
            print(f"[WARNING] Could not read datasets.yaml: {e}. Falling back to tasks directory listing.")
            if not TASKS_DIR.exists():
                print(f"[ERROR] {TASKS_DIR} does not exist.")
                return
            # Exclude corrupted/proxy variants and dotfiles
            tasks = [
                d.name for d in TASKS_DIR.iterdir()
                if d.is_dir() and not d.name.startswith(".") and not d.name.endswith("_c") and not d.name.endswith("_proxy")
            ]

    if not tasks:
        print("No tasks found to upload.")
        return

    for task in tasks:
        task_path = TASKS_DIR / task
        if not task_path.exists():
            print(f"[ERROR] Task directory {task_path} does not exist.")
            continue

        archive_name = f"{task}.tar.gz"
        archive_path = TASKS_DIR.parent / archive_name

        print(f"[PACK] {task_path} -> {archive_path}")
        with tarfile.open(archive_path, "w:gz") as tar:
            # Archive it such that it extracts back into tasks/task_name/
            tar.add(task_path, arcname=f"tasks/{task}")

        size_mb = archive_path.stat().st_size / 1e6
        print(f"[UPLOAD] {archive_name} ({size_mb:.2f} MB) -> dataset {rid}")
        api.upload_file(
            path_or_fileobj=str(archive_path),
            path_in_repo=archive_name,
            repo_id=rid,
            repo_type="dataset",
        )
        archive_path.unlink()
        print(f"[DONE] Uploaded {task}")


def download(task_name: str | None = None, force: bool = False) -> None:
    """Fetch task tarball(s) from the HF dataset repo and extract them.

    If a task directory already contains data, abort for that task unless ``force`` is set.
    """
    rid = repo_id()

    if task_name:
        tasks = [task_name]
    else:
        try:
            tasks = get_all_configured_tasks()
        except Exception as e:
            print(f"[WARNING] Could not read datasets.yaml: {e}. Falling back to Hugging Face file list.")
            # Fallback: List files in HF repo and find all *.tar.gz
            api = HfApi()
            files = api.list_repo_files(rid, repo_type="dataset")
            tasks = [f[:-7] for f in files if f.endswith(".tar.gz")]

    if not tasks:
        print("No tasks to download.")
        return

    dest = TASKS_DIR.parent
    for task in tasks:
        task_dir = TASKS_DIR / task
        if task_dir.exists() and any(task_dir.iterdir()) and not force:
            print(f"[SKIP] {task} already contains data. Use --force to overwrite.")
            continue

        archive_name = f"{task}.tar.gz"
        print(f"[FETCH] {archive_name} from dataset {rid}")
        try:
            local = hf_hub_download(repo_id=rid, filename=archive_name, repo_type="dataset")
            print(f"[EXTRACT] {archive_name} -> {dest}")
            with tarfile.open(local, "r:gz") as tar:
                tar.extractall(dest, filter="data")
            print(f"[DONE] {task} ready")
        except Exception as e:
            print(f"[ERROR] Failed to download/extract {task}: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download or upload individual task dataset tarballs via Hugging Face."
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Pack tasks and push to the HF dataset repo (maintainer only).",
    )
    parser.add_argument(
        "--task",
        type=str,
        help="Operate on a specific task only (e.g. credit-card-fraud).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite already-populated task directories (otherwise download skips them).",
    )
    args = parser.parse_args()

    if args.upload:
        upload(task_name=args.task)
    else:
        download(task_name=args.task, force=args.force)


if __name__ == "__main__":
    main()

