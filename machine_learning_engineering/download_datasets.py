"""Download Kaggle datasets and competitions for MLE-bench tasks."""

import os
import subprocess
import sys
import yaml
from pathlib import Path
from dotenv import load_dotenv

try:
    from kaggle.api.kaggle_api_extended import KaggleApi
except ImportError:
    KaggleApi = None


def load_env():
    """Load environment variables from .env file."""
    env_path = Path(__file__).parent.parent / ".env"
    if not env_path.exists():
        raise FileNotFoundError(f".env file not found at {env_path}")
    load_dotenv(env_path)
    api_key = os.getenv("KAGGLE_API_KEY")
    if not api_key:
        raise ValueError("KAGGLE_API_KEY not found in .env file")
    return api_key


def load_datasets_config():
    """Load datasets configuration from YAML."""
    config_path = Path(__file__).parent / "datasets.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def setup_kaggle_api(api_key):
    """Set up Kaggle API credentials."""
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(exist_ok=True)

    kaggle_json = kaggle_dir / "kaggle.json"
    kaggle_json.write_text(f'{{"username":"","key":"{api_key}"}}')
    kaggle_json.chmod(0o600)
    print(f"[INFO] Kaggle API credentials set up at {kaggle_json}")


def get_kaggle_api():
    """Initialize and return Kaggle API instance."""
    if not KaggleApi:
        return None
    try:
        api = KaggleApi()
        api.authenticate()
        return api
    except Exception as e:
        print(f"[WARNING] Could not authenticate Kaggle API: {str(e)}", file=sys.stderr)
        return None


def fetch_dataset_metadata(api, kaggle_id):
    """Fetch dataset metadata from Kaggle API."""
    if not api:
        return {}
    try:
        dataset_info = api.dataset_metadata(kaggle_id)
        return {
            "title": dataset_info.get("title", ""),
            "subtitle": dataset_info.get("subtitle", ""),
            "description": dataset_info.get("description", ""),
            "files": dataset_info.get("files", []),
        }
    except Exception as e:
        print(f"[WARNING] Could not fetch dataset metadata for {kaggle_id}: {str(e)}", file=sys.stderr)
        return {}


def fetch_competition_metadata(api, kaggle_id):
    """Fetch competition metadata from Kaggle API."""
    if not api:
        return {}
    try:
        competition_info = api.competition_info(kaggle_id)
        return {
            "title": competition_info.get("title", ""),
            "description": competition_info.get("description", ""),
            "enabledDate": competition_info.get("enabledDate", ""),
            "evaluationMetric": competition_info.get("evaluationMetric", ""),
        }
    except Exception as e:
        print(f"[WARNING] Could not fetch competition metadata for {kaggle_id}: {str(e)}", file=sys.stderr)
        return {}


def create_task_description(task_dir, task_name, task_config, metadata=None):
    """Create a task description file."""
    description_file = task_dir / "task_description.txt"

    if metadata is None:
        metadata = {}

    # Build description content with fetched metadata
    title = metadata.get("title") or task_config.get("description", f"ML task for {task_name}")
    evaluation_metric = metadata.get("evaluationMetric", "Depends on the task (see submission requirements on Kaggle)")
    full_description = metadata.get("description", "")

    # Extract files list if available
    files_info = ""
    if metadata.get("files"):
        files_info = "\n".join([f"  - {f.get('name', '')}" for f in metadata.get("files", [])[:10]])

    description_content = f"""# Task

{title}

# Task Type

{task_config.get('task_type', 'Tabular')}

# Task Name

{task_config.get('task_name', task_name)}

# Metric

{evaluation_metric}

# Description

{full_description if full_description else 'See Kaggle page for full details: https://www.kaggle.com'}

# Dataset Files

{files_info if files_info else 'See the downloaded files in this directory.'}
"""

    description_file.write_text(description_content)
    print(f"[INFO] Created task description: {description_file}")


def download_dataset(kaggle_id, task_name, task_config, api=None):
    """Download a Kaggle dataset."""
    task_dir = Path(__file__).parent / "tasks" / task_name
    task_dir.mkdir(parents=True, exist_ok=True)

    print(f"[DOWNLOAD] Dataset: {kaggle_id} -> {task_dir}")
    try:
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", kaggle_id, "-p", str(task_dir), "--unzip"],
            check=True,
            capture_output=True,
        )
        print(f"[SUCCESS] Downloaded dataset: {task_name}")

        # Fetch metadata from Kaggle API
        metadata = {}
        if api:
            print(f"[INFO] Fetching metadata for dataset: {kaggle_id}")
            metadata = fetch_dataset_metadata(api, kaggle_id)

        create_task_description(task_dir, task_name, task_config, metadata)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to download dataset {task_name}: {e.stderr.decode()}", file=sys.stderr)
        raise


def download_competition(kaggle_id, task_name, task_config, api=None):
    """Download a Kaggle competition."""
    task_dir = Path(__file__).parent / "tasks" / task_name
    task_dir.mkdir(parents=True, exist_ok=True)

    print(f"[DOWNLOAD] Competition: {kaggle_id} -> {task_dir}")
    try:
        result = subprocess.run(
            ["kaggle", "competitions", "download", "-c", kaggle_id, "-p", str(task_dir)],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print(f"[SUCCESS] Downloaded competition: {task_name}")

            # Fetch metadata from Kaggle API
            metadata = {}
            if api:
                print(f"[INFO] Fetching metadata for competition: {kaggle_id}")
                metadata = fetch_competition_metadata(api, kaggle_id)

            create_task_description(task_dir, task_name, task_config, metadata)
        else:
            print(f"[WARNING] Competition download may have failed or partially completed: {task_name}", file=sys.stderr)
            if result.stderr:
                print(f"  Error details: {result.stderr}", file=sys.stderr)
            print(f"  Note: You may need to accept competition terms at https://www.kaggle.com/c/{kaggle_id}", file=sys.stderr)
    except Exception as e:
        print(f"[WARNING] Exception downloading competition {task_name}: {str(e)}", file=sys.stderr)
        print(f"  Note: You may need to accept competition terms at https://www.kaggle.com/c/{kaggle_id}", file=sys.stderr)


def main():
    """Main execution."""
    print("=" * 60)
    print("Kaggle Dataset & Competition Downloader")
    print("=" * 60)

    api_key = load_env()
    setup_kaggle_api(api_key)

    # Initialize Kaggle API for metadata fetching
    api = get_kaggle_api()

    config = load_datasets_config()

    datasets = config.get("datasets", {})
    competitions = config.get("competitions", {})

    print(f"\n[INFO] Found {len(datasets)} dataset(s) and {len(competitions)} competition(s)")

    for task_name, task_config in datasets.items():
        kaggle_id = task_config["kaggle_id"]
        try:
            download_dataset(kaggle_id, task_name, task_config, api)
        except Exception as e:
            print(f"[ERROR] Failed to download dataset {task_name}: {str(e)}", file=sys.stderr)
            print(f"Continuing to next task...", file=sys.stderr)

    for task_name, task_config in competitions.items():
        kaggle_id = task_config["kaggle_id"]
        download_competition(kaggle_id, task_name, task_config, api)

    print("\n" + "=" * 60)
    print("Download Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
