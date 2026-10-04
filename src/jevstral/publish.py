"""Upload finished stages to a private Hugging Face model repository."""

from pathlib import Path

from huggingface_hub import HfApi

REPO_NAME = "jevstral-8b"
UPLOADED = ["final/**", "metrics.json", "log.jsonl", "calibration.json"]


def ensure_repo() -> str:
    """Create the private repository if it does not exist. Return its ID. This fails if the token cannot write."""
    api = HfApi()
    repo_id = f"{api.whoami()['name']}/{REPO_NAME}"
    api.create_repo(repo_id, repo_type="model", private=True, exist_ok=True)
    return repo_id


def token_report() -> str:
    auth = HfApi().whoami()["auth"]["accessToken"]
    return f"token '{auth.get('displayName')}', role {auth.get('role')}"


def publish_stage(stage: int, runs_dir: Path) -> str:
    """Upload main/stage{N} (the final checkpoint, metrics and log) to the folder stage{N} of the repository."""
    folder = runs_dir / "main" / f"stage{stage}"
    if not (folder / "final").exists():
        raise FileNotFoundError(f"{folder / 'final'} does not exist. Finish stage {stage} first.")
    repo_id = ensure_repo()
    commit = HfApi().upload_folder(
        repo_id=repo_id,
        folder_path=str(folder),
        path_in_repo=f"stage{stage}",
        allow_patterns=UPLOADED,
        commit_message=f"Upload stage {stage}",
    )
    return commit.commit_url
