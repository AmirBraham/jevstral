"""Modal entry points. All data, model and GPU work runs here, not on the local computer.

modal run modal_app.py::prepare_data
modal run modal_app.py::inspect_data
modal run modal_app.py::train --stage 1 --smoke
modal run --detach modal_app.py::train --stage 1
modal run --detach modal_app.py::calibrate
modal run modal_app.py::check_hub
modal run modal_app.py::publish --stage 1
modal run modal_app.py::track_finished_stage --stage 1
modal run modal_app.py::evaluate_stage --stage 2
"""

import subprocess
from pathlib import Path

import modal

app = modal.App("jevstral")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(groups=["train"])
    .env({"HF_HOME": "/cache/hf", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_python_source("jevstral")
)

hf_cache = modal.Volume.from_name("jevstral-hf-cache", create_if_missing=True)
data = modal.Volume.from_name("jevstral-data", create_if_missing=True)
runs = modal.Volume.from_name("jevstral-runs", create_if_missing=True)

DATA_DIR = Path("/data")
RUNS_DIR = Path("/runs")
VOLUMES = {"/cache": hf_cache, str(DATA_DIR): data, str(RUNS_DIR): runs}
SECRETS = [modal.Secret.from_name("huggingface")]
WANDB = modal.Secret.from_name("wandb")
HOUR = 3600


def persist() -> None:
    hf_cache.commit()
    runs.commit()


@app.function(image=image, volumes={str(DATA_DIR): data}, timeout=HOUR)
def prepare_data() -> None:
    from jevstral import suites

    for path in suites.fetch_all(DATA_DIR):
        print("verified", path)
    data.commit()


@app.function(image=image, volumes=VOLUMES, secrets=SECRETS, memory=32768, timeout=HOUR)
def inspect_data() -> None:
    """CPU only. Measure the delimiter embeddings and encode all stages without a model."""
    from jevstral.encode import Encoder
    from jevstral.model import delimiter_report, load_tokenizer
    from jevstral.stages import STAGES
    from jevstral.train import encoding_report

    print(delimiter_report(Encoder(load_tokenizer()).delimiter_ids))
    hf_cache.commit()
    print(encoding_report(list(STAGES.values()), DATA_DIR))


@app.function(
    image=image,
    gpu="H100",
    volumes=VOLUMES,
    secrets=[*SECRETS, WANDB],
    memory=65536,
    # No `retries`: Modal reruns preempted jobs itself, and a retry after an exception in our code fails again.
    timeout=8 * HOUR,
)
def train_stage(stage: int, smoke: bool, git_commit: str) -> None:
    from jevstral.stages import STAGES
    from jevstral.train import run_stage

    run_stage(STAGES[stage], DATA_DIR, RUNS_DIR, git_commit, smoke=smoke, persist=persist)


@app.function(image=image, gpu="H100", volumes=VOLUMES, secrets=SECRETS, memory=65536, timeout=2 * HOUR)
def evaluate_stage(stage: int) -> None:
    """Measure a finished stage on all four development files, without training."""
    import json

    from jevstral.stages import STAGES
    from jevstral.train import evaluate_checkpoint

    metrics = evaluate_checkpoint(stage, STAGES[4].dev_files, DATA_DIR, RUNS_DIR)
    runs.commit()
    print(json.dumps(metrics, indent=2))


@app.function(image=image, gpu="H100", volumes=VOLUMES, secrets=SECRETS, memory=65536, timeout=2 * HOUR)
def calibrate_stage4() -> None:
    from jevstral.calibrate import run_calibration

    run_calibration(DATA_DIR, RUNS_DIR, persist=persist)


@app.function(image=image, secrets=SECRETS, timeout=600)
def check_hub() -> None:
    """Verify that the Hugging Face token can write: create the private repository if it does not exist."""
    from jevstral.publish import ensure_repo, token_report

    print(token_report())
    print("repository ready:", ensure_repo())


@app.function(image=image, volumes={str(RUNS_DIR): runs}, secrets=SECRETS, timeout=HOUR)
def upload_stage(stage: int) -> None:
    from jevstral.publish import publish_stage

    print("uploaded:", publish_stage(stage, RUNS_DIR))


@app.function(image=image, volumes={str(RUNS_DIR): runs}, secrets=[WANDB], timeout=600)
def track_finished_stage(stage: int) -> None:
    """Send the log and metrics of a stage that trained without W&B (stage 1) to a W&B run."""
    from jevstral.tracking import upload_finished_stage

    print("run:", upload_finished_stage(RUNS_DIR / "main" / f"stage{stage}"))


def local_git_commit() -> str:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True).stdout.strip()
    return commit + ("-dirty" if dirty else "")


@app.local_entrypoint()
def train(stage: int, smoke: bool = False) -> None:
    train_stage.remote(stage, smoke, local_git_commit())


@app.local_entrypoint()
def calibrate() -> None:
    calibrate_stage4.remote()


@app.local_entrypoint()
def publish(stage: int) -> None:
    upload_stage.remote(stage)
