"""Weights & Biases tracking. Tracking is off when WANDB_API_KEY is not set."""

import json
import os
from pathlib import Path

PROJECT = "jevstral"


class Tracker:
    def __init__(self, run=None):
        self.run = run

    def log(self, entry: dict, step: int) -> None:
        if self.run is not None:
            self.run.log(entry, step=step)

    def finish(self, metrics: dict) -> None:
        if self.run is not None:
            self.run.summary.update(flatten(metrics))
            self.run.finish()


def start(name: str, config: dict, run_id: str | None = None) -> Tracker:
    """Start a run, or continue the run `run_id` after a resume. config["wandb_run_id"] gets the run ID."""
    if not os.environ.get("WANDB_API_KEY"):
        return Tracker()
    import wandb

    # A new run gets an ID from wandb. A resumed job passes the ID of its run and continues it.
    run = wandb.init(project=PROJECT, name=name, id=run_id, resume="allow" if run_id else None, config=config)
    config["wandb_run_id"] = run.id
    return Tracker(run)


def flatten(metrics: dict) -> dict:
    """{"v7/decision-v7/development.jsonl": {"accuracy": 0.9}} -> {"dev/decision-v7/accuracy": 0.9}"""
    return {
        f"dev/{path.split('/')[-2]}/{key}": value for path, values in metrics.items() for key, value in values.items()
    }


def upload_finished_stage(folder: Path) -> str:
    """Send the log and metrics of a stage that trained without tracking to a new run. Return the run URL."""
    config = json.loads((folder / "final" / "config.json").read_text())
    tracker = start(f"stage{config['stage']['number']}", config)
    if tracker.run is None:
        raise RuntimeError("WANDB_API_KEY is not set")
    for line in (folder / "log.jsonl").read_text().splitlines():
        entry = json.loads(line)
        tracker.log(entry, step=entry["step"])
    url = tracker.run.url
    metrics_file = folder / "metrics.json"
    tracker.finish(json.loads(metrics_file.read_text()) if metrics_file.exists() else {})
    return url
