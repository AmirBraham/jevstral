"""Save and load checkpoints: LoRA adapter, head, temperature and configuration."""

import json
import shutil
from pathlib import Path

import torch

from .model import BASE_REVISION, DecisionModel


def save(model: DecisionModel, directory: Path, config: dict, resume_state: dict | None = None) -> None:
    """Write the checkpoint to a temporary folder, then rename it. A stopped job never leaves half a checkpoint."""
    temporary = directory.with_name(directory.name + ".tmp")
    shutil.rmtree(temporary, ignore_errors=True)
    temporary.mkdir(parents=True)
    model.decoder.save_pretrained(temporary / "adapter")
    torch.save({"head": model.head.state_dict(), "temperature": model.temperature}, temporary / "head.pt")
    (temporary / "config.json").write_text(json.dumps(config, indent=2))
    if resume_state is not None:
        torch.save(resume_state, temporary / "resume.pt")
    shutil.rmtree(directory, ignore_errors=True)
    temporary.rename(directory)


def load(directory: Path, decoder, delimiter_ids: list[int], pad_id: int) -> DecisionModel:
    config = json.loads((directory / "config.json").read_text())
    if config["base_revision"] != BASE_REVISION:
        raise ValueError(f"{directory}: base revision {config['base_revision']}, this code uses {BASE_REVISION}")
    model = DecisionModel(decoder, delimiter_ids, pad_id, adapter_dir=directory / "adapter")
    head = torch.load(directory / "head.pt", map_location="cpu", weights_only=True)
    model.head.load_state_dict(head["head"])
    model.temperature = head["temperature"]
    return model


def set_temperature(directory: Path, temperature: float) -> None:
    head = torch.load(directory / "head.pt", map_location="cpu", weights_only=True)
    head["temperature"] = temperature
    torch.save(head, directory / "head.pt")
