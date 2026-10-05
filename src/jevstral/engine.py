"""Jevstral as an engine of the Decision Index harness (github.com/apolinario/decision-index).

Install:  pip install "jevstral[inference] @ git+https://github.com/AmirBraham/jevstral"
Run:      python -m decision_index run --engine jevstral.engine:JevstralEngine
The default checkpoint is the final model on the Hugging Face Hub; it is downloaded on first use.
"""

import json
from pathlib import Path

import torch
from decision_index.engines import Engine, Unsupported

from .encode import SERVE, RecordTooLong
from .inference import Predictor

MODEL_NAME = "jevstral-8b"
WEIGHTS_REPO = "AmirBraham/jevstral-8b"
FINAL = "stage4/final"  # the calibrated final checkpoint in WEIGHTS_REPO


def checkpoint_directory(checkpoint: str, revision: str) -> tuple[Path, str]:
    """A local checkpoint folder, or a Hub repository whose FINAL folder is downloaded. Returns (folder, source)."""
    if Path(checkpoint).is_dir():
        return Path(checkpoint), checkpoint
    from huggingface_hub import HfApi, snapshot_download

    sha = HfApi().model_info(checkpoint, revision=revision).sha
    local = snapshot_download(checkpoint, revision=sha, allow_patterns=[f"{FINAL}/*"])
    return Path(local) / FINAL, f"{checkpoint}@{sha}"


class JevstralEngine(Engine):
    name = "jevstral"
    latency = (
        "In-process wall time on one GPU: request encoding, tokenization and one batched forward pass for all "
        "questions of the request; excludes model loading."
    )

    def __init__(
        self,
        checkpoint: str = WEIGHTS_REPO,
        revision: str = "main",
        path: str = "bf16-merged",
        cache: bool = True,
        **options,
    ):
        super().__init__(**options)
        directory, source = checkpoint_directory(checkpoint, revision)
        self.predictor = Predictor(directory, path, cache=cache)
        config = json.loads((directory / "config.json").read_text())
        temperature = torch.load(directory / "head.pt", map_location="cpu", weights_only=True)["temperature"]
        self.provenance = {
            "kind": "trained",
            "model": MODEL_NAME,
            "weights": source,
            "base_model": config["base_model"],
            "base_revision": config["base_revision"],
            "git_commit": config["git_commit"],
            "inference_path": path,
            "state_cache": cache,
            "temperature": temperature,
            "limits": {"max_state_tokens": SERVE.max_state, "max_row_tokens": SERVE.max_row},
            "policy": "Each question is one row with the full state. Requests over the limits are refused, never cut.",
        }

    def __call__(self, state, questions):
        for key, q in questions.items():
            if q["type"] not in ("choice", "noul"):
                raise Unsupported(f"question {key}: type {q['type']} is not supported")
        try:
            probabilities = self.predictor(state, questions)
        except RecordTooLong as error:
            raise Unsupported(f"maximum context length: {error}") from error
        answers = {}
        for key, q in questions.items():
            p = probabilities[key]
            if q["type"] == "noul":
                answers[key] = {"type": "noul", "noul": p[1]}  # options are no, yes
            else:
                by_key = dict(zip(list(q["criteria"]), p, strict=True))
                answers[key] = {
                    "type": "choice",
                    "choice": max(by_key, key=lambda k: by_key[k]),
                    "probabilities": by_key,
                }
        return {"model": MODEL_NAME, "answers": answers}, None

    def synchronize(self):
        self.predictor.synchronize()

    def runtime(self):
        if not torch.cuda.is_available():
            return {"torch": torch.__version__, "device": "cpu"}
        return {"torch": torch.__version__, "device": "cuda", "gpu": torch.cuda.get_device_name(0)}
