"""Fit one temperature T on a pool of held-out questions."""

import json
from collections.abc import Callable
from pathlib import Path

import torch

from . import checkpoint, suites
from .encode import LONG, Encoder
from .metrics import summarize
from .model import load_text_decoder, load_tokenizer
from .records import to_record

# Kev's pool: datasets that no stage trains on. (file, sources to keep, expected question count)
POOL = (
    (
        "round3/transfer-r3/calibration.jsonl",
        frozenset(
            {"composition_holdout", "emotion", "legacy_holdout", "mmlu", "paws", "qnli", "sciq", "tweet_offensive"}
        ),
        448,
    ),
    ("v9/transfer-v9/development.jsonl", frozenset({"mmlu_pro"}), 200),
)
TEMPERATURES = [round(0.5 + 0.01 * i, 2) for i in range(451)]  # 0.50 to 5.00


def pool_records(data_dir: Path) -> list[dict]:
    records = []
    for path, sources, expected in POOL:
        selected = [record for record in suites.load(path, data_dir) if record["_meta"]["source"] in sources]
        questions = sum(len(record["questions"]) for record in selected)
        if questions != expected:
            raise ValueError(f"{path}: {questions} pool questions, expected {expected}")
        records += selected
    return records


def mean_nll(scores: list[torch.Tensor], labels: list[int], temperature: float) -> float:
    return sum(-(s / temperature).log_softmax(-1)[y].item() for s, y in zip(scores, labels, strict=True)) / len(labels)


def fit_temperature(scores: list[torch.Tensor], labels: list[int]) -> float:
    """The T in TEMPERATURES with the lowest mean negative log-likelihood."""
    return min(TEMPERATURES, key=lambda t: mean_nll(scores, labels, t))


def run_calibration(data_dir: Path, runs_dir: Path, persist: Callable[[], None] = lambda: None) -> dict:
    final = runs_dir / "main" / "stage4" / "final"
    encoder = Encoder(load_tokenizer())
    model = checkpoint.load(final, load_text_decoder(), encoder.delimiter_ids, encoder.pad)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.temperature = 1.0

    rows = [row for raw in pool_records(data_dir) for row in encoder.rows(to_record(raw), LONG)]
    labels = [row.target.index(1.0) for row in rows]
    scores = model.predict_scores(rows)
    temperature = fit_temperature(scores, labels)

    def report(t: float) -> dict:
        predictions = [((s / t).softmax(-1).tolist(), y) for s, y in zip(scores, labels, strict=True)]
        return {**summarize(predictions), "nll": mean_nll(scores, labels, t)}

    result = {
        "temperature": temperature,
        "pool_questions": len(rows),
        "before": report(1.0),
        "after": report(temperature),
    }
    checkpoint.set_temperature(final, temperature)
    (final.parent / "calibration.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    persist()
    return result
