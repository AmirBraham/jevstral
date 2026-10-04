"""Inference: answer one request, and compare the fast path with the training path."""

import json
import random
import statistics
import time
from pathlib import Path

import torch

from . import checkpoint, suites
from .encode import SERVE, Encoder, RecordTooLong
from .model import load_text_decoder, load_tokenizer
from .records import to_request
from .stages import STAGES

PATHS = ("fp32", "bf16-merged")
PARITY_RECORDS = 100  # for each development file
LATENCY_REQUESTS = 100  # for each request kind
WARMUP_REQUESTS = 3


class Predictor:
    """One request in, one probability list for each question out.

    path "bf16-merged": LoRA merged into the weights, bf16 decoder (the fast path).
    path "fp32": the training path, fp32 decoder with a separate LoRA adapter (the reference).
    """

    def __init__(self, directory: Path, path: str = "bf16-merged"):
        if path not in PATHS:
            raise ValueError(f"path must be one of {PATHS}")
        self.encoder = Encoder(load_tokenizer())
        ids, pad = self.encoder.delimiter_ids, self.encoder.pad
        if path == "fp32":
            self.model = checkpoint.load(directory, load_text_decoder(), ids, pad)
        else:
            self.model = checkpoint.load_merged(directory, load_text_decoder(), ids, pad, torch.bfloat16)
        self.model.to("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        self.path = path

    def __call__(self, state, questions: dict) -> dict[str, list[float]]:
        """Calibrated probabilities for each question key. Raises RecordTooLong for a request over the limits."""
        record = to_request(state, questions)
        if not record.questions:
            return {}
        rows = self.encoder.rows(record, SERVE)
        scores = self.model.predict_scores(rows)
        return {q.qid: s.softmax(-1).tolist() for q, s in zip(record.questions, scores, strict=True)}

    def synchronize(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()


def _sample(path: str, data_dir: Path, count: int, seed: int = 0) -> list[dict]:
    records = suites.load(path, data_dir)
    return random.Random(seed).sample(records, min(count, len(records)))


def _answers(predictor: Predictor, records: list[dict]) -> list[list[float]]:
    out = []
    for raw in records:
        try:
            probabilities = predictor(raw["state"], raw["questions"])
        except RecordTooLong:
            continue
        out += [probabilities[qid] for qid in raw["questions"]]
    return out


def _latency(predictor: Predictor, records: list[dict]) -> dict:
    for raw in records[:WARMUP_REQUESTS]:
        predictor(raw["state"], raw["questions"])
    times, tokens = [], []
    for raw in records:
        predictor.synchronize()
        start = time.perf_counter()
        predictor(raw["state"], raw["questions"])
        predictor.synchronize()
        times.append((time.perf_counter() - start) * 1000)
        record = to_request(raw["state"], raw["questions"])
        tokens.append(sum(len(row.ids) for row in predictor.encoder.rows(record, SERVE)))
    times.sort()
    return {
        "requests": len(times),
        "median_ms": round(statistics.median(times), 1),
        "p95_ms": round(times[int(0.95 * (len(times) - 1))], 1),
        "median_row_tokens": int(statistics.median(tokens)),
    }


def inference_report(directory: Path, data_dir: Path) -> dict:
    """Measure the latency of both paths, and the difference between their probabilities on development records."""
    short = _sample("v7/decision-v7/development.jsonl", data_dir, LATENCY_REQUESTS)
    long = _sample("documents-v1/development.jsonl", data_dir, LATENCY_REQUESTS)
    parity_records = [r for path in STAGES[4].dev_files for r in _sample(path, data_dir, PARITY_RECORDS, seed=1)]

    report: dict = {"latency": {}, "parity": {}}
    answers = {}
    for path in PATHS:
        predictor = Predictor(directory, path)
        report["latency"][path] = {"short": _latency(predictor, short), "long": _latency(predictor, long)}
        answers[path] = _answers(predictor, parity_records)
        del predictor
        torch.cuda.empty_cache()

    reference, fast = answers["fp32"], answers["bf16-merged"]
    differences = [max(abs(a - b) for a, b in zip(p, q, strict=True)) for p, q in zip(reference, fast, strict=True)]
    flips = sum(
        max(range(len(p)), key=p.__getitem__) != max(range(len(q)), key=q.__getitem__)
        for p, q in zip(reference, fast, strict=True)
    )
    report["parity"] = {
        "questions": len(differences),
        "max_abs_probability_difference": round(max(differences), 4),
        "mean_abs_probability_difference": round(statistics.mean(differences), 5),
        "changed_answers": flips,
    }
    (directory.parent / "inference.json").write_text(json.dumps(report, indent=2))
    return report
