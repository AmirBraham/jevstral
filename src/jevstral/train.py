"""Train one stage, then measure it on the development files."""

import json
import math
import random
import shutil
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import torch

from . import checkpoint, suites, tracking
from .augment import augment, minimal_pair
from .encode import LONG, Encoder, RecordTooLong, Row
from .metrics import summarize
from .model import BASE_MODEL, BASE_REVISION, DecisionModel, load_text_decoder, load_tokenizer
from .records import to_record
from .stages import (
    CHECKPOINT_SECONDS,
    LOG_EVERY_STEPS,
    MAX_GRAD_NORM,
    REPLAY_FILE,
    WARMUP_FRACTION,
    WEIGHT_DECAY,
    Stage,
)

SMOKE_RECORDS = 64
# OneCycleLR divides by zero if the warm-up (WARMUP_FRACTION * total steps) is 1 step or less.
SMOKE_STEPS = 20
SMOKE_DEV_RECORDS = 32


def stage_records(stage: Stage, data_dir: Path) -> list[dict]:
    records = [record for path in stage.files for record in suites.load(path, data_dir)]
    if stage.replay:
        records += random.Random(stage.seed).sample(suites.load(REPLAY_FILE, data_dir), stage.replay)
    return records


def epoch_items(records: list[dict], stage: Stage, epoch: int) -> list[dict]:
    """The training items of one epoch, in order. The same inputs always give the same items, so a resumed job
    continues with the same data."""
    order = list(range(len(records)))
    random.Random(f"{stage.seed}/{epoch}").shuffle(order)
    items = []
    for i in order:
        rng = random.Random(f"{stage.seed}/{epoch}/{i}")
        items.append(augment(records[i], rng))
        if rng.random() < stage.minimal_pair_probability:
            items += minimal_pair(records[i], rng)
    return items


def encoding_report(stages: list[Stage], data_dir: Path) -> str:
    """Encode the first epoch of each stage without a model. Report the row counts, the skipped records and one
    example row as text."""
    encoder = Encoder(load_tokenizer())
    lines = []
    example = None
    for stage in stages:
        rows, skipped = [], 0
        for raw in epoch_items(stage_records(stage, data_dir), stage, epoch=0):
            try:
                rows += encoder.rows(to_record(raw), stage.limits)
            except RecordTooLong:
                skipped += 1
        lengths = sorted(len(row.ids) for row in rows)
        lines.append(
            f"stage {stage.number}: {len(rows)} rows, {skipped} skipped records, "
            f"row tokens median {lengths[len(lengths) // 2]}, max {lengths[-1]}"
        )
        example = rows[0]
    if example is not None:
        lines.append("example row (last stage):")
        lines.append(encoder.tokenizer.decode(example.ids))
        lines.append(f"decide index {example.decide_index}, option indices {example.option_indices}")
        lines.append(f"target {example.target}")
    return "\n".join(lines)


def cross_entropy(scores: list[torch.Tensor], rows: list[Row]) -> torch.Tensor:
    """Mean over the questions of -sum(target * log p). A hard label has a one-hot target."""
    losses = [
        -(torch.tensor(row.target, device=score.device) * score.log_softmax(-1)).sum()
        for score, row in zip(scores, rows, strict=True)
    ]
    return torch.stack(losses).mean()


def evaluate(
    model: DecisionModel, encoder: Encoder, paths: tuple[str, ...], data_dir: Path, limit: int | None = None
) -> dict:
    results = {}
    for path in paths:
        rows, skipped = [], 0
        for raw in suites.load(path, data_dir)[:limit]:
            try:
                rows += encoder.rows(to_record(raw), LONG)
            except RecordTooLong:
                skipped += 1
        scores = model.predict_scores(rows)
        predictions = [
            (score.softmax(-1).tolist(), row.target.index(1.0))
            for score, row in zip(scores, rows, strict=True)
            if 1.0 in row.target
        ]
        results[path] = {**summarize(predictions), "skipped_records": skipped}
    return results


def run_stage(
    stage: Stage,
    data_dir: Path,
    runs_dir: Path,
    git_commit: str,
    smoke: bool = False,
    persist: Callable[[], None] = lambda: None,
) -> dict:
    """Train one stage. persist() is called after each checkpoint (Modal uses it to commit volumes)."""
    out = runs_dir / ("smoke" if smoke else "main") / f"stage{stage.number}"
    out.mkdir(parents=True, exist_ok=True)
    resume_dir = out / "resume"
    resuming = not smoke and resume_dir.exists()
    previous = runs_dir / "main" / f"stage{stage.number - 1}" / "final"
    if stage.number > 1 and not previous.exists():
        raise FileNotFoundError(f"stage {stage.number} starts from {previous}, which does not exist")

    torch.manual_seed(stage.seed)
    encoder = Encoder(load_tokenizer())
    decoder = load_text_decoder()
    if resuming:
        model = checkpoint.load(resume_dir, decoder, encoder.delimiter_ids, encoder.pad)
    elif stage.number > 1:
        model = checkpoint.load(previous, decoder, encoder.delimiter_ids, encoder.pad)
    else:
        model = DecisionModel(decoder, encoder.delimiter_ids, encoder.pad)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    print("delimiter row norms:", [round(n, 4) for n in model.delimiter_rows().norm(dim=-1).tolist()], flush=True)
    persist()  # keep the downloaded base model in the cache volume

    records = stage_records(stage, data_dir)
    if smoke:
        records = random.Random(stage.seed).sample(records, SMOKE_RECORDS)
    epochs = [epoch_items(records, stage, epoch) for epoch in range(stage.epochs)]
    steps = sum(math.ceil(math.ceil(len(items) / stage.batch) / stage.accumulation) for items in epochs)
    total_steps = min(steps, SMOKE_STEPS) if smoke else steps

    parameters = model.trainable_parameters()
    optimizer = torch.optim.AdamW(parameters, lr=stage.learning_rate, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=stage.learning_rate, total_steps=total_steps, pct_start=WARMUP_FRACTION
    )
    position = {"epoch": 0, "batch": 0, "step": 0, "skipped": 0}
    if resuming:
        state = torch.load(resume_dir / "resume.pt", map_location="cpu", weights_only=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        position = state["position"]

    config = {
        "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION,
        "data": {"hf_dataset": suites.HF_DATASET, "hf_revision": suites.HF_REVISION, "kev_commit": suites.KEV_COMMIT},
        "stage": asdict(stage),
        "records": len(records),
        "total_steps": total_steps,
        "trainable_parameters": sum(p.numel() for p in parameters),
        "git_commit": git_commit,
        "smoke": smoke,
    }
    run_id = json.loads((resume_dir / "config.json").read_text()).get("wandb_run_id") if resuming else None
    name = f"stage{stage.number}" + ("-smoke" if smoke else "")
    tracker = tracking.start(name, config, run_id)
    print(json.dumps(config, indent=2), flush=True)

    model.train()
    step, skipped = position["step"], position["skipped"]
    losses: list[float] = []
    last_save = time.monotonic()
    with (out / "log.jsonl").open("a") as log:
        for epoch in range(position["epoch"], stage.epochs):
            items = epochs[epoch]
            batches = [items[i : i + stage.batch] for i in range(0, len(items), stage.batch)]
            first = position["batch"] if epoch == position["epoch"] else 0
            for index in range(first, len(batches)):
                rows = []
                for raw in batches[index]:
                    try:
                        rows += encoder.rows(to_record(raw), stage.limits)
                    except RecordTooLong:
                        skipped += 1
                if rows:
                    loss = cross_entropy(model.scores(rows), rows)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"step {step}: the loss is {loss.item()}")
                    (loss / stage.accumulation).backward()
                    losses.append(loss.item())
                if (index + 1) % stage.accumulation and index + 1 < len(batches):
                    continue
                grad_norm = torch.nn.utils.clip_grad_norm_(parameters, MAX_GRAD_NORM, error_if_nonfinite=True)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if step % LOG_EVERY_STEPS == 0 or step == total_steps:
                    entry = {
                        "step": step,
                        "epoch": epoch,
                        "loss": sum(losses) / max(len(losses), 1),
                        "learning_rate": scheduler.get_last_lr()[0],
                        "grad_norm": float(grad_norm),
                        "skipped_records": skipped,
                    }
                    log.write(json.dumps(entry) + "\n")
                    log.flush()
                    print(entry, flush=True)
                    tracker.log(entry, step=step)
                    losses = []
                if step == total_steps:
                    break
                if not smoke and time.monotonic() - last_save > CHECKPOINT_SECONDS:
                    state = {
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "position": {"epoch": epoch, "batch": index + 1, "step": step, "skipped": skipped},
                    }
                    checkpoint.save(model, resume_dir, config, state)
                    persist()
                    last_save = time.monotonic()
            if step == total_steps:
                break

    checkpoint.save(model, out / "final", config)
    shutil.rmtree(resume_dir, ignore_errors=True)
    metrics = evaluate(model, encoder, stage.dev_files, data_dir, SMOKE_DEV_RECORDS if smoke else None)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2), flush=True)
    tracker.finish(metrics)
    persist()
    return metrics
