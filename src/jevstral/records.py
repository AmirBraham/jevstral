"""Kev records: parse them and change each question into a list of options."""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Question:
    qid: str
    qtype: str
    instructions: str
    options: tuple[str, ...]
    keys: tuple[str, ...]
    target: tuple[float, ...]  # probability for each option; one-hot for a hard label

    @property
    def label(self) -> int | None:
        """Index of the correct option, or None for a soft target."""
        return self.target.index(1.0) if 1.0 in self.target else None


@dataclass(frozen=True)
class Record:
    rid: str
    source: str
    state: str
    questions: tuple[Question, ...]


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def render(value, indent: int = 0) -> str:
    """Change a JSON value into labelled text. Same rules as render() in kev/api.py."""
    pad = "  " * indent
    if value is None:
        return ""
    if isinstance(value, str | int | float | bool):
        return str(value)
    if isinstance(value, list):
        return "\n".join(f"{pad}- {render(item, indent + 1).lstrip()}" for item in value)
    return "\n".join(
        f"{pad}{key}:\n{render(item, indent + 1)}" if isinstance(item, dict | list) else f"{pad}{key}: {render(item)}"
        for key, item in value.items()
    )


def option_text(name: str, description) -> str:
    return name if description in (None, "") else f"{name}: {render(description)}"


def to_record(raw: dict) -> Record:
    meta = raw.get("_meta", {})
    rid = meta.get("id", "")
    questions = tuple(_question(rid, qid, q) for qid, q in raw["questions"].items())
    if not questions:
        raise ValueError(f"{rid}: the record has no questions")
    return Record(rid=rid, source=meta.get("source", ""), state=render(raw["state"]), questions=questions)


def _question(rid: str, qid: str, q: dict) -> Question:
    criteria = q.get("criteria")
    if q["type"] in ("choice", "score") and not criteria:
        raise ValueError(f"{rid}/{qid}: a {q['type']} question needs criteria")
    if q["type"] == "choice":
        keys = tuple(criteria)
        options = tuple(option_text(key, description) for key, description in criteria.items())
    elif q["type"] == "noul":
        criteria = criteria or {}
        keys = ("false", "true")
        options = (option_text("no", criteria.get("false")), option_text("yes", criteria.get("true")))
    elif q["type"] == "score":
        keys = tuple(str(level) for level in range(len(criteria)))
        options = tuple(render(level) for level in criteria)
    else:
        raise ValueError(f"{rid}/{qid}: unknown question type {q['type']!r}")
    if not keys:
        raise ValueError(f"{rid}/{qid}: the question has no options")
    return Question(qid, q["type"], render(q["instructions"]), options, keys, _target(rid, qid, q, keys))


def _target(rid: str, qid: str, q: dict, keys: tuple[str, ...]) -> tuple[float, ...]:
    if q.get("target") is not None:
        weights = [float(q["target"].get(key, 0.0)) for key in keys]
        if sum(weights) <= 0:
            raise ValueError(f"{rid}/{qid}: the target gives no weight to an option")
        return tuple(weight / sum(weights) for weight in weights)
    if q["type"] == "noul":
        key = "true" if q["label"] else "false"
    elif q["type"] == "score":
        key = str(q["label"])
    else:
        key = q["label"]
    if key not in keys:
        raise ValueError(f"{rid}/{qid}: the label {q['label']!r} is not an option")
    return tuple(1.0 if candidate == key else 0.0 for candidate in keys)
