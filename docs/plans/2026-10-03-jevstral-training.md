# Jevstral Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train Ministral-3-8B-Base as a Kev-style decision model on Modal: four stages, then one calibration temperature.

**Architecture:** A small Python package (`src/jevstral`) holds pure data code (records, augmentation, encoding, metrics) and the model and training code. `modal_app.py` is the only file that knows Modal. The local computer runs only `modal` and `ruff`; all data, model and GPU work runs on Modal.

**Tech Stack:** Python 3.12, uv, PyTorch 2.14.1, transformers 5.18.0, PEFT 0.21.2, Modal 1.6, ruff.

**Spec:** `docs/specs/2026-10-03-jevstral-training-design.md`

---

## Rules for this plan

- **No local heavy work.** The owner's computer is slow. Do not install the `train` dependency group locally. Do not download the model or the data locally. Local checks are `ruff` only.
- **No unit tests.** This is the owner's decision. Each code task ends with a lint check. The Modal jobs in Tasks 9 to 12 are the real checks.
- **Learn as we build.** Each code task starts with an explanation in chat (the "Explain" step) and a concept note in `docs/concepts/`, in ASD-STE100 Simplified Technical English. Ask one short check question after each explanation.
- **Code was checked before this plan.** The modules were written and linted. `suites.py` was run once and verified all 11 files against Kev's manifests. The text-decoder loading, LoRA, trainable token rows and adapter save were checked on a tiny random Ministral 3 model.

## Verified facts used by the code

| Fact | Value |
|---|---|
| Base model revision | `d4883f9b36aa2e5d775730d3fdba3d30de51a8ef` |
| Checkpoint weight names | `language_model.model.*`, `vision_tower.*`, `multi_modal_projector.*` |
| `Ministral3Model.from_pretrained(repo)` | Gives random weights with no error. Load `Mistral3ForConditionalGeneration` and keep `.model.language_model`. |
| Tokenizer | BOS `<s>` = 1, pad `<pad>` = 11. `<SPECIAL_20>`..`<SPECIAL_23>` = 20..23, `<SPECIAL_26>` = 26. |
| Text `"<SPECIAL_20>"` with default tokenization | Becomes token 20. With `split_special_tokens=True` it stays plain text. |
| Calibration pool | transfer-r3 calibration, 8 sources: 448 questions. transfer-v9 development, `mmlu_pro`: 200 questions. |

## File map

| File | Responsibility |
|---|---|
| `pyproject.toml` | Project and dependency groups (`train` is for Modal only) |
| `README.md` | Setup and commands |
| `src/jevstral/suites.py` | Download pinned data, verify SHA-256 and record count |
| `src/jevstral/records.py` | Kev record → `Record` with option lists and targets |
| `src/jevstral/augment.py` | Shuffle, "none" options, distractors, minimal pairs |
| `src/jevstral/encode.py` | `Record` → token rows; delimiters; limits |
| `src/jevstral/model.py` | Text decoder, LoRA, pointer head, scoring |
| `src/jevstral/metrics.py` | Accuracy, Brier, ECE |
| `src/jevstral/checkpoint.py` | Save and load adapter, head, temperature, config |
| `src/jevstral/stages.py` | The four stage configurations |
| `src/jevstral/train.py` | Training loop, resume, evaluation, encoding report |
| `src/jevstral/calibrate.py` | Calibration pool and temperature fit |
| `modal_app.py` | Modal image, volumes and entry points |
| `docs/concepts/*.md` | STE concept notes |

### Task 1: Project setup

**Files:**
- Create: `pyproject.toml`, `README.md`, `src/jevstral/__init__.py`, `docs/concepts/decision-model.md`, `docs/concepts/jev-architecture.md`

- [ ] **Step 1: Explain** what a decision model is, and how the Jev essay maps to Jevstral. Use the two concept notes below. Check question: "Why is the latency of a decision model close to the time to first token of its backbone?"

- [ ] **Step 2: Write `docs/concepts/decision-model.md`**

```markdown
# Decision model

A decision model reads a document and a set of typed questions. It gives a probability for each option of each question. It does not write text.

## Input

- **State**: the document. It is text, or JSON that we change into labelled text.
- **Questions**. Each question has one of three types:
  - `choice`: named options.
  - `noul`: yes or no.
  - `score`: ordered levels, for example low, medium, high.

Every type becomes a list of options.

## Output

For each question, the model gives one probability for each option. The probabilities of one question add up to 1.

## Difference from a chat model

A chat model does two steps:

1. **Prefill**: it reads the full prompt in one parallel pass.
2. **Decode**: it writes the answer one token at a time. Each token is one more pass.

A decision model does only step 1. A small head reads the hidden states after the prefill and gives one score for each option. A softmax changes the scores into probabilities.

Results:

- The time is approximately the time to first token of the same backbone.
- The probability is a number from the model. It is not a text claim such as "90 % sure".
- The answer is always one of the given options.

## Why probabilities are useful

Software can act on a threshold. For example: if p ≥ 0.9, do the action automatically. If not, send the case to a person. This rule is correct only if the probabilities are calibrated. See `calibration.md`.
```

- [ ] **Step 3: Write `docs/concepts/jev-architecture.md`**

```markdown
# Jev architecture and Jevstral

The essay "Jev's Architecture Unmasked" (archerhume.com, 2026-09-17) gives seven ideas about how TypeSafe's Jev works. Kev follows most of them. This table compares the ideas with Jevstral.

| # | Idea | Jevstral |
|---|---|---|
| 1 | Read out probabilities. Do not generate text. | Pointer head on the hidden states (`pointer-head.md`). |
| 2 | Read the state once. Isolate the questions. | Each question has its own row, so isolation is exact (`encoding.md`). Reading the state once is a serving feature. It is not in scope. |
| 3 | Use a causal backbone. | Ministral 3 8B Base, a left-to-right decoder. |
| 4 | Let the options interact before the choice. | All options come before `<decide>`, so `<decide>` sees the full list. |
| 5 | Train the distribution, then calibrate. | Cross-entropy loss (a proper scoring rule), then one temperature (`calibration.md`). |
| 6 | Use sparse experts. | Not used. The backbone is dense. The other ideas do not need sparse experts. |
| 7 | Run the question branches as one batch. | A serving feature. It is not in scope. |

## Two open questions in the essay

- **Option slots or a pointer?** Kev and Jevstral use a pointer. See `pointer-head.md`.
- **How do option borders resist fake text?** The delimiters are special tokens. User text cannot make them. See `encoding.md`.
```

- [ ] **Step 4: Write `pyproject.toml`**

```toml
[project]
name = "jevstral"
version = "0.1.0"
description = "A decision model on Ministral 3 8B, trained with the Kev method"
requires-python = ">=3.12,<3.13"
dependencies = ["modal>=1.6,<2"]

[dependency-groups]
# Installed only in the Modal image (modal_app.py), not on the local computer.
train = ["torch==2.14.1", "transformers==5.18.0", "peft==0.21.2"]
dev = ["ruff>=0.14"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/jevstral"]

[tool.ruff]
line-length = 120
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP"]
```

- [ ] **Step 5: Write `README.md`**

````markdown
# Jevstral

Jevstral is a decision model. It reads one document and a set of typed questions. It gives a probability for each option of each question. It does not generate text.

Jevstral uses the method of [Kev](https://github.com/jaredpalmer/kev) on the backbone `mistralai/Ministral-3-8B-Base-2512`.

## Setup

The local computer runs only the `modal` command. All data, model and GPU work runs on Modal.

```
uv sync
uv run modal setup
uv run modal secret create huggingface HF_TOKEN=<your token>
```

## Train

```
uv run modal run modal_app.py::prepare_data
uv run modal run modal_app.py::inspect_data
uv run modal run modal_app.py::train --stage 1 --smoke
uv run modal run --detach modal_app.py::train --stage 1
```

Do stages 2, 3 and 4 in the same way. Then fit the temperature:

```
uv run modal run --detach modal_app.py::calibrate
```

## Documents

- `docs/specs/`: design.
- `docs/concepts/`: how each part works.
````

- [ ] **Step 6: Write `src/jevstral/__init__.py`**

```python
"""Jevstral: a decision model on Ministral 3 8B, trained with the Kev method."""
```

- [ ] **Step 7: Lock and install the light local environment**

```bash
uv lock && uv sync
```
Expected: `uv.lock` is created. `uv sync` installs only `modal` and `ruff` (default groups). It does not install torch.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml uv.lock README.md src/jevstral/__init__.py docs/concepts/decision-model.md docs/concepts/jev-architecture.md
git commit -m "Set up the project

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 2: Data download and records

**Files:**
- Create: `src/jevstral/suites.py`, `src/jevstral/records.py`

- [ ] **Step 1: Explain** where Kev's data lives (Hugging Face dataset and GitHub, both pinned), why we verify SHA-256, and how the three question types become option lists with a target. Check question: "A `noul` question has the label `true`. What is its target vector?" (Answer: `(0.0, 1.0)`, because the options are `no`, `yes`.)

- [ ] **Step 2: Write `src/jevstral/records.py`**

```python
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
```

- [ ] **Step 3: Write `src/jevstral/suites.py`**

```python
"""Download Kev's data at pinned revisions. Verify each file against its Kev manifest."""

import hashlib
import json
import urllib.error
import urllib.request
from pathlib import Path

from .records import read_jsonl

HF_DATASET = "jaredpalmer/kev-suites"
HF_REVISION = "cc4bac803e73112689ec327ffa481c519cbc7a05"
KEV_COMMIT = "84847f0a883d900f7de5b7a57eaa341ca7f9a6b4"

HF_URL = f"https://huggingface.co/datasets/{HF_DATASET}/resolve/{HF_REVISION}/{{path}}"
KEV_URL = f"https://raw.githubusercontent.com/jaredpalmer/kev/{KEV_COMMIT}/evals/{{path}}"

FILES = (
    "v7/decision-v7/train.jsonl",
    "v7/decision-v7/development.jsonl",
    "night2/dates_unknowable.jsonl",
    "documents-v1/train.jsonl",
    "documents-v1/development.jsonl",
    "hard-v1/train.jsonl",
    "hard-v1/development.jsonl",
    "devtools-v1/train.jsonl",
    "devtools-v1/development.jsonl",
    "round3/transfer-r3/calibration.jsonl",
    "v9/transfer-v9/development.jsonl",
)


def _download(url: str) -> bytes | None:
    """Return the content, or None if the server answers 404."""
    try:
        with urllib.request.urlopen(url, timeout=300) as response:
            return response.read()
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None
        raise


def fetch(path: str, data_dir: Path) -> Path:
    """Download one file, verify it and write it to data_dir/path."""
    suite, name = path.rsplit("/", 1)
    manifest = _download(KEV_URL.format(path=f"{suite}/manifest.json"))
    if manifest is None:
        raise FileNotFoundError(f"{suite}/manifest.json is not in the Kev repository")
    expected = json.loads(manifest)["files"][name]
    content = _download(HF_URL.format(path=path)) or _download(KEV_URL.format(path=path))
    if content is None:
        raise FileNotFoundError(f"{path} is not in {HF_DATASET} and not in the Kev repository")
    digest = hashlib.sha256(content).hexdigest()
    if digest != expected["sha256"]:
        raise ValueError(f"{path}: SHA-256 {digest} is not the manifest value {expected['sha256']}")
    count = sum(1 for line in content.splitlines() if line.strip())
    if count != expected["records"]:
        raise ValueError(f"{path}: {count} records, the manifest gives {expected['records']}")
    target = data_dir / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    return target


def fetch_all(data_dir: Path) -> list[Path]:
    return [fetch(path, data_dir) for path in FILES]


def load(path: str, data_dir: Path) -> list[dict]:
    return read_jsonl(data_dir / path)
```

- [ ] **Step 4: Lint**

```bash
uv run ruff check src modal_app.py && uv run ruff format --check src modal_app.py
```
Expected: `All checks passed!` and `already formatted`. Do not run the module locally. It runs on Modal in Task 9.

- [ ] **Step 5: Commit**

```bash
git add src/jevstral/records.py src/jevstral/suites.py
git commit -m "Add data download and record parsing

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 3: Augmentation

**Files:**
- Create: `src/jevstral/augment.py`, `docs/concepts/augmentation.md`

- [ ] **Step 1: Explain** augmentation as "find a shortcut, then add examples where the shortcut fails" (concept note below). Check question: "Why does a soft-target question get only a shuffle?"

- [ ] **Step 2: Write `docs/concepts/augmentation.md`**

```markdown
# Augmentation

Augmentation changes each training record a little, at random. It stops the model from learning shortcuts. It is used only in training.

## Two facts

1. A model learns any pattern that makes the loss lower. It cannot tell a real pattern from an accident in the data.
2. The softmax always puts 100 % on the given options. The model must choose one, also when no option is correct.

## The shortcuts and the changes

| Shortcut | Change | Probability |
|---|---|---|
| "The answer is at position X." | Shuffle the options. | Always (choice questions) |
| "Choose the best option, also when no option is correct." | Add "none of the above" and remove the correct option. The label becomes "none". | 0.10 |
| "When 'none' is there, choose it." (made by the change above) | Add "none of the above" and keep the correct option. | 0.12 |
| "All options are possible answers." | Add one unrelated option, for example "A recipe for pancakes". | 0.15 |

The three "add" changes do not occur together on one question.

## Minimal pairs (stage 1 only)

For 25 % of the records, we also add two copies of one choice question:

- Copy A: the correct option and a "none" option. The label is the correct option.
- Copy B: the same "none" option, and no correct option. The label is "none".

The two copies are the same except for one option. The only way to answer both correctly is to check if the correct option is present.

## Other rules

- Each epoch uses a new random draw. The model does not see the same version twice.
- A question with a soft target only gets a shuffle. Other changes would change the meaning of the target.
- Many wordings of "none" are used. Then the model cannot learn one wording as "the answer".
```

- [ ] **Step 3: Write `src/jevstral/augment.py`**

```python
"""Training-only changes to records. The rules and lists come from kev/data.py at the pinned Kev commit."""

import random

P_NONE_CORRECT = 0.10  # add "none of the above" and remove the correct option
P_NONE_WRONG = 0.12  # add "none of the above" and keep the correct option
P_DISTRACTOR = 0.15  # add one irrelevant option
MAX_OPTIONS = 255

# Many wordings, so the model cannot learn one wording as "the answer".
NONE_OPTIONS = (
    ("other", "None of the above"),
    ("other", "A reason that fits none of the above"),
    ("none", "None of these"),
    ("other", "Something else"),
    ("not_listed", "Not listed here"),
    ("none_of_the_above", None),
    ("other", "A category that fits none of the above"),
    ("other", "None of the listed options apply"),
    ("unknown", "Cannot be determined from the options given"),
    ("other", "Other"),
    ("none", None),
    ("other", "An answer not covered by the other options"),
    ("no_match", "No option matches"),
)
DISTRACTORS = {
    "weather": "Bad weather caused it",
    "purple": "The colour purple",
    "pancakes": "A recipe for pancakes",
    "taxes": "Unrelated: quarterly tax filing",
}


def augment(raw: dict, rng: random.Random) -> dict:
    """Return a changed copy of a record. Only choice questions change."""
    questions = {qid: _choice(q, rng) if q["type"] == "choice" else q for qid, q in raw["questions"].items()}
    return {**raw, "questions": questions}


def _choice(q: dict, rng: random.Random) -> dict:
    criteria, label = dict(q["criteria"]), q["label"]
    # A soft target gives weights to the original options. Do not add or remove options for it.
    if q.get("target") is None:
        draw = rng.random()
        nones = [(key, text) for key, text in NONE_OPTIONS if key not in criteria]
        distractors = [key for key in DISTRACTORS if key not in criteria]
        if draw < P_NONE_CORRECT:
            if len(criteria) > 2 and nones:
                key, text = rng.choice(nones)
                del criteria[label]
                criteria[key] = text
                label = key
        elif draw < P_NONE_CORRECT + P_NONE_WRONG:
            if len(criteria) < MAX_OPTIONS and nones:
                key, text = rng.choice(nones)
                criteria[key] = text
        elif draw < P_NONE_CORRECT + P_NONE_WRONG + P_DISTRACTOR:
            if len(criteria) < MAX_OPTIONS and distractors:
                key = rng.choice(distractors)
                criteria[key] = DISTRACTORS[key]
    keys = list(criteria)
    rng.shuffle(keys)
    return {**q, "criteria": {key: criteria[key] for key in keys}, "label": label}


def minimal_pair(raw: dict, rng: random.Random) -> list[dict]:
    """Return two records that differ only in one thing: the correct option is present or removed.

    Both copies have the same "none" option. Without the correct option, "none" is the answer.
    Return [] if the record has no choice question with at least 3 options and a hard label.
    """
    eligible = [
        (qid, q)
        for qid, q in raw["questions"].items()
        if q["type"] == "choice" and len(q["criteria"]) >= 3 and q.get("target") is None
    ]
    if not eligible:
        return []
    qid, q = rng.choice(eligible)
    nones = [option for option in NONE_OPTIONS if option[0] not in q["criteria"]] or [("none_of_these", None)]
    none_key, none_text = rng.choice(nones)
    keys = [*q["criteria"], none_key]
    rng.shuffle(keys)
    present = {**q, "criteria": {key: none_text if key == none_key else q["criteria"][key] for key in keys}}
    absent = {
        **present,
        "criteria": {key: text for key, text in present["criteria"].items() if key != q["label"]},
        "label": none_key,
    }
    return [{**raw, "questions": {qid: present}}, {**raw, "questions": {qid: absent}}]
```

- [ ] **Step 4: Lint**

```bash
uv run ruff check src modal_app.py && uv run ruff format --check src modal_app.py
```
Expected: `All checks passed!` and `already formatted`. Do not run the module locally. It runs on Modal in Task 9.

- [ ] **Step 5: Commit**

```bash
git add src/jevstral/augment.py docs/concepts/augmentation.md
git commit -m "Add training augmentation

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 4: Encoding

**Files:**
- Create: `src/jevstral/encode.py`, `docs/concepts/encoding.md`

- [ ] **Step 1: Explain** the row format, the delimiters, the `split_special_tokens` safety rule, BOS, and the two positions that the head reads. Check question: "Why must `<decide>` come after all the options?"

- [ ] **Step 2: Write `docs/concepts/encoding.md`**

````markdown
# Encoding

Encoding changes a record into token rows. Each question gets its own row.

## Row format

```
<s> <state> state text <q> instructions <opt> option 1 </opt> <opt> option 2 </opt> ... <decide>
```

- `<s>` is the Mistral BOS token. The tokenizer adds it by default, so normal model inputs start with it.
- A record with three questions gives three rows. Each row has the full state.
- A question cannot see another question, because it is in a different row.

## Delimiters

| Delimiter | Token | ID |
|---|---|---|
| `<state>` | `<SPECIAL_20>` | 20 |
| `<q>` | `<SPECIAL_21>` | 21 |
| `<opt>` | `<SPECIAL_22>` | 22 |
| `</opt>` | `<SPECIAL_23>` | 23 |
| `<decide>` | `<SPECIAL_26>` | 26 |

These are reserved placeholder tokens. Mistral gives them no name and no role: they fill the gaps between named control tokens such as `[TOOL_CALLS]` and `[AUDIO]`. We did not verify that the base model never saw them. For this reason:

- We train their embedding rows (see `lora.md`), so their meaning comes from our training.
- The `inspect_data` job measures their embedding norms against ordinary tokens.

**Safety rule.** By default, the tokenizer changes the text `<SPECIAL_20>` into token 20. Then a user could write a fake option border. We tokenize all user text with `split_special_tokens=True`. Then the text `<SPECIAL_20>` stays plain text.

## Positions that the head reads

- `decide_index`: the position of `<decide>`. It comes after all options, so it can see all of them.
- `option_indices`: the position of each `</opt>`. Each one comes after the text of its option, so it holds the meaning of that option.

## Limits

| Value | Stages 1 and 2 | Stages 3 and 4 |
|---|---|---|
| State tokens (`<s>` and `<state>` included) | 384 | 7,552 |
| Row tokens | 1,024 | 8,192 |

If a record is longer than a limit, we skip it and count it.

## Option text

| Type | Options |
|---|---|
| `choice` | `key: description`, or `key` |
| `noul` | `no`, `yes` (with a description if one is given) |
| `score` | the levels, in order |
````

- [ ] **Step 3: Write `src/jevstral/encode.py`**

```python
"""Change one record into token rows. Each question gets its own row."""

from dataclasses import dataclass

from .records import Record

# Reserved placeholder tokens of the Mistral tokenizer: no name, no role.
# User text cannot make them (see Encoder.text).
DELIMITERS = {
    "state": "<SPECIAL_20>",
    "question": "<SPECIAL_21>",
    "open": "<SPECIAL_22>",
    "close": "<SPECIAL_23>",
    "decide": "<SPECIAL_26>",
}


@dataclass(frozen=True)
class Limits:
    max_state: int  # tokens of <s>, <state> and the state text
    max_row: int  # all tokens of one row


SHORT = Limits(max_state=384, max_row=1024)
LONG = Limits(max_state=7552, max_row=8192)


@dataclass(frozen=True)
class Row:
    ids: list[int]
    decide_index: int
    option_indices: list[int]  # position of each </opt>
    target: tuple[float, ...]


class RecordTooLong(ValueError):
    pass


class Encoder:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.bos = tokenizer.bos_token_id
        self.pad = tokenizer.pad_token_id
        self.ids = {name: tokenizer.convert_tokens_to_ids(token) for name, token in DELIMITERS.items()}

    @property
    def delimiter_ids(self) -> list[int]:
        return list(self.ids.values())

    def text(self, text: str) -> list[int]:
        # split_special_tokens=True: if a user writes "<SPECIAL_20>", it stays plain text.
        return self.tokenizer(text, add_special_tokens=False, split_special_tokens=True)["input_ids"]

    def rows(self, record: Record, limits: Limits) -> list[Row]:
        state = [self.bos, self.ids["state"], *self.text(record.state)]
        if len(state) > limits.max_state:
            raise RecordTooLong(f"{record.rid}: state has {len(state)} tokens, limit {limits.max_state}")
        rows = []
        for question in record.questions:
            ids = [*state, self.ids["question"], *self.text(question.instructions)]
            option_indices = []
            for option in question.options:
                ids += [self.ids["open"], *self.text(option), self.ids["close"]]
                option_indices.append(len(ids) - 1)
            ids.append(self.ids["decide"])
            if len(ids) > limits.max_row:
                raise RecordTooLong(f"{record.rid}/{question.qid}: row has {len(ids)} tokens, limit {limits.max_row}")
            rows.append(Row(ids, len(ids) - 1, option_indices, question.target))
        return rows
```

- [ ] **Step 4: Lint**

```bash
uv run ruff check src modal_app.py && uv run ruff format --check src modal_app.py
```
Expected: `All checks passed!` and `already formatted`. Do not run the module locally. It runs on Modal in Task 9.

- [ ] **Step 5: Commit**

```bash
git add src/jevstral/encode.py docs/concepts/encoding.md
git commit -m "Add record encoding

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 5: Model

**Files:**
- Create: `src/jevstral/model.py`, `docs/concepts/pointer-head.md`, `docs/concepts/lora.md`

- [ ] **Step 1: Explain** the pointer head (slots against pointer) and LoRA (W + (α/r)·B·A, B starts at zero, parameter count). Point out why `load_text_decoder` loads the full checkpoint. Check question: "At step 0, before any update, is the LoRA model different from the base model? Why?"

- [ ] **Step 2: Write `docs/concepts/pointer-head.md`**

````markdown
# Pointer head

The pointer head changes hidden states into one score for each option.

## The problem

After the prefill, each position has one hidden vector of 4,096 numbers. We need K scores for K options. There are two methods.

## Method 1: option slots (not used)

Take the vector at `<decide>`. Multiply it by a fixed matrix with one row for each position, for example 256 rows. Score i means "the option at position i".

Problems:

- The number of options has a maximum (the number of rows).
- Each row must learn a position. Rows for high positions get almost no training.

## Method 2: pointer (used)

Each option has its own vector, at its `</opt>` token. Compare each option with `<decide>`:

```
score_i = q(h_decide) · k(h_option_i) / sqrt(256)
p       = softmax(scores / T)
```

`q` and `k` are two linear layers from 4,096 to 256. The same two layers score all options.

Results:

- There is no maximum number of options.
- A new option needs no new weights.
- The score comes from the text of the option, not from its position.

This is the same calculation as attention: `<decide>` is the query, and each option is a key.

## Size

2 × (4,096 × 256 + 256) = 2,097,664 parameters. The head starts from random weights.
````

- [ ] **Step 3: Write `docs/concepts/lora.md`**

````markdown
# LoRA

LoRA (low-rank adaptation) trains a small change to the model. The base weights do not change.

## The problem

Ministral 3 8B has approximately 8 billion weights. Training all of them needs a lot of GPU memory: the weights, the gradients and two optimizer values for each weight. We want to change the behaviour of the model with much less memory.

## The method

Take one weight matrix W of a linear layer, with size d_out × d_in. LoRA keeps W frozen and adds two small matrices:

```
W' = W + (alpha / r) · B · A
A: r × d_in
B: d_out × r
```

- r is the rank. We use r = 16.
- alpha is a scale. We use alpha = 32, so alpha / r = 2.
- B starts at zero. At step 0, W' = W, so the model starts the same as the base model.
- Dropout 0.05 is applied to the LoRA input during training.

## Size

One `q_proj` matrix is 4,096 × 4,096 = 16,777,216 weights. Its LoRA has 16 × (4,096 + 4,096) = 131,072 weights.

| Module | d_in → d_out | LoRA weights |
|---|---|---|
| `q_proj`, `o_proj` | 4,096 → 4,096 | 131,072 each |
| `k_proj`, `v_proj` | 4,096 → 1,024 | 81,920 each |
| `gate_proj`, `up_proj` | 4,096 → 14,336 | 294,912 each |
| `down_proj` | 14,336 → 4,096 | 294,912 |

One layer has 1,310,720 LoRA weights. 34 layers have 44,564,480 LoRA weights. That is approximately 0.5 % of the base model.

## Trainable token rows

The five delimiter tokens are new to the model. We also train their five embedding rows: 5 × 4,096 = 20,480 weights. All other embedding rows stay frozen.

## Checkpoints

A checkpoint holds only the LoRA weights, the token rows and the head. It does not hold the base model. Approximately 190 MB in fp32.
````

- [ ] **Step 4: Write `src/jevstral/model.py`**

```python
"""The decision model: Ministral 3 text decoder, LoRA, delimiter embeddings and pointer head."""

import math

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch import nn
from transformers import AutoTokenizer, Mistral3ForConditionalGeneration

from .encode import Row

BASE_MODEL = "mistralai/Ministral-3-8B-Base-2512"
BASE_REVISION = "d4883f9b36aa2e5d775730d3fdba3d30de51a8ef"
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
POINTER_DIM = 256
ROW_TOKENS_PER_PASS = 32768  # inference: the token budget of one batch of rows


def load_tokenizer():
    return AutoTokenizer.from_pretrained(BASE_MODEL, revision=BASE_REVISION)


def load_text_decoder(dtype: torch.dtype = torch.float32) -> nn.Module:
    """Load the full image-text checkpoint and keep only its text decoder.

    Do not use Ministral3Model.from_pretrained on this repository: it does not map the
    checkpoint names, and it gives random weights without an error.
    """
    full = Mistral3ForConditionalGeneration.from_pretrained(BASE_MODEL, revision=BASE_REVISION, dtype=torch.bfloat16)
    decoder = full.model.language_model
    del full
    decoder.config.use_cache = False
    return decoder.to(dtype)


class PointerHead(nn.Module):
    """Score each option: the dot product of q(<decide>) and k(</opt>)."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.q = nn.Linear(hidden_size, POINTER_DIM)
        self.k = nn.Linear(hidden_size, POINTER_DIM)

    def forward(self, decide: torch.Tensor, options: torch.Tensor) -> torch.Tensor:
        # decide: [hidden], options: [K, hidden] -> scores: [K]
        return self.k(options) @ self.q(decide) / math.sqrt(POINTER_DIM)


class DecisionModel(nn.Module):
    def __init__(self, decoder: nn.Module, delimiter_ids: list[int], pad_id: int, adapter_dir=None):
        super().__init__()
        decoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if adapter_dir is None:
            config = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                target_modules=LORA_TARGETS,
                trainable_token_indices={"embed_tokens": delimiter_ids},
            )
            self.decoder = get_peft_model(decoder, config)
        else:
            self.decoder = PeftModel.from_pretrained(decoder, str(adapter_dir), is_trainable=True)
        self.head = PointerHead(decoder.config.hidden_size)
        self.pad_id = pad_id
        self.temperature = 1.0  # set by calibration; always 1.0 during training

    @property
    def device(self) -> torch.device:
        return self.head.q.weight.device

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def scores(self, rows: list[Row]) -> list[torch.Tensor]:
        """Run the rows as one right-padded batch. Return the option scores of each row, divided by T."""
        length = max(len(row.ids) for row in rows)
        ids = torch.full((len(rows), length), self.pad_id, dtype=torch.long)
        mask = torch.zeros((len(rows), length), dtype=torch.long)
        for i, row in enumerate(rows):
            ids[i, : len(row.ids)] = torch.tensor(row.ids)
            mask[i, : len(row.ids)] = 1
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            hidden = self.decoder(input_ids=ids.to(self.device), attention_mask=mask.to(self.device)).last_hidden_state
        hidden = hidden.float()
        return [
            self.head(hidden[i, row.decide_index], hidden[i, row.option_indices]) / self.temperature
            for i, row in enumerate(rows)
        ]

    @torch.no_grad()
    def predict_scores(self, rows: list[Row]) -> list[torch.Tensor]:
        """Option scores (divided by T) for each row, in the input order. Rows go in batches of similar length."""
        self.eval()
        order = sorted(range(len(rows)), key=lambda i: len(rows[i].ids))
        scores: list[torch.Tensor] = [torch.empty(0)] * len(rows)
        start = 0
        while start < len(order):
            end = start + 1
            while end < len(order) and (end - start + 1) * len(rows[order[end]].ids) <= ROW_TOKENS_PER_PASS:
                end += 1
            batch = order[start:end]
            for i, score in zip(batch, self.scores([rows[i] for i in batch]), strict=True):
                scores[i] = score.cpu()
            start = end
        return scores


def delimiter_report(delimiter_ids: list[int]) -> str:
    """Compare the embedding norms of the delimiter tokens with the norms of ordinary tokens."""
    decoder = load_text_decoder(torch.bfloat16)
    norms = decoder.embed_tokens.weight.float().norm(dim=-1)
    ordinary = norms[1000:]
    lines = [f"ordinary tokens: mean norm {ordinary.mean():.4f}, std {ordinary.std():.4f}, min {ordinary.min():.4f}"]
    lines += [f"token {i}: norm {norms[i]:.4f}" for i in delimiter_ids]
    return "\n".join(lines)
```

- [ ] **Step 5: Lint**

```bash
uv run ruff check src modal_app.py && uv run ruff format --check src modal_app.py
```
Expected: `All checks passed!` and `already formatted`. Do not run the module locally. It runs on Modal in Task 9.

- [ ] **Step 6: Commit**

```bash
git add src/jevstral/model.py docs/concepts/pointer-head.md docs/concepts/lora.md
git commit -m "Add the decision model

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 6: Metrics and checkpoints

**Files:**
- Create: `src/jevstral/metrics.py`, `src/jevstral/checkpoint.py`

- [ ] **Step 1: Explain** accuracy, Brier score and ECE, and why `checkpoint.save` writes to a temporary folder first. Check question: "A model always gives 0.95 and is correct 70 % of the time. Is its ECE high or low?"

- [ ] **Step 2: Write `src/jevstral/metrics.py`**

```python
"""Accuracy, Brier score and expected calibration error (ECE)."""

BINS = 10


def summarize(predictions: list[tuple[list[float], int]]) -> dict:
    """predictions: (option probabilities, index of the correct option) for each question."""
    if not predictions:
        return {"questions": 0}
    correct, brier, top = [], [], []
    for probabilities, label in predictions:
        best = max(range(len(probabilities)), key=probabilities.__getitem__)
        correct.append(best == label)
        top.append(probabilities[best])
        brier.append(sum((p - (i == label)) ** 2 for i, p in enumerate(probabilities)))
    return {
        "questions": len(predictions),
        "accuracy": sum(correct) / len(correct),
        "brier": sum(brier) / len(brier),
        "ece": expected_calibration_error(top, correct),
    }


def expected_calibration_error(confidences: list[float], correct: list[bool]) -> float:
    """Put the predictions in 10 equal bins by confidence.

    Return the mean |accuracy - confidence| of the bins, weighted by bin size.
    """
    bins: list[list[int]] = [[] for _ in range(BINS)]
    for i, confidence in enumerate(confidences):
        bins[min(int(confidence * BINS), BINS - 1)].append(i)
    error = 0.0
    for members in bins:
        if members:
            accuracy = sum(correct[i] for i in members) / len(members)
            confidence = sum(confidences[i] for i in members) / len(members)
            error += len(members) / len(confidences) * abs(accuracy - confidence)
    return error
```

- [ ] **Step 3: Write `src/jevstral/checkpoint.py`**

```python
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
```

- [ ] **Step 4: Lint**

```bash
uv run ruff check src modal_app.py && uv run ruff format --check src modal_app.py
```
Expected: `All checks passed!` and `already formatted`. Do not run the module locally. It runs on Modal in Task 9.

- [ ] **Step 5: Commit**

```bash
git add src/jevstral/metrics.py src/jevstral/checkpoint.py
git commit -m "Add metrics and checkpoints

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 7: Stages and training loop

**Files:**
- Create: `src/jevstral/stages.py`, `src/jevstral/train.py`, `docs/concepts/training-stages.md`

- [ ] **Step 1: Explain** the four stages, replay, gradient accumulation, the one-cycle schedule, gradient clipping, and how resume gets the same data order (seeded `random.Random` per epoch and per record). Walk through `run_stage` from top to bottom. Check question: "Stage 3 has batch 2 and accumulation 4. How many records does one optimizer step use?"

- [ ] **Step 2: Write `docs/concepts/training-stages.md`**

```markdown
# Training stages

We train in four stages, as Kev-4B was trained. Each stage starts from the checkpoint of the stage before it.

| Stage | Data | What the model learns |
|---|---|---|
| 1. Base | `decision-v7`: 10 public classification datasets, generated policy cases and rule structures | The format and the basic decision skill |
| 2. Dates and missing evidence | `night2`: date cases and cases with the deciding sentence removed | Date arithmetic, and a uniform answer when the state does not contain the answer |
| 3. Documents | `documents-v1`: CFPB complaint narratives, up to approximately 7,000 tokens | Long, real documents |
| 4. Skills and devtools | `hard-v1` and `devtools-v1` | Long policies, trade-offs, probability, multi-hop reasoning, judging, developer tools |

## Replay

When a model trains on new data, it can forget the old data. This is "catastrophic forgetting". To prevent it, stages 2, 3 and 4 add records from `decision-v7` (2,000, 2,000 and 4,000).

## Batch and gradient accumulation

One optimizer step uses 8 records. Long records do not fit 8 in one pass, so we use gradient accumulation: we run 2 passes of 4 records, or 4 passes of 2 records, and add the gradients before the step.

## Learning rate

- Peak: 5e-5 in stage 1, 2e-5 in stages 2, 3 and 4. Later stages change the model less.
- Schedule: one-cycle. The rate goes up for the first 10 % of the steps, then goes down.

## Other settings

- AdamW, weight decay 0.01.
- The gradient norm is clipped at 1.0. This stops one bad batch from making a large change.
- Weights in fp32, calculations in bf16 (autocast).
- Gradient checkpointing: the model calculates some activations again in the backward pass. This uses less memory and more time.

## Smoke run

Before each stage, a smoke run trains 10 steps on 64 records. It finds load and memory errors in a few minutes, before a long run starts.

## Resume

The training saves a resume checkpoint every 30 minutes. If Modal stops the job, Modal starts it again, and the training continues from that checkpoint with the same data order.
```

- [ ] **Step 3: Write `src/jevstral/stages.py`**

```python
"""The four training stages of the Kev-4B recipe."""

from dataclasses import dataclass

from .encode import LONG, SHORT, Limits

REPLAY_FILE = "v7/decision-v7/train.jsonl"

WEIGHT_DECAY = 0.01
WARMUP_FRACTION = 0.1
MAX_GRAD_NORM = 1.0
LOG_EVERY_STEPS = 10
CHECKPOINT_SECONDS = 1800


@dataclass(frozen=True)
class Stage:
    number: int
    name: str
    files: tuple[str, ...]
    epochs: int
    learning_rate: float
    batch: int  # records in one forward pass
    accumulation: int  # forward passes in one optimizer step
    replay: int  # records sampled from REPLAY_FILE
    limits: Limits
    seed: int
    minimal_pair_probability: float
    dev_files: tuple[str, ...]  # development files of all suites trained so far


DEV_V7 = "v7/decision-v7/development.jsonl"
DEV_DOCUMENTS = "documents-v1/development.jsonl"
DEV_SKILLS = ("hard-v1/development.jsonl", "devtools-v1/development.jsonl")

STAGES = {
    1: Stage(
        number=1,
        name="base",
        files=(REPLAY_FILE,),
        epochs=2,
        learning_rate=5e-5,
        batch=4,
        accumulation=2,
        replay=0,
        limits=SHORT,
        seed=2,
        minimal_pair_probability=0.25,
        dev_files=(DEV_V7,),
    ),
    2: Stage(
        number=2,
        name="dates-missing-evidence",
        files=("night2/dates_unknowable.jsonl",),
        epochs=1,
        learning_rate=2e-5,
        batch=4,
        accumulation=2,
        replay=2000,
        limits=SHORT,
        seed=2,
        minimal_pair_probability=0.0,
        dev_files=(DEV_V7,),
    ),
    3: Stage(
        number=3,
        name="documents",
        files=("documents-v1/train.jsonl",),
        epochs=1,
        learning_rate=2e-5,
        batch=2,
        accumulation=4,
        replay=2000,
        limits=LONG,
        seed=2,
        minimal_pair_probability=0.0,
        dev_files=(DEV_V7, DEV_DOCUMENTS),
    ),
    4: Stage(
        number=4,
        name="skills-devtools",
        files=("hard-v1/train.jsonl", "devtools-v1/train.jsonl"),
        epochs=1,
        learning_rate=2e-5,
        batch=2,
        accumulation=4,
        replay=4000,
        limits=LONG,
        seed=1,
        minimal_pair_probability=0.0,
        dev_files=(DEV_V7, DEV_DOCUMENTS, *DEV_SKILLS),
    ),
}
```

- [ ] **Step 4: Write `src/jevstral/train.py`**

```python
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

from . import checkpoint, suites
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
SMOKE_STEPS = 10
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
    persist()
    return metrics
```

- [ ] **Step 5: Lint**

```bash
uv run ruff check src modal_app.py && uv run ruff format --check src modal_app.py
```
Expected: `All checks passed!` and `already formatted`. Do not run the module locally. It runs on Modal in Task 9.

- [ ] **Step 6: Commit**

```bash
git add src/jevstral/stages.py src/jevstral/train.py docs/concepts/training-stages.md
git commit -m "Add stages and the training loop

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 8: Calibration

**Files:**
- Create: `src/jevstral/calibrate.py`, `docs/concepts/calibration.md`

- [ ] **Step 1: Explain** the pool, the grid search over T, and why the fit uses raw scores and not rounded probabilities. Check question: "After calibration, does accuracy on the pool change?"

- [ ] **Step 2: Write `docs/concepts/calibration.md`**

````markdown
# Calibration

Calibration makes the probabilities match how often the model is correct. It does not change the answers.

## Definition

A model is calibrated if, for all cases where it gives approximately 0.8, it is correct approximately 80 % of the time. The same must be true at each probability level.

## The problem

During training, the loss pushes the scores of correct options up on the training data. The model learns to be very sure. On new data it is correct less often, but it stays equally sure. It is overconfident.

## The method: temperature

Divide all scores by one number T before the softmax:

```
p = softmax(scores / T)
```

Example with scores [4, 1, 0]:

| T | p |
|---|---|
| 1.0 | 0.94, 0.05, 0.02 |
| 2.41 | 0.68, 0.19, 0.13 |

- T > 1 makes the probabilities less sure. T < 1 makes them more sure.
- Division by a positive number does not change the order of the scores. The answer and the accuracy stay the same.

## How we find T

1. Run the trained model on 648 questions from datasets that it never trained on.
2. Try T from 0.50 to 5.00, in steps of 0.01.
3. Keep the T with the lowest mean negative log-likelihood.

**Why held-out data?** On training data, the model is overconfident and also correct, so a fit there gives a T that is too small.

**Why negative log-likelihood?** It is a proper scoring rule. The best mean value comes only from honest probabilities.

## How we measure it: ECE

Expected calibration error:

1. Put the predictions in 10 bins by the top probability.
2. In each bin, compare the mean probability with the accuracy.
3. Calculate the mean of the differences, weighted by the size of each bin.

0 is perfect.
````

- [ ] **Step 3: Write `src/jevstral/calibrate.py`**

```python
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
```

- [ ] **Step 4: Lint**

```bash
uv run ruff check src modal_app.py && uv run ruff format --check src modal_app.py
```
Expected: `All checks passed!` and `already formatted`. Do not run the module locally. It runs on Modal in Task 9.

- [ ] **Step 5: Commit**

```bash
git add src/jevstral/calibrate.py docs/concepts/calibration.md
git commit -m "Add temperature calibration

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 9: Modal app, data and inspection

**Files:**
- Create: `modal_app.py`

- [ ] **Step 1: Explain** Modal images (`uv_sync` with the `train` group), volumes, secrets, `retries`, `--detach`, and why the local entry point sends the git commit.

- [ ] **Step 2: Write `modal_app.py`**

```python
"""Modal entry points. All data, model and GPU work runs here, not on the local computer.

modal run modal_app.py::prepare_data
modal run modal_app.py::inspect_data
modal run modal_app.py::train --stage 1 --smoke
modal run --detach modal_app.py::train --stage 1
modal run --detach modal_app.py::calibrate
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
    secrets=SECRETS,
    memory=65536,
    timeout=8 * HOUR,
    retries=modal.Retries(max_retries=3, initial_delay=0.0),
)
def train_stage(stage: int, smoke: bool, git_commit: str) -> None:
    from jevstral.stages import STAGES
    from jevstral.train import run_stage

    run_stage(STAGES[stage], DATA_DIR, RUNS_DIR, git_commit, smoke=smoke, persist=persist)


@app.function(image=image, gpu="H100", volumes=VOLUMES, secrets=SECRETS, memory=65536, timeout=2 * HOUR)
def calibrate_stage4() -> None:
    from jevstral.calibrate import run_calibration

    run_calibration(DATA_DIR, RUNS_DIR, persist=persist)


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
```

- [ ] **Step 3: Lint**

```bash
uv run ruff check src modal_app.py && uv run ruff format --check src modal_app.py
```
Expected: `All checks passed!` and `already formatted`.

- [ ] **Step 4: Commit**

```bash
git add modal_app.py
git commit -m "Add the Modal app

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 5: Log in to Modal and add the Hugging Face secret** (the owner does this; it needs a browser and a token)

```bash
uv run modal setup
uv run modal secret create huggingface HF_TOKEN=<token from huggingface.co/settings/tokens>
```
Expected: `modal setup` opens a browser and saves a token. The secret command prints `Created a new secret 'huggingface'`.

- [ ] **Step 6: Download and verify the data on Modal** (CPU, a few minutes)

```bash
uv run modal run modal_app.py::prepare_data
```
Expected: 11 lines `verified /data/...`, from `v7/decision-v7/train.jsonl` to `v9/transfer-v9/development.jsonl`.

- [ ] **Step 7: Inspect on Modal** (CPU, downloads the base model once, approximately 10 to 20 minutes)

```bash
uv run modal run modal_app.py::inspect_data
```
Expected:
- A line for ordinary token norms, and one line for each delimiter token (20, 21, 22, 23, 26). Compare them with the owner: placeholder tokens that were never trained usually have a norm very different from ordinary tokens.
- One line for each stage with row counts and skipped records. Skipped records must be a small fraction (less than 1 %) of each stage. If a stage skips more, stop and check the limits.
- One decoded example row. Check with the owner that it shows `<s>`, `<SPECIAL_20>`, the state, `<SPECIAL_21>`, the instructions, the options between `<SPECIAL_22>` and `<SPECIAL_23>`, and `<SPECIAL_26>` at the end.

- [ ] **Step 8: Write the inspection results** into `docs/concepts/encoding.md` under a new heading `## Measured values`: the norms, and the row counts for each stage. Commit.

```bash
git add docs/concepts/encoding.md
git commit -m "Record the measured token norms and row counts

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 10: Stage 1

- [ ] **Step 1: Smoke run** (H100, approximately 10 minutes with the model load)

```bash
uv run modal run modal_app.py::train --stage 1 --smoke
```
Expected:
- A JSON configuration with `"total_steps": 10` and `"trainable_parameters"` near 46,700,000 (44,564,480 LoRA + 2,097,664 head + 20,480 token rows; PEFT can count the token rows in a different way).
- A log line at step 10 with a finite loss.
- Metrics for `v7/decision-v7/development.jsonl` on 32 records.
- No errors. If CUDA runs out of memory, stop and report it. Do not change hyperparameters without the owner.

- [ ] **Step 2: Full run** (H100, approximately 2 to 3 hours)

```bash
uv run modal run --detach modal_app.py::train --stage 1
```
The command returns. Follow the job in the Modal dashboard, or read the log:

```bash
uv run modal volume get jevstral-runs main/stage1/log.jsonl - | tail -5
```
Expected: the loss goes down during the first epoch.

- [ ] **Step 3: Read the stage 1 metrics**

```bash
uv run modal volume get jevstral-runs main/stage1/metrics.json -
```
Discuss with the owner. Reference: Kev-4B after all four stages has 0.873 accuracy on `decision-v7` development. A stage 1 result far below 0.8 needs investigation before stage 2.

### Task 11: Stages 2, 3 and 4

Do these steps for each stage N in order: 2, 3, 4. Each stage needs the `final` checkpoint of the stage before it.

- [ ] **Step 1: Explain** what stage N adds (see `docs/concepts/training-stages.md`).

- [ ] **Step 2: Smoke run**

```bash
uv run modal run modal_app.py::train --stage N --smoke
```
Expected: as in Task 10 Step 1. For stages 3 and 4, the rows are long (up to 8,192 tokens), so this is the memory check for long states.

- [ ] **Step 3: Full run**

```bash
uv run modal run --detach modal_app.py::train --stage N
```
Approximate times on one H100: stage 2 under 1 hour, stage 3 2 to 4 hours, stage 4 3 to 5 hours.

- [ ] **Step 4: Read the metrics**

```bash
uv run modal volume get jevstral-runs main/stageN/metrics.json -
```
Expected: `decision-v7` development accuracy does not drop by much after each stage (replay prevents forgetting). After stage 4, compare with Kev-4B development values: `decision-v7` 0.873, `documents-v1` 0.891, `hard-v1` 0.786, `devtools-v1` 0.739.

### Task 12: Calibration and results

- [ ] **Step 1: Fit the temperature** (H100, under 30 minutes)

```bash
uv run modal run --detach modal_app.py::calibrate
uv run modal volume get jevstral-runs main/stage4/calibration.json -
```
Expected: `pool_questions` is 648. `after.nll` is lower than or equal to `before.nll`. `after.accuracy` equals `before.accuracy`. Kev-4B's T is 2.41; a value in the range 1 to 4 is normal.

- [ ] **Step 2: Add a results section to `README.md`**

Add a `## Results` section with one table: the development metrics after each stage (accuracy, Brier, ECE for each file) and the calibration result (T, NLL and ECE before and after). Use the numbers from the `metrics.json` and `calibration.json` files. Write it in STE.

- [ ] **Step 3: Commit and push**

```bash
git add README.md
git commit -m "Add training results

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push
```
