"""The four training stages of the Kev-4B recipe."""

from dataclasses import dataclass

from .encode import LONG, SHORT, Limits

REPLAY_FILE = "v7/decision-v7/train.jsonl"

WEIGHT_DECAY = 0.01
WARMUP_FRACTION = 0.1
MAX_GRAD_NORM = 1.0
LOG_EVERY_STEPS = 10
CHECKPOINT_SECONDS = 900


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
