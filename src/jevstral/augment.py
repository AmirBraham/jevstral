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
