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
