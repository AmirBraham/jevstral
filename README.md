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
