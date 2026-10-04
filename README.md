# Jevstral

Jevstral is a decision model. It reads one document and a set of typed questions. It gives a probability for each option of each question. It does not generate text.

Jevstral uses the method of [Kev](https://github.com/jaredpalmer/kev) on the backbone `mistralai/Ministral-3-8B-Base-2512`.

## Setup

The local computer runs only the `modal` command. All data, model and GPU work runs on Modal.

```
uv sync
uv run modal setup
uv run modal secret create huggingface HF_TOKEN=<your token>
uv run modal secret create wandb WANDB_API_KEY=<your key>
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

## Back up to Hugging Face

The `huggingface` secret needs a token with write access. After a stage finishes:

```
uv run modal run modal_app.py::publish --stage 1
```

This uploads `final/`, `metrics.json` and `log.jsonl` of the stage to the folder `stage1/` of the private repository `<user>/jevstral-8b`.

## Monitor

Training sends the loss, learning rate and gradient norm to the Weights & Biases project `jevstral`, and the development metrics at the end of each stage. Stage 1 trained before this was added. To send its log to W&B:

```
uv run modal run modal_app.py::track_finished_stage --stage 1
```

## Documents

- `docs/specs/`: design.
- `docs/concepts/`: how each part works.
