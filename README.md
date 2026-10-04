# Jevstral

> **Disclaimer.** Jevstral is an independent personal project. It is not affiliated with, endorsed by or sponsored by Mistral AI. "Mistral" and "Ministral" are names of Mistral AI. This project only uses the open-weight model `mistralai/Ministral-3-8B-Base-2512`, which Mistral AI publishes under the Apache 2.0 license. Jevstral is also not affiliated with TypeSafe (the company that makes Jev) or with the author of Kev.

Jevstral is a decision model. It reads one document and a set of typed questions. It gives a probability for each option of each question. It does not generate text.

Example: for a support ticket, the question "Which team?" with the options `billing`, `shipping` and `returns` gets one probability for each option. Software can then act on a threshold, for example: if p ≥ 0.9, route the ticket automatically.

## Status

Work in progress. Stage 1 of 4 is training.

| Step | Status |
|---|---|
| Code for all stages and calibration | Done |
| Data download and verification | Done |
| Stage 1: base | Training |
| Stages 2, 3 and 4 | Not started |
| Calibration | Not started |
| Benchmarks against Kev | Not started |

## How it works

Jevstral uses the method of [Kev](https://github.com/jaredpalmer/kev), an open model that implements the design described in ["Jev's Architecture Unmasked"](https://archerhume.com/posts/jevs-architecture-unmasked). Kev uses Qwen backbones. Jevstral uses Ministral 3 8B Base.

| Part | Choice |
|---|---|
| Backbone | `mistralai/Ministral-3-8B-Base-2512`, text decoder only, frozen |
| Adapter | LoRA, rank 16, on all attention and MLP projections (44.6M weights) |
| Readout | Pointer head: scores each option against a `<decide>` token. No text generation. |
| Input | One token row for each question: `<s> <state> document <q> question <opt> option </opt> … <decide>` |
| Training | Four stages of supervised training with cross-entropy, Kev-4B recipe and Kev data |
| Calibration | One temperature, fitted on 648 questions from datasets that are not used in training |

The training data is Kev's data at pinned revisions. Each file is verified against the SHA-256 in Kev's manifests.

More detail:

- `docs/specs/`: the design.
- `docs/concepts/`: how each part works (decision model, encoding, augmentation, pointer head, LoRA, training stages, calibration).

## Setup

The local computer runs only the `modal` command. All data, model and GPU work runs on [Modal](https://modal.com).

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

## Credits

- [Kev](https://github.com/jaredpalmer/kev): the method, the training recipe and the data.
- ["Jev's Architecture Unmasked"](https://archerhume.com/posts/jevs-architecture-unmasked): the architecture analysis that Kev follows.
- [Mistral AI](https://mistral.ai): the open-weight base model, Ministral 3 8B Base (Apache 2.0).

The training datasets have their own licenses. See the Kev model cards and suite manifests.
