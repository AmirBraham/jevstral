# Jevstral

> **Disclaimer.** Jevstral is an independent personal project. It is not affiliated with, endorsed by or sponsored by Mistral AI. "Mistral" and "Ministral" are names of Mistral AI. This project only uses the open-weight model `mistralai/Ministral-3-8B-Base-2512`, which Mistral AI publishes under the Apache 2.0 license. Jevstral is also not affiliated with TypeSafe, the company that makes Jev.

Model weights: [AmirBraham/jevstral-8b](https://huggingface.co/AmirBraham/jevstral-8b) on Hugging Face (use the `stage4/` folder).

Jevstral is a decision model. It reads one document and a set of typed questions. It gives a probability for each option of each question. It does not generate text.

Example: for a support ticket, the question "Which team?" with the options `billing`, `shipping` and `returns` gets one probability for each option. Software can then act on a threshold, for example: if p ≥ 0.9, route the ticket automatically.

## Status

Training and calibration are done. The [Decision Index](https://huggingface.co/spaces/multimodalart/jev-decision-index) benchmark against other decision models is in progress.

## Results

Final model: the calibrated `stage4/` checkpoint.

### Accuracy on development splits

The model did not train on these items, but they come from the same sources as the training data. These are not benchmark results.

| Development set | Questions | Accuracy |
|---|---|---|
| `decision-v7`: classification and policy decisions | 1,468 | 0.881 |
| `documents-v1`: long consumer complaints | 920 | 0.891 |
| `hard-v1`: long policies, trade-offs, multi-hop, judging | 1,083 | 0.810 |
| `devtools-v1`: code review, commits, flaky tests, safety | 1,074 | 0.710 |

### Calibration

One temperature, T = 2.04, fitted on 648 questions from datasets that no training stage uses:

| | Before | After |
|---|---|---|
| Expected calibration error | 0.141 | **0.043** |
| Log loss | 1.018 | **0.848** |
| Accuracy | 0.679 | 0.679 |

Calibration changes the confidence, not the answers.

### Latency

One request at a time, bf16 weights with LoRA merged into the base weights:

| GPU | Requests | Median | p95 |
|---|---|---|---|
| RTX PRO 6000 (the Decision Index GPU) | 100 Decision Index requests | **45.5 ms** | 1,098 ms |
| H100 | 100 short development requests (121 tokens) | 27 ms | 33 ms |
| H100 | 100 long documents (819 tokens) | 37 ms | 190 ms |

Slow requests have many questions on one long document. Against the fp32 training path on 567 development questions, the bf16 path has a mean probability difference of 0.003 and changes 3 answers.

The results of each training stage are in [docs/training-log.md](docs/training-log.md).

## How it works

Jevstral follows the decision-model design described in ["Jev's Architecture Unmasked"](https://archerhume.com/posts/jevs-architecture-unmasked): a causal language model reads the document and the question once, and a small head scores the options. The backbone is Ministral 3 8B Base.

| Part | Choice |
|---|---|
| Backbone | `mistralai/Ministral-3-8B-Base-2512`, text decoder only, frozen |
| Adapter | LoRA, rank 16, on all attention and MLP projections (44.6M weights) |
| Readout | Pointer head: scores each option against a `<decide>` token. No text generation. |
| Input | One token row for each question: `<s> <state> document <q> question <opt> option </opt> … <decide>` |
| Training | Four stages of supervised training with cross-entropy |
| Calibration | One temperature, fitted on 648 questions from datasets that are not used in training |

The training data comes from public decision suites at pinned revisions (see Credits). Each file is verified against the SHA-256 in its suite manifest.

More detail:

- `docs/specs/`: the design.
- `docs/concepts/`: how each part works (decision model, encoding, augmentation, pointer head, LoRA, training stages, calibration).
- `docs/training-log.md`: the results of each training stage.

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
uv run modal deploy modal_app.py
uv run python modal_app.py train 1
```

`modal deploy` puts the app on Modal. `python modal_app.py train N` starts the stage on the deployed app and returns at once. The job does not depend on the local computer or its network. Deploy again after each code change.

Do stages 2, 3 and 4 in the same way. Then fit the temperature:

```
uv run python modal_app.py calibrate
```

## Back up to Hugging Face

The `huggingface` secret needs a token with write access. After a stage finishes:

```
uv run modal run modal_app.py::publish --stage 1
```

This uploads `final/`, `metrics.json` and `log.jsonl` of the stage to the folder `stage1/` of the repository `<user>/jevstral-8b`. The job creates the repository as private if it does not exist.

## Monitor

Training sends the loss, learning rate and gradient norm to the Weights & Biases project `jevstral`, and the development metrics at the end of each stage. Stage 1 trained before this was added. To send its log to W&B:

```
uv run modal run modal_app.py::track_finished_stage --stage 1
```

## License

Apache 2.0 for the code and the weights. See [LICENSE](LICENSE). The training datasets have their own licenses; see the dataset cards and the suite manifests.

## Credits

- [Kev](https://github.com/jaredpalmer/kev): the method, the training recipe and the data. Jevstral is a port of Kev to a Mistral backbone. It is not affiliated with the author of Kev.
- ["Jev's Architecture Unmasked"](https://archerhume.com/posts/jevs-architecture-unmasked): the architecture analysis behind the design.
- [Mistral AI](https://mistral.ai): the open-weight base model, Ministral 3 8B Base (Apache 2.0).

