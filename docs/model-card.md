---
language: en
library_name: peft
base_model: mistralai/Ministral-3-8B-Base-2512
base_model_relation: adapter
pipeline_tag: text-classification
tags:
  - decision-model
  - calibration
  - lora
  - multiple-choice
  - ministral
datasets:
  - jaredpalmer/kev-suites
---

# Jevstral 8B

> **Disclaimer.** Jevstral is an independent personal project. It is not affiliated with, endorsed by or sponsored by Mistral AI. "Mistral" and "Ministral" are names of Mistral AI. This model is an adapter on the open-weight model `mistralai/Ministral-3-8B-Base-2512` (Apache 2.0). Jevstral is also not affiliated with TypeSafe (the company that makes Jev) or with the author of Kev.

Jevstral is a decision model. It reads one document (the *state*) and a set of typed questions. It gives a probability for each option of each question, in one forward pass. It does not generate text.

Code: [github.com/AmirBraham/jevstral](https://github.com/AmirBraham/jevstral) (private while in progress).

## Status

Work in progress. This repository holds one folder for each finished training stage.

| Folder | Stage | Status |
|---|---|---|
| `stage1/` | Base: `decision-v7` | Done |
| `stage2/` | Dates and missing evidence: `night2` | Done |
| `stage3/` | Documents: `documents-v1` (CFPB complaints) | Done |
| `stage4/` | Skills and developer tools: `hard-v1`, `devtools-v1` | Training |
| – | Calibration (one temperature) | Not started |
| – | Decision Index 0.2.1 benchmark, accuracy and latency | Not started |

The checkpoints are not calibrated yet. Their temperature is 1.0. Do not use their probabilities as calibrated values.

## Model details

| | |
|---|---|
| Model type | Decision model: a causal language model run prefill-only, with a pointer head over the options |
| Backbone | `mistralai/Ministral-3-8B-Base-2512`, revision `d4883f9b`, text decoder only (34 layers, hidden size 4,096). Frozen. The vision encoder is not used. |
| Adapter | LoRA, rank 16, α 32, dropout 0.05, on `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` in all layers: 44,564,480 weights |
| Delimiter tokens | Five reserved tokens of the Mistral tokenizer: `<SPECIAL_20>` (state), `<SPECIAL_21>` (question), `<SPECIAL_22>` (option start), `<SPECIAL_23>` (option end), `<SPECIAL_26>` (decide). Their embedding rows are trained: 20,480 weights. In the base model these rows are all zeros, so stage 1 started them as samples of the real embedding distribution. |
| Head | Pointer head: `q` and `k` linear layers from 4,096 to 256. The score of an option is `q(h_decide) · k(h_option_end) / 16`. 2,097,664 weights. |
| Trained weights | 46,682,624 in total, approximately 0.6 % of the decoder |
| Language | English |
| Method | [Kev](https://github.com/jaredpalmer/kev), which follows ["Jev's Architecture Unmasked"](https://archerhume.com/posts/jevs-architecture-unmasked) |

### Input format

Each question becomes its own token row, with a full copy of the state:

```
<s> <state> state text <q> instructions <opt> option 1 </opt> <opt> option 2 </opt> ... <decide>
```

Question types:

| Type | Options |
|---|---|
| `choice` | One option for each named criterion |
| `noul` | `no`, `yes` |
| `score` | One option for each ordered level |

User text is tokenized with `split_special_tokens=True`. Thus text in the state cannot make a delimiter token.

## Training

All training data comes from Kev at pinned revisions (`jaredpalmer/kev-suites` at `cc4bac80`, and the Kev repository at `84847f0a`). Each file was verified against the SHA-256 in Kev's manifests.

| Stage | Data | Records | Epochs | Peak LR | Batch | Replay from `decision-v7` |
|---|---|---|---|---|---|---|
| 1 | `decision-v7` train | 12,576 | 2 | 5e-5 | 4 × 2 | – |
| 2 | `night2/dates_unknowable` | 1,425 | 1 | 2e-5 | 4 × 2 | 2,000 |
| 3 | `documents-v1` train | 5,219 | 1 | 2e-5 | 2 × 4 | 2,000 |
| 4 | `hard-v1` and `devtools-v1` train | 11,320 | 1 | 2e-5 | 2 × 4 | 4,000 |

Common settings: cross-entropy loss over the options of each question (soft targets where given), AdamW with weight decay 0.01, one-cycle schedule with 10 % warm-up, gradient clipping at norm 1.0, fp32 weights with bf16 autocast, gradient checkpointing. Training augmentation follows Kev: option shuffle, "none of the above" options (correct and wrong), irrelevant distractors, and minimal pairs in stage 1.

Hardware: one NVIDIA H100 80 GB on Modal. Stage 1 took approximately 70 minutes, stage 2 approximately 10 minutes, stage 3 approximately 40 minutes.

## Results so far

Development splits. The model did not train on these items, but they come from the same sources as the training data. These are not benchmark results.

| Development set | Questions | After stage 1 | After stage 2 | After stage 3 | Kev-4B, final |
|---|---|---|---|---|---|
| `decision-v7` | 1,468 | 0.881 | 0.884 | 0.883 | 0.873 |
| `documents-v1` | 920 | – | 0.851 | **0.899** | 0.891 |
| `hard-v1` | 1,083 | – | 0.517 | – | 0.786 |
| `devtools-v1` | 1,074 | – | 0.563 | – | 0.739 |

Values are accuracy. "–" means not measured at that stage. Kev-4B values are from its model card. `hard-v1` and `devtools-v1` are the targets of stage 4.

Expected calibration error before calibration: 0.086 on `decision-v7` and 0.065 on `documents-v1` after stage 3.

## Planned benchmark

The [Decision Index 0.2.1](https://huggingface.co/spaces/multimodalart/jev-decision-index) (38 benchmarks in five areas) already scores Jev (57.9), Kev 4B (34.6) and Laya (6.0). Jevstral will run the same suite with the public harness, and the results will report accuracy and latency.

## Intended use

- Typed decisions over English documents: classification, routing, triage, policy and eligibility checks.
- Research on decision models and calibration.

## Out-of-scope use

- Text generation, chat or summarization. The model only scores the options that it is given.
- Fully automated decisions with legal, medical, financial, employment or similar effects on people, without human review.
- Languages other than English.

## Limitations

- Not calibrated yet (see Status).
- Weak before stage 4 on temporal and numeric reasoning and on long policies with exceptions (`hard-v1` development: 0.319 and 0.450).
- Trained states have at most 7,552 tokens. Longer states are not validated.
- Some training sources (for example BANKING77) also appear in public benchmarks.

## How to load

The checkpoints use a custom head and input format. Load them with the code in the Jevstral repository (`jevstral.checkpoint.load`). Each `stageN/final/` folder has:

- `adapter/`: the LoRA weights and the delimiter rows (PEFT format).
- `head.pt`: the pointer head and the temperature.
- `config.json`: the base revision, the data revisions, the stage settings and the git commit.

## License

The license for these weights is not chosen yet. The base model is Apache 2.0. The training datasets have their own licenses; see the Kev model cards and suite manifests.

## Credits

- [Kev](https://github.com/jaredpalmer/kev): the method, the recipe and the data.
- ["Jev's Architecture Unmasked"](https://archerhume.com/posts/jevs-architecture-unmasked): the architecture analysis.
- [Mistral AI](https://mistral.ai): the open-weight base model.
