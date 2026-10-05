---
license: apache-2.0
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

> **Disclaimer.** Jevstral is an independent personal project. It is not affiliated with, endorsed by or sponsored by Mistral AI. "Mistral" and "Ministral" are names of Mistral AI. This model is an adapter on the open-weight model `mistralai/Ministral-3-8B-Base-2512` (Apache 2.0). Jevstral is also not affiliated with TypeSafe, the company that makes Jev.

Jevstral is a decision model. It reads one document (the *state*) and a set of typed questions. It gives a probability for each option of each question, in one forward pass. It does not generate text.

Code: [github.com/AmirBraham/jevstral](https://github.com/AmirBraham/jevstral).

## Status

Training and calibration are done. **Use the `stage4/` folder**: it is the final, calibrated model (T = 2.04). The folders `stage1/` to `stage3/` are intermediate checkpoints, kept for reproducibility. They are not calibrated.

Decision Index 0.2.1: **34.3** (rank 36 of 74), median latency **28.3 ms** on one RTX PRO 6000. See Benchmark.

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
| Design | Follows ["Jev's Architecture Unmasked"](https://archerhume.com/posts/jevs-architecture-unmasked) |

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

All training data comes from public decision suites at pinned revisions: the dataset `jaredpalmer/kev-suites` at `cc4bac80` and its source repository at `84847f0a`. Each file was verified against the SHA-256 in its suite manifest.

| Stage | Data | Records | Epochs | Peak LR | Batch | Replay from `decision-v7` |
|---|---|---|---|---|---|---|
| 1 | `decision-v7` train | 12,576 | 2 | 5e-5 | 4 × 2 | – |
| 2 | `night2/dates_unknowable` | 1,425 | 1 | 2e-5 | 4 × 2 | 2,000 |
| 3 | `documents-v1` train | 5,219 | 1 | 2e-5 | 2 × 4 | 2,000 |
| 4 | `hard-v1` and `devtools-v1` train | 11,320 | 1 | 2e-5 | 2 × 4 | 4,000 |

Common settings: cross-entropy loss over the options of each question (soft targets where given), AdamW with weight decay 0.01, one-cycle schedule with 10 % warm-up, gradient clipping at norm 1.0, fp32 weights with bf16 autocast, gradient checkpointing. Training augmentation: option shuffle, "none of the above" options (correct and wrong), irrelevant distractors, and minimal pairs in stage 1.

Hardware: one NVIDIA H100 80 GB on Modal. Stage 1 took approximately 70 minutes, stage 2 approximately 10 minutes, stage 3 approximately 45 minutes, stage 4 approximately 80 minutes.

## Results

### Accuracy on development splits

The model did not train on these items, but they come from the same sources as the training data. These are not benchmark results.

| Development set | Questions | Accuracy |
|---|---|---|
| `decision-v7`: classification and policy decisions | 1,468 | 0.881 |
| `documents-v1`: long consumer complaints | 920 | 0.891 |
| `hard-v1`: long policies, trade-offs, multi-hop, judging | 1,083 | 0.810 |
| `devtools-v1`: code review, commits, flaky tests, safety | 1,074 | 0.710 |

### Calibration

One temperature, T = 2.04, fitted on 648 questions from datasets that no training stage uses (`transfer-r3` calibration split, 8 sources, and 200 MMLU-Pro questions from `transfer-v9` development):

| | Before | After |
|---|---|---|
| Expected calibration error | 0.141 | **0.043** |
| Log loss | 1.018 | **0.848** |
| Brier score | 0.454 | **0.418** |
| Accuracy | 0.679 | 0.679 |

### Latency

One request at a time, bf16 weights with LoRA merged into the base weights:

| GPU | Requests | Median | p95 |
|---|---|---|---|
| RTX PRO 6000 | 100 Decision Index requests | **45.5 ms** | 1,098 ms |
| H100 | 100 short development requests (121 tokens) | 27 ms | 33 ms |
| H100 | 100 long documents (819 tokens) | 37 ms | 190 ms |

The results of each training stage are in the [training log](https://github.com/AmirBraham/jevstral/blob/main/docs/training-log.md).

## Benchmark

[Decision Index 0.2.1](https://huggingface.co/spaces/multimodalart/jev-decision-index): 38 benchmarks in five areas, chance-corrected (0 is random guessing, 100 is perfect). Full run of the suite (150,317 requests) with the [public harness](https://github.com/apolinario/decision-index) on one NVIDIA RTX PRO 6000. All requests were answered, with no errors.

| Area (weight) | Jevstral 8B | Kev 4B | Kev 9B | Jev |
|---|---|---|---|---|
| Knowledge and reasoning (25.8 %) | **24.5** | 22.9 | 26.2 | 51.4 |
| Language understanding (25.8 %) | **34.6** | 35.3 | 41.7 | 62.0 |
| Retrieval and classification (20.0 %) | **39.3** | 41.0 | 43.7 | 55.4 |
| Tools and automation (18.3 %) | **52.9** | 52.6 | 54.5 | 75.1 |
| Arts and human taste (10 %) | **14.8** | 17.9 | 22.4 | 37.7 |
| **Decision Index** | **34.3** | 34.6 | 38.5 | 57.9 |

Rank: 36 of 74. Other models: public leaderboard, data of 2026-10-01.

Latency, one request at a time: median **28.3 ms**, p95 185 ms, mean 79 ms (bf16 weights, LoRA merged, the document read once for all its questions).

Notes: BANKING77 is in the Decision Index and in the training data. 200 MMLU-Pro questions were used to fit the temperature; the temperature does not change the answers. Jev is a hosted API, so its latency includes the network.

Full results: [`docs/benchmark/`](https://github.com/AmirBraham/jevstral/tree/main/docs/benchmark) in the code repository.

## Intended use

- Typed decisions over English documents: classification, routing, triage, policy and eligibility checks.
- Research on decision models and calibration.

## Out-of-scope use

- Text generation, chat or summarization. The model only scores the options that it is given.
- Fully automated decisions with legal, medical, financial, employment or similar effects on people, without human review.
- Languages other than English.

## Limitations

- One temperature for all tasks. Some tasks can still be over- or underconfident.
- Temporal and numeric reasoning is the weakest skill (`hard-v1` development: 0.621).
- Trained states have at most 7,552 tokens. Longer states are not validated.
- Some training sources (for example BANKING77) also appear in public benchmarks.

## How to load

The checkpoints use a custom head and input format. Load them with the code in the Jevstral repository (`jevstral.checkpoint.load`). Each `stageN/final/` folder has:

- `adapter/`: the LoRA weights and the delimiter rows (PEFT format).
- `head.pt`: the pointer head and the temperature.
- `config.json`: the base revision, the data revisions, the stage settings and the git commit.

## License

The weights (LoRA adapter, delimiter rows and pointer head) and the code are released under the [Apache 2.0 license](https://github.com/AmirBraham/jevstral/blob/main/LICENSE). The base model, `mistralai/Ministral-3-8B-Base-2512`, is also Apache 2.0. The training datasets have their own licenses; see the dataset cards and the suite manifests.

## Credits

- [Kev](https://github.com/jaredpalmer/kev): the method, the recipe and the data. Jevstral is a port of Kev to a Mistral backbone. It is not affiliated with the author of Kev.
- ["Jev's Architecture Unmasked"](https://archerhume.com/posts/jevs-architecture-unmasked): the architecture analysis.
- [Mistral AI](https://mistral.ai): the open-weight base model.
