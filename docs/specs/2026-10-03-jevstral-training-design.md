# Jevstral training design

Date: 2026-10-03
Status: draft for review

## 1. Goal

Jevstral is a decision model. It reads one document (the state) and a set of typed questions. It gives a probability for each option of each question. It does not generate text.

Jevstral uses the method of Kev (github.com/jaredpalmer/kev). Kev uses Qwen backbones. Jevstral uses `mistralai/Ministral-3-8B-Base-2512`.

This spec covers training only:

1. Load and verify the training data.
2. Train the four Kev stages on Modal.
3. Fit the calibration temperature.

### 1.1 Non-goals

- Serving, an HTTP API and a prefix cache.
- Benchmarks other than the dev-set metrics in section 9.
- Image states. The vision encoder is not loaded.
- Unit tests and CI.

### 1.2 Learning goal

The owner of this project uses it to learn how Kev works. Each module has a short concept note in `docs/concepts/`. The notes use ASD-STE100 Simplified Technical English.

## 2. Background

A decision model replaces text generation with a readout. The model reads the input in one forward pass. A small head then reads the hidden states and gives one score per option. A softmax changes the scores into probabilities.

The design follows "Jev's Architecture Unmasked" (archerhume.com, 2026-09-17), as Kev does:

| Essay idea | Jevstral |
|---|---|
| Read out probabilities, do not generate | Pointer head on the hidden states |
| Share the state, isolate the questions | One row for each question (section 5). Sharing is a serving feature, out of scope. |
| Causal backbone | Ministral 3 8B Base |
| Options interact before the choice | All options come before `<decide>` |
| Train the distribution, then calibrate | Cross-entropy loss, then one temperature |
| Sparse experts | Not used. The backbone is dense. |

## 3. Repository layout

```
jevstral/
├── pyproject.toml
├── modal_app.py            Modal entry points
├── src/jevstral/
│   ├── records.py          Kev record format to typed dataclasses
│   ├── suites.py           download pinned data, verify SHA-256
│   ├── augment.py          training-only record changes
│   ├── encode.py           record to token rows
│   ├── model.py            backbone, LoRA, delimiter embeddings, pointer head
│   ├── train.py            training loop for one stage
│   ├── stages.py           configuration of the four stages
│   ├── metrics.py          accuracy, Brier score, ECE
│   ├── calibrate.py        temperature fit
│   └── checkpoint.py       save and load
└── docs/
    ├── concepts/
    └── specs/
```

Rules:

- Only `model.py` knows the backbone.
- Only `modal_app.py` knows Modal.
- `records.py`, `augment.py`, `encode.py` and `metrics.py` do not use a GPU.

## 4. Data

### 4.1 Sources

All data comes from Kev. We do not generate or label new data.

| Name | Location | Use |
|---|---|---|
| `kev-suites` | Hugging Face dataset `jaredpalmer/kev-suites`, revision `cc4bac803e73112689ec327ffa481c519cbc7a05` | Most training and calibration files |
| `kev-repo` | GitHub `jaredpalmer/kev`, commit `84847f0a883d900f7de5b7a57eaa341ca7f9a6b4`, folder `evals/` | Manifests, and files that are not in `kev-suites` |

| Suite | Files |
|---|---|
| `v7/decision-v7` | `train.jsonl`, `development.jsonl` |
| `night2` | `dates_unknowable.jsonl` |
| `documents-v1` | `train.jsonl`, `development.jsonl` |
| `hard-v1` | `train.jsonl`, `development.jsonl` |
| `devtools-v1` | `train.jsonl`, `development.jsonl` |
| `round3/transfer-r3` | `calibration.jsonl` |
| `v9/transfer-v9` | `development.jsonl` |

`suites.py` reads each file from `kev-suites` first. If the file is not there, it reads the file from `kev-repo`.

### 4.2 Verification

Each suite folder in `kev-repo` has a `manifest.json`. The manifest gives the SHA-256 and the record count of each file.

`suites.py` does these steps for each file:

1. Download the file.
2. Calculate its SHA-256.
3. Compare the SHA-256 and the record count with the manifest.
4. If a value is different, stop with an error that names the file.

The verified files go to the Modal volume `jevstral-data`.

### 4.3 Record format

Jevstral uses the Kev record format without changes:

```json
{"state": "text, or a JSON object or array",
 "questions": {"<id>": {"type": "choice | noul | score",
                        "instructions": "...",
                        "criteria": "...",
                        "label": "...",
                        "target": "optional soft target"}}}
```

`records.py` changes each question into one list of options:

| Type | Options | Label |
|---|---|---|
| `choice` | One option for each criteria key. Text: `key: description`, or `key` if there is no description. | Index of the label key |
| `noul` | `no`, `yes`. A description from `criteria.false` or `criteria.true` is added if it exists. | 0 for false, 1 for true |
| `score` | One option for each level, in order | Level index |

A JSON state becomes labelled text with the rules of `render()` in `kev/api.py`: one `key: value` line for each field, and `- item` lines for a list.

Some records have a `target`. A `target` is a soft label, for example a uniform distribution for a question that the state cannot answer. The loss uses the target when it exists (section 7.2).

`records.py` stops with an error if a label is not one of the options.

## 5. Encoding

Each question becomes one token row:

```
<s> <state> state text <q> instructions <opt> option 1 </opt> <opt> option 2 </opt> ... <decide>
```

A record with three questions gives three rows. Each row has the full state. A question cannot see another question.

`<s>` is the Mistral BOS token. The tokenizer adds it by default, so normal model inputs start with it.

### 5.1 Delimiter tokens

The five delimiters are tokens from the Mistral tokenizer, not text. User text cannot make these tokens. This stops fake options in the state or in the option text.

We use five reserved placeholder tokens. Mistral gives them no name and no role. We did not verify that the base model never saw them:

| Delimiter | Token | ID |
|---|---|---|
| `<state>` | `<SPECIAL_20>` | 20 |
| `<q>` | `<SPECIAL_21>` | 21 |
| `<opt>` | `<SPECIAL_22>` | 22 |
| `</opt>` | `<SPECIAL_23>` | 23 |
| `<decide>` | `<SPECIAL_26>` | 26 |

By default, the tokenizer changes the text `<SPECIAL_20>` into token 20. We tokenize all user text with `split_special_tokens=True`. Then that text stays plain text.

We always train the five embedding rows of these tokens (section 6.3). The `inspect_data` job measures their embedding norms and the norms of ordinary tokens. This measurement is for learning only. It does not change the training.

### 5.2 Limits

| Value | Stages 1 and 2 | Stages 3 and 4 |
|---|---|---|
| State tokens, `<s>` and `<state>` included | 384 | 7,552 |
| Row tokens (state and one question) | 1,024 | 8,192 |

These are the Kev limits. If a record is longer than a limit, we skip the full record. Kev does the same in training. The training log gives the number of skipped records.

Evaluation and calibration use the long limits for all files.

### 5.3 Positions in the row

`encode.py` returns these values for each row:

- `ids`: the token IDs.
- `decide_index`: the position of `<decide>`.
- `option_end_indices`: the position of each `</opt>`.
- `label` or `target`.

## 6. Model

### 6.1 Backbone

- Model: `mistralai/Ministral-3-8B-Base-2512`, pinned to one revision.
- We load only the text decoder: 34 layers, hidden size 4,096. We do not use the vision encoder.
- To load the decoder, we load the full `Mistral3ForConditionalGeneration` checkpoint and keep `model.language_model`. Do not use `Ministral3Model.from_pretrained` on this repository. It does not map the weight names, and it gives random weights without an error.
- All backbone weights are frozen.
- Training uses fp32 weights with bf16 autocast. Gradient checkpointing is on.
- GPU: one H100 80 GB.

### 6.2 LoRA

| Setting | Value |
|---|---|
| Rank | 16 |
| Alpha | 32 |
| Dropout | 0.05 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` in all layers |

### 6.3 Delimiter embeddings

The five embedding rows of the delimiter tokens are trainable (PEFT `trainable_token_indices`). All other embedding rows stay frozen.

The original five rows are all zeros (measured by `inspect_data`). Stage 1 starts each row as a sample from a normal distribution with the mean and full covariance of the trained embedding rows. Rows with norm 0 are not used for the mean and covariance.

### 6.4 Pointer head

```
h_d   = hidden state at <decide>
h_i   = hidden state at the </opt> of option i
s_i   = q(h_d) · k(h_i) / sqrt(256)
p     = softmax(s / T)
```

`q` and `k` are linear layers from 4,096 to 256. During training, T is 1. The head starts from random weights.

## 7. Training

### 7.1 Augmentation

`augment.py` changes training records only. It uses a new random draw in each epoch. The seed is fixed.

| Change | Probability |
|---|---|
| Shuffle the options of each `choice` question | Always |
| Add "none of the above" and remove the correct option. The label becomes "none of the above". | 0.10 |
| Add "none of the above" and keep the correct option | 0.12 |
| Add one irrelevant distractor option | 0.15 |
| Stage 1 only: add a minimal pair for a `choice` record. One copy has "none of the above" and the correct option. One copy has "none of the above" and no correct option. | 0.25 |

The three "add" changes do not occur together on one question. The augmentation lists (the "none" option texts and the distractors) come from `kev/data.py` at the pinned Kev commit.

### 7.2 Loss

- Without a target: cross-entropy between the option probabilities and the label.
- With a target: cross-entropy between the option probabilities and the target distribution.

The loss is the mean over the questions in the batch.

### 7.3 Optimizer

| Setting | Value |
|---|---|
| Optimizer | AdamW |
| Weight decay | 0.01 on the LoRA weights, the head and the delimiter rows |
| Schedule | One-cycle, 10 % warm-up |
| Gradient clip | Global norm 1.0 |

### 7.4 Stages

Each stage starts from the checkpoint of the previous stage. Stage 1 starts from the base model.

| Stage | Data | Records | Epochs | Peak LR | Batch | Replay from `decision-v7` train | Seed |
|---|---|---|---|---|---|---|---|
| 1. Base | `decision-v7` train | 12,576 | 2 | 5e-5 | 4 × 2 accumulation | – | 2 |
| 2. Dates and missing evidence | `night2/dates_unknowable` | 1,425 | 1 | 2e-5 | 4 × 2 | 2,000 | 2 |
| 3. Documents | `documents-v1` train | 5,219 | 1 | 2e-5 | 2 × 4 | 2,000 | 2 |
| 4. Skills and devtools | `hard-v1` train and `devtools-v1` train | 11,320 | 1 | 2e-5 | 2 × 4 | 4,000 | 1 |

The batch unit is one record. A record can have more than one question row.

The Kev-4B card gives the seed for stages 1 and 4 only. We use seed 2 for stages 2 and 3.

## 8. Calibration

`calibrate.py` fits one temperature T after stage 4.

### 8.1 Pool

The pool has 648 questions. The model does not train on these datasets.

- 448 questions from `round3/transfer-r3` `calibration.jsonl`. Keep only these sources: `composition_holdout`, `emotion`, `legacy_holdout`, `mmlu`, `paws`, `qnli`, `sciq`, `tweet_offensive`.
- 200 questions from `v9/transfer-v9` `development.jsonl`. Keep only the source `mmlu_pro`.

If the filtered counts are not 448 and 200, stop with an error.

### 8.2 Fit

1. Run the stage 4 model on the pool with T = 1. Keep the option scores.
2. Find the T that gives the lowest mean negative log-likelihood. Search T in [0.5, 5.0].
3. Save T in the checkpoint.

T does not change the answer with the highest probability. It changes only the probabilities.

## 9. Metrics

`metrics.py` calculates these values:

- Accuracy: the option with the highest probability is the label.
- Brier score.
- Expected calibration error (ECE) with 10 equal bins on the highest probability.

After each stage, the model runs on the `development.jsonl` file of each suite trained so far. The log gives the metrics for each suite.

## 10. Modal

### 10.1 Resources

| Resource | Name | Content |
|---|---|---|
| Volume | `jevstral-hf-cache` | Base model weights, approximately 17 GB |
| Volume | `jevstral-data` | Verified data files |
| Volume | `jevstral-runs` | Checkpoints, logs and metrics |
| Secret | `huggingface` | `HF_TOKEN` |
| GPU | H100 80 GB | Training and calibration |

### 10.2 Functions

| Function | GPU | Work |
|---|---|---|
| `prepare_data` | No | Download and verify all files (section 4) |
| `inspect_data` | No | Measure the delimiter embedding norms. Encode all stages and report row counts and skipped records. |
| `train_stage` | Yes | Train one stage. Arguments: stage number, `smoke` flag, git commit. |
| `calibrate_stage4` | Yes | Fit T (section 8) |

The local entry points `train` and `calibrate` start `train_stage` and `calibrate_stage4`. `train` sends the local git commit to the job.

The local computer runs only the `modal` command. It does not install torch, and it does not download the model or the data.

### 10.3 Smoke run

Before each stage, run `train_stage` with `smoke=True`. The smoke run uses 64 records and 10 optimizer steps. It must finish without errors before the full stage starts.

### 10.4 Commands

```
modal run modal_app.py::prepare_data
modal run modal_app.py::inspect_data
modal run modal_app.py::train --stage 1 --smoke
modal run --detach modal_app.py::train --stage 1
...
modal run --detach modal_app.py::calibrate
```

## 11. Checkpoints and errors

### 11.1 Checkpoint content

Each checkpoint folder has these files:

- `adapter/`: the LoRA weights (PEFT format).
- `head.pt`: the pointer head weights, the delimiter embedding rows and T.
- `config.json`: the base model revision, the data revision, the stage configuration, the seed and the git commit.
- `resume.pt`: the optimizer state, the scheduler state and the position (epoch, batch, step). Only the intermediate checkpoint has this file.

### 11.2 Error handling

| Condition | Action |
|---|---|
| SHA-256 or record count is different from the manifest | Stop. Give the file name. |
| A calibration pool count is not 448 or 200 | Stop. Give the file name. |
| A label is not one of the options | Stop. Give the record ID. |
| Loss or gradient norm is not finite | Stop. Give the step. |
| The record is too long (section 5.2) | Skip the record. Count it in the log. |
| The checkpoint has a different base model revision | Stop. Give the two revisions. |
| Modal stops the job | Resume from the last intermediate checkpoint. |

The training saves an intermediate checkpoint every 30 minutes. `train_stage` resumes from the last intermediate checkpoint of that stage if one exists.

### 11.3 Log

Each stage writes `log.jsonl` to its run folder. Each line has the step, the loss, the learning rate and the gradient norm.

## 12. Estimated cost

The total is 8 to 12 H100 hours, approximately 30 to 50 USD on Modal. Stages 3 and 4 use the most time because their states are long.

## 13. Concept notes

We write these notes while we build the modules:

| Note | Module |
|---|---|
| `decision-model.md` | All |
| `jev-architecture.md` | All. It compares the essay with Jevstral. |
| `encoding.md` | `encode.py` |
| `augmentation.md` | `augment.py` |
| `pointer-head.md` | `model.py` |
| `lora.md` | `model.py` |
| `training-stages.md` | `train.py`, `stages.py` |
| `calibration.md` | `calibrate.py` |

## 14. Later work

These items are not in this spec:

- Benchmarks against Kev 1.0 (transfer-v4, breadth-v1).
- Long-document check on `longdoc-v1`.
- Serving with a shared state cache.
- French data and multilingual evaluation.
- A test for yes/no label bias.
