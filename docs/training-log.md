# Training log

This document records the results of each training stage. For the final model, see the Results section of the README.

All values are accuracy on development splits, before calibration (T = 1). "–" means not measured after that stage.

## Accuracy after each stage

| Development set | Questions | Stage 1 | Stage 2 | Stage 3 | Stage 4 |
|---|---|---|---|---|---|
| `decision-v7` | 1,468 | 0.881 | 0.884 | 0.883 | 0.881 |
| `documents-v1` | 920 | – | 0.851 | 0.899 | 0.891 |
| `hard-v1` | 1,083 | – | 0.517 | – | 0.810 |
| `devtools-v1` | 1,074 | – | 0.563 | – | 0.710 |

Stage 2 has no development split for its own skills (dates and missing evidence).

Reference: the final Kev-4B model scores 0.873, 0.891, 0.786 and 0.739 on these four sets (from its model card).

## Expected calibration error after each stage

| Development set | Stage 1 | Stage 2 | Stage 3 | Stage 4 |
|---|---|---|---|---|
| `decision-v7` | 0.065 | 0.070 | 0.086 | 0.066 |
| `documents-v1` | – | 0.027 | 0.065 | 0.044 |
| `hard-v1` | – | 0.249 | – | 0.064 |
| `devtools-v1` | – | 0.328 | – | 0.077 |

## `hard-v1` by family

| Family | Stage 2 | Stage 4 |
|---|---|---|
| Ambiguous | 0.560 | 0.880 |
| Judge | 0.543 | 0.904 |
| Long policy | 0.450 | 0.830 |
| Multi-hop | 0.516 | 0.895 |
| Probability | 0.535 | 0.735 |
| Temporal and numeric | 0.319 | 0.621 |
| Trade-off | 0.640 | 0.780 |

## `devtools-v1` by source

| Source | Stage 2 | Stage 4 |
|---|---|---|
| Aegis | 0.663 | 0.806 |
| CodeReviewer | 0.500 | 0.727 |
| CommitPackFT | 0.585 | 0.759 |
| FlakeFlagger | 0.500 | 0.680 |
| Prompt injection (not trained) | 0.607 | 0.687 |
| When2Call (not trained) | 0.487 | 0.540 |

## Training runs

| Stage | Optimizer steps | Time on one H100 |
|---|---|---|
| 1 | 3,790 | ≈ 70 min |
| 2 | 429 | ≈ 10 min |
| 3 | 903 | ≈ 45 min |
| 4 | 1,915 | ≈ 80 min |
| Calibration | – | ≈ 5 min |

Loss curves: Weights & Biases project `jevstral` (private).

## Incidents

- **Delimiter embeddings.** The five reserved tokens that Jevstral uses as delimiters have all-zero embedding rows in Ministral 3. Stage 1 starts them as samples of the real embedding distribution. See `docs/concepts/lora.md`, section 9.
- **Stage 4, first run.** A network failure on the local computer cancelled the job at step 680, before its first resume checkpoint. Full stages now run on the deployed Modal app and save a resume checkpoint every 15 minutes.
