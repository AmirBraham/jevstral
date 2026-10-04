# Training stages

We train in four stages, as Kev-4B was trained. Each stage starts from the checkpoint of the stage before it.

| Stage | Data | What the model learns |
|---|---|---|
| 1. Base | `decision-v7`: 10 public classification datasets, generated policy cases and rule structures | The format and the basic decision skill |
| 2. Dates and missing evidence | `night2`: date cases and cases with the deciding sentence removed | Date arithmetic, and a uniform answer when the state does not contain the answer |
| 3. Documents | `documents-v1`: CFPB complaint narratives, up to approximately 7,000 tokens | Long, real documents |
| 4. Skills and devtools | `hard-v1` and `devtools-v1` | Long policies, trade-offs, probability, multi-hop reasoning, judging, developer tools |

## Replay

When a model trains on new data, it can forget the old data. This is "catastrophic forgetting". To prevent it, stages 2, 3 and 4 add records from `decision-v7` (2,000, 2,000 and 4,000).

## Batch and gradient accumulation

One optimizer step uses 8 records. Long records do not fit 8 in one pass, so we use gradient accumulation: we run 2 passes of 4 records, or 4 passes of 2 records, and add the gradients before the step.

## Learning rate

- Peak: 5e-5 in stage 1, 2e-5 in stages 2, 3 and 4. Later stages change the model less.
- Schedule: one-cycle. The rate goes up for the first 10 % of the steps, then goes down.

## Other settings

- AdamW, weight decay 0.01.
- The gradient norm is clipped at 1.0. This stops one bad batch from making a large change.
- Weights in fp32, calculations in bf16 (autocast).
- Gradient checkpointing: the model calculates some activations again in the backward pass. This uses less memory and more time.

## Smoke run

Before each stage, a smoke run trains 20 steps on 64 records. It finds load and memory errors in a few minutes, before a long run starts.

## Resume

The training saves a resume checkpoint every 30 minutes. If Modal stops the job, Modal starts it again, and the training continues from that checkpoint with the same data order.
