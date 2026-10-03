# Decision model

A decision model reads a document and a set of typed questions. It gives a probability for each option of each question. It does not write text.

## Input

- **State**: the document. It is text, or JSON that we change into labelled text.
- **Questions**. Each question has one of three types:
  - `choice`: named options.
  - `noul`: yes or no.
  - `score`: ordered levels, for example low, medium, high.

Every type becomes a list of options.

## Output

For each question, the model gives one probability for each option. The probabilities of one question add up to 1.

## Difference from a chat model

A chat model does two steps:

1. **Prefill**: it reads the full prompt in one parallel pass.
2. **Decode**: it writes the answer one token at a time. Each token is one more pass.

A decision model does only step 1. A small head reads the hidden states after the prefill and gives one score for each option. A softmax changes the scores into probabilities.

Results:

- The time is approximately the time to first token of the same backbone.
- The probability is a number from the model. It is not a text claim such as "90 % sure".
- The answer is always one of the given options.

## Why probabilities are useful

Software can act on a threshold. For example: if p ≥ 0.9, do the action automatically. If not, send the case to a person. This rule is correct only if the probabilities are calibrated. See `calibration.md`.
