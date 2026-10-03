# Calibration

Calibration makes the probabilities match how often the model is correct. It does not change the answers.

## Definition

A model is calibrated if, for all cases where it gives approximately 0.8, it is correct approximately 80 % of the time. The same must be true at each probability level.

## The problem

During training, the loss pushes the scores of correct options up on the training data. The model learns to be very sure. On new data it is correct less often, but it stays equally sure. It is overconfident.

## The method: temperature

Divide all scores by one number T before the softmax:

```
p = softmax(scores / T)
```

Example with scores [4, 1, 0]:

| T | p |
|---|---|
| 1.0 | 0.94, 0.05, 0.02 |
| 2.41 | 0.68, 0.19, 0.13 |

- T > 1 makes the probabilities less sure. T < 1 makes them more sure.
- Division by a positive number does not change the order of the scores. The answer and the accuracy stay the same.

## How we find T

1. Run the trained model on 648 questions from datasets that it never trained on.
2. Try T from 0.50 to 5.00, in steps of 0.01.
3. Keep the T with the lowest mean negative log-likelihood.

**Why held-out data?** On training data, the model is overconfident and also correct, so a fit there gives a T that is too small.

**Why negative log-likelihood?** It is a proper scoring rule. The best mean value comes only from honest probabilities.

## How we measure it: ECE

Expected calibration error:

1. Put the predictions in 10 bins by the top probability.
2. In each bin, compare the mean probability with the accuracy.
3. Calculate the mean of the differences, weighted by the size of each bin.

0 is perfect.
