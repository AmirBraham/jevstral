# Pointer head

The pointer head changes hidden states into one score for each option.

## The problem

After the prefill, each position has one hidden vector of 4,096 numbers. We need K scores for K options. There are two methods.

## Method 1: option slots (not used)

Take the vector at `<decide>`. Multiply it by a fixed matrix with one row for each position, for example 256 rows. Score i means "the option at position i".

Problems:

- The number of options has a maximum (the number of rows).
- Each row must learn a position. Rows for high positions get almost no training.

## Method 2: pointer (used)

Each option has its own vector, at its `</opt>` token. Compare each option with `<decide>`:

```
score_i = q(h_decide) · k(h_option_i) / sqrt(256)
p       = softmax(scores / T)
```

`q` and `k` are two linear layers from 4,096 to 256. The same two layers score all options.

Results:

- There is no maximum number of options.
- A new option needs no new weights.
- The score comes from the text of the option, not from its position.

This is the same calculation as attention: `<decide>` is the query, and each option is a key.

## Size

2 × (4,096 × 256 + 256) = 2,097,664 parameters. The head starts from random weights.
