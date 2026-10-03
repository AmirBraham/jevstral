# Augmentation

Augmentation changes each training record a little, at random. It stops the model from learning shortcuts. It is used only in training.

## Two facts

1. A model learns any pattern that makes the loss lower. It cannot tell a real pattern from an accident in the data.
2. The softmax always puts 100 % on the given options. The model must choose one, also when no option is correct.

## The shortcuts and the changes

| Shortcut | Change | Probability |
|---|---|---|
| "The answer is at position X." | Shuffle the options. | Always (choice questions) |
| "Choose the best option, also when no option is correct." | Add "none of the above" and remove the correct option. The label becomes "none". | 0.10 |
| "When 'none' is there, choose it." (made by the change above) | Add "none of the above" and keep the correct option. | 0.12 |
| "All options are possible answers." | Add one unrelated option, for example "A recipe for pancakes". | 0.15 |

The three "add" changes do not occur together on one question.

## Minimal pairs (stage 1 only)

For 25 % of the records, we also add two copies of one choice question:

- Copy A: the correct option and a "none" option. The label is the correct option.
- Copy B: the same "none" option, and no correct option. The label is "none".

The two copies are the same except for one option. The only way to answer both correctly is to check if the correct option is present.

## Other rules

- Each epoch uses a new random draw. The model does not see the same version twice.
- A question with a soft target only gets a shuffle. Other changes would change the meaning of the target.
- Many wordings of "none" are used. Then the model cannot learn one wording as "the answer".
