# Encoding

Encoding changes a record into token rows. Each question gets its own row.

## Row format

```
<s> <state> state text <q> instructions <opt> option 1 </opt> <opt> option 2 </opt> ... <decide>
```

- `<s>` is the Mistral BOS token. The base model always saw it at the start of its input.
- A record with three questions gives three rows. Each row has the full state.
- A question cannot see another question, because it is in a different row.

## Delimiters

| Delimiter | Token | ID |
|---|---|---|
| `<state>` | `<SPECIAL_20>` | 20 |
| `<q>` | `<SPECIAL_21>` | 21 |
| `<opt>` | `<SPECIAL_22>` | 22 |
| `</opt>` | `<SPECIAL_23>` | 23 |
| `<decide>` | `<SPECIAL_26>` | 26 |

These are unused placeholder tokens. Their embeddings are trained (see `lora.md`).

**Safety rule.** By default, the tokenizer changes the text `<SPECIAL_20>` into token 20. Then a user could write a fake option border. We tokenize all user text with `split_special_tokens=True`. Then the text `<SPECIAL_20>` stays plain text.

## Positions that the head reads

- `decide_index`: the position of `<decide>`. It comes after all options, so it can see all of them.
- `option_indices`: the position of each `</opt>`. Each one comes after the text of its option, so it holds the meaning of that option.

## Limits

| Value | Stages 1 and 2 | Stages 3 and 4 |
|---|---|---|
| State tokens (`<s>` and `<state>` included) | 384 | 7,552 |
| Row tokens | 1,024 | 8,192 |

If a record is longer than a limit, we skip it and count it.

## Option text

| Type | Options |
|---|---|
| `choice` | `key: description`, or `key` |
| `noul` | `no`, `yes` (with a description if one is given) |
| `score` | the levels, in order |
