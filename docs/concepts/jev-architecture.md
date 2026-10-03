# Jev architecture and Jevstral

The essay "Jev's Architecture Unmasked" (archerhume.com, 2026-09-17) gives seven ideas about how TypeSafe's Jev works. Kev follows most of them. This table compares the ideas with Jevstral.

| # | Idea | Jevstral |
|---|---|---|
| 1 | Read out probabilities. Do not generate text. | Pointer head on the hidden states (`pointer-head.md`). |
| 2 | Read the state once. Isolate the questions. | Each question has its own row, so isolation is exact (`encoding.md`). Reading the state once is a serving feature. It is not in scope. |
| 3 | Use a causal backbone. | Ministral 3 8B Base, a left-to-right decoder. |
| 4 | Let the options interact before the choice. | All options come before `<decide>`, so `<decide>` sees the full list. |
| 5 | Train the distribution, then calibrate. | Cross-entropy loss (a proper scoring rule), then one temperature (`calibration.md`). |
| 6 | Use sparse experts. | Not used. The backbone is dense. The other ideas do not need sparse experts. |
| 7 | Run the question branches as one batch. | A serving feature. It is not in scope. |

## Two open questions in the essay

- **Option slots or a pointer?** Kev and Jevstral use a pointer. See `pointer-head.md`.
- **How do option borders resist fake text?** The delimiters are special tokens. User text cannot make them. See `encoding.md`.
