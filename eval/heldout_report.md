# Evaluation report

cases: 15 | judge: claude-sonnet-4-5

| metric | overall | EN | AR | cross-lingual |
|---|---|---|---|---|
| retrieval hit@k | 12/12 | 7/7 | 5/5 | 3/3 |
| citation correct | 10/10 | 6/6 | 4/4 | 3/3 |
| faithfulness SUPPORTED | 10/10 | 6/6 | 4/4 | 3/3 |
| abstained on out-of-corpus | 3/3 | 2/2 | 1/1 | — |
| false abstention (answerable) | 2/12 | 1/7 | 1/5 | 0/3 |

## Cases needing attention

- **held_2bfbfbcb** (heldout:v1) What is the deportation deposit and when can it be refunded?
  - FALSE abstention
- **held_ebfd9b19** (heldout:v1) ما هي الخدمة التنفيذية وكيف يمكن الاستفادة منها؟
  - FALSE abstention
