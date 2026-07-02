# Evaluation report

cases: 39 | judge: claude-sonnet-4-5

| metric | overall | EN | AR | cross-lingual |
|---|---|---|---|---|
| retrieval hit@k | 33/33 | 18/18 | 15/15 | 9/9 |
| citation correct | 33/33 | 18/18 | 15/15 | 9/9 |
| faithfulness SUPPORTED | 32/33 | 17/18 | 15/15 | 9/9 |
| abstained on out-of-corpus | 6/6 | 3/3 | 3/3 | — |
| false abstention (answerable) | 0/33 | 0/18 | 0/15 | 0/9 |

## Cases needing attention

- **case_8b7bd6d1** (faq:faq_cat6_en) How can I request the local transfer of a domestic employee?
  - faithfulness PARTIAL: The phone number +973 17506055 is attributed to sources [3][6], but those sources describe the 'Domestic Employee Change of Occupation' process, not the local transfer process for domestic employees
