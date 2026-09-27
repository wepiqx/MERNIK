# MTP draft duels — full ledger (PrismCoder native MTP, 2026-09-27)

Same model (`PrismCoder-9B-MERNIK-5600-SMSE`), same HE protocol
(temp 1.0 / top_p 0.95 / top_k 20 / presence 0.0 / max 2048) — only
`--spec-*` flags vary. Battery time is the speed metric; verdict must
hold if speculation is lossless.

## Results

| Run | Flags | Battery | HE pass@1 | HE+ | Empties |
|:----|:------|--------:|:---------:|:---:|:-------:|
| base | — | 60.8 min (22.3 s/task) | 74.39% (122/164) | 70.7 | 17 |
| mtp2 | draft-mtp, n-max 2 | 40.4 min (14.8 s/task) | 67.07% (110/164) | 62.8 | 26 — draft adds hesitation |
| mtp3p80 | draft-mtp, n-max 3, p-min 0.80 | 40.5 min (14.8 s/task) | 68.90% (113/164) | 65.9 | 27 |
| mtp8p80 | draft-mtp, n-max 8, p-min 0.80 | 45.6 min (16.7 s/task) | 72.56% (119/164) | 67.7 | 22 |
| mtp5ngram | draft-mtp + ngram-mod, n-max 5, p-min 0.75 | 39.9 min (14.6 s/task) | 71.34% (117/164) | 65.2 | ? |
| mtp16p80 | draft-mtp, n-max 16, p-min 0.80 (sanmai combo) | — server died on load | — | — | — VRAM wall: 16-token draft KV doesn't fit 8GB. Needs bigger card. |

## Reading

- Speed: combo (ngram-mod) wins overall ×1.53 (14.6 vs 22.3 s/task).
  Longer drafts cost more than they gain here (n8 slower than n2).
- Verdict: all draft runs sit 67–73 vs base 74.39. NOT proven
  lossless — pattern says systematic draft penalty (−2 to −7pp),
  exact size TBD by n16 + replicates. Sampling noise is ±3.6pp, so
  mtp8 (72.56) is inside noise, mtp2 (67.07) outside it.
- MTP tax (464MB pinned head vs trunk budget) vs speedup: speedup
  reclaims the tax in wall-clock; verdict cost is the open wound.

Background: SAGA.md (graft saga). Method: MERNIK slow ring.
