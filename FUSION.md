# FUSION — layer-interleaved Frankensteins (2026-09-15, tri 2026-09-16)

Even blocks from model A, odd blocks from B, globals from A. Tri-mode:
per-block donor map (`--map`) + optional soup-averaged backbone (`--soup`).
Same skeleton required (verified: 427 common tensors, 0 shape mismatches
across OxCoder-9B, NeoHorse-1-9B, Ornith-1.5-9B-MTP; Ornith's 15 extra
tensors are the MTP head, ignored).

Tool: `scripts/fuse_layers.py` (streaming, ~100 MB RAM; `--a/--b/--c`,
`--map "15:b,31:c"`, `--soup "0-8" --soup-from "a,c"`).

## Specimens

| Specimen | Recipe | Status |
|:---------|:-------|:-------|
| OxOrnith-FUSED-5100 (SMAPE) | Ox even / Ornith odd, dual imatrix | **HE 39.63%** (PPL 8.26, 79 empties) — lives crippled: parents score 83–88 here. Capability is holistic, not layer-swappable. GPQA-rec 47.98% (=Neo): recognition intact, reasoning halved. |
| OxOrnith-V2 (divergent swap) | Ox base + Ornith layers 24,25,26,31 (max imatrix divergence) | GPQA-rec 49.5%, **HE 87.80%** — +48pp over blind v1 (39.63), ties Ox parent (88.41). Inverted compass CANONIZED. |
| RINIQ-M1-MERNIK-5100 (weight compass) | Ox base + Ornith blks 15,19,23,27 + Neo blk 31; SMAPE-5100 dual-imatrix (Ox+Neo max) | PPL **7.6242** (ctx1024,n64) ~ Ox-SMAPE-5100 7.567, < Neo-SMAPE-5100 7.801. GPQA-rec **51.01%**. **HE 90.85% (149/164), 7 empties — BEATS Ox-MSE-6500 parent (90.24%) at 5.0 vs 6.8 GB 🏆**. Weight compass wins the duel of the compasses. |
| RINIQ-M2-MERNIK-5100 (V2+Neo31) | Ox base + Ornith blks 24,25,26 + Neo blk 31; same quant | PPL **7.5819** — best of all fusions, ~Ox parent. GPQA-rec **50.00%**. **HE 91.46% (150/164), 6 empties — CURRENT CROWN 👑**, beats M1 (90.85%) and parent (90.24%). Imatrix compass takes the verdict column; weight compass keeps recognition (51.01%). |
| RINIQ-M3 (soup-backbone) | M1 recipe + blks 0–8 weight-averaged Ox+Neo | SMAPE-5100: PPL **7.7111**, GPQA-rec **50.00%**, **HE 86.59% (142/164), 13 empties**. Soup buried: −4.9pp vs M2. |
| RINIQ-M2-SMAPE-6500 (wrong utility, scar kept) | M2-BF16, SMAPE @6500: Q4 91 drowned + Q8 149 — barbell at big budget | PPL 7.5730, **HE 89.63% (147/164), 5 empties** — worse than M2-5100. Budget law violated by builder. |
| RINIQ-M2-MSE-6500 (the real hunt) | M2-BF16, MSE @6500: spread middle, no drowned floor | PPL **7.5263**, GPQA-rec **48.48%**, **HE 91.46% (150/164), 8 empties — EXACT TIE with M2-5100**. +1.4 GB bought fidelity (PPL −0.056) and zero capability: ceiling hit for this recipe. Still +1.2pp over Ox-MSE-6500 parent at same size. Bet log: 93.8 predicted (builder) — busted, scar kept; 95 trousers bet — void, trousers stay. Path to 95 is not more GB: own-imatrix, qwen38-quant, M4 ablation. |
| RINIQ-RR-BF16 (control) | round-robin (rebuildable one-liner) | file removed for disk |
| RINIQ-M7 (auto rank-vote) | Orn{10,14,15,16,18}, first auto-fused monster | SMAPE-5100 dual-wiki: PPL 7.6033, GPQA 49.49%, **HE 88.41% (145/164), HE+ 86.0 (−2.4, tidiest tied with V2)**. Auto compass works (beats M3/M4b, below M2/M4a). Method validated, not crowned. |
| RINIQ-M6 (Ornith base) | Orn-MTP base (all 442, MTP kept) + Ox blks 16,24,25,26,31; NO Neo — mirror test | SMAPE-5600-trunk: PPL 8.5137, GPQA 46.97%, **HE 83.54% (137/164), 23 empties**. BASE RULES: Orn globals drag −8pp vs M2; Ox blocks couldn't save it. Combo Ox-base was right from V2. Prediction 87.5 busted downward, scar kept. |
| RINIQ-M4a (Orn-center) | M2 + blk 16←Orn | SMAPE-5100: PPL 7.5898, GPQA **51.01%** (=M1), **HE 91.46% (150/164), 4 empties (best decisiveness)** — ties M2 on verdict, M1 on recognition. Orn-center works, no regression. |
| RINIQ-M4b (drop-25) | M2 with blk 25→Ox (24,26 Orn + 31 Neo stay) | SMAPE-5100: PPL **7.5743** (best), GPQA 48.48%, **HE 89.63% (147/164), 4 empties**. A/B answered: Orn-25 STRONGER than Ox-25 by 1.8pp — qwen38's low rank of 25 was a PPL-lens illusion. Ballast hypothesis dead, scar kept. |
| RINIQ-M4c (Orn-30) | M2 + blk 30←Orn | SMAPE-5100: PPL 7.5863, GPQA 47.98%, **HE 90.85% (149/164), 6 empties — ties M1**. Seam Orn-30/Neo-31 NOT poisonous; 87.5 prediction busted upward. Cheap columns lied again (both weak, verdict strong). |
| RINIQ-M5 (max-Orn: 15,19,23,24,25,26,27 + Neo 31) | kitchen sink, both compasses merged | 🔨 quant retry running (night chain died on full disk — 100%, scar; freed 18 GB) |

## Why

Attention from one finetune, blocks from another — which half carries the
soul? If the fusion speaks coherently: capability is layer-local and
swappable. If it babbles: finetunes are holistic. Either answer is a paper.

## Rules

- Same arch family only (shape gate runs first, mismatches abort).
- No MTP heads in v1 (trunk fusion first).
- Every specimen gets PPL smoke before any verdict.

## Compass finding (negative, important)

OxCoder vs Ornith layer-importance maps correlate at **0.9962** — the
imatrix recipe dominates, the finetune is invisible. An imatrix compass
for guided fusion is BLIND. Real compass: per-model damage labels
(teacher). Bonus explanation: SMAPE transfers across the family precisely
because the map is shared.

## Duel of imatrix: dual-wiki vs own-qwen38 (resolved 2026-09-17)

Same M2-BF16, same SMAPE-5100, only the lens differs.

| Lens | PPL | GPQA-rec | HE pass@1 | Empties |
|:-----|----:|:--------:|:---------:|:-------:|
| dual-wiki max(Ox,Neo) | **7.5819** | **50.00%** | 91.46% (150/164) | 6 |
| own-qwen38 single | 7.6130 | 47.47% | **92.07% (151/164) 👑** | 8 |

Lost everything, won the verdict — the most delicious outcome, as predicted.
PPL↔HE correlation on RINIQ: BROKEN (91.46@7.58 < 92.07@7.61). Capability
has its own address; the canon survives. qwen38 lens is the grail for code;
single targeted imatrix beats noisy dual-max. imatrix rank agreement was
0.9954 — the 0.5% reorder is worth +0.6pp. Small compasses steer big ships.
(The old plan above — compare importances, then (a) tri-imatrix or (b) v2 —
resolved straight to (a): full qwen38-quant won the duel. The per-layer
deltas that guided M4 live in the weight-compass section below.)

## HE+ rigor ledger (EvalPlus 80×, CPU rescore of saved samples, 2026-09-18)

| Build | HE | HE+ | Drop |
|:------|---:|:---:|:----:|
| Q38 👑 | 92.07 | **87.8** | 3.7 |
| M2 | 91.46 | 87.2 | 3.0 — tidiest code |
| M4a | 91.46 | 87.2 | 4.3 |
| M1 | 90.85 | 85.4 | 4.8 — most fragile |
| M4b | 89.63 | 84.1 | 4.9 |
| M3 | 86.59 | 83.5 | 3.1 |
| M4c | 90.85 | 85.4 | 4.8 |
| M5 (kitchen sink) | 91.46 | 84.8 | 6.1 — boldest, most fragile |
| M2-MSE-6500 | 91.46 | 85.4 | 4.8 |
| M2-SMAPE-6500 | 89.63 | 83.5 | 5.5 |
| V2 | 87.80 | 85.4 | 2.4 — tidiest code in series |
| V1 | 39.63 | 39.0 | 0.0 — crippled but honest |

Crown holds under strict tests; gap halves (1.3→0.6pp). For scale: GPT-4
drops ~13pp on HE+, mid models 15–25. Our worst drop is 4.9. Footnote: HE+
Base recomputed in evalplus harness runs ~0.5pp under our runner — drops
are apples-to-apples inside one harness.

Parent rigor (battlefield archives rescored, 2026-09-18):

| Build | HE | HE+ | Drop |
|:------|---:|:---:|:----:|
| Neo Q6_K (stock) | 82.32 | 79.9 | 2.4 |
| Neo SMAPE-5100 | 82.93 | 79.3 | 3.6 |
| Neo MSE-5100 (clean r2; r1 was the dead-server 18.3% scar) | 79.88 | 76.8 | 3.1 |
| Neo Q4_K_M (stock) | 76.83 | 76.2 | 0.6 — tidiest hands in the lab |

Law: flat stock quants drop 0.6–3.6, hybrids 3.0–6.1. Allocation boldness
costs rigor — the pie writes braver and more fragile code. Ox parents have
no saved samples; fresh batteries queued after M6.

## Utility duel: RMSE enters the ring (2026-09-18)

Zoo said RMSE beats MSE on PPL (−0.8 @1.7B) but never got a slow-ring
verdict. RINIQ-M2-6500-RMSE-Q38: PPL 7.5411, GPQA **50.00%**,
**HE 88.41% (145/164)** — takes recognition, loses verdict by 3pp.
Budget law extended to three utilities.

## MIX utility: first verdict (2026-09-19)

Per-group gain (kings-MSE + smape-base, `_scale()` normalized):
RINIQ-M2-5100-MIX — PPL 7.6001, GPQA 48.99%, **HE 89.63% (147/164)**.
Novel geometry (wide Q5, no Q8), mid-pack verdict. Doesn't beat MSE.
Stays as an option, not a crown.

## Smoothing v1: collapse (2026-09-19, scar)

in_sum2 smoothing (α=1.0, probe −22% fake-damage) + norm folding produced
a 1.7B quant with PPL **523897** — model destroyed. Fold direction/layout
(GGUF dim order?) wrong somewhere; probe optimism didn't transfer.
Parked until layout forensics + numeric block-drift verification pass.

**Update 2026-09-19 — smoothing DEAD for K-quants (scar kept).**
Drift guardrail measured 6e-08 (math exact), yet smoothed quant scored
PPL **124.36 vs 68.71 baseline**. Root cause: K-quant blocks are FLAT
(256 contiguous elements mixing channels) — per-channel scaling makes
in-block ranges *wider*, murdering small channels. Smoothing helps only
formats with channel-aligned groups (GPTQ/AWQ-style). Our quants can't
exploit it. smooth_probe.py/smooth_quant.py parked; the `_scale()` lesson
(normalize before comparing) survives them.

## RSQ positional heuristics: don't transfer (2026-09-19)

RSQ's First-N / First&Last-N (positional token importance) ported via
`llama-imatrix --chunks/--from-chunk/--in-file` merge, same M2-BF16 SMAPE-5100:

| Lens | PPL |
|:-----|----:|
| dual-wiki max(Ox,Neo) | **7.5819** |
| full qwen38 (all chunks) | 7.6130 |
| First&Last-32 merged | 7.6255 |
| First-32 | 7.6270 |

Fewer chunks = noisier importance, monotonically worse. RSQ's claim lives
in GPTQ+Hessian land; our K-quant queue wants all the data. AttnCon
(attention-weighted) still untested — needs attention dumps, parked.

## Next: MiniCPM5-2B distillate duel (queued)
`Akahsizrr/AIAAH-1-RL-DPO-iter1` (2B distillate, +10% on vendor tests, no
HE+ published) vs regular MiniCPM5-2B — both through OUR quants
(SMAPE-1700 + smape_frag-1700), same PPL/HE/HE+ protocol. Distillate arrives
as safetensors (`model.safetensors`, do-not-touch while downloading) →
GGUF convert → quant → duel. Honest constraint: cross-arch block fusion
(MiniCPM llama-arch vs Qwen) is IMPOSSIBLE with layer surgery (shape gate) —
a 2B RINIQ needs same-arch donors, not yet seen. Fusion plans wait for those.

## Next: 9B teacher on forge after 16 GB dual-channel

M2 damage sweep (200 units Q5→Q2) died at baseline on 5 GB single-channel:
Vulkan can't load the 6.2 GB Q5 (device lost, not OOM). Verdict: 9B teachers
wait for the 16 GB dual-channel upgrade (shared RAM doubles + bandwidth ×2);
main rig stays off teacher duty — forge owns it.

## Lab notes (kept)

- Norms audit on RINIQ-M1-MERNIK-5100: all 105 norm tensors intact F32,
  21×F16 are ssm_alpha smalls. The TD lesson is NOT violated — M3's PPL
  gap (7.71 vs 7.58/7.62) is soup/content, not norms.

## Weight compass (2026-09-16, no labels, no inference)

Per-tensor BF16-decoded rel-L2 across the three donors, by sub-block.
Ox–Neo are near-identical (0.002–0.007 — same bones, light code finetunes);
Ox–Orn ≈ Neo–Orn (0.02–0.09, FFN ~2× attn, growing with depth). Ornith is
the family outlier — all divergence is distance-to-Ornith.

- Backbone (all three agree, soup-safe): blks 0–8.
- Specialized (finetune soul, never average): FFN peak blks 15–19
  (Orn-dist 0.085–0.092), then 23, 27, 11, 31.
- Correction: first map attempt compared raw bytes, not weights (0.14–0.18
  garbage) — honest scar, recomputed via ml_dtypes.bfloat16 decode.
- V2's imatrix pick (24,25,26: Orn-dist 0.076–0.079) sits just below the
  weight peak — two different compasses, adjacent answers. M1 (weight pick)
  won the duel on HE (90.85% > 90.24% parent) — FUSION HF repo opens once
  M3–M4 crown the final recipe.
