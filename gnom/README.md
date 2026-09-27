# GNOM — learned damage allocator (Gnom-0.2 home)

Gnom predicts measured quantization damage per tied group from cheap
static features, so the queue can spend megabytes where a *learned*
model says it matters — not where a hand formula guesses.

## Layout

```
gnom/
  README.md            this file
  labels/              teacher labels (damage sweeps), one file per run:
    spark17_damage_q4.jsonl   Spark-1.7B, Q5_K base → Q4_K drops (112+BASE)
    spark17_damage_q3.jsonl   Spark-1.7B, Q5_K base → Q3_K drops (112+BASE)
    damage_4b.jsonl           Spark-4B, Q5_K base → Q4_K drops (66+BASE)
    damage_v3_q5q2.jsonl      MiniCPM5-2B, Q5_K base → Q2_K drops (168+BASE)
    damage_minicpm.jsonl      MiniCPM5-2B, second run (168+BASE)
    damage_kaggle.jsonl       tiny probe set (7, held out / skipped)
  features/            per-model feature matrices (built, deterministic):
    <model>.npz         X per tensor (15-dim, raw) + names + model stats
  models/              trained nets (.npz) + netpred.json for duels
  build_features.py    imatrix + weights → features (one model at a time)
  train_gnom.py        pool → variants → report (random + leave-one-model-out)
```

Label line format: `{"unit": "<type>@<layer>[+...]", "tensors": [...],
"ppl": float, "damage": float}` plus one `{"unit": "BASELINE", "ppl": f}`.
Damage is stored absolute; training divides by base PPL (relative).
Tier bpws in pool.json: base 5.5 (Q5_K) everywhere; drops 4.5 (Q4_K),
3.44 (Q3_K), 2.5 (v3 Q2-class, exact sub-tier unverified — curve
feature approximate, flagged).

## Features (15-dim, `tinynet.featurize`)

6 type one-hot (attn_gate, attn_qkv, attn_output, ffn_gate, ffn_up,
ffn_down) + layer_pos + log10(n_elements) + timp_mean + timp_max +
ssim_fragility + energy_concentration + 3 weight stats
(log10(std), clipped kurtosis, log10(max/mean)).

Scale-invariance (the −99 detonation lesson) is applied at TRAIN time,
per model: timp as in-model z-score, logN vs model median. Raw values
stay in the .npz; standardization lives in the trainer.

## Models (variants, same param budget ~300)

- **trunk-only** — shared MLP, all labels pooled.
- **trunk+model-bias** — mixed-effects lifted groups→models (James-Stein).
- **moe-shared** — 1 shared + 6 routed experts (`tinynet`, Flash-Next mirror).
- **timp-baseline** — must-beat reference (raw timp_sum rank).

## Exams

- **random-split Spearman** — continuity with Gnom-0.1 (old best 0.309).
- **leave-one-model-out Spearman** — THE exam: train on 2 models, test
  on the 3rd unseen (emulates zero-shot to 9B). Groups never leak.
- **Duel bar: LOMO ≥ 0.45.** Below it we don't duel (0.3 lost already:
  netdmg 70.94). Above it: `netpred.json` → `--utility netdmg` on
  Spark-4B → PPL smoke + HE verdict vs SMAPE/MSE.

## Results (Gnom-0.2, 2026-09-26, n=626 pooled)

| Variant | random-split | LOMO (minicpm/spark17/spark4b) |
|:--------|------------:|:-------------------------------|
| trunk | 0.170 | −0.056 (−0.28/−0.09/+0.20) |
| trunk+model-bias | 0.226 | **+0.060** (0.14/0.06/−0.02) |
| moe-shared | 0.228 | −0.099 |
| timp_sum baseline | 0.195 | — |

**FAIL (bar 0.45).** Mapping verified 626/626 — no plumbing bug.
Label scales explain it: spark17-q4 std 0.77 vs noise floor ~1.0,
minicpm std 0.024–0.057 (microscopic). PPL-damage at these drop sizes
is mostly noise + model-specific physics; no static feature set ranks
it across models. Narrowed finding (mailbox 2026-09-28, SMALL verified):
per-group PPL damage at one small drop (q4: std 0.7726 BELOW the ~1.0
noise floor) was unmeasurable, and static features ranked it near zero
ACROSS models (LOMO ≤ 0.06). Within-model heldout stays positive
(0.17–0.23) — and Gnom was always a modulator ON TOP of imatrix
importance, never a replacement, so "does not TRANSFER between models",
not "damage is not importance". Next: KLD-damage
labels (`--kld-base` already in the sweep; Soulfate gone, run our own
KLD battery) — same pipeline, `damage_kld` key, no rework.

## SSIM-damage (third metric, 2026-09-26)

`ssim_damage.py`: per-group (1−SSIM) at drop tier, fake-quant, CPU-only
(112 labels in ~3 min, no GPU). Key finding: rank-corr(SSIM-dmg,
PPL-dmg) = **0.041** — orthogonal views; structure ≠ sharpness.
Learnability (1.7B, random split): trunk **0.978**, timp 0.626.
Caveat: SSIM derives from the weights, so high score partly inverts
the computation — value is as a cheap SURROGATE (instant estimate vs
minutes of fake-quant), not as discovery. The decider: which rank
predicts KLD-damage / HE when those land.

Perceptual SSIM (`--perceptual`: column damage weighted by imatrix
input energy): rank-corr with plain SSIM 0.784, with PPL 0.001.
Refines within the structural view (stretches the tail 0.026 vs
0.0088) but doesn't bridge to PPL — improvement, not breakthrough.
MS-SSIM (`--ms`: geometric blend over block sizes 64/256/1024):
rank-corr with plain SSIM **0.998** — scales agree, zero new info.
Killed in 5 minutes, scar kept: on weights, coarse drift and fine
texture rank identically.
GMSD (`--gmsd`: gradient-magnitude similarity pooled by STD — weakest
link dominates): rank-corr with SSIM **0.991**, with PPL 0.036. Dead
too: deviation pooling doesn't reorder vs mean pooling. Scoreboard
4/4: all structural metrics rank together, all orthogonal to PPL.

## KLD-damage (first labels, 2026-09-27)

112 labels (1.7B Q5→Q4, two-pass run_ppl fix). Arbitration:
rank-corr(KLD,PPL) = **0.018**, rank-corr(KLD,SSIM) = 0.142 — KLD walks
its own samurai path, orthogonal to both. Three mutually blind views.
Learnability (random split, full set): modelbias **0.609** vs timp 0.104
— KLD signal an order more learnable than PPL ever was (0.23 max).
Caveat: single model, no LOMO yet. Awaiting forge Q3-KLD (curve) +
4B-KLD (LOMO).

## KLD-ready

Label loader is keyed by metric: `damage` today, `damage_kld` tomorrow
(`group_damage_sweep.py` already writes both with `--kld-base`).
No retraining-plumbing changes needed when KLD labels land.

## Why not finetune Laya

Laya is a 322M classifier in its own framework (GPU training, чужой
стек) for decision tasks. Gnom is 150–300 params of numpy on CPU,
trained in minutes, predicting a number the queue consumes directly.
Right tool, right size. (Laya honest-culture fans though — ECE scars.)
