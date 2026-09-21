# 🦁 ZOO — battlefield utility-metric experiments

Playground copy of ASHQ1 where the queue's quality metric is replaceable.
Goal: beat theoretical MSE — with another formula, or with measurement.

Stand: **Spark-X2.5-1.7B @1000 MiB** (quant ~2 min, PPL ~3 min),
`llama-perplexity -ngl 99 -c 1024 -b 512 --seed 7`, wikitext-2-raw.
Baseline: **MSE 71.50 ± 0.74**.

Usage: `--utility NAME [--cv-w X --linf-w Y --huber-delta D]`
(`classifier._gain`, `main.py`, origin code untouched).

## Scoreboard (1.7B @1000, ctx 1024, seed 7)

| Metric | Distribution | PPL | KLD vs F16 | Verdict |
|:-------|:-------------|:---:|:----------:|:--------|
| mse (baseline) | Q4 105 / Q5 8 / Q6 56 | 71.50 ± 0.74 | 0.3456 | PPL-theater, same fidelity as all below |
| smape_frag (SMAPE modulated by measured fragility, w=1.0) | Q4 124 / Q5 5 / Q6 14 / Q8 26 | **68.49 ± 0.70** | 0.3506 | dethroned twice: by relief in PPL, by KLD in fidelity |
| smape_frag + relief-pins (14 groups ceiling Q4) | Q4 126 / Q5 1 / Q6 16 / Q8 26 | **66.70 ± 0.68** | 0.3492 | 🥇 PPL-champion RETRACTED as fidelity claim: identical KLD to MSE. Sharpening, not law. Scar kept as pledged |
| netdmg (SMAPE + relief + Gnom mixed-effects modulator) | Q4 131 / Q5 2 / Q6 4 / Q8 32 | 70.94 ± 0.73 | — | ❌ (PPL; KLD unneeded — lost the duel already) |
| smape | Q4 122 / Q5 1 / Q6 24 / Q8 22 | 68.71 ± 0.71 | 0.3495 | same fidelity as MSE |
| Q8_0 reference | — | 60.91 ± 0.62 | 0.0139 | quasi-lossless band; the only real fidelity tier here |

## KLD verdict (the paradigm nuke)

KLD-vs-F16 battery (stock `--kl-divergence`, our ctx1024 span): Q8 0.0139 vs
ALL @1000 quants at 0.345–0.351 — identical within noise, while PPL spans
66.70–71.50. The metric works (two orders between Q8 and quants), so the
reading is unavoidable: **the entire zoo scoreboard (MSE→SMAPE→FRAG→RELIEF,
−4.8 PPL of "progress") optimized distribution sharpness, not fidelity.**
Soulfate24 called it; receipts above. Consequences:
- PPL is demoted to canary everywhere. KLD is the rank column from here on.
- Relief 66.70: PPL win with zero KLD delta = sharpening until proven otherwise.
  Champion status retracted (scar kept per pledge).
- Concentration paradigm (damage-Gini) survives as a *distributional* fact but
  its motivation needs re-proof under KLD-damage sweeps (planned): if ΔPPL ≠
  fidelity, Gini-of-ΔPPL may not predict the right utility either.
- Teacher labels (PPL-damage) are now suspect as training targets for fidelity.
  Gnom work pauses until KLD-damage labels exist.
| rmse | Q4 111 / Q5 1 / Q6 47 / Q8 10 | 70.68 ± 0.73 | 🥈 beats MSE, loses to SMAPE |
| logcosh | Q4 104 / Q5 15 / Q6 50 | 72.24 ± 0.75 | loses to MSE |
| smape_ssim (BOSS: relative lens on measured damage) | Q4 122 / Q5 5 / Q6 18 / Q8 24 | 69.52 ± 0.72 | beats plain SSIM (−0.7) and RMSE, loses to pure SMAPE. Lesson: relative-on-damage works, but fake-quant SSIM is noisier than theory |
| ssim (fake-quant) | Q4 107 / Q5 6 / Q6 48 / Q8 8 | 70.21 ± 0.72 | measured signal works — beats MSE and RMSE |
| ssim (REAL roundtrips F16→T→F16) | same, 4 tensors differ | 70.23 ± 0.72 | real damage is 4× smaller than emulation but ranks the same — emulation lied about scale, not order. Replacement caps here; only modulation (FRAG) wins |
| pw_ssim (perceptual: damage × column energy) | Q4 106 / Q5 7 / Q6 52 / Q8 4 | 72.35 ± 0.75 | ❌ LOSES to baseline. Autopsy: real-pw ≡ MSE (0 tensor diffs) — column energy IS in_sum2, the source of timp. Weighting importance by importance is circular. LPIPS dream closed |
| huber (δ=3e-4) | == mse | — | dead: monotonic in Δ, dud at this δ |
| hybrid cv/linf | == mse at any weight | — | **dead by construction** (see below) |

Flag series (`--allow-q3-or-lower`, done): RMSE-1000-Q3 **84.37** vs
MSE-1000-Q3 91.14 (RMSE wins flag-vs-flag too) — but both far worse than
no-flag (70–71): sub-4 dungeons cost more than Q8 penthouses gain.
RMSE-600-Q3: **13611.77** — total collapse, the floor is lava.

## Cross-model validation

- **Spark-4B @4000**: SMAPE 30.64 ± 0.29 vs MSE 31.25 vs Q8 30.86.
  Barbell holds on bigger dense weights (Q4 3 / Q5 2 / Q8 212 vs MSE's
  Q5 3 / Q6 6 / Q8 208). SMAPE is no 1.7B artifact.
- **Ling-3.0-tiny MoE 8B @5000** (equal size, Δ43KB): MSE 13.0681 ± 0.105 vs
  SMAPE 13.1424 ± 0.106. SMAPE loses the direction (+0.07, ~0.7σ tie) —
  no barbell magic on experts.
- **MiniCPM5-2B dense 2.6B @1700** (llama arch, 42 layers): MSE 13.3271 ± 0.103
  vs SMAPE 13.4563 ± 0.104. SMAPE loses (+0.13, ~1.2σ) ON A SMALL DENSE MODEL.
- **Ornith-1.5-9B-MTP @6500**: MSE 8.6341 👑 vs TD+SMAPE 8.7631 vs
  BU+SMAPE 8.7845. The barbell LOSES on 9B (+0.15, ~2.4σ): dumping 101
  tensors to Q4 costs more than 58 Q8 upgrades gain. Bigger model =
  angrier "meat" — even "junk" layers carry connections. TD+SMAPE's
  triple-decker (F16 core 564MB + Q8 + Q4 floor) takes second — top-down's
  first non-last finish ever.
- Tally: SMAPE wins Spark 1.7B+4B only; loses Ornith-9B, Ling-8B-MoE,
  MiniCPM5-2.6B. Size hypothesis DEAD (MiniCPM is small dense) — it was
  family/pipeline all along: both Sparks share the Qwen3.8 calibration +
  imatrix recipe. SMAPE is a Spark-pipeline weapon, not a general one.
  Relief (pure measurement, no utility prior) remains the only untested
  transfer candidate.

## Dead ends (with autopsy)

- **hybrid cv** (concentration discount `1/(1+cv_w·cv²)`): tied tensors are tied
  *by equal in_sum2* → identical timp → cv ≡ 0. No-op by construction.
- **hybrid linf** (worst-case-error boost): multiplier depends only on the tier
  pair, and the LINF table is near-geometric like MSE (×44.75 vs ×45.4) —
  near-uniform scaling, zero reorderings even at weight 100.
- **pw_ssim** (perceptual: damage × column energy): real run ≡ MSE
  (0 tensor diffs) — column energy IS in_sum2, the source of timp.
  Weighting importance by importance is circular. LPIPS dream closed.
- Lesson: only signals that vary **per group** can reorder the queue.
  Per-step transforms (mse/rmse/huber/…) only reshuffle step preferences.

## Seed invariance

FRAG/SMAPE/RMSE/MSE on seeds 7/8/21/42 give byte-identical PPLs
(68.4883×3, 68.7103×2, 70.6772×2, 71.5029×5). The harness is fully
deterministic — seed variation carries zero information. Validate by
domain slice, not by seed.

## SSIM probe (running)

`ssim_probe.py`: fake-quantizes F16 weights (affine, per-block:
Q8 8b/32, Q6 6b/256, Q5 5b/256, Q4 4b/256) and measures block-SSIM
(luminance × contrast × structure) vs original → `models/ssim_table.npz`.
Next: `--utility ssim` with gain = ΔSSIM — a *measured* per-(tensor, tier)
signal. Bonus: layer fragility map (1−SSIM@Q4) as an importance lens.

## Gnom-0.1 (the tiny net)

MLP that predicts measured PPL damage per tied group from 12 cheap features
(type one-hot, layer pos, log size, timp mean/max, SSIM fragility, energy
concentration). Symbolically 440 neurons (12→200→160→64→15→1); the combat
version is 12→8→4→1 (~150 params, honest size for ~100 labels).

Recipe (from literature: RealMLP, regularization cocktails, SNN): Mish,
dropout default 0.0 (hurts at this scale — verified: RMSE 0.18 vs 0.24),
β2=0.95, best-epoch revert, target standardization, feature jitter
(anti-parrot) + sample weights ∝ |damage|.
Status: plumbing proven (smoke R²=0.995, synthetic heldout Spearman 0.979).
Teacher (`damage.jsonl` from `scripts/group_damage_sweep.py`, pipelined
quant||PPL) DONE ×2 (v1 Q5→Q4 + v2 Q5→Q3). Best result: mixed-effects
(trunk + per-group bias) on v2, heldout Spearman 0.309 vs timp −0.10 —
first positive signal, but `netdmg` duel FAILED (70.94): 0.3 isn't enough
to steer the queue. Zero-shot on 9B detonated (−99s) → scale-invariant
features fixed the scale (−0.22..+0.09, sane) — validation pending.

## Teacher (damage sweep)

v1 (Q5→Q4 drops, 112 labels, DONE): damage range −2.07..+3.30, mean +0.12,
std 0.77; 44 negatives, 61 near-zero (|d|<0.3). Pain centers: mid-layer
attn (attn_output@11 +3.30); relief centers: early ffn_gate+up pairs.
BUT: label noise floor ≈1.0 (diff of two σ≈0.7 PPLs) sits ABOVE the effect
std — correlations with everything are ~0 (timp −0.05, |d| −0.12, size
+0.02, layer −0.09). Gnom trained on v1 learns nothing (heldout −0.33).
Autopsy: single small step → signal < noise. v2 (Q5→Q3 drops, DONE, 112
labels): range −7.81..+5.83, std 1.52 — SNR fixed, but damage is still
~unpredictable from static features (timp −0.10, Q4-vs-Q3 corr 0.091).
Conclusion: damage is NOT importance — and barely anything static.

Sensitivity sweep per tied group on 1.7B (quantize one group down, measure
ΔPPL, ~100 groups × 3 min one-time) → utility = measured damage/MB.

## Zero-shot Gnom on Ornith-9B (FAILED, mechanism found)

Ran 1.7B-trained mixed_v2 trunk on 128 Ornith tied groups (no Ornith labels
exist — pure zero-shot, biases zeroed): pred range −99..+0.38, mean −12.97.
Not relief — violent extrapolation. Mechanism: `xsd[logN]=0.22` — at 1.7B all
tensors are same-ish size, the trunk learned a huge weight on a near-constant
feature; Ornith tensors are ~1000× bigger → +15–20σ inputs → Mish detonates.
timp scale differs too (other imatrix). Verdict: Gnom-0.1 is a 1.7B-local
model BY CONSTRUCTION. Fix = scale-invariant features (logN vs model median,
timp as in-model z-score) + retrain. No new labels needed.
Automatically prices the embedding blind spot and sub-4 toxicity.

## Risks

Single seed / single model (1.7B reads raw, no jinja — weird beast) /
single domain (wiki). Any zoo champion gets re-checked on 4B before
entering the main codebase.
