# GNOM active loop (RL-flavored): full plan

Status: PLAN (2026-09-27). Supervised base first (KLD labels incoming),
loop second. Co-designed with a friend — three nuances below are theirs.

## The loop

```
Gnom proposes TOP-N units  →  Teacher measures (KLD+PPL batch, GPU)
        ↑                                   ↓
   net update  ←←←←←←←←←←←←←←←  labels appended
```

1. **Propose:** acquisition = uncertainty + predicted damage. Uncertainty
   = ensemble disagreement (train K nets on bootstraps — numpy-cheap)
   or MoE-expert spread. Rank all unmeasured units, take top-N.
2. **Measure:** `group_damage_sweep.py --units tag1,tag2,...` (subset mode)
   with `--kld-base`. Batch = one GPU session, no babysitting.
3. **Update:** append labels, retrain trunk (warm-start from prev params),
   re-rank. Stop when heldout plateaus 2 rounds (anti-overthink gate,
   same as boost()).

## 1. Reward = USEFUL surprise (friend's fix for noisy-TV)

Pure surprise `|KLD_act − KLD_pred|` hacks itself: the agent camps on
chaotic, fundamentally unpredictable units (white-noise TV). Reward:

```
R = α·|KLD_act − KLD_pred| + β·KLD_act,   KLD_act capped at collapse
```

The agent hunts where it errs AND where quantization truly bites
(structure-changing, non-catastrophic damage). α/β tuned on the first
pooled KLD set; start α=1, β=0.5. Collapse zone (PPL explodes) scores 0
— measuring lava twice teaches nothing.

## 2. State space (friend converged on our features independently)

Already in the 15-dim vector: arch type one-hot, layer pos, logN,
timp stats, weight std/kurtosis/max-mean. MISSING, add:
- **outlier fraction**: share of |w| > 6σ (outliers suffer most) — one
  line in `featurize`, deterministic.
- **budget/step**: how many units already quantized (loop-time feature,
  not in .npz).

## 3. Teacher is the bottleneck → batch, don't stream

Single-unit round-trips waste GPU on load/unload. Propose TOP-N (start
N=8), teacher scores the batch in one session, net updates once per
batch. Our sweep already pipelines quant(CPU)+PPL(GPU); subset mode
makes it a batch oracle. Expected: 112-unit model in ~14 batches.

## Order of work

1. `--units` subset filter in the sweep (enabler, tiny).
2. Outlier-fraction feature in `featurize` + rebuild features.
3. Supervised KLD base (train_gnom --metric damage_kld) when labels land.
4. Ensemble disagreement acquisition + loop controller script.
5. First closed loop on 1.7B, verdict = labels-per-Spearman-point curve.
