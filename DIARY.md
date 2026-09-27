# SESSION DIARY — RINIQ-NEXT → SMSE → KLD → Gnom (2026-09-22 → 27)

Full chronicle of this work session. If anything breaks, this file
restarts the context. Ledger files: README.md / MIMO.md / RINIQ-NEXT.md /
FUSION.md / SAGA.md / MTP.md / gnom/ (this repo), HF cards (Hub).

## 1. Orientation (Sep 22)

Read all READMEs: MERNIK = measure-first quantization protocol
(priority-queue allocation + two rings: fast allocator / slow capability;
three columns PPL-canary / KLD-rank / tasks-verdict). Satellites:
FUSION.md (layer monsters M1–M7), MIMO.md (MiMo distillate workbench),
RINIQ-NEXT.md (Ox×MiMo fusions), SAGA.md (MTP graft), README-ZOO.md
(utility duels, PPL demoted after KLD paradigm nuke).

## 2. RINIQ-NEXT chaos → M8/N2 (Sep 22–23)

- PPL slate MiMo (chain.log): 4500-SMAPE 10.5343, 4500-BALANCE 9.4092,
  6500-MSE 8.7655. N1m HE lost to stopper kill; chain5 closed N1 79.27%,
  N1m PPL 7.96, Q6 PPL 9.1362.
- Weight compass died (GGUFReader loads all → OOM risk); pkill -f killed
  own shell (lesson: never pkill patterns present in own cmdline).
- Deleted RINIQ-N1m-BF16 (18GB, recipe logged).
- **fuse_layers --map BUG FOUND**: 2 donors + --map silently fell back to
  odd/even interleave → N1/N1m were half-foreign interleaves, not grafts.
  Fixed (base-A-plus-overrides) + `--dry` + `--d` (4th donor).
- Built M8 (M2+MiMo31) and N2 (M2+MiMo15–17) from M2-BF16 as base
  (Neo/Orn baked in, no BF16s needed). N2 ties M2 crown: 91.46%.
- M8: MiMo31 loses to Neo31 by 8.5pp (82.93%) — one block decides.
- N1m rerun: 63.41%, 51 empties — PPL-liar crown (best PPL 7.96).

## 3. BALANCE → SMSE (Sep 23)

- New utility `balance` (geometric SMAPE×MSE blend, MIX _scale lesson):
  MiMo-4500 PPL 9.4092 vs SMAPE 10.5343; HE 54.88% vs 29.88% (+25pp,
  TOX-like lava rescue). Renamed BALANCE→**SMSE** (SMAPE+MSE).
- 5100: PPL tie 8.7389/8.7435, HE 71.34% vs 70.12% (+1.2pp), 3 empties —
  small-budget crown. 6500: 8.7275/Tie PPL, HE 72.56% vs 75.00% (−2.4pp,
  MSE owns big; middle Q6 67 vs 41 decides, not penthouses).
- Sub-4 Law confirmed (Neo-3650-Q2 23.78% ≈ MiMo-4500 29.88%).
- SRMSE (SMAPE×RMSE) added: bit-identical file to SMSE-5100 (md5!) —
  same allocation, verdicts 66.46/70.12 across runs = sampling-noise demo.
- Ox-6500-SMSE (thank-you build for reviewer amo1mor3): PPL 7.5410,
  **HE 91.46% BEATS Ox-MSE crown 90.24%**, HE+ 87.8. Uploaded + replied.

## 4. Tissue surgery + machine duels (Sep 23)

- fuse_layers tissue mode (`15:b:ffn`, ranges, `ffn+attn`, `ln` norms):
  N4a FFN-only 91.46% (5 empties, soul in FFN), N4b attn-only 89.02%,
  N4c norms-only 89.63% (best PPL 7.5795, mild liar).
- autofuse rank-vote improved (--d/--id): map MiMo{10,14,15,16,27}.
  N3 (M2+map): 88.41% — machine 0-2 (M7 same 88.41%). N4a-Q38-SMSE
  stacking: 89.02% — wins don't sum.
- N2a/b/c dissection dropped (N3 arbitrated 17 for free).

## 5. GPU OC saga (Sep 23)

- GTX 1070 OC +150/+800 via nvidia-ml-py (Wayland): 31→36 t/s,
  gpu_burn clean. OC-taint scare: identical files scored 71.34/66.46 →
  arbitration rerun 70.12 → sampling noise, no harm. Lesson: repeats
  before taint claims. dinit service persists OC across reboots.

## 6. Gnom-0.2 (Sep 24–26)

- Pooled 626 PPL-damage labels (1.7B/2B/4B) → LOMO best 0.067 (bar 0.45):
  FAIL. Damage is noise + model-local physics. Thesis closed w/ receipts.
- SSIM-damage (CPU, 112 labels/3min): rank-corr vs PPL 0.041 (orthogonal);
  learnability 0.978 (surrogate value only). Perceptual 0.784, MS-SSIM
  0.998 (dead), GMSD 0.991 (dead), structure-vs-full 0.995 (dead).
- KLD-damage: two-pass run_ppl fix (new llama.cpp: --kl-divergence
  REPLACES perplexity). First labels: rank-corr(KLD,PPL)=0.018,
  (KLD,SSIM)=0.142 — samurai path. Learnability 0.609 vs timp 0.104!
- Active-loop RL-PLAN.md (friend co-design): useful-surprise reward,
  16th feature (outlier fraction), --units/--timeout/--ngl in sweep.

## 7. SRIQ + Prism + Qwengram (Sep 25–27)

- SRIQ (LoRA-child of MiMo): own cal file (1.11MB, trimmed parquet),
  Q6 + own imatrix, HE 64.63% both temps (temp moves decisiveness only),
  SMSE-5100 63.41% (−1.2pp vs Q6, mirror of MiMo).
- Prism-coder-9B (tool router): BF16 converted (18.4GB, 442 tensors),
  Q6 PPL 9.2602 / HE 79.27%. SMSE-5600 (MTP floor!): PPL 9.3339 /
  HE 74.39% (−4.9pp vs Q6).
- MTP draft duels (first global test, SAGA/MTP.md): base 74.39 →
  mtp2 67.07 → mtp3p80 68.90 → mtp8p80 72.56 → mtp5ngram 71.34
  (fastest 14.6 s/task). Speed/verdict curve, not free lunch. n16 died
  on load (VRAM wall).
- Qwengram repro: model+32GB sidecar downloading, fork built (CPU).

## 8. Infra scars (all kept)

pkill-self, partial-file chain race (guard: full size + Done),
100%-disk kills (BF16 graveyard rule: keep M2 + sources only),
earlyoom resurrections (dinit), OC-taint false alarm, HE default
protocol drift (see audit below), fuse --map blind interleave.

## 9. External audit (2026-09-27, small agent) — STATUS 2026-09-28

Findings A1 (HE defaults vs protocol), A2 (dual-imatrix divergence),
A3 (unanchored global regex rules), A4 (dry-run audit discarded),
B1 (per-type×tier KLD curve kills the zoo), B2 (no skip-ahead),
B3 (Wilson+McNemar duel.py), C (uncommitted 4 days, 98% disk,
ledger drift).

State after the follow-up pass: A1 done (protocol.py, defaults fixed,
metadata sidecar), A2 done (lens divergence measured, refuses to build
split-brain), A3 done+extended (anchored globals), A4 done (preflight.py,
runs on every invocation), B3 done (duel.py, all standings re-adjudicated),
C done except the commit (needs the operator). B1 still queued on KLD
labels. B2 implemented, found to be a no-op in practice (see 9.3).

### 9.1 The verdict column is mostly noise (scripts/duel.py)

Paired McNemar + Wilson CI on the per-task vectors already on disk
(`*.jsonl_results.jsonl`, human_eval runner = the numbers in the tables):

| Claim (README/FUSION) | Δ | discordant | p | verdict |
|:---|---:|:---:|:---:|:---|
| M2-Q38 92.07 beats M2-MSE-6500 91.46 (crown) | +0.6 | 8 / 7 | **1.00** | NOISE |
| M2-Q38 beats M4a 91.46 | +0.6 | 7 / 6 | **1.00** | NOISE |
| M2-Q38 beats M5 91.46 | +0.6 | 7 / 6 | **1.00** | NOISE |
| N2 91.46 ties M2-Q38 92.07 | −0.6 | 8 / 9 | **1.00** | NOISE |
| MiMo-5100 SMSE +1.2pp "small-budget crown" | +1.2 | 22 / 20 | 0.88 | NOISE |
| MiMo-6500 SMSE −2.4pp "loses big budgets" | −2.4 | 20 / 24 | 0.65 | NOISE |
| Prism Q6 vs SMSE-5600 −4.9pp "family-local" | +4.9 | 21 / 13 | 0.23 | weak |
| **MiMo-4500 SMSE 54.88 vs SMAPE 29.88** | **+25** | **53 / 12** | **<0.0001** | **REAL** |

One law survives: the sub-4 lava rescue. Everything inside ±3pp does not.
The RINIQ crown table (92.07/91.46/91.46/91.46/90.85) is a four-way
statistical tie; the "budget law" legs (5100 +1.2, 6500 −2.4) are both
null. Noise is ±3.6pp per their own MTP note, so ±1-2pp crown changes were
never measurable at n=164. Nothing is retracted — the *mechanisms* (FFN
soul, distant islands, seam friction) rest on empties counts and larger
deltas (M8 −8.5pp), which do survive; only the crown pecking order does not.

Harness law confirmed and now enforced in duel.py: EvalPlus `base_status`
and the human_eval runner disagree by ~1 task on the SAME completions
(FUSION.md footnote). duel.py refuses cross-harness pairs.

### 9.2 The 1D law (found by preflight, not by reading)

llama.cpp writes 1D tensors as F32 no matter what `--tensor-type` says.
Proof: MiniCPM5-2B, 85 norm tensors assigned Q4_K/IQ4_XS by the top-down
queue → 85 F32 in the dry run. RINIQ-M2 additionally has 24 `ssm_conv1d`
(shape [4,8192]) that also come back F32.

Consequences, all now fixed in code:
- The queue was spending decisions on tensors that cannot land (75 phantom
  steps in one top-down run) and pricing them at the assigned tier's bpw.
- **The norms shield and the "norms are load-bearing walls" law are about a
  precision the binary never spends.** Whatever `--pin-norms`/TDN measure
  is a budget-redirect effect, not norm precision. The −3.7pp HE from
  "downgrading 30 norm tensors to Q4" cannot be a norm-quantization effect:
  those tensors were never quantized. Re-read that entry as an
  allocation-breadth effect.
- After the fix the estimate matches the binary tensor-for-tensor:
  MiniCPM5-2B @1700 bias +0.00%, RINIQ-M2 @5100 bias −0.00% (was +0.03%
  with 24 unknown-F32 tensors).
- `--legacy-1d` reproduces the old tables (verified: identical geometry).

Also confirmed and kept: llama.cpp matches `--tensor-type` with
`std::regex_search` + first-match-wins (src/llama-quant.cpp:717), so the
priority sort in config_generator is correct. The hazard was only that
global rules were `.*ttype.*` substring patterns — now anchored to
`(?:^|\.)ttype\.=`.

### 9.3 B2 (budget tail) — implemented, but a no-op where it matters

`_polish_slack` walks the whole reachable ladder window, not just cur+1, so
the tail of the budget is not stranded. Measured on MiniCPM5-2B across
1500–2600 × {mse, smape, smse}: it never fires, because the single-rung
greedy already lands within 0.1–7 MiB of target. The 40 MiB gap at @2400
is not ladder granularity — it is the **structural ceiling**: bottom-up caps
every class at CLASS_MAX_TIER (Q8_0), which sums to exactly 2359.4 MiB on
this model. `--size 2600` silently produced a 2359 MiB file; now it says so
loudly and names the blocking classes. On an 8 GB card that is the
difference between a model that loads and one that does not.

### 9.4 Still open

- B1: per-(type × tier) KLD curve. 78 numbers instead of 626 per-tensor
  regressions, and the only measured-learnable target so far (SSIM 0.978).
  Queued on the KLD labels.
- C: 76 GB of `gnom/ref_*.dat` still on a 98% disk (two 38 GB KLD
  reference caches) — not deleted, they are not mine to delete.
- Commit pending: 4+ days of work uncommitted, MIMO.md / RINIQ-NEXT.md
  were in .gitignore until this pass.

