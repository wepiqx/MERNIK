---
license: apache-2.0
base_model: wepiqx/ASHQ1
language:
- en
tags:
- quantization
- gguf
- llama-cpp
- imatrix
- mernik
- gnom
- humaneval
- evalplus
- benchmarking
- code-generation
- model-merging
- layer-fusion
- calibration
- kld
- perplexity
- consumer-gpu
---

# MERNIK — Measure-First Quantization Protocol

**MERNIK** ("the one who measures") is the evolution of the ASHQ1 battlefield zoo: fewer
utility duels, more verdicts. The method (priority-queue allocation) is built; MERNIK is how
we prove and certify it.

Think of MERNIK as a **finetune of ASHQ1**: same base weights (queue, pins, tied groups),
retrained objective (measure-first protocol, columns that carry their own ring), and
expanded capabilities (slow capability ring, KLD-aware teachers, the zoo bench, and the
water-fill allocator).

---

## 0. TL;DR — what the last weeks produced

### The 5100 experiment (2026-10-27): water-fill beats NETPURE on PPL and mean/median KLD; **HE deliberately not run**

This was the week the allocator stopped carrying its own architecture table and
started reading the model through `llama-quantize --dry-run` (`llama_inspect.py`).
The same week added the adaptive sub-Q4 floor, recorded 38 greedy baseline
allocations across 10 models, and built the water-fill prototype. The 5100 MiB
rung is the first measured head-to-head of that prototype against the greedy
Gnom build on the same Ling-3.0-tiny base, imatrix and corpus.

| arm | MiB | PPL | KLD mean | KLD median | KLD max |
|:---|---:|---:|---:|---:|---:|
| stock Q4_K_M | 4600 | 13.327 | 0.114020 | 0.048512 | 14.354789 |
| NETPURE 5100 (greedy Gnom) | 5091 | 13.4271 ±0.10880 | 0.089046 | 0.037072 | 9.071382 |
| **WATERFILL 5100** (Gnom amplitude, α=2.4) | 5116 | **13.0611 ±0.10497** | **0.063126 ±0.000371** | **0.025847** | 9.515678 |
| stock Q6_K (ceiling, not the same budget) | 6198 | 12.69 | 0.022102 | 0.008468 | 4.612892 |

Water-fill buys **−2.7% PPL, −29% mean KLD, −30% median KLD** at +25 MiB over
NETPURE, while max KLD is 4.9% worse (9.52 vs 9.07). It spends the middle —
Q5_K 1914 / Q4_K 1519 / Q6_K 1150 / Q8_0 184 / F16 8 MiB — exactly the region
the 4500 rung cannot reach. The result is fidelity per byte, not a clean sweep:
the tail is the price of the mean.

The same prototype at 4500 MiB also beats NETPURE-4500: PPL **13.1785** vs
13.503, mean KLD **0.074614** vs 0.095092, median **0.030849** vs 0.039573,
at +9 MiB. That is −2.4% PPL / −22% mean / −22% median; max KLD is 16% worse
(10.11 vs 8.72) but still below stock Q4_K_M's 14.35.

**Why HE was not launched.** The 4500 battery had already put the capability
column on the floor (130/164 vs stock 131/164, duel 0/3 NOISE). A second
164-task battery at 5100 costs ~100 minutes of the single GPU and has a
13-task resolution, so it would produce another draw, not a decision. The 5100
question was fidelity per byte; PPL and KLD answer it. HE stays reserved for
builds where capability is the open question — not for every fidelity
comparison.

### The first MoE build (2026-10-25): Ling NETPURE-4500 wins every KLD column at 93 MiB less weight — mean 0.0951, max 8.72; **HE 0/3 NOISE, fidelity win stands** (not Q6: stock Q6_K sits at 0.0221 mean / 4.61 max)

The first MoE family is on the shelf: `Ling-3.0-tiny-MERNIK-4500-NETPURE.gguf` (4,725,910,720 B = 4507 MiB) against the vendor `Ling-3.0-tiny-Q4_K_M.gguf` (4,823,894,944 B = 4600 MiB), both on the protocol corpus (`wiki.test.raw`, 297 chunks, Q8-base ref 47.7 GB, base PPL 12.9500, `-c 1024 -b 128 --seed 7`). The 4500 rung is the comfortable floor: the Q4_K base with the F32 pins is 4454 MiB, so 4000/4200 need the sub-Q4 fallback (automatic when the Q4_K floor does not fit; `--dont-allow-q3` refuses it). Stock **Q6_K** (`Ling-3.0-tiny-Q6_K.gguf`, 6,499,262,368 B = 6198 MiB) is the fidelity ladder's top rung, same protocol.

| metric (wiki vs Q8 ref) | stock Q4_K_M | MERNIK 4500 NETPURE | stock Q6_K | delta (ours vs Q4) |
|:---|---:|---:|---:|---:|
| weight | 4600 MiB | **4507 MiB** | 6198 MiB | **−93 MiB (−2.0%)** |
| PPL (canary) | **13.327** (+2.91%) | 13.503 (+4.27%) | **12.69** | +0.176 (+1.3%) |
| mean KLD | 0.114020 ±0.000619 | **0.095092 ±0.000525** | 0.022102 ±0.000163 | **−16.6%** |
| median KLD | 0.048512 | **0.039573** | 0.008468 | **−18.4%** |
| max KLD | 14.354789 | **8.715995** | 4.612892 | **−39.3%** |
| RMS Δp | 9.405 ±0.054% | **8.588 ±0.052%** | 4.347 ±0.038% | **−0.82 pp** |
| same top p | 85.799 ±0.090% | **86.875 ±0.087%** | 93.647 ±0.063% | **+1.08 pp** |

**The absolute numbers are the record.** 0.0951 mean is the lowest of any Q4-class build on file (4B NETPURE: 0.1769; 4B stock: 0.1666; 9B arms: 0.1209–0.1262) and 8.72 is the lowest max KLD of any non-Q8 arm (4B NETPURE: 14.17; 9B arms: 25.5–30.0). Only the Q8 references and the near-ceiling 4B 4050-GNOMF16 control (mean 0.0623) sit lower on mean; cross-model absolutes are not a ranking. Within its own control the direction is unambiguous: **every KLD column improves, at 93 MiB less weight.** The canary disagrees (PPL +1.3%), which is exactly why this project labels PPL a canary and dropped the plain-PPL pass from the teacher entirely — the pre-registered fidelity column is KLD and it does not get overturned by the canary.

**Q6_K refutes the "Q5/Q6 level" guess.** At 6198 MiB stock Q6_K sits at mean 0.0221 / max 4.61 — **4× closer to base** than our 0.0951, and its PPL (12.69) beats stock Q4 (13.33). So the NETPURE-4500 is a Q4-class build, not Q5/Q6; the win is *fidelity per byte*, not fidelity per quant. The honest claim: the best Q4-class arm on the shelf, at Q4 weight, 93 MiB lighter than the stock Q4 it beats.

**Provenance.** The allocator is the sighted ranker: the training pool now carries 821 units — 512 dense (Spark X2.6-4B Q4/Q3/Q2) + 309 real Ling units — with 160 tied-group predictions for the MoE (90 structural tensors skipped by the 1D/F32/rank-1 law). The Ling leg is also the first ranker retrain that changes a *different* model's allocation: on Ornith it retiers 30/442 tensors (rank-corr +0.42 with the dense-only netpred), Q8→Q5 in the middle, layer 0/31 attention up, 1–4 down, same total size — recorded in the Fracture card, not here.

**HE closed (2026-10-25, same sampling: temp 1.0, top_k 20, top_p 0.95, min_p 0.05, max_tokens 8192, ctx 16384, `--reasoning-effort high`, `--cache-ram 2048`).** Stock Q4_K_M: HE **131/164 (79.9%)**, HE+ **128/164 (78.0%)**, empties 24. MERNIK NETPURE-4500: HE **130/164 (79.3%)**, HE+ **124/164 (75.6%)**, empties 29. Duel `scripts/duel.py`: **0/3 — NOISE** (HE p=1.00, HE+ p=0.60, empties p=0.38). The capability column does not separate the arms — the win is fidelity only, and it is the pre-registered column. This is the instrument's floor (SD ~5.9 tasks, resolution 13) plus saturation above Q4, not the model's ceiling.

### 3D smoke — my test (2026-10-25): NETPURE writes a working voxel scene first try; Q4_K_M writes broken geometry even on second run

**Files (`assets/probes/3d/` — also see `probe_info.md` for full prompt + settings):**
- `NETPURE.html` (22688 B): working scene — `Scene`, `WebGLRenderer`, `OrbitControls`, legend, coherent voxel geometry (`screenshot_netpure_20-16-02.png`).
- `Q4_K_M.html` (17518 B): broken geometry — scattered blocks, missing elements, no coherent layout (`screenshot_q4_20-15-51.png`).
- `probe_info.md`: full prompt + settings.

### The 9B verdict (2026-10-23): a zero-shot Gnom statistically ties the imatrix-trained SMSE — blindness costs nothing. The vendor stock row closed same-day: **task-tie at +470 MiB, fidelity collapse** (KLD +34% vs the best arm, worst tail, worst canary)

The pre-registered 9B batteries (below) closed on 2026-10-22; the vendor control (plain Q4_K_M, no imatrix, no allocator) closed 2026-10-23 16:17. Duel `scripts/duel.py` (McNemar exact, α=0.05, same protocol, 164 tasks):

| duel | HE | HE+ | empties | verdict |
|:---|:---|:---|:---|:---|
| **NETPURE (Gnom, zero-shot) vs SMSE (imatrix)** | 137 vs **140** (p=0.66) | 129 vs **133** (p=0.48) | **14 vs 19** (fewer better, p=0.23) | **0/3 — NOISE** |
| **stock Q4_K_M (vendor) vs NETPURE** | 137 vs 137 (p=1.00) | 125 vs 129 (p=0.60) | 18 vs 14 (p=0.45) | **0/3 — NOISE** |
| **stock Q4_K_M (vendor) vs SMSE** | 137 vs 140 (p=0.69) | 125 vs 133 (p=0.17) | 18 vs 19 (p=1.00) | **0/3 — NOISE** |

| arm | bet | HE (runner) | HE+ (evalplus) | empties | PPL wiki | KLD wiki vs Q8 ref | KLD wiki vs BF16 ref | KLD rlmini (interim) |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|
| **Gnom (netpure)** — ranker never saw the 9B, **no imatrix** | Q8 on the early peaks + Q4 floor | 137/164 (83.54%) | 129/164 (78.66%) | **14** | 8.7973 ±0.0626 | 0.12267 ±0.00199 | 0.12150 ±0.00197 | 0.2889 ±0.0064 |
| **SMSE** — imatrix-driven, trained on Ornith's own data | Q5 pyramid through the middle | **140/164 (85.37%)** | **133/164 (81.10%)** | 19 | **8.7349** ±0.0619 | 0.12620 ±0.00206 | 0.12427 ±0.00203 | 0.2915 ±0.0065 |
| DEEP (opposite-bet control) | — | — | — | — | 8.7760 | 0.12394 | 0.12225 | 0.2972 (worst, as intended) |
| LAW (hand-law control) | — | not run | not run | — | **8.7879 ±0.0625** | 0.12270 | **0.12094** (top) | — |
| **stock Q4_K_M (vendor)** — no imatrix, no allocator, **+470 MiB** | frozen llama.cpp pattern (Q6×35, **Q8×0**) | 137/164 (83.54%) | 125/164 (76.22%) | 18 | **8.8456** ±0.0626 (worst) | — (ref deleted) | **0.16304** ±0.00228 (worst of five) | — |

**What the tie is worth.** The 0/3 columns read "crown must not move" — and the tie has teeth: **the learned ranker, blind (zero-shot, no imatrix, never trained on the 9B), holds an even score on both pass columns against an allocator trained on this model's own imatrix.** The pre-registered outcome is **(c) tie on capability** — the next fight is below Q4, where the arms diverge on 117/442 tensors (open).

**Decisiveness axis (pre-registered as a separate column):** Gnom delivers **fewer empties — 14 vs 19** (5-task lead, p=0.23 — *a lead, not a result*, same discipline as the 4B p=0.035). And the direction **reverses the 4B empties law**: there, empty count grew monotonically with the depth of the upgrade (blk.27 → 9, blk.7 → 13, blk.0 → 18 empties "protect the earliest blocks and the model becomes more hesitant"). Here the arm that puts bits on the earliest blocks delivers **fewer** refusals at 97-tensor intervention scale. The 4B empties-monotonic claim does not survive its first big-intervention test — demoted to small-intervention artifact (recorded with both numbers, kept as history).

**Fidelity columns — landed (2026-10-23 06:52, both ref legs closed, zero failures).** KLD on the protocol corpus (302-chunk wiki.test) measured twice per arm: against a **Q8-base ref** and a **BF16-base ref** (the 4B protocol's tier). The ref-tier debate is answered, and boringly: **the tier moves every arm by the same small constant (+0.0012…+0.0019, mean +0.0016, 4/4 same sign) and flips no ranking** — NETPURE/LAW hold the top-2 on both refs (their ≤0.00056 order swap is deep inside σ), DEEP between, SMSE the furthest from base on BOTH refs. So future batteries rank identically on either tier; the honest "distance to base" number is the BF16 one; the Q8 ref is ~4× cheaper to build (~22 min vs ~75 min on this box, byte-identical 73.6 GB file format). Deviation note by the protocol law: the 4B batteries measured against a BF16 ref — both tiers are now on file and the offset is measured, not assumed.

**The cross-metric tension (noise-level, stated — it is the honest texture of this verdict):** SMSE wins the canary (PPL 8.7349 vs 8.7973, ~0.7σ) and the task column by raw count (140 vs 137, p=0.66) — but it is the arm drifted **furthest from base** on the protocol KLD (0.12427 vs 0.12150 = 0.98σ vs BF16 ref; same sign vs the interim rlmini ref, 0.4σ → 3/3 refs). NETPURE — blind, imatrix-free, never trained on the 9B — is the one closest to base. Both directions live inside noise, which is exactly why the verdict column stays a tie.

**The hand-law control also earned its slot:** LAW is top-of-ranking on the BF16 ref (0.12094) and co-top on Q8 (0.12270) — never scored on tasks by design, but the hand-law allocation is not fidelity-nonsense; its PPL hole closed at 8.7879 ±0.0625.

### The stock control row: plain Q4_K_M (2026-10-23, running now)

The row the top never had: `Ornith-1.5-9B-Q4_K_M.gguf` (5,780,090,240 B), quantized from
the BF16 base by plain `llama-quantize Q4_K_M` — **no imatrix, no allocator**, the vendor
heuristic exactly as anyone gets it. Through the full shelf right now: HE (164 tasks,
same protocol) → HE+ (evalplus) → PPL (wiki canary) → KLD (wiki, vs the BF16 ref).
Numbers land here when the run closes; the duel column pair is already armed
(`stock vs netpure`, `stock vs smse`, McNemar exact).

**The vendor fingerprint, counted from the file** (`scripts/audit_tiers.py --per-layer`,
the same tool over all three arms):

| build | file | tensors above Q4 | notes |
|:---|---|:---|:---|
| stock Q4_K_M | 5.78 GB | **Q6 ×35**, Q8 ×0 | static llama.cpp pattern: `attn_qkv`+`ffn_down` Q6 on layers 0,1,2,6,9,12,18,21,24,28–31 (`attn_v` on the full-attention layers 3,15,27,31,32) + output head; **layers 4,5,7,8,… at full Q4** |
| Gnom (netpure) | 5.31 GB | **Q8 ×23**, Q5 ×1 | Q8 = first blocks' `attn_qkv`/`attn_gate`/`ssm_alpha`/`ssm_beta` layers 1–4 + heads — the exact zone the Fracture labels call the epicenter |
| SMSE (imatrix) | 5.31 GB | **Q5 ×82**, Q8 ×8 | the activation-magnitude pyramid through attention projections + gates |

Three sentences of physics from that table: (1) the stock carries **+470 MiB more than
either allocator** and places every upgrade by a frozen layer-list, not by a measurement —
not one tensor above Q6, no Q8 anywhere; (2) layers 0–2 happen to be covered, but
**layers 4–8 — the measured damage epicenter of this family — sit at untouched full Q4**;
(3) both allocator builds spend a *smaller* budget and still aim Q8 at the zone the
blame lives in. **The answer landed same day (16:17):** the stock ties BOTH allocator
builds on tasks (0/3 both duels: HE p=1.00/0.69, HE+ p=0.60/0.17, empties p=0.45/1.00)
while paying +470 MiB — and **collapses the fidelity board: KLD 0.16304 = +34% over the
best arm, the worst tail of all five builds (max 30.60), the worst canary PPL 8.8456,
and more refusals than Gnom (18 vs 14)**. The frozen pattern buys nothing and pays the
physics price; "fewer bytes, same scores" is now measured, not assumed.

*Sources:* `scripts/audit_tiers.py` audit dumps 2026-10-23; battery in flight
`shelf_9b_stockq4.log`, sidecar-stamped when closed.

*Sources:* `eval_results/humaneval_ornith5300{netpure,smse}*` (sidecar-stamped, git `8cce6d2`), duel run 2026-10-23; HE `shelf_9b.log`, fidelity legs `shelf_9b_kld.log` (rlmini/Q8-forge interim), `shelf_9b_kld_wiki.log` (Q8 ref, closed 04:24), `shelf_9b_ref_ab.log` (BF16 ref, closed 06:52); refs `assets/refs/ref_ornith{q8,bf16}_wiki.dat` — 73,599,454,980 B each, byte-identical format (BF16 ref kept on disk); stock control row `shelf_9b_stockq4.log` + `eval_results/humaneval_ornithstockq4km*` (in flight 2026-10-23, sidecars land with the run, fingerprint via `scripts/audit_tiers.py`); HE column = runner's per-task `passed` (`.jsonl_results.jsonl`); evalplus `base_status` disagrees by the usual ~0.5pp harness gap (136/139).

### Pre-registered: the 9B test (written 2026-10-22, before the result)

The 4B shelf could not separate the allocations (0/3 columns on four of five pairs). The 9B
can: at the Q4 floor (5300 MiB) the two allocations differ in **97 tensors of 442**, so the
verdict column finally has the power to tell them apart.

| | Q8_0 | Q5_K | Q4_K | the bet |
|:---|:---|:---|:---|:---|
| SMSE (imatrix) | layer 32 only | **82** (layers 0, 3, 7, 13–31) | 168 | a broad Q5 pyramid through the middle |
| **Gnom (zero-shot)** | layers 1–4 + 32 | 1 | **230** | *depth is safe at Q4; bits only on the first blocks* |

(layer 32 in Q8 in both arms is the allocator's **fallback** for the last block, not a
decision of either method.)

**Provenance:** the ranker behind the Gnom arm was trained on the 1.7B/4B rlmini legs only —
**the 9B leg was not in its training set**, so this is a genuine zero-shot on an unseen model.

**Pre-registered outcomes** (recorded before the batteries finished):

| result | what it means |
|:---|:---|
| **Gnom ≥ SMSE on HE/HE+** | the law transfers to a hybrid architecture it never saw — the first capability win for the learned allocator |
| **SMSE clearly better** | "early layers are fragile" is transformer-specific and fails on the recurrent/SSM hybrid: the deep blocks cannot hold Q4 here. The ranker gets a *boundary of applicability* — worth more than a win |
| **tie on capability** | as on Spark: capability does not separate them and the win stays quality-per-megabyte. Then the next fight is **below Q4**, where the arms diverge on 117/442 tensors |
| **empties trend deeper** (recorded as a third, separate axis) | on the 4B, empties were monotonic in the depth of the upgrade (SMSE blk.27 → 9 empties, LAWONLY blk.7 → 13, Gnom blk.0 → 18). If it repeats on the 9B, the first blocks govern *decisiveness*, not just signal — and the right answer would be Q8 for the first blocks, but not F16 for them wholesale |

**Falsification controls, built before the result** (`models/netpred_ornith_law.json`,
`models/netpred_ornith_deep.json`): if the "deep" arm — the *opposite* bet — also wins,
then the win was not about depth at all. One command from a build, no re-training.

**The named risk.** If the 9B hybrid's depth genuinely cannot hold Q4, that is not the law
failing: the ranker was trained on Spark (1.7B/4B) and knows nothing about recurrent blocks.
The 9B's *own* labels do support the bet (57.2% of the damage in layers 0–5, 8.9% in ≥24,
blocks 0–2 at 7.4× the middle) — so a collapse would indict the *transfer*, not the
measurement.

---

**One model, one byte budget, three beliefs about where the bits go.** Spark-X2.6-4B
(released hours after its weights leaked; same `spark2_5` architecture as X2.5,
post-training only). All three MERNIK arms are the **same 2254.8 MiB** with the **same tier
multiset** (Q4_K ×213 + Q5_K ×4) — they differ only in *which* tensors got the Q5.

| allocation | who chose | PPL (wiki) | max KLD vs BF16 | mean Δp |
|:---|:---|---:|---:|---:|
| stock Q4_K_M (2479.8 MiB) | llama.cpp heuristic | 20.6148 | 17.65 | −1.015% |
| SMSE | the imatrix (layer 27) | 19.2660 | 17.77 | −0.733% |
| LAWONLY | the depth law, by hand (layer 7) | 18.6080 | 18.71 | −0.445% |
| **Gnom (netpure)** | **the ranker trained on measured damage (layer 0)** | **18.3646** | **14.17** | **−0.306%** |

The learned ranker beats the static lens **and** a hand-picked reading of its own law — and
by 24% on the tail. The monotone prior cannot see the layer-0 spike; the ranker can.

**The PPL ladder** — `llama-perplexity` on wikitext-2-raw test, ctx 1024, seed 7. Same
corpus, same harness; PPL is the **canary**, but it is the cheapest cross-build comparison we
have, so it gets written down:

| build | weight | PPL | note |
|:---|---:|---:|:---|
| **MERNIK-2250-Gnom** | 2254.8 MiB | **18.3646** | Gnom-driven (Q5 on layer 0) |
| MERNIK-2250-LAWONLY | 2254.8 MiB | 18.6080 | law picked by hand (layers 7, 11) |
| MERNIK-4050-GNOMF16 | 4053.4 MiB | 18.6228 | F16 on the fragile layers (46 tensors) |
| *BF16 base* (anchor) | 7848.7 MiB | *18.8263* | zero-loss reference, not a quant |
| stock Q8_0 | 4172.3 MiB | 19.0307 | "almost lossless" (KLD 0.0061 — 27× closer to BF16) |
| MERNIK-2250-SMSE | 2254.8 MiB | 19.2660 | imatrix-driven (Q5 on layer 27) |
| stock Q4_K_M | 2479.8 MiB | 20.6148 | vendor mix (853 MiB in Q6) |

The three 2255 MiB arms beat the stock Q8_0 **on PPL while being 46% smaller** — but the
KLD column below says plainly what Q8 buys: it is **27× closer to the BF16 base**
(mean KLD 0.0061 vs 0.1769) and keeps the argmax on 96.94% of positions against our ~82%.
Q8 is the honest "touch nothing" reference; our arms are the cheap ones, and what they
trade away is written down in the same table.

**The KLD column, in full** — every build against the **same BF16 reference** (wiki test,
302 chunks, ctx 1024, `llama-perplexity --kl-divergence --kl-divergence-base`):

| metric | stock Q4_K_M | stock Q8_0 | MERNIK-2250-SMSE | MERNIK-2250-Gnom | MERNIK-2250-LAWONLY | MERNIK-4050-GNOMF16 |
|:---|---:|---:|---:|---:|---:|---:|
| Mean KLD | 0.166641 ±   0.000872 | 0.006139 ±   0.000161 | 0.180843 ±   0.000904 | 0.176940 ±   0.000894 | 0.178306 ±   0.000894 | 0.062301 ±   0.000481 |
| Maximum KLD | 17.646482 | 6.612764 | 17.770252 | 14.172658 | 18.709267 | 13.859949 |
| 99.9% KLD | 4.042008 | 0.442399 | 4.074759 | 4.134612 | 3.953914 | 2.491301 |
| 99.0% KLD | 1.346914 | 0.059898 | 1.493514 | 1.490102 | 1.489377 | 0.624270 |
| 95.0% KLD | 0.582286 | 0.016562 | 0.644909 | 0.637474 | 0.640173 | 0.209655 |
| 90.0% KLD | 0.386221 | 0.009498 | 0.425906 | 0.414667 | 0.419712 | 0.127790 |
| Median KLD | 0.077591 | 0.001394 | 0.083268 | 0.080211 | 0.081173 | 0.025654 |
| Mean Δp | -1.015 ± 0.024 | -0.083 ± 0.005 | -0.733 ± 0.024 | -0.306 ± 0.024 | -0.445 ± 0.024 | 0.089 ± 0.015 |
| Maximum Δp | 99.747 | 97.502 | 99.924 | 99.639 | 99.818 | 98.523 |
| 99.9% Δp | 55.175 | 12.304 | 63.900 | 67.559 | 65.434 | 46.425 |
| 99.0% Δp | 25.209 | 4.206 | 28.549 | 30.309 | 29.513 | 18.499 |
| 95.0% Δp | 10.534 | 1.671 | 11.624 | 12.468 | 12.188 | 7.753 |
| 90.0% Δp | 5.344 | 0.869 | 5.888 | 6.464 | 6.336 | 4.214 |
| 75.0% Δp | 0.560 | 0.117 | 0.695 | 0.907 | 0.838 | 0.680 |
| Median Δp | -0.015 | -0.000 | -0.009 | -0.002 | -0.004 | -0.000 |
| Minimum Δp | -99.741 | -98.934 | -99.072 | -99.094 | -98.129 | -99.536 |
| RMS Δp | 9.444 ± 0.055 | 1.904 ± 0.046 | 9.539 ± 0.052 | 9.398 ± 0.053 | 9.460 ± 0.052 | 5.817 ± 0.046 |
| Same top p | 82.199 ± 0.097 | 96.940 ± 0.044 | 81.482 ± 0.099 | 81.769 ± 0.098 | 81.644 ± 0.099 | 88.864 ± 0.080 |


The imatrix carries across the model version almost unchanged (**Spearman 0.9964**, top-50
overlap 100% between X2.5 and X2.6), so the X2.5 labels applied to X2.6 without
re-measuring. That is why the arms above are comparable at all.

### The verdict column — all five arms, all pairs

| arm | weight | HE | HE+ | empties | PPL | max KLD | same-top |
|:---|---:|---:|---:|---:|---:|---:|---:|
| stock Q8_0 | 4172.3 MiB | 140/164 (85.37%) | 132/164 (80.49%) | 13 | 19.0307 | 6.61 | **96.94%** |
| stock Q4_K_M | 2479.8 MiB | 139/164 (84.76%) | 132/164 (80.49%) | 14 | 20.6148 | 17.65 | 82.20% |
| MERNIK-2250-LAWONLY | 2254.8 MiB | 137/164 (83.54%) | 127/164 (77.44%) | 13 | 18.6080 | 18.71 | 81.64% |
| MERNIK-2250-Gnom | 2254.8 MiB | 136/164 (82.93%) | 130/164 (79.27%) | 18 | **18.3646** | **14.17** | 81.77% |
| MERNIK-2250-SMSE | 2254.8 MiB | 136/164 (82.93%) | 130/164 (79.27%) | 9 | 19.2660 | 17.77 | 81.48% |
| MERNIK-4050-GNOMF16 | 4053.4 MiB | 134/164 (81.71%) | 129/164 (78.66%) | 16 | 18.6228 | 13.86 | 88.86% |

Every pair, arbitrated by `scripts/duel.py` (McNemar exact, α = 0.05):

| duel | HE | HE+ | empties | verdict |
|:---|:---|:---|:---|:---|
| stock Q4_K_M vs Q8_0 | p=1.00 | p=1.00 | p=1.00 | **0/3 — noise** |
| stock Q4_K_M vs GNOMF16 | p=0.38 | p=0.66 | p=0.77 | **0/3 — noise** |
| Q8_0 vs GNOMF16 | p=0.34 | p=0.73 | p=0.58 | **0/3 — noise** |
| Gnom vs SMSE | p=1.00 | p=1.00 | **p=0.035** | 1/3 — empties only |
| Gnom vs LAWONLY | p=1.00 | p=0.73 | p=0.33 | **0/3 — noise** |

**What the whole shelf actually says (honest reading).**

1. **At this scale the verdict column cannot separate any of the allocations** — 0 of 3 columns
   for four of the five pairs. The three 2250 MiB arms (SMSE / Gnom / LAWONLY) score *the
   same* HE (136 / 136 / 137) and even swap tasks symmetrically (Gnom vs SMSE: 15 stock-only
   vs 15 Gnom-only discordant pairs, p=1.00). Which three tensors get the Q5 does not move
   capability.
2. **The one column that did separate is empties, and it points against us:** Gnom gives up on
   **18** tasks vs SMSE's **9** (p=0.035) — and it is *monotonic in depth*: the deeper the
   upgrade (SMSE blk.27/35 → 9 empties; LAWONLY blk.7 → 13; Gnom blk.0 → 18). Protect the
   earliest blocks and the model becomes **more hesitant**. But: that p=0.035 is 1 of 15
   comparisons run tonight, so it does not survive multiple-comparison correction
   (Bonferroni 0.0033) — it is a **lead to chase**, not a result.
3. **Q8_0 — the near-lossless reference — is NOT better than Q4_K_M** (140 vs 139 HE, all
   columns p=1.00). At 4B scale capability is indifferent to fidelity across this whole range,
   which kills the "just use Q8" intuition and vindicates spending the bytes elsewhere.
4. **So the win is not capability — it is quality-per-megabyte:** the same verdict
   (indistinguishable), **9.1% less weight**, a **20% cheaper tail** (max KLD 14.17 vs 17.65)
   and **2.25 better PPL**. The PPL/KLD advantages are real measurements that do **not**
   convert into capability at this intervention size.

**And the honest bottom line for the write-up:** the *damage law* (first blocks carry the
damage) is a **measurement** — it is what the KLD labels say, and it transfers across
architectures. Whether spending bits there buys **capability** is a separate question that at
4B/3-tensor scale the instrument cannot answer. It stays open, and the heavier arms
(46 tensors in F16, the 9B with a looser budget, the sub-4 regime) are where it can be
tested.

**And the limit of the claim.** Moving 3 tensors out of 290 is below what the verdict column
can resolve (13-task floor). The law has to show up where the intervention is larger — the
F16-blend arm (46 tensors in F16) or the 9B, where the budget is looser.

**What is in the files** (read straight from the GGUF headers — the allocation, not a
claim):

| | stock Q4_K_M | MERNIK-2250 (any arm) | stock Q8_0 | MERNIK-4050-GNOMF16 |
|:---|:---|:---|:---|:---|
| Q4_K | 180 (1620.8 MiB) | 213 (2011.7 MiB) | — | — |
| Q5_K | — | 4 (237.2 MiB) | — | — |
| Q6_K | 37 (853.1 MiB) | — | — | 44 |
| Q8_0 | — | — | 217 | 85 |
| F16 | — | — | — | 46 (1510 MiB) |
| attn_proj | Q4_K×54 Q6_K×18 | Q4_K×70 Q5_K×2 | — | — |
| ffn_down | Q4_K×18 **Q6_K×18** | Q4_K×36 | — | — |
| gate | Q4_K×36 | Q4_K×35 Q5_K×1 | — | — |

The stock mix spends **853 MiB in Q6** (and gives `ffn_down` the lion's share of it); the
2250 arms spend **237 MiB in Q5** on four tensors. 3.6× less high-tier weight, **better
perplexity**, 9% smaller file — that is the project thesis in two rows: *measured importance
beats blind spreading*. The whole gap between SMSE and Gnom is **three tensors**:
SMSE upgrades `blk.27.attn_gate/qkv` + `blk.35.attn_output`; Gnom upgrades
`blk.0.attn_gate/qkv` + `blk.11.attn_output` — same count, same bytes, and that is the
*entire* difference between 19.2660 and 18.3646. Whether that shows up in the verdict column
is an open question: three tensors out of 290 is a small intervention, and the verdict column
has a 13-task floor. The heavier arms (46 tensors in F16, the 9B) are where the law can
actually be tested.

**The law behind it** (measured on rlmini KLD labels, three models, transfers zero-shot):

| model | share of damage in layers 0–5 | in layers ≥24 |
|:---|---:|---:|
| Spark-1.7B-Q4 (measured) | 43.2% | 3.2% |
| Spark-4B-Q4 (measured) | 36.4% | 7.4% |
| **Ornith-1.5-9B (ranker zero-shot)** | **48.3%** | **6.4%** |

---

## 1. The two rings (revised 2026-10-16)

MERNIK strictly separates allocation signals from capability certification:

| Ring | Columns | Role | Cadence |
|:---|:---|:---|:---|
| **Fast / Allocator** | **PPL only** | per-decision canary for the queue | every run |
| **Slow / Verdict** | **HE, HE+, KLD** | the only columns authorized to crown or kill a build | once per release |

> **Rule:** Allocator signals never certify. Capability scores never steer. Mixing them
> leads to the PPL disease.
>
> **2026-10-16 revision.** KLD moved from the allocator ring to the verdict ring: the KLD
> column is a *full-model* measurement against the unquantized base over the whole corpus
> (302 chunks, ~24 min/build), not a cheap per-decision signal. PPL is now the only
> fast-ring column — and it is a **canary**: cheap, always available, and frequently
> misleading (it saturates on reasoning-critical layers and can be bought by sharpening). A
> build that only wins on PPL has won nothing.
>
> KLD's blind spots stand as recorded: fixed span, fixed chunks, Flash-Attention — blind to
> long context, instruction drift, and tool formats. It is a fidelity column, not a
> capability column.

*Decision rules.* **Divergence** — if metrics disagree, KLD wins allocator arguments; if
they agree, trust the pair. **Silence is signal** — empty completions scale with weakness
(Ox 4–5 → Neo 20–27 → Q2_K 63 → S2 115; Mellum-10000 24 → 7500 35 → Q6 39); weak models go
silent rather than babble. Decisiveness is governed by the finetune, not model size (on
Qwen3.5 bones, Neo hesitates while OxCoder commits). **Saturation rule** — bases at ~98%
cannot resolve fine differences; capability duels need bases at 60–85% or they only answer
the collapse threshold.

### The rules (all of them, in one place)

1. **Allocator signals never certify. Capability scores never steer.** Mixing them is how
   the PPL disease started.
2. **PPL is a canary** — never quoted without that label, and never as a result.
3. **Verdicts travel with MDE.** A difference of 5 tasks out of 164 is a direction, not a
   rank. Below ~8–10 tasks nothing is reproducible.
4. **The verdict column certifies preservation, not rank** (nine-way tie at the top: 151,
   then eight on 150).
5. **Every claim carries a number, a date, and a source** (a file, a repo link, a battery
   tag). A claim without all three is a story, not a measurement.
6. **One table per subject.** The same numbers must not appear in two sections — link
   instead. Family tables live on their own cards for exactly this reason.
7. **Dead results stay filed**, with the number that killed them — that is how we avoid
   re-testing them.
8. **A censored unit is never imputed.** TERMINAL records (`damage: null`,
   `retryable: false`) are visible and excluded by construction, not filled in.
9. **Cross-family verdicts are void** — different models, different rulers.

---

## 2. The laws (all of them, in one place)

Everything we have learned about *where* quantization damage lives and *what* to do about
it. Each law carries the number that proves it.

### L0 — The sub-4 law (confirmed 2026-09-23)

Any tier below Q4 hurts, and **fewer is better**. Sub-4 exposure at ~4 GB is lava in
every family: MiMo-4500-SMAPE (PPL 10.53 / HE 29.88%) lands exactly where
NeoHorse-3650-Q2 did (10.36 / 23.78%) — same physics, different bones. SMSE-4500
(9.41 / **54.88%**, +25pp) proves that limiting sub-4 exposure rescues the build. TOX
(×2.0 effective-MSE penalty, `constants.py`) called it; SMSE proved it on the verdict
column.

### L1 — The budget law (SMAPE small, MSE big)

SMAPE owns small budgets, MSE owns big ones. Direction only, not rank — the legs differ by
5–6 tasks, below the 13-task resolution floor (6500: 141 vs 135; 5100: 136 vs 131). The
lava leg survives on 3/3 columns plus effect size. **Above Q4, utilities debate inside the
floor; below Q4, physics decides.**

Floor-first confirmation (Mellum, 2026-10-10): the MoE floor sits at **7428 MiB** — no build
below it without sub-Q4 tiers (5500 arithmetically impossible; 5688 even with Q3 allowed).
At 6500: SMSE 126 vs recovery 123 vs stock Q4 125 — the whole shelf is one tie-zone, and
the weights differ (ours 1 GB lighter).

### L2 — The depth law: the first blocks carry the damage (2026-10-21)

**Where this came from (first person).** I checked my own first hand-made quant —
`gemma-4-12B-it-SHQ6`, June 2026, a manual allocation I wrote before any of this was
measurable. Its card already said it: *layers 0-1 are 3228× more important than middle
layers (23, 35); attention projections on the first layer (`attn_q`, score 5.11e8) dominate
the importance distribution.* The community noticed the effect without knowing the cause
("truly black magic", and it beat unsloth's Q4_K_XL while being smaller and faster).

Six months later the same thing showed up again — **on families that differ enormously
architecturally**:

| model | family | damage in layers 0–5 | in layers ≥24 | how measured |
|:---|:---|---:|---:|:---|
| gemma-4-12B-it | dense, 48 layers | — | — | manual imatrix, June 2026 (SHQ6) |
| Spark-X2.5-1.7B | hybrid SWA, 28 layers | **43.2%** | 3.2% | KLD labels (teacher) |
| Spark-X2.5-4B | hybrid SWA, 36 layers | **36.4%** | 7.4% | KLD labels (teacher) |
| Ornith-1.5-9B | hybrid + SSM, 33 layers | **57.2%** | 8.9% | KLD labels (teacher, blind to the ranker) |

A ranker trained on the 1.7B/4B legs predicted the Ornith profile **before it was
measured** (48.3% vs 6.4% predicted; 57.2% vs 8.9% measured) — **zero-shot transfer across
architecture**, direction and magnitude both right.

On the 9B the hot band is 7.4× the middle of the stack: mean damage in blocks 0–2 is
+0.00217 against +0.00029 for blocks 10–30. The fingerprint is stable too: the hottest
blocks on the 9B are **0, 1, 2, 3 and 11** — and the Gnom arm on the 4B independently
picked **`blk.0` and `blk.11`**. Different families, different sizes, different instruments
(the imatrix by hand in June, the teacher in October, the ranker as a prediction) — the same
fingerprint.

So the law is not "a property of that model": **the first blocks are where quantization
damage lives, and the error there has not yet been smoothed by the layers above it.**

The mechanism: activation magnitude grows with depth, so the imatrix keeps handing bits to
the *late* layers while measured reasoning damage lives *early*. The two allocations
disagree on purpose, and the disagreement is the experiment:

| Ornith-9B | SMSE @5310 | Gnom @5310 | SMSE @4000 | Gnom @4000 |
|:---|---:|---:|---:|---:|
| Q8 | 15 | **30** | 15 | **32** |
| Q6 | 0 | 3 | 0 | 3 |
| Q5 | **88** | 4 | 44 | 1 |
| Q4 | 162 | **228** | 78 | **113** |
| IQ2 (sub-4) | — | — | 117 | 110 |
| tensors moved | — | 103/442 | — | 117/442 |

SMSE builds a **pyramid** (88 Q5 in the middle); Gnom says *"Q4 everywhere, bits only to
the early peaks"* — and its leg is **leaner** (4 Q5 vs 88), which is L0 ("fewer is better")
applied to the upper tiers. At 4000 both arms fall into sub-4 (lava), so the meaningful
duel is at the Q4 floor.

### L3 — Which block exactly is architecture-specific — and that is why a learned ranker wins

The zone is always the first blocks; *which* one spikes is a property of the architecture:
layer 0 vs the average of layers 1–3 is **4.6× (4B Q4), 3.3× (4B Q3), 2.3× (1.7B Q4)**,
but only **1.1× on the 9B**, where the hot band is 0–2 with `blk.1` on top.

A **monotone prior** ("damage decays with depth") cannot express a band or a spike — it puts
the first block at rank 1 or 2 and stops thinking. The learned ranker can, because it learns
the *profile*, not the trend: on the 4B it upgrades `blk.0.attn_gate/qkv` where the
hand-picked law build takes `blk.7`. That single decision is worth **−0.24 PPL**
(18.3646 vs 18.6080) — the gap between the learned allocator and the best monotone
reading of its own law. Read L2 and L3 together: the law says *early*, the ranker says
*which* early.

### External confirmation (the literature caught up)

The law above was found by hand (June 2026) and measured (October 2026). Four independent
papers published in the same window describe the same physics:

- **[arXiv:2608.08188](https://arxiv.org/abs/2608.08188)** — *Quantization Degradation in LLMs:
  A Signal-Noise Perspective* (CASIA, 8 Aug 2026). Degrades to two linked processes: how
  errors arise inside a module, and **how they accumulate across layers** (R(l+1) ≈ G(l)·R(l) +
  Q(l)). The *propagated* error dominates at every depth, the per-layer source error becomes
  secondary as errors build up — and their closing sentence is our thesis in their words:
  *"the goal is to control the errors that drive downstream degradation, rather than to
  minimize all errors uniformly."*
- **[arXiv:2504.09629](https://arxiv.org/abs/2504.09629)** — *Quantization Error Propagation:
  Revisiting Layer-Wise PTQ* (Apr 2025): the activation difference **grows exponentially with
  depth** — the mathematical form of the snowball.
- **[arXiv:2505.15909](https://arxiv.org/abs/2505.15909)** — *Is (Selective) Round-To-Nearest
  Quantization All You Need?* (Kogan, MIT, 21 May 2025). Selective precision by layer: for
  smaller Llama models perplexity improves fastest when the **first** layers stay in 8-bit; and
  crucially **which** layers pay is architecture-specific — *middle* for Llama-3.1-405B, *last*
  for Phi-3-mini and Dolly-v2. That is exactly our L3: the zone is early, the exact block is
  the model's business — which is why a **learned** ranker beats a hand-picked rule.
- **[arXiv:2411.07191](https://arxiv.org/abs/2411.07191)** — *The Super Weight in Large Language
  Models* (Jul 2025): the few weights whose damage destroys the model sit in `mlp.down_proj`
  of the **early layers**, consistently across Llama, Mistral and OLMo.

Read together: the mechanism (errors born early are amplified through the depth), the practical
consequence (keep the early blocks precise), and the caveat (which block is architecture-
specific) are all established independently. MERNIK's contribution is the *measurement
instrument* — per-tensor damage labels on a reasoning corpus, and a ranker that transfers the
rule to models it has never seen.

### L4 — The 1D/F32 law

The binary writes 1D tensors (norms, biases) and `ssm_conv1d` as **F32 whatever the rules
say**, so the allocator pins them outside the budget. `--pin-norms` was removed as a
proven no-op (it changed 0 tensors); the old "norms are load-bearing precision" reading
was **budget-redirect, not precision**. The pre-flight audit catches exactly this class
of mistake.

### L5 — Saturation rule

Bases at ~98% cannot resolve fine differences (97% vs 96%). Capability duels require bases
at 60–85%, or they only answer the collapse threshold.

---

## 3. Quick Start & CLI

### Basic execution

```bash
pip install -r requirements.txt

# 1. Preview tier allocation & dry-run (~1 sec)
python main.py --model M.gguf --imatrix M.imatrix.gguf --size 6800

# 2. Execute actual quantization
python main.py --model M.gguf --imatrix M.imatrix.gguf --size 6800 --run
```

Stock llama.cpp binaries required (`llama-quantize`, `llama-perplexity`, `llama-server`);
override with `LLAMA_QUANTIZE`, `LLAMA_PPL`, `LLAMA_SERVER`.

### Flags

| Flag | Description |
|:-----|:------------|
| `--model M.gguf` | Source GGUF — required for allocation; not needed for verify-only |
| `--imatrix I.gguf` | Importance lens; repeatable. In the normal path `llama-quantize` gets the **first** file for both the queue and the binary; extra files only check lens divergence. To combine lenses for real, build one merged imatrix with `llama-imatrix`, then pass it alone |
| `--imatrix-method max\|mean` | Legacy-only (`--legacy-modes --imatrix-legacy-combine`); has no effect in the normal path |
| `--imatrix-tol F` | Max relative importance spread across repeated `--imatrix` files before refusal (default 1%) |
| `--size MIB` | Target file size in MiB (default: 6800). The Q4_K floor is tried first; if it does not fit, the allocator falls back to IQ2_XXS, then IQ1_S |
| `--dont-allow-q3` | Keep the Q4_K floor even when `--size` is smaller; fail instead of dropping to IQ2/IQ1 |
| `--output O.gguf` | Output path (default: `<model-basename>-MERNIK.gguf` in the current directory) |
| `--run` | Execute `llama-quantize`; without it: allocation + dry-run only |
| `--utility NAME` | Queue gain: `mse` (default), `smse`, `smape`, `gnom` (active path). `netpure`/`netfd` are deprecated aliases for `gnom` |
| `--netpred P` | Gnom predicted-damage table (required by `--utility gnom`) |
| `--legacy-modes` | Unlock archived utilities and flags for reproduction only — see `LEGACY.md` |
| `--top-down` | Start quantizable 2D groups at F16, downgrade cheapest-loss-first to fit, then spend the slack. Structural pins (MTP/embd/routers) and 1D F32 stay outside the queue |
| `--verify gpqa\|he\|all` | Slow-ring verify on a fresh `--run` build or an existing `--output` (busy port aborts LOUDLY) |
| `--verify-tag TAG` | Result tag (default: from `--output` basename) |
| `--show-config` | Print the tensor map and exit before dry-run |
| `--dump-alloc PATH` | Write the final tensor map + stats as JSON; real dry-run geometry is included only when preflight ran |
| `--no-preflight` / `--strict-preflight` | Skip the dry-run audit, or abort if any assigned tier cannot fire (default: audit runs, ~1s) |

Archived flags (`--pin-small`, `--max-tier`, `--free-pins`, `--legacy-1d`,
`--imatrix-legacy-combine`, `--allow-q3-or-lower`, plus the legacy utility
knobs) are hidden from `--help` and require `--legacy-modes`; see `LEGACY.md`.

### Utility modes

Gnom is the active allocator. The three core lenses stay because they define the
baseline ladder; every historical lens is archived in `LEGACY.md` behind
`--legacy-modes`.

| mode | belief | status |
|:---|:---|:---|
| `mse` (default) | relative-blind absolute gain; tiers spread evenly (Q5/Q6-heavy middle) | baseline; best PPL/KLD on dense 9B |
| `smape` | barbell: junk to the Q4 floor, kings to Q8 penthouses | owns small budgets, loses big ones (L1) |
| `smse` | geometric mean of SMAPE-relative and MSE-absolute halves; carries the sub-4 toxicity through its absolute half | best hand-tuned baseline; owns small+medium budgets |
| **`gnom`** | **per-group predicted damage is the gain; `--top-down` is the F16-blend direction, bottom-up is the floor direction** | **active path; smooth ladder from 4500 to 9000 MiB on Ling-3.0-tiny** |

`netpure` and `netfd` are deprecated aliases for `gnom` (bottom-up and top-down
respectively).

### Recently added (2026-09-28 → 2026-10-16, all CPU-cheap, all verified)

- `scripts/fuse_layers.py --expect "15:b,..."` — gates the resolved donor map (ranges
  expanded, tissue suffixes checked); exits 2 with a diff before one output byte. Plus a
  soup shape-guard: mismatched donor shapes fail LOUD, never a corrupt file.
- `scripts/group_damage_sweep.py --units TAGS --timeout S --ngl N` — subset sweeps; failures
  go to `*.failures.jsonl`, second failure of a unit becomes a TERMINAL censored label
  (`damage: null`, never retried, never imputed). `--kld-only` skips the plain-PPL pass and
  emits the 5-column KLD family.
- `--max-tier F16` (archived behind `--legacy-modes`) lets kings rise above Q8_0 without
  touching `constants.py`; it reproduces the raised-ceiling geometry byte-exact in dry-run.
- MoE golden geometries: `Mellum2.1 @7500 smse` (Q4 171 / routers F16 ×28) and `huber`
  (Q4 163 / Q5 34) — regression net over a Mixture-of-Experts allocation. The golden set
  now has 16 geometries (model-dependent cases skip when the GGUF is absent) plus 8/8
  provenance checks.
- `scripts/run_humaneval.py`: `HE_CHUNK="start,count"` slices the sorted task list; `HE_CTX=N`
  replaces the protocol default `-c`; `serve_sha256` sidecar key + `scripts/twin.py`
  pre-battery twin gate (same sha → reuse, else measure).
- `--cache-ram 0` on HE chains (164 distinct prompts share nothing, so the prompt cache only
  eats RAM — unbounded on hybrid archs); `LLAMA_PPL_BATCH` env overrides the sweep's hardcoded
  `-b 512` (forge iGPU OOMs a 9B at 512; 128 holds).
- `scripts/audit_tiers.py` — tier auditing:

```bash
python scripts/audit_tiers.py --model M-6500.gguf
python scripts/audit_tiers.py --model M-6500.gguf --big 10
python scripts/audit_tiers.py --model M-6500.gguf --layer 31
python scripts/audit_tiers.py --model M-6500.gguf --per-layer   # every layer
```
- `scripts/consolidate_labels.py` — builds the clean label pool (corpus-tagged, dedup-checked)
  so a mixed archive can never leak into a training run again.
- `scripts/shelf_x26.sh` — resumable measurement queue (one GPU job at a time, skips finished
  arms): KLD vs BF16 → HE batteries → HE+ per arm.

### Example: mapping the α-axis (dry-runs, CPU seconds)

```bash
for A in 0.1 0.25 0.5 0.75 0.9; do
  python main.py --model M.gguf --imatrix M.imatrix.gguf --size 3500 \
    --utility powalpha --pow-alpha $A --legacy-modes   # archived α-axis
done
python main.py --model M.gguf --imatrix M.imatrix.gguf --size 3500 \
  --utility logratio --legacy-modes       # archived: extreme barbell
python main.py --model M.gguf --imatrix M.imatrix.gguf --size 3500 \
  --utility smse --hinge-w 0.5 --legacy-modes  # archived sub-4 boost
```

---

## 4. Engine mechanics

### Queue & allocation principles

The default `main.py` path: every tensor starts at a floor tier based on its functional
class. A global max-heap drains the target budget best-first by
sum(importance) × Δ / MiB. Tied groups (identical imatrix energy) upgrade as a single unit
with summed importance. Structural pins — MTP heads (→ Q8_0), output/token embeddings
(→ Q5_K), MoE routers (assigned F16, F32 in file) — sit outside the budget queue. The
water-fill prototype below keeps the same groups and pins but replaces the heap.

### Certify vs report (tool boundary)

Three tools, two jobs. `scripts/manifest.py` and `scripts/ledger.py` **REPORT**: build→model
identity (three keys: score+time+family) and standings generated from artefacts — every
number travels with its identity status. `scripts/duel.py` **ARBITRATES**: paired McNemar
across the columns, refuses to average across harnesses. Nothing certifies except a
significant duel; everything else is a report.

### Pre-flight audit & the size ceiling

Every invocation runs a dry-run audit (~1s) comparing intended vs actually-written tiers per
tensor — the only thing that can catch a `--tensor-type` rule that cannot fire. Bottom-up
caps every class at CLASS_MAX_TIER, so a model can saturate below `--size`: MiniCPM5-2B
caps at exactly 2359.4 MiB, and `--size 2600` silently produced a 2359 MiB file. It now
says so loudly and names the blocking classes.

### Norms shield & attention core insights (re-labelled 2026-09-28)

TD-MSE vs MSE: −3.7pp HE with identical PPL (7.7701 vs 7.7695). The old reading ("norms
are load-bearing walls for code") is **mechanically impossible** (L4): no norm was ever
requantized. What the numbers measure is budget redirect: top-down routing moved ~5 MiB
between tied attention groups while the "shield" guarded precision that was never spent. The
Norms Shield (TDN) +1.2pp stands as a measured effect with a redirect mechanism, not a
precision mechanism.

### Distribution comparison @ 6500 MiB (427 tensors, incl. 177 F32 forced by the binary)

```
Tier      BU-MSE (spread)    BU-SMAPE (barbell)   TD-MSE       TD-SMAPE (F16 core)
F16       177 (2 MiB)        177 (2 MiB)          147 (2 MiB)  85 (949 MiB)
Q4_K      48 (972 MiB)       91 (1989 MiB)        71 (972)     236 (2368 MiB)
Q5_K      28 (1994 MiB)      3 (1345 MiB)         28 (1983)    7 (1411 MiB)
Q6_K      66 (2114 MiB)      7 (249 MiB)          71 (2127)    14 (486 MiB)
Q8_0      108 (1417 MiB)     149 (2913 MiB)       110 (1417)   85 (1287 MiB)
```

Takeaway: MSE fills the middle (Q5+Q6 = 94 tensors). SMAPE hollows it (10 tensors) to
double the Q4 floor and gain 41 extra Q8 penthouses.

### Water-fill prototype (`scripts/waterfill.py`)

A separate allocator experiment, not the default path: it reuses the same tied
groups and pins as Gnom, but replaces the greedy heap with a discrete
rate-distortion allocation. Gnom supplies the per-group damage amplitude `G`,
and the model assumes

```text
D_i(t) = G_i * 2^(-alpha * (b_t - b0_i))
```

where `b_t` is bits/weight at tier `t`. Negative `G` is clipped to zero and
pinned to the floor. The floor is Q4_K and the ceiling is F16 (the usual
per-class cap does not apply); tier choice is a discrete Lagrangian, and
leftover bytes are spent by a greedy tail fill. `alpha=2.4` is calibrated on
paired Ling Q4→Q3 labels (median 2.54 / 2.23).

On Ling-3.0-tiny, against the NETPURE builds at the same target:

| budget | build | MiB | PPL | KLD mean | KLD median | KLD max |
|---:|:---|---:|---:|---:|---:|---:|
| — | stock Q4_K_M (vendor) | 4600 | 13.327 | 0.114020 | 0.048512 | 14.354789 |
| 4500 | NETPURE 4500 | 4507 | 13.503 | 0.095092 | 0.039573 | **8.715995** |
| 4500 | **WATERFILL 4500** | 4516 | **13.1785** | **0.074614** | **0.030849** | 10.109997 |
| 5100 | NETPURE 5100 | 5091 | 13.4271 | 0.089046 | 0.037072 | **9.071382** |
| 5100 | **WATERFILL 5100** | 5116 | **13.0611** | **0.063126** | **0.025847** | 9.515678 |

Against stock Q4_K_M at 4600 MiB, **WATERFILL 4500 is 84 MiB smaller and
better on every column**: PPL −1.1%, mean KLD −35%, median −36%, max −30%.
Against NETPURE at 4500 MiB: mean KLD −22%, median −22%, PPL −2.4%, at
+9 MiB; max KLD is 16% worse (still below stock Q4_K_M's 14.35). At 5100 MiB
against NETPURE: mean −29%, median −30%, PPL −2.7%, at +25 MiB; max KLD is
4.9% worse. That is a fidelity-per-byte result, not a clean sweep. The
4500/6500/9000 allocations and result files are kept in the local lab
provenance (`provenance/waterfill/`, gitignored) and mirrored to the
Hugging Face repo.

---

## 5. Gnom — the learned ranker

**What it is.** A tiny neural ranker (150 params, CPU, seconds) that learns which tensors
carry quantization damage instead of guessing importance from proxy signals. It is trained
on **measured** damage labels, then ranks models it has never seen.

**The labels** — [`MERNIK-Fracture`](https://huggingface.co/datasets/wepiqx/MERNIK-Fracture):
per-tensor KLD damage on a reasoning corpus (`rl_reasoning_mini`), five columns per row
(mean / max / median KLD + Δp RMS + same-top), deterministic, uniform schema across models
and tiers. The 9B file took ~13 days of continuous sweeping; the 1.7B/4B legs minutes each.
**Cheap instruments are for the shelf; the expensive one is for the model you care about.**

**The corpus ships with the repo** — `corpus/rl_reasoning_mini.raw` (343 prompts,
479,633 B, sha256 `82b2e153…de400866fbe`; the same bytes the forge and GTX-1070 legs
measured against). Rebuilding a reference against it is one
`llama-perplexity --save-all-logits` call on your machine, so a volunteer run reproduces
the exact measurement stack.

**What the labels bought** (clean rlmini pool, 512 units, 768 tensors, 0 censored):

| exam | result |
|:---|:---|
| strict leave-one-MODEL-out, trunk variant | **LOMO 0.801** (bar 0.45, ×6 MDE) — held-out rl17 0.805, held-out rl4b 0.798 |
| same, modelbias | 0.689 PASS |
| same, moeshared | 0.179 FAIL |
| static imatrix proxy (timp) | **−0.103** (the learned ranker beats the proxy by a mile, on unseen models) |
| mixed 4-fold exam | moeshared LOMO 0.579, rank LOMO-acc 0.760 |

The old "no transfer" verdict was an artifact of **contaminated labels**; on clean data the
transfer exists and is large. See [L2](#l2--the-depth-law-2026-10-21-early-layers-carry-the-damage)
and [L3](#l3--the-layer-0-spike-a-learned-ranker-beats-a-monotone-prior) for what the
ranker actually learned, and [§0](#0-tldr--what-the-last-weeks-produced) for what it buys
on a real build.

**The teacher** (5-column, KLD-only, 2026-10): `group_damage_sweep.py --kld-only` skips
the plain-PPL pass (it lies) and scores five columns per unit against the unquantized
reference on `rl_reasoning_mini`: `damage_kld` (mean), `damage_kld_max` (tail
detonations), `damage_kld_median` (core), `damage_dp_rms` (decisiveness),
`damage_same_top` (rank). `ppl`/`damage` are `null` — not measured.

* **Machine split:** the 9B (Ornith) sweeps on the **16 GB Vega-11 forge**
  (memory-bandwidth-bound; a 9B is unreasonable to torture through a 1070), the 1.7B and 4B
  on the **GTX 1070**.
* **Cost:** 9B ≈ 90 min/unit × 210 = **~13 days** continuous (the crown-jewel file);
  1.7B ≈ 7.5 h; 4B ≈ 5 h.
* **Result:** the [`MERNIK-Fracture`](https://huggingface.co/datasets/wepiqx/MERNIK-Fracture)
  dataset — complete, clean, uniform schema — the origin of the project's current paradigm.

**Adding a leg to the pool:**

1. sweep the model: `group_damage_sweep.py --model M.gguf --imatrix I.gguf --kld-base
   ref.dat --kld-only --drop-tier Q4_K` (and Q3/Q2 for the curve);
2. `build_features.py --model M.gguf --imatrix I.gguf --out gnom/features/<m>.npz`;
3. `scripts/consolidate_labels.py` → clean pool + `pool.json`;
4. add the row to the pool file, retrain, check the bars (0.45 / 0.60).

---

## 6. Slow-ring protocol (HumanEval execution)

Server (one model at a time, GPU is a strict queue):

```bash
llama-server -m MODEL.gguf --port 28082 -ngl 99 -c 8192 --jinja --log-disable
```

Sampling (protocol default):

```text
temperature 1.0, top_p 0.95, top_k 20, min_p 0.0,
presence_penalty 0.0, repetition_penalty 1.0, max_tokens 2048
```

`presence_penalty 1.5` (vendor recipe) breaks thinking templates; `0.0` verified. Preflight
`/health` before every battery; mid-run watchdog every 10 tasks (dead server aborts LOUDLY,
never scores empties silently). Long-ctx servers accumulate commit (KV + prompt cache never
shrink) — chunked batteries (`HE_CHUNK`, fresh server per ~30 tasks, stitch in order) plus
`--cache-ram 0` (164 distinct prompts share nothing). Family sampling: thinking families run
their vendor recipe (K2: temp 1.0 + top_p 0.95 only, top_k 40 / min_p 0.05 = llama defaults,
ctx 16384, `--reasoning-effort high`; Spark-X2.6: temp 1.0, top_p 0.95, top_k 0, min_p 0.05,
effort high, ctx 16384, max_tokens 8192) — cross-family verdict comparisons stay void,
different rulers. Runner: `scripts/run_humaneval.py` (env `HE_*`, `HE_CHUNK`, `HE_CTX`,
`HUMANEVAL_OUT`, `HUMANEVAL_SERVER`; `HUMANEVAL_SERVE_MODEL` = self-serve with auto-kill).
Eval: `human_eval.evaluation.evaluate_functional_correctness`, pass@1, k=[1]. Results:
`eval_results/humaneval_<tag>.jsonl`. Strictness upgrade: every battery is rescored with
EvalPlus HumanEval+ (80× tests, CPU) — the HE+ column.

Reproduce any number below (runner + rescorer ship in this repo):

```bash
export HUMANEVAL_SERVE_MODEL=/path/to/MODEL.gguf HUMANEVAL_OUT=eval_results/humaneval_mytag.jsonl
export HE_TEMP=1.0 HE_TOP_P=0.95 HE_TOP_K=20 HE_MIN_P=0.0 HE_PRESENCE=0.0 HE_MAX_TOKENS=2048
python scripts/run_humaneval.py          # serves, runs 164 tasks, kills server, writes .meta.json sidecar
evalplus.evaluate humaneval --samples eval_results/humaneval_mytag.jsonl   # HE+ column
```

The `.meta.json` sidecar (model path, git sha, full sampling, timestamps) is written at
battery time — it is what makes a new battery verifiable instead of
permanent-unverifiable.

Answered 2026-09-29: the same file twice at temp 0.0 scored 98/164 twice, per-task verdicts
164/164 identical, completions 163/164 byte-identical. Greedy does not narrow the verdict
column — it **reproduces** it. The column can stop being sampled: temp-0 batteries are
certificates, temp-1.0 batteries are draws. (Caveat kept: the sampler skips RNG at temp ≤ 0,
but logits still depend on batch composition — so far, no spread observed.)

### Fast-ring diagnostic: GPQA-Recognition

Recognition decisiveness (first-token P(letter) over top-30, no CoT). Via
`scripts/gpqa_duel.py`.

| Build | GPQA-rec | HE pass@1 | Note |
|:------|---------:|:---------:|:-----|
| Neo-MSE-6500 | 51.5% | 85.98% | Thinker spreads first-token mass |
| Neo-SMAPE-5100 | 48.0% | 82.93% | Compact barbell |
| Ox-MSE-6500 | 49.5% | 90.24% | Decisive stabs the letter |
| Ox-SMAPE-5100 | 49.0% | 88.41% | High task execution |

---

## 7. Reference results (links — the tables live on the family cards)

| family | card | load-bearing line |
|:---|:---|:---|
| NeoHorse-1-9B | [wepiqx/NeoHorse-1-9B-MERNIK-GGUF](https://huggingface.co/wepiqx/NeoHorse-1-9B-MERNIK-GGUF) | 6500-MSE 141/164 (85.98%); 5100-SMAPE 136/164 ties stock Q6_K 135/164 at −2.2 GB |
| OxCoder-9B | [wepiqx/OxCoder-9B-MERNIK-GGUF](https://huggingface.co/wepiqx/OxCoder-9B-MERNIK-GGUF) | SMSE-6500 150/164 (91.46%), Q8 148/164, SMAPE-5100 145/164 — one cluster, no crown inside it |
| RINIQ series (layer-fused) | [wepiqx/RINIQ-GGUF](https://huggingface.co/wepiqx/RINIQ-GGUF) | nine-way statistical tie (151, then eight on 150) — no crown, the top is shared |
| Mellum2.1-12B-A2.5B | [wepiqx/Mellum2.1-12B-A2.5B-Thinking-MERNIK-GGUF](https://huggingface.co/wepiqx/Mellum2.1-12B-A2.5B-Thinking-MERNIK-GGUF) | 10000-SMSE 134/164 beats stock Q6 119/164 by +15 at −0.7 GB (**3/3 SIGNIFICANT**); 7500-SMSE ties stock Q4_K_M 125/125 (triple p=1.0) |
| FrogNano-4B | family card (lab notes) | the α-axis laboratory: 110→117→120→122→109, and stock Q6 wins the temp-0 duel 3/3 (−16 tasks real) |
| K2-Horizon-7B | family card | thinking family, saturated shelf: 5500 SMSE 147/164 vs stock Q4_K_M 152/164, Q6 148/164 — all inside the noise floor |
| **Spark-X2.6-4B** | family card | **the current generation** — see [§0](#0-tldr--what-the-last-weeks-produced) |

**The noise budget, as a number.** Verdict noise is a binomial draw — SD 5.9 tasks/run at
p=0.70, 8.3 for a difference of two runs. That IS the ±3.6pp band and IS the 8-task MTP
re-run gap. Below ~8–10 tasks nothing is reproducible at n=164 — a property of the
instrument, not a measurement error. Verdicts travel with MDE; the column certifies
preservation, not rank (nine-way tie at the top: 151, eight on 150).

**Verdict ledger** (the duels that shaped the method — all pre-KLD-paradigm, kept so the
same experiments are not re-run):

| duel | numbers | verdict |
|:---|:---|:---|
| MiMo-4500 SMSE vs SMAPE (allow-q3) | PPL 9.4092 vs 10.5343; HE **54.88% vs 29.88%** | **lava rescue** (+25pp) — L0 |
| MiMo-5100 SMSE vs stock | PPL 8.7389 vs 8.7435 (tie); HE 71.34% vs 70.12% (+1.2pp), 3 empties | **0/3** (HE p=0.88, HE+ the other way, empties p=0.13) — NOISE, crown demoted |
| MiMo-6500 SMSE vs stock | PPL 8.7275 vs 8.7655 (tie); HE 72.56% vs 75.00% (−2.4pp) | SMSE **loses** big: at 6500 the middle decides (MSE Q6 67 vs SMSE 41), not penthouses |
| Mellum-10000 SMSE vs stock Q6 | HE 134 vs 119, −0.7 GB | **3/3 SIGNIFICANT** — the one real quant-vs-stock win |
| FrogNano-3500 stock Q6 vs ours | temp-0 duel, −16 tasks | stock wins 3/3 — RL-sharpened base, allocation win absent |
| netdmg (Gnom modulator) vs SMSE | — | **0/3** — manners without a verdict; became `netpure` |
| RINIQ crown | 151, then eight on 150 | **nine-way tie** — no crown, the top is shared |

---

## 8. Advanced experiments and lab notes

The repo carries several long-form notes that sit beside the code. They are lab
notebooks, not required reading: each records one line of experiments, including
the scars and dead ends. This README is the active documentation; `LEGACY.md` is
the only one needed to reproduce archived flags.

| file | what it is |
|:---|:---|
| `DIARY.md` | chronological session journal (2026-09-22 → 27), the context-restore file |
| `FUSION.md` | layer-interleaved fusions: donor maps, soup, shape guards, specimens |
| `MIMO.md` | MiMo-V2.6-Distill-Qwen-9B workbench and its distillate notes |
| `MTP.md` | speculative-decoding draft duels (speed vs verdict) |
| `README-ZOO.md` | utility-metric playground: the older MSE/SMAPE/SMSE duels |
| `RINIQ-NEXT.md` | Ox × MiMo fusion lineage, recipes and corrections |
| `SAGA.md` | the OxCoder+MTP graft saga, attempt by attempt |
| `LEGACY.md` | archived utility modes and removed flags, frozen for reproduction |

### MTP head grafting

Transplant of a 15-tensor (464 MB) MTP donor head onto OxCoder-9B via `scripts/graft_mtp3.py`.
Draft engages (5–23% acceptance); economics negative (31→20 t/s on mismatched head).
Mechanics proven; full saga in `SAGA.md`.

### Teacher & Gini (v3, 2026-09-19)

Damage-sweep labels train the Gini allocator. v3 complete: 168/168 labels (Q5_K→Q2_K drops,
MiniCPM5-2B, forge): damage −0.046…+0.515, mean 0.070, Gini **0.364**, only 3 reliefs. Late
units carry the signal (ffn_gate@41 +0.51, ffn_down@41 +0.33, ffn@39/40) + early attn@2. vs
older teacher (Gini 0.498, max 0.173): bigger drops hit harder (max 3×) but spread wider
(Gini lower) — signal with teeth, less concentrated. MLP un-paused: retrain Gnom on v3 →
netdmg duel. Gini answers to Soulfate24. *(Superseded as the main instrument by the KLD
teacher and Gnom — §5.)*

### KLD battery & relief

KLD-vs-ref was the allocator's rank column (Soulfate24 battery: fixed span, fixed chunks,
Flash-Attention). Measured relief ceilings (`--relief`) pin sweet-spot groups at Q4 and
release their budget elsewhere. It is now a full-model verdict column (§1) — the same
mechanism, measured against the BF16 base instead of a per-decision proxy.

### Ceiling & inversion probes (2026-09-30/10-01)

- **Ceiling map:** `RINIQ-M2-MAXF16` (9.01 GB, F16 core 123) measured PPL 7.7357 (worse than
  6500-MSE's 7.5263), HE 137/164, HE+ 129 — +4 GB over the 5 GB cluster buys −13 tasks.
  Kings saturate (Q8→F16 gains nothing) while the Q4 tail drags: at big budgets the MIDDLE
  decides.
- **Inversion (adversarial control, `--invert` / `--invert-keep K`):** full priority flip at
  8350 (PPL 7.6960, HE 145/164, HE+ 137) and at 5100 (PPL 7.5845, HE 145/164, HE+ 137) —
  duels vs honest arms 0/3 and 0/3. Flipped noise ties, so the lens carries nothing
  resolvable.
- **Doctrine consequence:** the FLOOR is the whole game, priority among Q4+ is noise (L0/L1).

### F16-blend (`--utility gnom --top-down`)

At bigger budgets the useful move is not "which tensor gets one more tier" but **"which
tensor must not be touched at all"**. Gnom top-down keeps the fragile blocks in full precision
and demotes by damage per byte saved. On X2.6-4B at 4050 MiB: 46 tensors in F16 (1510 MiB),
PPL 18.6228 — beating a solid stock Q8_0 (4172 MiB, 19.0307) while being smaller. This is
L2/L3 turned into an allocator mode.

---

## 9. Architecture support

MERNIK has no supported-architecture list and no hand-maintained architecture
table. Earlier versions required the architecture to be declared and kept a
manual mapping; that path is gone. Today MERNIK runs `llama-quantize --dry-run`
and consumes what the C++ loader reports: `general.architecture`, llama.cpp's
generic metadata keys (`<arch>.block_count`, `<arch>.expert_count`,
`<arch>.expert_feed_forward_length`, `<arch>.nextn_predict_layers`) and the
full tensor inventory with shapes and source types. There is no preset table
and no architecture-name fallback: if llama.cpp loads the model, MERNIK can
allocate it; if llama.cpp cannot load it, both tools fail loudly before a byte
is written.

The broad-coverage claim is a theory rather than an exhaustive test: not every
architecture has been measured yet. That is not a support limit — an untested
architecture is an untested path, not an unsupported one. There is no whitelist
to reject it, and it should work as long as llama.cpp reports the standard
metadata and tensor inventory. Tested examples are `qwen35` (OxCoder, MiMo,
Ornith, FrogNano),
`bailingmoe3` (Ling), `mellum`, `spark2_5`, `k2-horizon`, and `llama`
(MiniCPM5). That is evidence, not a whitelist.

The same contract covers custom llama.cpp forks and their models: point
`LLAMA_QUANTIZE` at the fork's binary and MERNIK reads the fork's loader output
instead of assuming an upstream version. The dry run reads headers only, not
tensor data (Ling BF16: ~0.2 s, ~77 KiB of output). `llama_inspect.py` is the
adapter; the old `model_reader.py` is gone.

Special cases are derived from the same metadata:

- **MTP:** `nextn_predict_layers`, or `nextn`/`blk.32` tensors, pin the head
  outside the regular layer count.
- **MoE:** `expert_count > 0` or `*_exps` tensors; K-quant padding uses
  `expert_feed_forward_length`.
- **1D and `ssm_conv1d`:** llama.cpp writes them F32, so the allocator pins
  them outside the budget.

---

## 10. Ecosystem & repositories

- 📦 [NeoHorse-1-9B-MERNIK-GGUF](https://huggingface.co/wepiqx/NeoHorse-1-9B-MERNIK-GGUF)
- 📦 [OxCoder-9B-MERNIK-GGUF](https://huggingface.co/wepiqx/OxCoder-9B-MERNIK-GGUF)
- 📦 [MiMo-V2.6-Distill-Qwen-9B-GGUF-MERNIK](https://huggingface.co/wepiqx/MiMo-V2.6-Distill-Qwen-9B-GGUF-MERNIK)
- 📦 [RINIQ-GGUF](https://huggingface.co/wepiqx/RINIQ-GGUF)
- 📦 [RINIQ-NEXT-GGUF](https://huggingface.co/wepiqx/RINIQ-NEXT-GGUF)
- 📦 [Ornith-1.5-9B-MTP-ASHQ1-GGUF](https://huggingface.co/wepiqx/Ornith-1.5-9B-MTP-ASHQ1-GGUF)
- 📦 [Spark-X2.5-1.7B-ASHQ1-GGUF](https://huggingface.co/wepiqx/Spark-X2.5-1.7B-ASHQ1-GGUF)
- 📦 [Mellum2.1-12B-A2.5B-Thinking-MERNIK-GGUF](https://huggingface.co/wepiqx/Mellum2.1-12B-A2.5B-Thinking-MERNIK-GGUF)
- 📊 [MERNIK-Fracture](https://huggingface.co/datasets/wepiqx/MERNIK-Fracture) — **the labels**:
  KLD-only, 5-column, reasoning-corpus damage (the origin of L2/L3)
- 📊 [MERNIK-rlmini-kld](https://huggingface.co/datasets/wepiqx/MERNIK-rlmini-kld) — early KLD labels
- 📊 [MERNIK-gnom-labels](https://huggingface.co/datasets/wepiqx/MERNIK-gnom-labels) — legacy PPL-era
  labels (retired prototype)

Lineage: Apache-2.0. Built with [Soulfate24](https://huggingface.co/Soulfate24), whose KLD
battery ended our PPL theater. Field journals with every scar: `FUSION.md` (monsters),
`README-ZOO.md` (utility duels), `SAGA.md` (MTP graft), `LEGACY.md` (archived
modes and removed paths), `DIARY.md`.

---

---
