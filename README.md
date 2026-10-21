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
expanded capabilities (slow capability ring, KLD-aware teachers, relief ceilings, the zoo
bench, and norms shields).

---

## 0. TL;DR — what the last weeks produced

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
beats blind spreading*. And the whole gap between SMSE and Gnom is **three tensors**:
SMSE upgrades `blk.27.attn_gate/qkv` + `blk.35.attn_output`; Gnom upgrades
`blk.0.attn_gate/qkv` + `blk.11.attn_output` — same count, same bytes, and that is the
*entire* difference between 19.2660 and 18.3646.

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

Any tier below Q4 hurts, and **fewer is better**. `--allow-q3-or-lower` at ~4 GB is lava in
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

### L2 — The depth law (2026-10-21): early layers carry the damage

Damage concentrates in the **early blocks**; deep ones are safe. Measured on the rlmini KLD
labels: 43.2% / 36.4% of all damage sits in layers 0–5 of the 1.7B / 4B, and only
3.2% / 7.4% in layers ≥24. A ranker trained on those two models predicted the same
profile for a third it had never seen (Ornith-1.5-9B, hybrid SSM, 33 layers: 48.3% vs
6.4%) — **zero-shot transfer across architecture**.

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

### L3 — The layer-0 spike: a learned ranker beats a monotone prior

Layer 0 is not merely "early", it is an **outlier**: mean damage in layer 0 vs the average
of layers 1–3 is **2.3× (1.7B Q4), 1.6× (1.7B Q3), 4.6× (4B Q4), 3.3× (4B Q3)**. A
monotone prior ("damage decays with depth") cannot express that — it puts the first block
at rank 1 or 2. The learned ranker can, because it learns the *profile*, not the trend: it
upgrades `blk.0.attn_gate/qkv` where the hand-picked law build takes `blk.7`. That single
decision is worth **−0.24 PPL** (18.3646 vs 18.6080) — the gap between the learned
allocator and the best monotone reading of its own law.

### L4 — The 1D/F32 law

The binary writes 1D tensors (norms, biases) and `ssm_conv1d` as **F32 whatever the rules
say**, so `--pin-norms` is a proven no-op (it warns when it changes 0 tensors) and the old
"norms are load-bearing precision" reading was **budget-redirect, not precision**. The
pre-flight audit exists to catch exactly this class of mistake.

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
| `--model M.gguf` | Source weights (BF16/F16) — required |
| `--imatrix I.gguf` | Imatrix file; repeatable (`--imatrix A --imatrix B --imatrix-method max\|mean`) |
| `--imatrix-method max\|mean` | How repeat imatrix lenses combine (default: max) |
| `--size MIB` | Target file size in MiB (the primary budget knob) |
| `--output O.gguf` | Output path (default: `<model>-MERNIK.gguf`) |
| `--run` | Execute `llama-quantize` (without it: dry-run estimate only) |
| `--utility NAME` | Queue gain metric — see the modes table below |
| `--mix-base NAME` | MIX utility: gain for non-king groups (`mse`/`rmse`/`smape`/`logcosh`) |
| `--mix-top-frac F` | MIX utility: fraction of top-importance king groups (default: 0.2) |
| `--top-down` | Reverse allocation: everything from F16, downgrade cheapest-loss-first |
| `--pin-norms` | Top-down native norms shield (proven no-op — warns when it changes 0 tensors) |
| `--allow-q3-or-lower` | CAN_Q3 types (`ffn_gate/up/down`, `attn_output`, `ssm_out`) may drop to IQ2_XXS |
| `--pin-small MIB` | Pin every tensor under MIB MiB (F16 weight) at F16, outside the budget (bottom-up only). Measured: Mellum k/v heads (2.3 MB F16) cost 126 MiB for 56 tensors — protection priced loudly, 0 = off |
| `--relief labels.jsonl --relief-thr -0.5` | Groups below threshold get ceiling Q4; budget flows elsewhere |
| `--relief-thr F` | Relief cutoff (default: −0.5) |
| `--max-tier T` | Runtime override of CLASS_MAX_TIER for every class (e.g. F16 lets kings rise above Q8_0); bottom-up only, `constants.py` untouched |
| `--free-pins` | EXPERIMENTAL: output/token_embd/MTP/routers join the budget pool |
| `--squeeze` | SQUEEZE mode (separate path): only IQ1_S or F16 per tensor, bottom-up only |
| `--legacy-1d` | LEGACY: price 1D tensors (norms/biases) and `ssm_conv1d` as quantizable F16 — reproduces pre-2026-09-28 budgets |
| `--imatrix-tol F` / `--imatrix-legacy-combine` | Refuse when lens files diverge beyond tolerance (default 1%); legacy flag restores max/mean-combine with a split-brain binary |
| `--verify gpqa\|he\|all` | Slow-ring verify on fresh build or existing `--output` (busy GPU port aborts LOUDLY, never steals) |
| `--verify-tag TAG` | Result tag (default: from `--output` basename) |
| `--show-config` / `--show-floors` | Print tensor config / hard tier floors |
| `--no-preflight` / `--strict-preflight` | Skip the dry-run audit, or abort if any assigned tier cannot fire (default: audit runs, ~1s) |
| `--ssim-table P` / `--ptable P` / `--frag-w F` | ssim / pw_ssim tables + smape_frag modulation weight (default 0.5) |
| `--huber-delta D` | huber/logcosh scale splitting small vs large deltas (default 3e-4) |
| `--pow-alpha A` | The α-axis itself: g = ec^A−en^A (1=MSE, 0.5=RMSE, →0 log-ratio) |
| `--hinge-w W` | Smooth sub-4 boost multiplier on any mode (bias on the decision, not the loss); 0 = off |
| `--cv-w F` / `--linf-w F` | Hybrid (dead by construction): concentration discount / worst-case boost weights |
| `--netpred P` / `--netdmg-w F` | Gnom predicted damage per group + modulator weight (default 0.5) |
| `--invert` | ADVERSARIAL CONTROL (operator's idea): negate importance — kings down, junk up. If it still scores, importance is bunk; if it collapses, importance is causal |
| `--invert-keep K` | Soft invert: top-K importance groups exempt (small-budget dial; inert at big budgets where the ceiling binds, not priority) |
| `--aggro F` | [deprecated] Use `--size` instead |
| `--verbose` | Detailed output |

### Utility modes — one row per belief

Each mode is a *belief about where damage lives*. This table is the extension point for new
modes.

| mode | belief | status / evidence |
|:---|:---|:---|
| `mse` (default) | relative-blind absolute gain; tiers spread evenly (Q5/Q6-heavy middle) | best PPL/KLD on dense 9B |
| `smape` | barbell: junk to the Q4 floor, kings to Q8 penthouses | owns small budgets, loses big ones (L1) |
| `smse` | geometric mean of the SMAPE-relative and MSE-absolute halves; SMAPE's `(ec+en)` denominator silently cancels the sub-4 toxicity penalty, so the blend carries real toxicity through its absolute half | the best hand-tuned mode; owns small+medium budgets (MiMo-4500 lava rescue, +25pp) |
| `rmse` | perfectionist rescale of MSE | zoo: beats MSE on PPL twice; RINIQ-M2-6500-RMSE-Q38: PPL 7.5411, GPQA 50.00%, **HE 88.41%** — takes recognition, trails MSE by 3pp on verdict (sub-floor: direction, not rank) |
| `srmse` | SMAPE-relative × compressed absolute (sqrt squeezes toxicity ×1.41 instead of ×2.0) | demoted 2026-09-28; MiMo-5100 duel 0/3 (109–115 on a bit-identical file) — manners without verdict. Filed, not crowned |
| `huber` | fat middle: Huber loss on deltas pushes Q4/Q6 stragglers into Q5_K | best of six at FrogNano-3500 (122/164; SMSE 118, MSE 109, logratio 110, α0.25 117, α0.75 120); Mellum-7500 123 vs SMSE 125, duel 0/3 — the middle does not decide there |
| `logratio` | left edge of the α-axis: log(ec/en) — extreme barbell, sub-4 toxicity survives structurally | 110/164 on FrogNano-3500 |
| `powalpha` (+ `--pow-alpha A`) | the axis itself: g = ec^A−en^A | 117 → 120 as A goes 0.25 → 0.75; MSE (A=1) collapses to 109. The axis is **not monotonic**; the pure-absolute endpoint is the worst seat |
| `recovery` | asymmetric relative: (ec−en)/ec, fraction of current damage recovered | ties SMSE at Mellum-6500 (123 vs 126) |
| `mix` (+ `--mix-base`) | kings by MSE, rest by another base metric; formulas live on different scales (mse Δ ~1e-3 vs smape ~1.2), so `_scale()` normalizes every formula to O(1) first — without it smape junk outbids mse kings 800:1 and the mix collapses | dry-run geometry is new (wide Q5 spread, no Q8); verdict queued |
| `ssim` / `smape_ssim` / `smape_frag` / `pw_ssim` | SSIM-flavoured variants | table-driven, `--ssim-table` / `--ptable` |
| `netdmg` | modulator: Gnom-predicted damage as a bias on the lens | duel vs SMSE came back 0/3 — manners, not a verdict |
| **`netpure`** | **the queue's gain IS the predicted damage; group weight 1.0 → demotion order is damage per byte saved; the static lens is not re-inserted** | X2.6-4B @2250: PPL **18.3646** (vs 19.2660 SMSE), max KLD vs BF16 **14.17 vs 17.77** (−20%) |
| **`netfd`** | **F16-blend for bigger budgets: the fragile blocks are kept in full precision instead of upgraded a tier (top-down only)** | X2.6-4B @4050: 46 tensors in F16 (1510 MiB), PPL **18.6228** — beats a solid stock Q8_0 (4172 MiB, 19.0307) **while being smaller** |

`--hinge-w` (sub-4 boost multiplier): 105/164 on FrogNano, −13 vs plain SMSE — the "free
boost" backfired, filed as dead.

### Recently added (2026-09-28 → 2026-10-16, all CPU-cheap, all verified)

- `scripts/fuse_layers.py --expect "15:b,..."` — gates the resolved donor map (ranges
  expanded, tissue suffixes checked); exits 2 with a diff before one output byte. Plus a
  soup shape-guard: mismatched donor shapes fail LOUD, never a corrupt file.
- `scripts/group_damage_sweep.py --units TAGS --timeout S --ngl N` — subset sweeps; failures
  go to `*.failures.jsonl`, second failure of a unit becomes a TERMINAL censored label
  (`damage: null`, never retried, never imputed). `--kld-only` skips the plain-PPL pass and
  emits the 5-column KLD family.
- `--max-tier F16` lets kings rise above Q8_0 without touching `constants.py` (reproduces
  the raised-ceiling geometry byte-exact in dry-run; 13/13 golden geometries still green).
- MoE golden geometries: `Mellum2.1 @7500 smse` (Q4 171 / routers F16 ×28) and `huber`
  (Q4 163 / Q5 34) — first regression net over a Mixture-of-Experts allocation
  (14/14 green + 8/8 provenance).
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
    --utility powalpha --pow-alpha $A   # concentration↔distribution knob
done
python main.py --model M.gguf --imatrix M.imatrix.gguf --size 3500 \
  --utility logratio                     # left edge: extreme barbell
python main.py --model M.gguf --imatrix M.imatrix.gguf --size 3500 \
  --utility smse --hinge-w 0.5           # sub-4 boost on any mode
```

---

## 4. Engine mechanics

### Queue & allocation principles

Every tensor starts at a floor tier based on its functional class. A global max-heap drains
the target budget best-first by sum(importance) × Δ / MiB. Tied groups (identical imatrix
energy) upgrade as a single unit with summed importance. Structural pins — MTP heads
(→ Q8_0), output/token embeddings (→ Q5_K), MoE routers (assigned F16, F32 in file) — sit
outside the budget queue.

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

## 8. Advanced experiments

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

### F16-blend (`--utility netfd`) — the next generation

At bigger budgets the useful move is not "which tensor gets one more tier" but **"which
tensor must not be touched at all"**. `netfd` keeps the fragile blocks in full precision
and demotes by damage per byte saved. On X2.6-4B at 4050 MiB: 46 tensors in F16 (1510 MiB),
PPL 18.6228 — beating a solid stock Q8_0 (4172 MiB, 19.0307) while being smaller. This is
L2/L3 turned into an allocator mode.

---

## 9. Supported architectures

| Arch | Detection | Key Features |
|:-----|:----------|:-------------|
| `qwen35` | SSM + QKV | Hybrid attention, SSM layers, GQA, MTP head pinning |
| `mellum2` | MoE (`exps` tensors) | Mixture of Experts, GQA, routers out-of-budget (F32 in file, 28/28 incl. stock) |
| `bailingmoe3` | KDA+MLA + MoE | Ling-3.0 family: routed + shared experts |
| `granite` | `general.architecture` | Dense GQA, 40 layers, split Q/K/V |
| `spark2_5` | `general.architecture` | Dense + hybrid sliding-window attention (1 full + 3 SWA) |
| `k2_horizon` | `general.architecture` (`k2-horizon` alias) | Dense GQA thinking family, 36 layers, split attn_q/k/v; vendor sampling (temp 1.0 + top_p 0.95 only), ctx 16384, `--reasoning-effort high` (NOT qat — without the row it misdetects as gemma4) |
| `lfm2moe` | `general.architecture` | Liquid LFM2.5 hybrid: 30 short-conv + 10 GQA blocks, MoE (top-4/64), 131K ctx; `shortconv` weight-stats class |
| `gemma4` | layer-scale norms | QAT support, Q4_K attention floor |
| llama (generic) | tensor names | Dense GQA (MiniCPM5, NeoHorse, etc.) |


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
`README-ZOO.md` (utility duels), `SAGA.md` (MTP graft), `DIARY.md`.

---

---
