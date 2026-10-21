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

# MERNIK — Measure-First Quantization Protocol (v2)

> **This is the reorganized draft.** The published README is [`README.md`](README.md); this
> file is where the new shape is being built, so no detail is ever lost in the move. When it
> settles, it replaces the old file.
>
> **How this file is meant to be used:** write here first (notebook-style, numbers and
> claims as they arrive), then read it back and fold it into `README.md`. Every section
> ends with a *→ where new material goes* note. See [§11](#11-how-to-extend-this-readme).

**MERNIK** ("the one who measures") is the evolution of the ASHQ1 battlefield zoo: fewer
utility duels, more verdicts. The method (priority-queue allocation) is built; MERNIK is how
we prove and certify it.

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

**The PPL ladder** (wikitext-2-raw test, ctx 1024, seed 7 — same corpus, same harness):

| build | weight | PPL | note |
|:---|---:|---:|:---|
| **MERNIK-2250-Gnom** | 2254.8 MiB | **18.3646** | best arm; the smallest one |
| MERNIK-2250-LAWONLY | 2254.8 MiB | 18.6080 | law picked by hand (layers 7, 11) |
| MERNIK-4050-GNOMF16 | 4053.4 MiB | 18.6228 | F16 on the fragile layers (46 tensors) |
| *BF16 base* (anchor) | 7848.7 MiB | *18.8263* | zero-loss reference, not a quant |
| stock Q8_0 | 4172.3 MiB | 19.0307 | "almost lossless" — beaten by all three of ours |
| MERNIK-2250-SMSE | 2254.8 MiB | 19.2660 | imatrix-driven (layer 27) |
| stock Q4_K_M | 2479.8 MiB | 20.6148 | vendor mix (853 MiB in Q6) |

Three of our quants beat the stock Q8_0 **while being smaller**; the smallest (2255 MiB)
beats everything including the BF16 anchor's own perplexity. "Just quantize to Q8" is not
the better strategy — deciding *what* to keep at full precision is.

**The law behind it** (measured on rlmini KLD labels, three models, transfers zero-shot):

| model | share of damage in layers 0–5 | in layers ≥24 |
|:---|---:|---:|
| Spark-1.7B-Q4 (measured) | 43.2% | 3.2% |
| Spark-4B-Q4 (measured) | 36.4% | 7.4% |
| **Ornith-1.5-9B (ranker zero-shot)** | **48.3%** | **6.4%** |

→ *new results: this table, then a link to the family card. Never a family table here —
those live on their own cards (§7).*

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
> **2026-10-16 revision.** KLD moved from the allocator ring to the verdict ring: the
> KLD column is a *full-model* measurement against the unquantized base over the whole
> corpus (302 chunks, ~24 min/build), not a cheap per-decision signal. PPL is now the only
> fast-ring column — and it is a **canary**: cheap, always available, and frequently
> misleading (it saturates on reasoning-critical layers and can be bought by sharpening).
> A build that only wins on PPL has won nothing.

*Decision rules* (unchanged): **Divergence** — if metrics disagree, KLD wins allocator
arguments; if they agree, trust the pair. **Silence is signal** — empty completions scale
with weakness (Ox 4–5 → Neo 20–27 → Q2_K 63 → S2 115; Mellum-10000 24 → 7500 35 → Q6 39);
weak models go silent rather than babble. **Saturation rule** — bases at ~98% cannot
resolve fine differences; capability duels need bases at 60–85% or they only answer the
collapse threshold.

→ *new metric: decide its ring here, in one line, with the reason.*

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

Direction only, not rank — the legs differ by 5–6 tasks, below the 13-task resolution
floor (6500: 141 vs 135; 5100: 136 vs 131). The lava leg survives on 3/3 columns plus
effect size. Above Q4, utilities debate inside the floor; below Q4, physics decides.

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

Bases at ~98% cannot resolve fine differences (97% vs 96%). Capability duels require
bases at 60–85%, or they only answer the collapse threshold.

→ *new law: append here — name, date, the number that proves it, and the model it was
measured on.*

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

Each mode is a *belief about where damage lives*. The table is the extension point for new
modes.

| mode | belief | status / evidence |
|:---|:---|:---|
| `mse` (default) | relative-blind absolute gain; tiers spread evenly (Q5/Q6-heavy middle) | best PPL/KLD on dense 9B |
| `smape` | barbell: junk to the Q4 floor, kings to Q8 penthouses | owns small budgets, loses big ones (L1) |
| `smse` | geometric mean of the SMAPE-relative and MSE-absolute halves | the best hand-tuned mode; owns small+medium budgets (MiMo-4500 lava rescue, +25pp) |
| `rmse` | perfectionist rescale of MSE | zoo: beats MSE on PPL twice; RINIQ-M2-6500-RMSE-Q38 HE 88.41% (sub-floor: direction, not rank) |
| `srmse` | SMAPE-relative × compressed absolute | demoted 2026-09-28 (0/3) — filed, not crowned |
| `huber` | fat middle: Q4/Q6 stragglers migrate to Q5_K | best of six at FrogNano-3500 (122/164); ties at Mellum-7500 |
| `logratio` | left edge of the α-axis: log(ec/en), extreme barbell | 110/164 on FrogNano-3500 |
| `powalpha` (+ `--pow-alpha A`) | the axis itself: g = ec^A−en^A | 117→120 as A goes 0.25→0.75, MSE (A=1) collapses to 109 |
| `recovery` | asymmetric relative: (ec−en)/ec, fraction of damage recovered | ties SMSE at Mellum-6500 (123 vs 126) |
| `mix` (+ `--mix-base`) | kings by MSE, rest by another base metric | per-group mix; formulas must be O(1)-normalized first |
| `ssim` / `smape_ssim` / `smape_frag` / `pw_ssim` | SSIM-flavoured variants | table-driven, `--ssim-table` / `--ptable` |
| `netdmg` | modulator: Gnom-predicted damage as a bias on the lens | duel vs SMSE came back 0/3 — manners, not a verdict |
| **`netpure`** | **the queue's gain IS the predicted damage; group weight 1.0 → demotion order is damage per byte saved; the static lens is not re-inserted** | X2.6-4B @2250: PPL **18.3646** (vs 19.2660 SMSE), max KLD vs BF16 **14.17 vs 17.77** (−20%) |
| **`netfd`** | **F16-blend for bigger budgets: the fragile blocks are kept in full precision instead of upgraded a tier (top-down only)** | X2.6-4B @4050: 46 tensors in F16 (1510 MiB), PPL **18.6228** — beats a solid stock Q8_0 (4172 MiB, 19.0307) **while being smaller** |

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
- `scripts/run_humaneval.py`: `HE_CHUNK="start,count"` slices the sorted task list;
  `HE_CTX=N` replaces the protocol default `-c`; `serve_sha256` sidecar key +
  `scripts/twin.py` pre-battery twin gate (same sha → reuse, else measure).
- `--cache-ram 0` on HE chains (164 distinct prompts share nothing, so the prompt cache
  only eats RAM); `LLAMA_PPL_BATCH` env overrides the sweep's hardcoded `-b 512`
  (forge iGPU OOMs a 9B at 512; 128 holds).
- `scripts/audit_tiers.py` — tier auditing tool: `--model M.gguf [--big 10] [--layer 31]
  [--per-layer]`.
- `scripts/consolidate_labels.py` — builds the clean label pool (corpus-tagged, dedup-checked)
  so a mixed archive can never leak into a training run again.
- `scripts/shelf_x26.sh` — resumable measurement queue (one GPU job at a time, skips
  finished arms): KLD vs BF16 → HE batteries → HE+ per arm.

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

→ *new mode: one row in the modes table. New flag: one row in the flags table.*

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
between tied attention groups while the "shield" guarded precision that was never spent.
The Norms Shield (TDN) +1.2pp stands as a measured effect with a redirect mechanism, not a
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
carry quantization damage, instead of guessing importance from proxy signals. It is trained
on **measured** damage labels, then ranks models it has never seen.

**The labels** — [`MERNIK-Fracture`](https://huggingface.co/datasets/wepiqx/MERNIK-Fracture):
per-tensor KLD damage on a reasoning corpus (`rl_reasoning_mini`), five columns per row
(mean / max / median KLD + Δp RMS + same-top), deterministic, uniform schema across
models and tiers. The 9B file took ~13 days of continuous sweeping; the 1.7B/4B legs
minutes each. **Cheap instruments are for the shelf; the expensive one is for the model you
care about.**

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

**Adding a leg to the pool** (the extension point):

1. sweep the model: `group_damage_sweep.py --model M.gguf --imatrix I.gguf --kld-base
   ref.dat --kld-only --drop-tier Q4_K` (and Q3/Q2 for the curve);
2. `build_features.py --model M.gguf --imatrix I.gguf --out gnom/features/<m>.npz`;
3. `scripts/consolidate_labels.py` → clean pool + `pool.json`;
4. add the row to the pool file, retrain, check the bars (0.45 / 0.60).

→ *new model in the pool: one row in the pool file + a line in the tables above.*

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
`/health` before every battery; mid-run watchdog every 10 tasks (dead server aborts
LOUDLY, never scores empties silently). Long-ctx servers accumulate commit (KV + prompt
cache never shrink) — chunked batteries (`HE_CHUNK`, fresh server per ~30 tasks, stitch in
order) plus `--cache-ram 0` (164 distinct prompts share nothing). Family sampling:
thinking families run their vendor recipe (K2: temp 1.0 + top_p 0.95 only, top_k 40 /
min_p 0.05 = llama defaults, ctx 16384, `--reasoning-effort high`; Spark-X2.6: temp 1.0,
top_p 0.95, top_k 0, min_p 0.05, effort high, ctx 16384, max_tokens 8192) — cross-family
verdict comparisons stay void, different rulers. Runner: `scripts/run_humaneval.py` (env
`HE_*`, `HE_CHUNK`, `HE_CTX`, `HUMANEVAL_OUT`, `HUMANEVAL_SERVER`;
`HUMANEVAL_SERVE_MODEL` = self-serve with auto-kill). Eval:
`human_eval.evaluation.evaluate_functional_correctness`, pass@1, k=[1]. Results:
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

Answered 2026-09-29: the same file twice at temp 0.0 scored 98/164 twice, per-task
verdicts 164/164 identical, completions 163/164 byte-identical. Greedy does not narrow the
verdict column — it **reproduces** it. The column can stop being sampled: temp-0 batteries
are certificates, temp-1.0 batteries are draws. (Caveat kept: the sampler skips RNG at
temp ≤ 0, but logits still depend on batch composition — so far, no spread observed.)

---

## 7. Reference results (links — the tables live on the family cards)

| family | card | load-bearing line |
|:---|:---|:---|
| NeoHorse-1-9B | [wepiqx/NeoHorse-1-9B-MERNIK-GGUF](https://huggingface.co/wepiqx/NeoHorse-1-9B-MERNIK-GGUF) | 6500-MSE 141/164 (85.98%); 5100-SMAPE 136/164 ties stock Q6_K 135/164 at −2.2 GB |
| OxCoder-9B | [wepiqx/OxCoder-9B-MERNIK-GGUF](https://huggingface.co/wepiqx/OxCoder-9B-MERNIK-GGUF) | SMSE-6500 150/164 (91.46%), Q8 148/164, SMAPE-5100 145/164 — one cluster, no crown inside it |
| RINIQ series (layer-fused) | [wepiqx/RINIQ-GGUF](https://huggingface.co/wepiqx/RINIQ-GGUF) | nine-way statistical tie (151, then eight on 150) — no crown, the top is shared |
| Mellum2.1-12B-A2.5B | [wepiqx/Mellum2.1-12B-A2.5B-Thinking-MERNIK-GGUF](https://huggingface.co/wepiqx/Mellum2.1-12B-A2.5B-Thinking-MERNIK-GGUF) | 10000-SMSE 134/164 beats stock Q6 119/164 by +15 at −0.7 GB (**3/3 SIGNIFICANT**) |
| FrogNano-4B | family card (lab notes) | the α-axis laboratory: 110→117→120→122→109, and stock Q6 wins the temp-0 duel 3/3 |
| K2-Horizon-7B | family card | thinking family, saturated shelf: 5500 SMSE 147/164 vs stock Q4_K_M 152/164, Q6 148/164 — all inside the noise floor |
| **Spark-X2.6-4B** | family card | **the current generation** — see [§0](#0-tldr--what-the-last-weeks-produced) |

**The noise budget, as a number.** Verdict noise is a binomial draw — SD 5.9 tasks/run at
p=0.70, 8.3 for a difference of two runs. That IS the ±3.6pp band and IS the 8-task MTP
re-run gap. Below ~8–10 tasks nothing is reproducible at n=164 — a property of the
instrument, not a measurement error. Verdicts travel with MDE; the column certifies
preservation, not rank.

→ *new family: one row here + a card link.*

---

## 8. Advanced experiments

### MTP head grafting

Transplant of a 15-tensor (464 MB) MTP donor head onto OxCoder-9B via
`scripts/graft_mtp3.py`. Draft engages (5–23% acceptance); economics negative (31→20 t/s on
mismatched head). Mechanics proven; full saga in `SAGA.md`.

### Teacher & Gini (v3, 2026-09-19)

Damage-sweep labels train the Gini allocator. v3 complete: 168/168 labels (Q5_K→Q2_K drops,
MiniCPM5-2B, forge): damage −0.046…+0.515, mean 0.070, Gini 0.364, only 3 reliefs. Late
units carry the signal (ffn_gate@41 +0.51, ffn_down@41 +0.33, ffn@39/40) + early attn@2.
vs older teacher (Gini 0.498, max 0.173): bigger drops hit harder (max 3×) but spread
wider (Gini lower) — signal with teeth, less concentrated. Gini answers to Soulfate24.
*(Superseded as the main instrument by the KLD teacher and Gnom — §5.)*

### Ceiling & inversion probes (2026-09-30/10-01)

- **Ceiling map:** `RINIQ-M2-MAXF16` (9.01 GB, F16 core 123) measured PPL 7.7357 (worse
  than 6500-MSE's 7.5263), HE 137/164, HE+ 129 — +4 GB over the 5 GB cluster buys −13
  tasks. Kings saturate (Q8→F16 gains nothing) while the Q4 tail drags.
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

→ *new experiment: a short block here, with the numbers.*

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

Scaling: Spark-1.7B @1000 (zoo stand) → Spark-4B @4000 (PPL 31.25 vs Q8 30.86) →
Ornith-9B-MTP @6500 (PPL 8.6541) → NeoHorse-9B @6500 (HE 141 vs Q6_K 135) →
Mellum2.1-MoE @10000 (HE 134 vs stock Q6 119, 3/3 SIGNIFICANT) → FrogNano-4B @3500
(stock wins temp-0 3/3) → K2-Horizon-7B @5500 (saturated shelf) → **Spark-X2.6-4B @2250
(18.3646 PPL, −20% max KLD vs the imatrix arm at equal weight)**.

→ *new architecture: one row here.*

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

→ *new repo or dataset: one link here.*

---

## 11. How to extend this README

The file is organised so that every new thing has exactly one home. When you add
something, put it in the place below and nowhere else.

| what you added | where it goes |
|:---|:---|
| a **new utility mode** | §3 modes table — one row: what it believes, where it won/lost |
| a **new CLI flag** | §3 flags table — one row |
| a **new law** | §2 — name, date, the number that proves it, the model it was measured on |
| a **new result** | §0 tables (the short version) + §7 (the family-card link) |
| a **new model in the label pool** | §5 — the pool step list + the transfer table |
| a **new architecture** | §9 — one row |
| a **new repo / dataset / model** | §10 — one link |
| a **new ring or column** | §1 — the ring table, with the date of the revision |

**House rules (they are what keep this readable):**

1. **Every claim carries a number, a date, and a source** (a file, a repo link, or a battery
   tag). A claim without all three is a story, not a measurement.
2. **Allocator signals never certify.** If a number is PPL, it is labelled a canary.
3. **One table per subject.** The same numbers must not appear in two sections — link
   instead. (Family tables live on their own cards for exactly this reason.)
4. **Verdicts travel with MDE.** A difference of 5 tasks out of 164 is a direction, not a
   rank.
5. **Dead results stay filed.** A mode that lost is still a row in the modes table, marked
   with the number that killed it — that is how we avoid re-testing it.

**Workflow for a big change:** write it in this file first (notebook-style, numbers as they
arrive), read it back, then fold it into `README.md`. Nothing is ever deleted from the
published file before the replacement is in place.

→ *a new convention (a column, a ring, a rule): add it to the tables in §1 and to the
house rules above.*
