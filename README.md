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

**MERNIK** ("the one who measures") is the evolution of the ASHQ1 battlefield zoo: fewer utility duels, more verdicts. The method (priority queue allocation) is built; MERNIK is how we prove and certify it.

Think of MERNIK as a **finetune of ASHQ1**: same base weights (queue, pins, tied groups), retrained objective (measure-first protocol, three-column verdicts), and expanded capabilities (slow capability ring, KLD-aware teachers, relief ceilings, the zoo bench, and norms shields).

---

## 1. Core Architecture & Evaluation Protocol

### The Two Rings

MERNIK strictly separates allocation signals from capability certification:

| Ring | Role | Trigger / Cadence |
|:-----|:-----|:------------------|
| **Fast / Allocator** | Per-decision signal for the queue (imatrix, KLD-damage sweeps) | Every run |
| **Slow / Capability** | Scored finals, fixed seeds, exact capability metrics | Once per release |

> **Rule:** Allocator signals never certify. Capability scores never steer. Mixing them leads to the PPL disease.

### Three-Column Verdicts

Every performance claim must present all three columns or stay unverified:

1. **PPL (Canary):** Cheap, demoted everywhere (often lies by sharpening).
2. **KLD vs Ref (Rank Column):** Used strictly for allocator decisions (Soulfate24 battery: fixed span, fixed chunks, Flash-Attention). Triage only — blind to long context, instruction drift, and tool formats.
3. **Tasks (Verdict Column):** The Slow Ring. The *only* column authorized to crown or kill a build.

### Decision Rules & Signal Interpretation

* **Divergence:** If metrics diverge, KLD wins allocator arguments; if they agree, trust the pair.
* **Silence is Signal:** Empty completions scale with weakness across clean runs (Ox: 4–5 → Neo: 20–27 → Q2_K: 63 → S2: 115). Weak models go silent rather than babble. Decisiveness is governed by the **finetune**, not model size (e.g., on Qwen3.5 bones, Neo hesitates while OxCoder commits).
* **Saturation Rule:** Base models at ~98% cannot resolve fine differences (97% vs 96%). Capability duels require bases at 60–85%, or they only answer the collapse threshold.

---

## 2. Quick Start & CLI Usage

### Basic Execution

```bash
pip install -r requirements.txt

# 1. Preview tier allocation & dry-run (~1 sec)
python main.py --model M.gguf --imatrix M.imatrix.gguf --size 6800

# 2. Execute actual quantization
python main.py --model M.gguf --imatrix M.imatrix.gguf --size 6800 --run
```

Requires stock llama.cpp binaries (`llama-quantize`, `llama-perplexity`, `llama-server`). Custom paths via `LLAMA_QUANTIZE`, `LLAMA_PPL`, `LLAMA_SERVER`.

### Full CLI Flags Reference

| Flag | Description |
|:-----|:------------|
| `--model M.gguf` | Source weights (BF16/F16) — required |
| `--imatrix I.gguf` | Imatrix file; repeatable (`--imatrix A --imatrix B --imatrix-method max\|mean`) |
| `--size MIB` | Target file size in MiB (the primary budget knob) |
| `--output O.gguf` | Output path (default: `<model>-MERNIK.gguf`) |
| `--run` | Execute `llama-quantize` (without it: dry-run estimate only) |
| `--utility NAME` | Queue gain metric: `mse` (default) / `rmse` / `smape` / `logcosh` / `ssim` / `smape_ssim` / `smape_frag` / `mix` / `smse` (`balance` = deprecated alias) / `srmse` |
| `--mix-base NAME` | MIX utility: gain for non-king groups (`mse`/`rmse`/`smape`/`logcosh`) |
| `--mix-top-frac F` | MIX utility: fraction of top-importance king groups (default: 0.2) |
| `--top-down` | Reverse allocation: everything from F16, downgrade cheapest-loss-first |
| `--pin-norms` | Top-down native norms shield (norms stay F16, outside budget) |
| `--allow-q3-or-lower` | CAN_Q3 types (`ffn_gate/up/down`, `attn_output`, `ssm_out`) may drop to IQ2_XXS |
| `--relief labels.jsonl --relief-thr -0.5` | Groups below threshold get ceiling Q4; budget flows elsewhere |
| `--verify gpqa\|he\|all` | Slow-ring verify on fresh build or existing `--output` (busy GPU port aborts LOUDLY, never steals) |
| `--verify-tag TAG` | Result tag (default: from `--output` basename) |
| `--show-config` / `--show-floors` | Print tensor config / hard tier floors |
| `--no-preflight` / `--strict-preflight` | Skip the dry-run audit, or abort if any assigned tier cannot fire (default: audit runs, ~1s) |
| `--legacy-1d` | LEGACY: price 1D tensors (norms/biases) and ssm_conv1d as quantizable F16 — reproduces pre-2026-09-28 budgets (binary writes them F32 regardless) |
| `--imatrix-tol F` / `--imatrix-legacy-combine` | Refuse when lens files diverge beyond tolerance (default 1%); legacy flag restores max/mean-combine with a split-brain binary |
| `--squeeze` | SQUEEZE mode (separate path): only IQ1_S or F16 per tensor, bottom-up only |
| `--free-pins` | EXPERIMENTAL: output/token_embd/MTP/routers join the budget pool |
| `--cv-w F` / `--linf-w F` | Hybrid (dead by construction): concentration discount / worst-case boost weights |
| `--huber-delta D` | huber/logcosh scale splitting small vs large deltas (default 3e-4) |
| `--ssim-table P` / `--ptable P` / `--frag-w F` | ssim / pw_ssim tables + smape_frag modulation weight (default 0.5) |
| `--netpred P` / `--netdmg-w F` | Gnom predicted damage per group + modulator weight (default 0.5) |
| `--aggro F` | [deprecated] Use `--size` instead |
| `--verbose` | Detailed output |

### Recently added (2026-09-28, all CPU-cheap, all verified)

- `scripts/fuse_layers.py --expect "15:b,..."` — gates the resolved
  donor map (ranges expanded, tissue suffixes checked); exits 2 with a
  diff before one output byte. Catches the silent-fallback class that
  bit N1/N1m. Plus a soup shape-guard: mismatched donor shapes fail
  LOUD, never a corrupt file.
- `scripts/group_damage_sweep.py --units TAGS --timeout S --ngl N` —
  subset sweeps (tags, not indices) with per-PPL timeout; failures go
  to `*.failures.jsonl`, second failure of a unit becomes a TERMINAL
  censored label (`damage: null`, never retried, never imputed).
- `--pin-norms` warns when it changes 0 tensors (proven no-op: the
  binary writes 1D F32 regardless). The size ceiling says so loudly
  instead of silently capping.

### Tier Auditing Tool

```bash
python scripts/audit_tiers.py --model M-6500.gguf
python scripts/audit_tiers.py --model M-6500.gguf --big 10
python scripts/audit_tiers.py --model M-6500.gguf --layer 31
```

---

## 3. Quantization Engine Mechanics

### Queue & Allocation Principles

Every tensor starts at a floor tier based on its functional class. A global max-heap drains the target budget best-first by sum(importance) × Δ / MiB. Tied groups (identical imatrix energy) upgrade as a single unit with summed importance. Structural pins — MTP heads (→ Q8_0), output/token embeddings (→ Q5_K), MoE routers (→ F16) — sit outside the budget queue.

### Certify vs Report (tool boundary)

Three tools, two jobs. `scripts/manifest.py` and `scripts/ledger.py`
REPORT: build→model identity (three keys: score+time+family) and
standings generated from artefacts — every number travels with its
identity status. `scripts/duel.py` ARBITRATES: paired McNemar across
all three columns, refuses to average across harnesses. Nothing
certifies except a significant duel; everything else is a report.

### Pre-flight Audit & the 1D Law

Every invocation runs a dry-run audit (~1s) comparing intended vs
actually-written tiers per tensor — the only thing that can catch a
`--tensor-type` rule that cannot fire. The 1D law: the binary writes
1D tensors (norms, biases) and `ssm_conv1d` as F32 whatever the rules
say, so `--pin-norms` is a proven no-op (warns when it changes 0
tensors) and the old "norms are load-bearing precision" reading was
budget-redirect, not precision.

### The Size Ceiling (user trap)

Bottom-up caps every class at CLASS_MAX_TIER, so a model can saturate
below `--size`: MiniCPM5-2B caps at exactly 2359.4 MiB, and `--size 2600`
silently produced a 2359 MiB file. It now says so loudly and names the
blocking classes — check the dry-run before assuming the budget bit.

### Utility Lenses: MSE vs SMAPE vs RMSE vs MIX vs SMSE vs SRMSE

* **MSE (default / spread):** Relative-blind absolute gain. Tiers spread evenly (Q5/Q6-heavy middle). Best PPL/KLD on dense 9B.
* **SMAPE (barbell):** Junk to the Q4 floor, kings to Q8 penthouses. Owns small budgets, loses big ones (budget law).
* **RMSE:** Perfectionist rescale of MSE. Zoo: beats MSE on PPL twice. First verdict (RINIQ-M2-6500-RMSE-Q38): PPL 7.5411, GPQA 50.00%, **HE 88.41%** — takes recognition, trails MSE by 3pp on verdict (sub-floor: direction, not rank).
* **MIX (per-group):** Kings by MSE, rest by `--mix-base`. Formulas live on different scales (mse Δ ~1e-3 vs smape ~1.2), so `_scale()` normalizes every formula to O(1) first — without it smape junk outbids mse kings 800:1 and the mix collapses. Dry-run geometry is new (wide Q5 spread, no Q8); verdict queued.
* **SMSE (blend, ex-BALANCE):** Geometric mean of the SMAPE-relative and MSE-absolute halves (both O(1)-normalized, MIX lesson). SMAPE's `(ec+en)` denominator silently cancels the sub-4 toxicity penalty; the blend carries real toxicity through its absolute half. First verdict (MiMo-4500, allow-q3): PPL 9.4092 vs SMAPE 10.5343 (−1.12), **HE 54.88% vs 29.88% (+25pp)** — rescue from the lava. Second verdict (MiMo-5100): PPL 8.7389 vs 8.7435 (tie), HE 71.34% vs 70.12% (+1.2pp), 3 empties — DUEL 0/3 (HE p=0.88, HE+ p=0.76 pointing the other way, empties p=0.13): NOISE, crown demoted 2026-09-28. Numbers stay, verdict withdrawn. Third verdict (MiMo-6500): PPL 8.7275 vs 8.7655 (tie), HE 72.56% vs 75.00% (−2.4pp) — LOSES big: at 6500 the middle decides (MSE Q6 67 vs SMSE 41), not penthouses (SMSE had more Q8, 125 vs 108). Verdict: SMSE owns small+medium budgets, MSE owns big. The bet behind it: any tier below Q4 hurts, and fewer is better.
* **SRMSE (SMAPE×RMSE, demoted 2026-09-28):** Relative lens × COMPRESSED absolute (sqrt squeezes toxicity ×1.41 instead of ×2.0). Duel MiMo-5100-SRMSE came back 0/3 (109-115, sampling spread on a bit-identical file) — manners without verdict. Filed, not crowned.

### Distribution Comparison @ 6500 MiB (427 tensors, incl. 177 F32 forced by the binary)

```
Tier      BU-MSE (spread)    BU-SMAPE (barbell)   TD-MSE       TD-SMAPE (F16 core)
F16       177 (2 MiB)        177 (2 MiB)          147 (2 MiB)  85 (949 MiB)
Q4_K      48 (972 MiB)       91 (1989 MiB)        71 (972)     236 (2368 MiB)
Q5_K      28 (1994 MiB)      3 (1345 MiB)         28 (1983)    7 (1411 MiB)
Q6_K      66 (2114 MiB)      7 (249 MiB)          71 (2127)    14 (486 MiB)
Q8_0      108 (1417 MiB)     149 (2913 MiB)       110 (1417)   85 (1287 MiB)
```

Takeaway: MSE fills the middle (Q5+Q6 = 94 tensors). SMAPE hollows it (10 tensors) to double the Q4 floor and gain 41 extra Q8 penthouses.

### Norms Shield & Attention Core Insights (RE-LABELED 2026-09-28)

TD-MSE vs MSE: −3.7pp HE with identical PPL (7.7701 vs 7.7695). Old
reading ("norms are load-bearing walls for code") is MECHANICALLY
IMPOSSIBLE: the binary writes 1D tensors as F32 whatever the tier says
(preflight-proven), so no norm was ever requantized. What the numbers
measure is budget redirect: top-down routing moved ~5 MiB between tied
attention groups while the "shield" guarded precision that was never
spent. Norms Shield (TDN) +1.2pp stands as a measured effect with a
redirect mechanism, not a precision mechanism. `--pin-norms` now warns
when it changes 0 tensors (mailbox SPEC, implemented in main.py).

---

## 4. Slow-Ring Protocol (HumanEval Execution)

Server (one model at a time, GPU is a strict queue):

```bash
llama-server -m MODEL.gguf --port 28082 -ngl 99 -c 8192 --jinja --log-disable
```

Sampling:

```text
temperature 1.0, top_p 0.95, top_k 20, min_p 0.0,
presence_penalty 0.0, repetition_penalty 1.0, max_tokens 2048
```

`presence_penalty 1.5` (vendor recipe) breaks thinking templates; `0.0` verified. Preflight `/health` before every battery; mid-run watchdog every 10 tasks (dead server aborts LOUDLY, never scores empties silently). Runner: `scripts/run_humaneval.py` (env `HE_*`, `HUMANEVAL_OUT`, `HUMANEVAL_SERVER`; `HUMANEVAL_SERVE_MODEL` = self-serve with auto-kill). Eval: `human_eval.evaluation.evaluate_functional_correctness`, pass@1, k=[1]. Results: `eval_results/humaneval_<tag>.jsonl`. Strictness upgrade: every battery is rescored with EvalPlus HumanEval+ (80× tests, CPU) — the HE+ column below.

Answered 2026-09-29: the same file twice at temp 0.0 scored 98/164 twice,
per-task verdicts 164/164 identical, completions 163/164 byte-identical.
Greedy does not narrow the verdict column — it REPRODUCES it. The column
can stop being sampled: temp-0 batteries are certificates, temp-1.0
batteries are draws. (Caveat kept: the sampler skips RNG at temp ≤ 0, but
logits still depend on batch composition — so far, no spread observed.)

---

## 5. Benchmark Results & Standings

The noise budget, as a number: verdict noise is a binomial draw —
SD 5.9 tasks/run at p=0.70, 8.3 for a difference of two runs. That IS
the ±3.6pp band and IS the 8-task MTP re-run gap. Below ~8–10 tasks
nothing is reproducible at n=164 — a property of the instrument, not a
measurement error. Verdicts travel with MDE; the column certifies
preservation, not rank (nine-way tie at the top: 151, eight on 150).

### The Budget Law

SMAPE owns small budgets; MSE owns big budgets — as LEDGER
measurements (identity of pre-sidecar artefacts: permanent-unverifiable,
see manifest; the lava leg survives on 3/3 columns + effect size).
Direction only, not rank: the legs differ by 5–6 tasks, below the
13-task resolution floor (6500: 141 vs 135; 5100: 136 vs 131).

### The Sub-4 Law (confirmed 2026-09-23)

Any tier below Q4 hurts, and fewer is better. `--allow-q3-or-lower` at ~4 GB is lava in every family: MiMo-4500-SMAPE (PPL 10.53 / HE 29.88%) lands exactly where NeoHorse-3650-Q2 did (10.36 / 23.78%) — same physics, different bones. SMSE-4500 (9.41 / **54.88%**, +25pp) proves limiting sub-4 exposure rescues the build. TOX (×2.0 effective-MSE penalty, still in `constants.py`) called it; SMSE proves it on the verdict column.

### Reference Results I: NeoHorse-1-9B — full table lives on the family card

[wepiqx/NeoHorse-1-9B-MERNIK-GGUF](https://huggingface.co/wepiqx/NeoHorse-1-9B-MERNIK-GGUF).
Load-bearing lines: 6500-MSE 141/164 (85.98%), 5100-SMAPE 136/164
(82.93%) ties stock Q6_K 135/164 at −2.2 GB. Gaps between neighbours are
1–6 tasks — direction, not rank.

### Reference Results II: OxCoder-9B — full table lives on the family card

[wepiqx/OxCoder-9B-MERNIK-GGUF](https://huggingface.co/wepiqx/OxCoder-9B-MERNIK-GGUF).
Load-bearing lines: SMSE-6500 150/164 (91.46%), Q8 148/164, SMAPE-5100
145/164 — one cluster, no crown inside it.

### Monster Standings: The RINIQ Series — full table lives on the family card

Layer-fused models (OxCoder base + donor blocks; auto-maps via
`scripts/auto_fuse.py`). Files and verdicts:
[wepiqx/RINIQ-GGUF](https://huggingface.co/wepiqx/RINIQ-GGUF), recipes in
`RINIQ-NEXT.md`. Load-bearing line: nine-way statistical tie (151, then
eight builds on 150) — no crown, the top is shared.

### Fast-Ring Diagnostic: GPQA-Recognition

Recognition decisiveness (first-token P(letter) over top-30, no CoT). Via `scripts/gpqa_duel.py`.

| Build | GPQA-rec | HE pass@1 | Note |
|:------|---------:|:---------:|:-----|
| Neo-MSE-6500 | 51.5% | 85.98% | Thinker spreads first-token mass |
| Neo-SMAPE-5100 | 48.0% | 82.93% | Compact barbell |
| Ox-MSE-6500 | 49.5% | 90.24% | Decisive stabs the letter |
| Ox-SMAPE-5100 | 49.0% | 88.41% | High task execution |

---

## 6. Advanced Experiments

### MTP Head Grafting

Transplant of a 15-tensor (464 MB) MTP donor head onto OxCoder-9B via `scripts/graft_mtp3.py`. Draft engages (5–23% acceptance); economics negative (31→20 t/s on mismatched head). Mechanics proven; full saga in `SAGA.md`.

### Teacher & Gini (v3 DONE 2026-09-19)

Damage-sweep labels (`scripts/group_damage_sweep.py`) train the Gini allocator. **v3 complete: 168/168 labels** (Q5_K→Q2_K drops, MiniCPM5-2B, forge): damage −0.046…+0.515, mean 0.070, Gini **0.364**, only 3 reliefs. Late units carry the signal (ffn_gate@41 +0.51, ffn_down@41 +0.33, ffn@39/40) + early attn@2. vs older teacher (Gini 0.498, max 0.173): bigger drops hit harder (max 3×) but spread wider (Gini lower) — signal with teeth, less concentrated. MLP un-paused: retrain Gnom on v3 → netdmg duel. Gini answers to Soulfate24.

### KLD Battery & Relief

KLD-vs-ref is the rank column (Soulfate24 battery). Measured relief ceilings (`--relief`) pin sweet-spot groups at Q4.

---

## 7. Supported Architectures & Multi-Scale Scaling

| Arch | Detection | Key Features |
|:-----|:----------|:-------------|
| `qwen35` | SSM + QKV | Hybrid attention, SSM layers, GQA, MTP head pinning |
| `mellum2` | MoE (`exps` tensors) | Mixture of Experts, GQA, routers pinned F16 |
| `bailingmoe3` | KDA+MLA + MoE | Ling-3.0 family: routed + shared experts |
| `granite` | `general.architecture` | Dense GQA, 40 layers, split Q/K/V |
| `spark2_5` | `general.architecture` | Dense + hybrid sliding-window attention (1 full + 3 SWA) |
| `gemma4` | layer-scale norms | QAT support, Q4_K attention floor |
| llama (generic) | tensor names | Dense GQA (MiniCPM5, NeoHorse, etc.) |

Scaling: Spark-1.7B @1000 (zoo stand) → Spark-4B @4000 (PPL 31.25 vs Q8 30.86) → Ornith-9B-MTP @6500 (PPL 8.6541) → NeoHorse-9B @6500 (HE 141 vs Q6_K 135 — 6 tasks, direction only).

---

## 8. Ecosystem & Repositories

- 📦 [NeoHorse-1-9B-MERNIK-GGUF](https://huggingface.co/wepiqx/NeoHorse-1-9B-MERNIK-GGUF)
- 📦 [OxCoder-9B-MERNIK-GGUF](https://huggingface.co/wepiqx/OxCoder-9B-MERNIK-GGUF)
- 📦 [MiMo-V2.6-Distill-Qwen-9B-GGUF-MERNIK](https://huggingface.co/wepiqx/MiMo-V2.6-Distill-Qwen-9B-GGUF-MERNIK)
- 📦 [RINIQ-GGUF](https://huggingface.co/wepiqx/RINIQ-GGUF)
- 📦 [RINIQ-NEXT-GGUF](https://huggingface.co/wepiqx/RINIQ-NEXT-GGUF)
- 📦 [Ornith-1.5-9B-MTP-ASHQ1-GGUF](https://huggingface.co/wepiqx/Ornith-1.5-9B-MTP-ASHQ1-GGUF)
- 📦 [Spark-X2.5-1.7B-ASHQ1-GGUF](https://huggingface.co/wepiqx/Spark-X2.5-1.7B-ASHQ1-GGUF)

Lineage: Apache-2.0. Built with [Soulfate24](https://huggingface.co/Soulfate24), whose KLD battery ended our PPL theater. Field journals with every scar: `FUSION.md` (monsters), `README-ZOO.md` (utility duels), `SAGA.md` (MTP graft).
