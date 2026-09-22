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
| `--utility NAME` | Queue gain metric: `mse` (default) / `rmse` / `smape` / `logcosh` / `ssim` / `smape_ssim` / `smape_frag` / `mix` |
| `--mix-base NAME` | MIX utility: gain for non-king groups (`mse`/`rmse`/`smape`/`logcosh`) |
| `--mix-top-frac F` | MIX utility: fraction of top-importance king groups (default: 0.2) |
| `--top-down` | Reverse allocation: everything from F16, downgrade cheapest-loss-first |
| `--pin-norms` | Top-down native norms shield (norms stay F16, outside budget) |
| `--allow-q3-or-lower` | CAN_Q3 types (`ffn_gate/up/down`, `attn_output`, `ssm_out`) may drop to IQ2_XXS |
| `--relief labels.jsonl --relief-thr -0.5` | Groups below threshold get ceiling Q4; budget flows elsewhere |
| `--verify gpqa\|he\|all` | Slow-ring verify on fresh build or existing `--output` (busy GPU port aborts LOUDLY, never steals) |
| `--verify-tag TAG` | Result tag (default: from `--output` basename) |
| `--show-config` / `--show-floors` | Print tensor config / hard tier floors |

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

### Utility Lenses: MSE vs SMAPE vs RMSE vs MIX

* **MSE (default / spread):** Relative-blind absolute gain. Tiers spread evenly (Q5/Q6-heavy middle). Best PPL/KLD on dense 9B.
* **SMAPE (barbell):** Junk to the Q4 floor, kings to Q8 penthouses. Owns small budgets, loses big ones (budget law).
* **RMSE:** Perfectionist rescale of MSE. Zoo: beats MSE on PPL twice. First verdict (RINIQ-M2-6500-RMSE-Q38): PPL 7.5411, GPQA 50.00%, **HE 88.41%** — takes recognition, loses verdict by 3pp to MSE. Budget law extended.
* **MIX (per-group):** Kings by MSE, rest by `--mix-base`. Formulas live on different scales (mse Δ ~1e-3 vs smape ~1.2), so `_scale()` normalizes every formula to O(1) first — without it smape junk outbids mse kings 800:1 and the mix collapses. Dry-run geometry is new (wide Q5 spread, no Q8); verdict queued.

### Distribution Comparison @ 6500 MiB (427 tensors, incl. 177 F16 norms/1D)

```
Tier      BU-MSE (spread)    BU-SMAPE (barbell)   TD-MSE       TD-SMAPE (F16 core)
F16       177 (2 MiB)        177 (2 MiB)          147 (2 MiB)  85 (949 MiB)
Q4_K      48 (972 MiB)       91 (1989 MiB)        71 (972)     236 (2368 MiB)
Q5_K      28 (1994 MiB)      3 (1345 MiB)         28 (1983)    7 (1411 MiB)
Q6_K      66 (2114 MiB)      7 (249 MiB)          71 (2127)    14 (486 MiB)
Q8_0      108 (1417 MiB)     149 (2913 MiB)       110 (1417)   85 (1287 MiB)
```

Takeaway: MSE fills the middle (Q5+Q6 = 94 tensors). SMAPE hollows it (10 tensors) to double the Q4 floor and gain 41 extra Q8 penthouses.

### Norms Shield & Attention Core Insights

TD-MSE: downgrading ~30 small norm tensors to Q4 left PPL unaffected (7.7701 vs 7.7695) but dropped HumanEval by −3.7pp. Norms are free real estate for perplexity, but load-bearing walls for code. Norms Shield (TDN): native shielding recovers +1.2pp; post-hoc forcing disrupts greedy paths — shields must be native. Audit: RINIQ builds keep all 105 norm tensors intact F32.

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

---

## 5. Benchmark Results & Standings

### The Budget Law

SMAPE owns small budgets; MSE owns big budgets.

| Target Budget | MSE Utility | SMAPE Utility | Winning Lens |
|:--------------|:------------|:--------------|:-------------|
| 6500 MiB | PPL 7.7695 / HE 85.98% | PPL 7.8252 / HE 82.32% | MSE spread (+3.7pp) |
| 5100 MiB | PPL 7.7962 / HE 79.88% | PPL 7.8012 / HE 82.93% | SMAPE barbell (+3.0pp) |

### Reference Results I: NeoHorse-1-9B (official BF16 98.17% @ undisclosed ctx)

| Build | File Size | PPL (ctx1024) | KLD vs Q8-proxy | HE pass@1 | Key Finding |
|:------|----------:|:-------------:|:---------------:|:---------:|:------------|
| MERNIK-6500-MSE | 6.83 GB | 7.7695 | 0.0453 | 85.98% (141/164) | Outperforms stock Q6_K |
| MERNIK-6500-SMAPE | 6.83 GB | 7.8252 | 0.0476 | 82.32% (135/164) | Ties stock Q6_K |
| MERNIK-6500-TD-MSE | 6.83 GB | 7.7701 | 0.0452 | 82.32% | −3.7pp code loss vs MSE |
| MERNIK-6500-TDN-MSE | 6.83 GB | 7.7701 | — | 83.54% | Norms shield recovers +1.2pp |
| MERNIK-6500-TDN-SMAPE | 6.83 GB | — | — | 79.27% | Post-hoc shield disrupts greedy path |
| MERNIK-6500-TD-SMAPE | 6.83 GB | 7.8184 | 0.0505 | 82.32% | Holds Q6 level with 236 tensors at Q4 |
| Q6_K (stock) | 7.36 GB | 7.9419 | 0.0118 | 82.32% (135/164) | Stock baseline |
| MERNIK-5100-SMAPE | 5.36 GB | 7.8012 | 0.0586 | 82.93% (136/164) | Q6-class code at −2.2 GB 👑 |
| MERNIK-5100-MSE | 5.36 GB | 7.7962 | 0.0588 | 79.88% (131/164) | Budget law holds by 3pp |
| Q4_K_M (stock) | 5.62 GB | 7.7824 | 0.0875 | 76.83% (126/164) | Beaten by MERNIK-5100 |
| Q2_K (stock) | 3.83 GB | 90.0284 💀 | 2.6354 💀 | 0.00% | Full collapse |
| MERNIK-3650-Q2 | 3.65 GB | 10.3586 | 0.4706 | 23.78% (39/164) | Recovers sub-4-bit collapse |

### Reference Results II: OxCoder-9B (finetune duel, same Qwen3.5 bones)

| Build | File Size | PPL (ctx1024) | HE pass@1 | Note |
|:------|----------:|:-------------:|:---------:|:-----|
| Ox-MSE-6500 | 6.83 GB | 7.5125 | 90.24% (148/164) | Distillate leads benchmark |
| Ox-SMAPE-5100 | 5.36 GB | 7.5670 | 88.41% (145/164) | Compact build beats big NeoHorse |
| Ox-Q6_K (stock) | 7.36 GB | 7.6758 | 84.15% (138/164) | Allocation wins over stock flat |

Full card: [wepiqx/OxCoder-9B-MERNIK-GGUF](https://huggingface.co/wepiqx/OxCoder-9B-MERNIK-GGUF).

### Monster Standings: The RINIQ Series

Layer-fused models (OxCoder base + donor blocks; auto-maps via `scripts/auto_fuse.py`). Files: [wepiqx/RINIQ-MERNIK-GGUF](https://huggingface.co/wepiqx/RINIQ-MERNIK-GGUF).

```bash
python scripts/fuse_layers.py --a OX --b ORN --c NEO --map "15:b,..."
python scripts/auto_fuse.py --a OX --b ORN --ia OX.imatrix --ib ORN.imatrix --rule rank-vote --out M7.gguf
```

| Build | Size | PPL | GPQA-rec | HE pass@1 | HE+ (EvalPlus 80×) | Empties |
|:------|-----:|:---:|:--------:|:---------:|:-----------------:|:-------:|
| RINIQ-M2-MERNIK-5100-Q38 👑 | 5.0 GB | 7.6130 | 47.47% | **92.07% (151/164)** | **87.8** (−3.7) | 8 |
| RINIQ-M2-MERNIK-5100 | 5.0 GB | 7.5819 | 50.00% | **91.46% (150/164)** | 87.2 (−3.0) | 6 |
| RINIQ-M4a-MERNIK-5100 | 5.0 GB | 7.5898 | **51.01%** | **91.46% (150/164)** | 87.2 (−4.3) | **4** |
| RINIQ-M1-MERNIK-5100 | 5.0 GB | 7.6242 | **51.01%** | 90.85% (149/164) | 85.4 (−4.8) | 7 |
| RINIQ-M4c-MERNIK-5100 | 5.0 GB | 7.5863 | 47.98% | 90.85% (149/164) | 85.4 (−4.8) | 6 |
| RINIQ-M4b-MERNIK-5100 | 5.0 GB | **7.5743** | 48.48% | 89.63% (147/164) | 84.1 (−4.9) | 4 |
| RINIQ-M3-MERNIK-5100 | 5.0 GB | 7.7111 | 50.00% | 86.59% (142/164) | 83.5 (−3.1) | 13 |
| RINIQ-M2-SMAPE-6500 | 6.8 GB | 7.5730 | 48.48% | 89.63% (147/164) | 83.5 (−5.5) | 5 |
| RINIQ-M2-MSE-6500 | 6.8 GB | 7.5263 | 48.48% | **91.46% (150/164)** | 85.4 (−4.8) | 8 |
| RINIQ-M5-MERNIK-5100 | 5.0 GB | 7.6587 | **52.02%** | **91.46% (150/164)** | 84.8 (−6.1) | 7 |
| RINIQ-M6-MERNIK-5100 | 5.6 GB | 8.5137 | 46.97% | 83.54% (137/164) | 79.9 (−3.6) | 23 |
| RINIQ-M7-MERNIK-5100 | 5.0 GB | 7.6033 | 49.49% | 88.41% (145/164) | 86.0 (−2.4) | 11 — first auto-fused: mid-pack, method validated |
| RINIQ-M2-MERNIK-5100-MIX | 5.0 GB | 7.6001 | 48.99% | 89.63% (147/164) | 85.4 (−3.6) | 6 — novel geometry, mid-pack |
| RINIQ-M2-RMSE-Q38-6500 | 6.8 GB | 7.5411 | **50.00%** | 88.41% (145/164) | 82.9 (−4.9) | 6 — RMSE's first verdict: takes GPQA, loses HE by 3pp |

Sub-block donor mapping: M1 = Ox + Orn 15,19,23,27 + Neo 31. M2 = Ox + Orn 24,25,26 + Neo 31. M3 = M1 + blks 0–8 soup. M4a = M2 + 16←Orn. M4b = M2 with 25←Ox. M4c = M2 + 30←Orn. M5 = M1∪M2 (Orn 15,19,23,24,25,26,27). M6 = Orn base + Ox 16,24,25,26,31 (mirror, BASE RULES). M7 = auto rank-vote Orn{10,14,15,16,18}.

### External Reference: JackOD-9B-Coder (different harness!)

Corporate 4-way omnimerge_v2 over the same Qwen3.5-9B ancestor. Quoted from their README — greedy temp 0.0, mixed banks (their own † admits it). Same-harness duel (LiveCodeBench v6 55 hard) queued.

| Build | HE | HE+ | LCB v6 55 hard |
|:------|---:|:---:|:--------------:|
| RINIQ-M2-MERNIK-5100-Q38 (ours) | **92.07** | **87.8** | ⏳ |
| JackOD-9B-Coder (Q6_K, theirs) | 88.41 | 82.32 | **78.18** |

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

Scaling: Spark-1.7B @1000 (zoo stand) → Spark-4B @4000 (PPL 31.25 vs Q8 30.86) → Ornith-9B-MTP @6500 (PPL 8.6541) → NeoHorse-9B @6500 (HE 85.98% > Q6_K 82.32%).

---

## 8. Ecosystem & Repositories

- 📦 [NeoHorse-1-9B-MERNIK-GGUF](https://huggingface.co/wepiqx/NeoHorse-1-9B-MERNIK-GGUF)
- 📦 [OxCoder-9B-MERNIK-GGUF](https://huggingface.co/wepiqx/OxCoder-9B-MERNIK-GGUF)
- 📦 [RINIQ-MERNIK-GGUF](https://huggingface.co/wepiqx/RINIQ-MERNIK-GGUF)
- 📦 [Ornith-1.5-9B-MTP-ASHQ1-GGUF](https://huggingface.co/wepiqx/Ornith-1.5-9B-MTP-ASHQ1-GGUF)
- 📦 [Spark-X2.5-1.7B-ASHQ1-GGUF](https://huggingface.co/wepiqx/Spark-X2.5-1.7B-ASHQ1-GGUF)

Lineage: Apache-2.0. Built with [Soulfate24](https://huggingface.co/Soulfate24), whose KLD battery ended our PPL theater. Field journals with every scar: `FUSION.md` (monsters), `README-ZOO.md` (utility duels), `SAGA.md` (MTP graft).
