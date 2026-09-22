# MIMO — MiMo-V2.6-Distill-Qwen-9B workbench (2026-09-22)

Xiaomi MiMo SFT distillate of Qwen3.5-9B (77.4B SFT tokens, 27.2B
loss-bearing; 27% visual). Agentic generalist, not a code specialist.
SFT ≈ continued-pretraining scale — deltas vs base are huge
(SWE Pro 32→44.6, TerminalBench 27→37.1).

Donors/files in `/mnt/Vsio/Downloads/`:
- `MiMo-V2.6-Distill-Qwen-9B-bf16.gguf` (17 GB)
- `MiMo-V2.6-Distill-Qwen-9B-imatrix.gguf` (own, 4.9 MB)
- `MiMo-V2.6-Distill-Qwen-9B-calibration-v6.txt` (own cal, 1.1 MB)
- `MiMo-V2.6-Distill-Qwen-9B-MERNIK-5100.gguf` (SMAPE, own lens)
- `MiMo-V2.6-Distill-Qwen-9B-Q6_K.gguf` (stock baseline)
- `RINIQ-N1-BF16.gguf` (Ox base + MiMo 15,16,17, via fuse_layers.py)
- `RINIQ-N1m-BF16.gguf` (MiMo base + Ox 15,16,17, mirror)
- `RINIQ-N1-MERNIK-5100.gguf` / `RINIQ-N1m-MERNIK-5100.gguf` (SMAPE-5100, dual-max lens)

MiMo ships its own jinja (macro-based, 3.9 KB) — NOT the Qwen
froggeric template (16 KB). `--jinja` works natively on all runs.

Sampling (Qwen family protocol): temp 1.0 / top_p 0.95 / top_k 20,
presence 0.0, max_tokens 2048. No Gemma leftovers (those lived only in
chain scripts). HE runner now logs battery time (start stamp + min/task).

## Imatrix compass (MiMo vs Ox vs Neo, 248 common tensors)

| Pair | Pearson | Rank agree |
|:-----|--------:|-----------:|
| mimo–ox | 0.9989 | 0.9975 |
| mimo–neo | 0.9992 | 0.9977 |
| ox–neo | 0.9999 | 0.9999 |

Same law as FUSION: recipe dominates, finetune nearly invisible. But the
divergence clusters mid-layer FFN **15,16,17,19,13,27,14,18,23,11** — the
weight-compass "finetune soul" zone. Scale: MiMo cal runs 1.6× hotter
(mean 1.9e5 vs 1.2e5) — quantize on its own imatrix, never mixed.
(Dual-max(Ox,MiMo) is a fiction: max always picks MiMo's hotter scale.
N1 dry-run with dual lens == pure-MiMo geometry bit for bit.)

## Verdicts (slow ring: HumanEval pass@1, temp 1.0 / top_p 0.95 / top_k 20, presence 0.0, max 2048)

| Build | Size | PPL (ctx1024) | HE pass@1 | HE+ |
|:------|-----:|:-------------:|:---------:|:---:|
| MiMo-5100-SMAPE (own imatrix) | 5.0 GB | 8.7435 | 70.12% (115/164) | 66.5% |
| RINIQ-N1 (Ox+MiMo15–17, dual lens) | 5.0 GB | 8.0572 | ⏳ chain5 | ⏳ |
| RINIQ-N1m (MiMo+Ox15–17, mirror) | 5.0 GB | 7.9604 | ⏳ chain5 | ⏳ |
| MiMo-Q6_K (stock) | 7.2 GB | ⏳ chain5 | ⏳ chain5 | ⏳ |
| Ox-SMAPE-5100 (ref) | 5.0 GB | 7.5670 | 88.41% (145/164) | — |
| Neo-SMAPE-5100 (ref) | 5.0 GB | 7.8012 | 82.93% (136/164) | — |

Dry-run geometry @5100 (SMAPE, own lens): F16 177 / Q4 226 / Q5 13 /
Q6 11 / Q8 0 — all floor, zero penthouses (same as Ox-SMAPE-5100 shape).

Empties: MiMo 7 (Ox-like decisiveness, not Neo hesitation).
Speed: MiMo ~20 s/task (vs 40+ Gemma-12B, 60+ RINIQ duels) — decisive,
no thinking-chewing. Timer now logged per battery.

## Published-code duel (different harnesses — theater, same-harness HE decides)

| Bench | OxCoder-9B | MiMo-Distill |
|:------|-----------:|-------------:|
| SWE Verified | 73.5 | 61.1 (avg@3) |
| SWE Pro | 49.1 | 44.6 (avg@3) |
| TerminalBench 2.1 | 49.6–50.8 | 37.1 |
| GPQA Diamond | 86.9 | ??? (unreported) |

Base Qwen3.5-9B itself differs between the two tables (SWE-V 53.2 vs
60.0) — harness strictness (OpenHands + anti-hacking, no net) vs avg@3
inflation. Cross-table comparison is void; only same-harness duels count.

## Scars (chain discipline)

- 2026-09-22: heavy CPU quant launched parallel to GPU HE battery → RAM
  pressure → server death, 30-min battery progress DISCARDED. Second time
  (same kill took Q6-HE at task 60). Rule (README:118, "GPU is a strict
  queue") now enforced as: one heavy job at a time, sequential chains
  only (chain5: PPL×3 then HE×3).
- earlyoom (root, main rig) unkillable without local sudo (root locked
  after wrong attempts) — lives on, chain discipline is the shield.
- Server deaths mid-long-run (tasks 25/30/101/110) on 8 GB VRAM:
  6.4 GB weights + KV c4096 + fragmentation → random OOM. Mitigations:
  `--cache-reuse 256`, `-c 4096` (c8192 KV doesn't fit: create_context
  fail, proven). c8192 works on 9B/5 GB builds (RINIQ era).

## Next

- [x] Q6_K stock baseline: quant done, PPL+HE in chain5
- [ ] Weight compass MiMo vs Ox (both BF16 local, no GPU)
- [x] RINIQ-NEXT direct: Ox base + MiMo 15,16,17 (N1 built, chain5 verdicts)
- [x] RINIQ-NEXT mirror: MiMo base + Ox 15,16,17 (N1m built, chain5 verdicts)
- [ ] Neo blk31 donor: no Neo BF16 local (only imatrix) — skip or extract
      blk31 from RINIQ-M2-BF16 (verbatim Neo bytes)
- [ ] LiveCodeBench v6 (backlog): MiMo's arena (SWE/terminal), Ox's too
