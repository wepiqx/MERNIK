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
| MiMo-4500-SMAPE (own imatrix, allow-q3) | 4.5 GB | 10.5343 | 29.88% (49/164) | 26.2 — lava zone (Neo-3650-Q2: 23.78%), sub-4 at ~4GB |
| MiMo-4500-SMSE (own imatrix, allow-q3) | 4.5 GB | **9.4092** | **54.88% (90/164)** | **53.0** — rescue +25pp over SMAPE, TOX-like save |
| MiMo-5100-SMSE (own imatrix) | 5.0 GB | 8.7389 | 71.34% (117/164) | 65.9 — 3 empties. Duel 0/3 columns (HE p=0.88, HE+ p=0.76 against, empties p=0.13): the +1.2pp is NOISE, crown demoted 2026-09-28. Decisiveness edge (3 vs 7) stands as observation |
| MiMo-6500-MSE (own imatrix) | 6.5 GB | 8.7655 | **75.00% (123/164)** | **68.9** — beats Q6_K by +3pp at −0.7 GB |
| MiMo-6500-SMSE (own imatrix) | 6.5 GB | 8.7275 | 72.56% (119/164) | 67.1 — LOSES to MSE by 2.4pp: PPL tie, verdict fail. Burn wins; refinement: SMSE had MORE Q8 (125 vs 108) yet lost — at big budgets the MIDDLE decides (MSE Q6 67 vs SMSE 41) |
| MiMo-Q6_K (stock) | 7.2 GB | 9.1362 | 71.95% (118/164) | 66.5% |
| Ox-SMAPE-5100 (ref) | 5.0 GB | 7.5670 | 88.41% (145/164) | — |
| Neo-SMAPE-5100 (ref) | 5.0 GB | 7.8012 | 82.93% (136/164) | — |

Fusions (N1/N1m) live in `RINIQ-NEXT.md`.

Geometry tells the rescue story (audit_tiers):

| Tier | 4500-SMAPE | 4500-SMSE | 5100-SMAPE | 5100-SMSE |
|:-----|----------:|----------:|----------:|----------:|
| sub-Q4 | **107** (104×IQ2_XXS!) | **66** (59×IQ2_XXS, rest IQ3/IQ4) | 0 | 0 |
| Q4 | 4 | 6 | 206 | 197 |
| Q5 | 2 | 27 | 11 | 29 |
| Q6 | 0 | 82 | 11 | 0 |
| Q8 penthouses | **118** | **0** | 0 | 0 |
| HE | 29.88% | 54.88% | 70.12% | 71.34% |

SMSE never affords a penthouse (Q8 = 0 at both budgets), halves the
dungeon, and rebuilds the middle (Q5+Q6 = 109 @4500). "Penthouses only
if affordable" — in numbers. Bonus catch: SMAPE-4500 evicted even the
free F16 residents (22→0), SMSE-4500 kept 8 — total war vs discipline.
Open risk @6500: MSE's 75% rides 108 Q8 penthouses; if SMSE arrives
with zero Q8 there and loses, the law extends to "penthouses required
at big budgets". Dry-run geometry at quant time will tell first.

Dry-run geometry @5100 (SMAPE, own lens): F16 177 / Q4 226 / Q5 13 /
Q6 11 / Q8 0 — all floor, zero penthouses (same as Ox-SMAPE-5100 shape).

Empties: MiMo 7 (Ox-like decisiveness, not Neo hesitation).
Speed: MiMo ~20 s/task (vs 40+ Gemma-12B, 60+ RINIQ duels) — decisive,
no thinking-chewing. Timer now logged per battery.
Battery times: N1 77.1 min (28.2 s/task); MiMo-5100 ~50 min (~18 s/task).
Fusion decisiveness (N1: 24 empties, seam friction) → `RINIQ-NEXT.md`.

HE+ rescore runs after every battery (CPU, EvalPlus 80×) — strictness
column. Layer swaps (15–17) may move other benchmarks (SWE, terminal,
vision) in either direction — those arenas are queued, not claimed.
If a build behaves weird on your hardware, open a discussion with setup
+ task — every scar goes in the ledger.

## Field notes (manual testing, 2026-09-22)

- Thinks a lot: long reasoning traces, lower t/s than OxCoder but FEELS
  faster (decisive, no hesitation). Part of the gap was CPU contention
  on the test box — clean comparison gives MiMo ~+1 tok/s over RINIQ/Ox.
- Agent-scaffold leak: in the pi agent framework, a bare "Privet" (hello
  in Russian) makes it confabulate a task (portfolio brief) — operator
  aborted manually upon noticing ("Operation aborted" was the operator,
  not the model).
  Other frameworks fine; RINIQ/Ox/Neo never do this. Read: MiMo's agent
  training (TerminalBench/Toolathlon scaffolds) misfires on agent-style
  system prompts — training artifact, harness-dependent, not quant damage.
- Temp sensitivity: temp 0.6 breaks tool-call format adherence (confuses
  file-create vs bash); temp 1.0 works. Vendor temp 1.0 mandatory for
  agentic tasks — lower temp commits to the wrong pattern confidently.
- `--reasoning-effort` WORKS (unique in family): low → <5k thinking
  tokens, xhigh → ~15k. Controllable depth/speed dial out of the box.
  Ox/Neo ignore these flags (thinking stripped).
- 3D-snake field test, first attempt BROKEN (critical: wall.push coords,
  isHead args, food destructuring [] vs {}, drawObj syntax, undefined
  dir0/tSince; logic: matrices rewritten, WebGL buffers, input).
  OxCoder baseline: wrote 3D models but NEVER got it running.
  Iterations-to-working TBD.
- low-effort FIRST TRY: WORKING 3D snake (renders, orbit camera, Russian
  UI, score/game-over flow; not quite playable). Total 11k tokens vs
  17k for the broken xhigh attempt.
  CONFOUND (honest): prompts differed by ONE word (originally in
  Russian) — xhigh got "make an HTML 3D snake...", low got "make a
  WORKING HTML 3D snake...".
  The win may belong to the word, not the effort level. Unconfounded
  A/B (same prompt, low vs xhigh) still open.
- opencode harness: 400-line file, noticed a bug immediately, fixing
  it himself. Harness-independent thinking, second framework confirmed.
- RINIQ-N1 field test: thinking depth lands BETWEEN parents (Ox shallow
  < N1 medium < MiMo deep) — mid-layer blocks 15–17 appear to govern
  reasoning depth. 3D-snake attempts: 1st = 2D game, 2nd = semi-working
  3D, 3rd = working snake (ultra slow but works), each attempt <4k
  tokens. N1m report pending.
- Test conditions: llama-server web UI, normal sampling (temp 1.0),
  no looping observed. First runs only — slight-breakage chance noted.

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
House rule: HE is used for comparison, not for score — if a small hybrid
scores roughly like stock Q6 on the same harness, most capabilities likely
survived quantization. The verdict column certifies preservation, not rank.

Field temp law (Ox-SMSE-6500, 3D-snake first-try): temp 0.0 = working code
at once; 0.6/1.0 needed retries. Hypothesis: low temp for codegen,
high temp for agents/reasoning. (MiMo agentic tasks mandate temp 1.0 —
same split from the other side.)

SRIQ-Q6 (LoRA-child of MiMo, Chinese traces): HE 64.63% (106/164) at
temp 1.0 vs **78.66% (129/164) at temp 0.0** — greedy moves the VERDICT
+14pp here (2/3 columns significant), decisiveness flat (6 vs 8 empties).
CORRECTION 2026-09-28 (manifest duel falsified the old "identical score"
entry — I never counted the t0 file, scar kept). Below MiMo-Q6 (71.95%)
at temp 1.0; PPL skipped (wiki canary invalid for Chinese reasoning).
Own SRIQ imatrix done.
SRIQ-5100-SMSE (own imatrix): 63.41% (104/164), HE+ 56.7, **1 empty** —
most decisive build in the lab, but −1.2pp vs Q6: mirror image of MiMo
(+1.2pp). SMSE buys decisiveness everywhere, verdicts are family-local.

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
- [ ] Fusions (N1/N1m), weight compass, LCB: see `RINIQ-NEXT.md`
- [ ] LiveCodeBench v6 (backlog): MiMo's arena (SWE/terminal), Ox's too
