# RINIQ-NEXT — Ox × MiMo layer fusions (2026-09-22)

M2-formula on a new donor pair: OxCoder-9B × MiMo-V2.6-Distill-Qwen-9B
(same Qwen3.5-9B bones). Donor card: `MIMO.md`. Imatrix compass picked
MiMo blocks **15,16,17** (top divergence zone; weight-compass "finetune
soul" region). Dual-max(Ox,MiMo) lens == pure-MiMo geometry (hotter
scale) — documented, not hidden.

## Recipes

- **N1 (direct)**: Ox base + globals, MiMo blks 15,16,17.
  `fuse_layers.py --a Ox --b MiMo --map "15:b,16:b,17:b"`.
- **N1m (mirror, M6 lesson)**: MiMo base + globals, Ox blks 15,16,17.
- ⚠️ CORRECTION 2026-09-23 (scar): N1/N1m were **not** base+3-blocks.
  `fuse_layers.py` with 2 donors + `--map` silently fell back to the
  legacy odd/even interleave for unmapped blocks (`fuse-n1.log` map:
  `0:a,1:b,2:a,…` — V1-style half-foreign models misread as grafts).
  Fixed: explicit `--map` now means base-A-plus-overrides; legacy
  interleave only when no `--map` is passed. Added `--dry` + `--d`
  (fourth donor). The TRUE base+blocks builds are M8/N2 below.
- **M8 (single-swap A/B, V2→M2 precedent)**: RINIQ-M2 base (= Ox +
  Orn 24,25,26 + Neo31 baked in) + globals, MiMo blk **31**.
  `fuse_layers.py --a RINIQ-M2-BF16 --b MiMo --map "31:b"`.
  Edge position = minimal seam surface. If MiMo31 ≥ Neo31, the
  "one block decides" law extends to a second donor pair.
- **N2 (mashup 🙏)**: RINIQ-M2 base + MiMo blks **15,16,17**.
  `fuse_layers.py --a RINIQ-M2-BF16 --b MiMo --map "15:b,16:b,17:b"`.
  Donor islands (15–17 vs 24–26) are far apart — hypothesis: distant
  islands interfere less than one contiguous foreign block.
- Neo BF16 is NOT local (only imatrix) — but Neo31 travels verbatim
  inside RINIQ-M2-BF16, so M2-as-base replaces three donors at once.
  Ornith BF16s live in `shq-program/models/` if ever needed raw.

## Verdicts (slow ring: HumanEval pass@1, temp 1.0 / top_p 0.95 / top_k 20, presence 0.0, max 2048)

Power note (2026-09-28): at n=164 the slow ring resolves only large
effects (noise ±3.6pp; the table is a nine-way tie — 151, eight on 150).
This column certifies preservation, not rank. Crowns below are
ledger shorthand, not measurable order.

| Build | Size | PPL (ctx1024) | HE pass@1 | HE+ |
|:------|-----:|:-------------:|:---------:|:---:|
| RINIQ-N1 (interleave, see correction) | 5.0 GB | 8.0572 | 79.27% (130/164) | 76.2% |
| RINIQ-N1m (interleave, see correction) | 5.0 GB | 7.9604 | **63.41% (104/164)** | 63.4 (−0.0) — 51 empties: best PPL, worst verdict. PPL-liar crown 👑 |
| RINIQ-M8 (M2+MiMo31) | 5.0 GB | 7.6479 | 82.93% (136/164) | 79.3 (−3.6) — 15 empties: MiMo31 loses to Neo31 by 8.5pp |
| RINIQ-N2 (M2+MiMo15–17) 🙏 | 5.0 GB | 7.6813 | **91.46% (150/164)** | 85.4 (−6.1) — 8 empties: TIES M2 crown |
| RINIQ-N4a (M2+MiMo15–17 FFN-only) 👑 | 5.0 GB | 7.6276 | **91.46% (150/164)** | **86.6 (−4.9)** — **5 empties**: soul is in FFN, tidier than N2 (−6.1) |
| RINIQ-N4b (M2+MiMo15–17 attn-only) | 5.0 GB | 7.6163 | 89.02% (146/164) | **84.1 (−4.9)** — 12 empties: soul divisible, FFN-dominant (−2.4pp without it) |
| RINIQ-N4c (M2+MiMo15–17 ln-only) | 5.0 GB | **7.5795** | 89.63% (147/164) | **85.4 (−4.2)** — 7 empties: best canary, sub-crown verdict (mild liar) |
| RINIQ-N4a-Q38-SMSE (soul+lens+blend) | 5.0 GB | 7.6617 | 89.02% (146/164) | 84.8 (−4.2) — 14 empties: STACKING LOSES. Three separate winners (tissue 91.46, qwen38 +0.6pp on M2, SMSE +1.2pp) interfere combined — wins don't sum |
| RINIQ-N3 (M2+MiMo10,14,15,16,27 rank-vote) | 5.0 GB | 7.6313 | 88.41% (145/164) | 82.9 (−5.5) — 9 empties: machine goes 0-2 (M7 88.41% same score!)
| Ox-SMAPE-5100 (ref) | 5.0 GB | 7.5670 | 88.41% (145/164) | — |
| MiMo-5100-SMAPE (ref) | 5.0 GB | 8.7435 | 70.12% (115/164) | 66.5% |

Dry-run geometry M8/N2 @5100 (SMAPE, Ox+Neo dual-max lens, identical):
F16 21 / F32 177 (norms) / Q4 209 / Q5 5 (1361 MiB) / Q6 15 / Q8 0 —
all floor, zero penthouses (same barbell as MiMo-5100).

N1 sits at the exact midpoint ((88.41+70.12)/2 = 79.27) — now explained:
as a half-foreign interleave, the midpoint is the NULL hypothesis, not a
finding about 3 blocks. The real 3-block verdicts are M8/N2. Seam
friction (24 empties) still stands as the mechanism signal.

## Tissue surgery (technique, 2026-09-23)

`fuse_layers.py` takes per-tissue donor maps: `15:b:ffn` grafts only
FFN-kind tensors of block 15 (rest of the block stays on A). Kinds:
`ffn` / `attn` / `ssm` / `norm` (see `kind_of`) + `ln` (any `*norm*`
tensor — attn_norm, post_attention_norm, ssm_norm). Ranges
(`15-17:b:ffn`), multi-tissue (`15:b:ffn+attn`), `--dry` (map check
without the 18 GB write), `--d` (fourth donor). Verified via dry maps;
whole-block `--map` behavior unchanged (base-A-plus-overrides since the
N1 scar fix).

Builds (all: M2 base + MiMo tissue, SMAPE-5100 Ox+Neo dual-max lens):

| Build | Map | PPL | HE | Empties |
|:------|:----|----:|:--:|:-------:|
| N4a (soul) | `15-17:b:ffn` | 7.6276 | 91.46% | 5 |
| N4b (control) | `15-17:b:attn` | 7.6163 | 89.02% | 12 |
| N4c (gain staging) | `15-17:b:ln` | **7.5795** (best!) | ⏳ HE running | ⏳ |

Soul verdict so far: divisible, FFN-dominant (N4a ties crown, N4b −2.4pp).
N4c sets the trap: best canary of the series vs unknown verdict — norms
portable (discovery) or incompatible (strongest PPL-lies case).

## Decisiveness: the seam-friction hypothesis — ARBITRATED 2026-09-23

| Build | Empties |
|:------|--------:|
| Ox | 4–5 |
| MiMo | 7 |
| N1 (interleave) | **24** |
| M8 (M2+MiMo31) | 15 |
| N2 (M2+MiMo15–17) | 8 |
| N1m (interleave) | **51** |
| N3 (rank-vote 5-block) | 9 |
| N4a (FFN-only) | **5** |
| N4b (attn-only) | 12 |

N1m arbitration LANDED, both ways at once: the mirror (smooth Ox blocks
in MiMo globals) did NOT restore decisiveness — 51 empties, worst in the
lab, seams rule both ways. But block COUNT doesn't rule: N2 carries THREE
foreign blocks with only 8 empties (M2-level). Reconciliation stands and
sharpens: friction ∝ donor behavioral distance × seam surface. One edge
swap (M8, position 31) already costs 15 empties; three distant islands
(N2) cost 8; half-foreign interleaves (N1 24, N1m 51) collapse.

## Verdicts meaning (2026-09-23 night chain)

- **M8 — clean negative A/B.** Same M2, only blk31 Neo→MiMo: PPL
  7.582→7.648, HE 91.46→82.93 (−8.5pp). The V2→M2 law (+3.7pp for one
  block) holds in reverse: blk31 is load-bearing, and Neo31 specifically
  (thinking donor's output block?) beats MiMo31. "Any 31 works" dead.
- **N2 — distant-islands hypothesis CONFIRMED.** M2 + MiMo 15–17 ties
  the M2 crown exactly (91.46%, 150/164, 8 empties): donor islands far
  apart (15–17 vs 24–26) integrate without verdict loss. The mashup
  prayed for 🙏 delivered. Cost: rigor — HE+ drop −6.1 (M5-flavored
  fragility) vs M2's −3.0. Boldness tax, same law as the old series.
- **N1m — PPL-liar crown.** Best PPL of the N-trio (7.96), worst verdict
  (63.41%), 51 empties, HE+ flat 63.4 (honest but weak). Strongest
  canary-vs-verdict split on record. Field notes (hallucination,
  3D-snake fail) now have numbers.
- **MiMo-Q6_K closes the baseline** at 71.95% (+1.8pp over the 5GB
  SMAPE-5100 for +2.2GB — allocation wins over flat stock again).

## Field notes (manual testing)

- RINIQ-N1 thinking depth lands BETWEEN parents (Ox shallow < N1 medium
  < MiMo deep) — mid-layer blocks 15–17 appear to govern reasoning
  depth. Second independent vote for 15–17 as the "thinking" region.
- N1 3D-snake: 1st = 2D game, 2nd = semi-working 3D, 3rd = working
  snake (ultra slow but works), each attempt <4k tokens. N1m pending.

## Field verdict: N1m fails behaviorally

N1m (best PPL of the three, 7.96) hallucinates in manual testing —
jumps block to block, thinks little or nothing, answers fast, failed
3D-snake completely. Strongest "PPL lies" case on record: canary
crowned it, field buried it. Seam incompatibility confirmed
behaviorally, not just in empties.

## Scars

- 2026-09-22: N1m battery reached [164/164], stopper script killed the
  runner before write_jsonl — verdict lost, full rerun required.
  Stopper now fires on pass@1/file, never on task count.

## Where we are & what for (2026-09-23 evening)

Goal: beat regular RINIQ on code (M2-Q38 92.07%) with an Ox×MiMo fusion —
or prove exactly why it can't be done. Current best: N2 ties M2-5100
(91.46%) but loses rigor (HE+ 85.4 vs 87.2). Open fronts:

1. **Tissue surgery** (running: N4a HE now): which tissue carries the
   soul — FFN (compass bet), attention (compatibility), norms (gain
   staging)? If one tissue holds the crown alone, fusions get surgical.
2. **SMSE-6500 universal test** (queued): does the blend own big budgets
   too, or do penthouses rule there (MSE-6500 has 108 Q8)?
3. **SRMSE duel** (queued): SMAPE×RMSE — "SMAPE with manners", beats
   SMAPE everywhere?
4. **Thank-you Ox-6500-SMSE** (queued): for the reviewer, if SMSE holds.

No HF weight uploads until something beats M2-Q38 — ties are not a
reason to ship 5 GB.

## Next

- [x] Night chain DONE 2026-09-23: PPL M8 7.6479 / N2 7.6813; HE M8
  82.93% / N2 91.46% (ties M2) / N1m 63.41% / Q6_K 71.95%; HE+ logged.
- [ ] N2a/b/c dissection DROPPED (machine-first: N3 arbitrates 17 for free)
- [x] N3-auto DONE: PPL 7.6313 / HE 88.41% / HE+ 82.9 — machine 0-2,
  rank-vote ceiling at parent level (M7 scored exactly 88.41% too)
- [ ] N4a (FFN-only, PPL 7.6276 best N-canary, HE running) / N4b
  (attn-only) / N4c (norms-only, highest risk) — tissue mode live
  (`15:b:ffn`, ranges `15-17:b:ffn`, multi `ffn+attn`, `ln` for norms).
- [ ] SMSE-6500 → Ox-6500-SMSE → SRMSE-5100 duel (chained behind tissue).
- [ ] Open `wepiqx/RINIQ-NEXT-GGUF` with N2 (+M8 as honest negative?)
- [ ] Weight compass MiMo vs Ox (needs mmap rewrite — GGUFReader loads
  everything; OOM risk as-is, see 2026-09-23 scar)
- [x] N2-Q38 ANSWERED by artefact (manifest duel): 143/164 (87.20%) loses
  to plain N2 (150) — 0/3 columns, p=0.17. The qwen38 lens has one null
  win (M2 +0.6pp, p=1.00) and one null loss: not the grail.
- Donor watch: DavidAU Defiant-Fable (Qwen3.5-9B) SKIPPED — creative/
  abliterated lane, ablation often costs intelligence. Waiting on
  same-bones coding/agent finetunes instead.
- Prism-coder-9B (tool-routing QLoRA, same bones): idea — graft
  token_embd + output (untouched pins = syntax boundary) from Prism
  onto M2, plus early blocks (backbone zone). Needs BF16 first
  (safetensors downloading).
- Prism-Q6 verdict: PPL 9.2602 / HE 79.27% (130/164) / HE+ 73.2 /
  20 empties — codes decently for a router (between MiMo 70 and Ox 88).
  Emb-graft queued behind verdict.
- Prism-5600-SMSE: PPL 9.3339 / HE 74.39% (122/164) / HE+ 70.7 /
  17 empties — LOSES to Q6 by 4.9pp. SMSE magic is family-local
  (MiMo +1.2pp, Prism −4.9pp). MTP draft duels running (n2/n3+p.8/
  n8+p.8/ngram-mod combo).
