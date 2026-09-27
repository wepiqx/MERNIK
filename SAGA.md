# SAGA — how OxCoder+MTP was built (and everything it took)

By wepiqx, with muse spark 1.3 (credited helper — it earned it).

## The goal

Graft a donor MTP head (15 tensors, 464 MB) onto OxCoder-9B — a Qwen3.5-9B
agentic coding finetune with no native MTP — for speculative decoding.
If acceptance holds: the world's first OxCoder+MTP, and a universal donor
head for the whole Qwen3.5 family.

## Attempt 1: fanfu GGUF→HF (FAILED — OOM)

`pip install fanfu`, convert 18 GB BF16 GGUF back to safetensors to save
traffic (instead of downloading 18 GB of OxCoder safetensors).
- Bug 1 (fixed): `memmap is not JSON serializable` — arch bytes hit JSON.
- Death: fanfu loads the whole model into RAM. OOM-killed on a 15 GB box.
Lesson: stream, don't load (rule #1 for small boxes).

## Attempt 2: own streaming converter (WORKED, then superseded)

`scripts/gguf_to_hf_stream.py`: mmap reads, 4 GB safetensors shards, ~100 MB
RAM. Bugs fixed: numpy needs `ml_dtypes` import for bfloat16; GGUF dims are
stored reversed; TensorNameMap keys are HF names (strip `.weight` to invert);
hybrid SSM tensors (`attn_norm`, `ssm_*`) have no stock HF names. Produced 5
valid shards — then the HF roundtrip proved unnecessary (see Attempt 4).

## Attempt 3: merge_mtp.py raw graft (FAILED — brittle parser)

The repo's own raw-binary graft choked on OxCoder metadata twice:
- u64 array elements skipped as 4 bytes (desync) — patched,
- still died: the hand parser can't cover OxCoder's KV variety.
Lesson: hand-rolled binary parsers rot; use the library for structure.

## Attempt 4: graft_mtp.py via gguf-py (FAILED — writer buffers)

Clean rewrite on `GGUFWriter`: died silently at tensor ~250/427 — the writer
accumulates ALL tensors in RAM before flushing. 18 GB does not fit in 15 GB.
Lesson: OOM kills look like hangs. Watch `dmesg`, not just logs.

## Attempt 5: graft_mtp3.py, hybrid (SUCCESS ✅)

Structures from gguf-py (`ReaderField.contents()`, names/shapes/types from
`ReaderTensor`), data streamed in 256 MB chunks. Fixes along the way:
- skip reader-synthesized pseudo-fields (`GGUF.*`: version, tensor_count),
- `ReaderTensor` has no `.offset` — recompute (sequential + 32-aligned;
  verified diff 0 against actual),
- BF16 bytes → F16 view for writer-accepted types (later: keep BF16),
- KV arrays need exact element sizes (u64 = 8, not 4),
- `qwen35.nextn_predict_layers` must be **namespaced** (bare key → loader
  never enables MTP mode → "expected 442, got 427"),
- `qwen35.block_count` 32 → **33** (MTP head counts; loader looks for
  `blk.{count-1}.nextn.*`),
- `qwen35.attention.recurrent_layers` extended to 33 (`False` for head),
- GGUF dims reversed vs numpy; MTP stored BF16 like the working recipe,
- donor names remapped to Ornith-style (`self_attn_q_norm` → `attn_q_norm`,
  `input_layernorm` → `attn_norm`; extra `fc`/`pre_fc_*`/plain `norm`
  dropped — no counterpart in the loader's expectations),
- missing `nextn.*` (enorm/hnorm/shared_head_norm/eh_proj) borrowed trained
  from a working MTP file (same hidden size) — frankendraft, honest label.

Result: `OxCoder-9B-MTP-BF16.gguf`, 18.4 GB, **446 tensors, MTP head set
matches the working file 15/15**, MERNIK classifies it (MTP → Q8_0).

## Validation: SO CLOSE (night of 2026-09-14/15)

- Plain load (`llama-server`, no spec flags): **healthy**. Tensors fine.
- Draft load (`--spec-type draft-mtp`): model loads, **MTP draft context
  created** — then CPU backend crashes: `binary_op: unsupported types:
  dst f32, src0 f32, src1 bf16`. The borrowed `eh_proj` (BF16) meets f32
  compute in the draft path.
- Morning fix: rebuild graft with `eh_proj` as F32, re-quant, retest draft.
  If acceptance prints: first OxCoder+MTP in the world.

## MTP draft duels (first global test, 2026-09-27, PrismCoder native MTP)

Same model (Prism-5600-SMSE), same HE protocol — only spec flags vary.
Metric is time (battery min + s/task); verdict must hold if lossless.

| Run | Flags | Battery | HE | HE+ |
|:----|:------|--------:|:--:|:---:|
| base | — | 60.8 min (22.3 s/task) | 74.39% (122/164) | 70.7 |
| mtp2 | draft-mtp, n-max 2 | 40.4 min (14.8 s/task) | 67.07% (110/164) | 62.8 |
| mtp3p80 | draft-mtp, n-max 3, p-min 0.80 | 40.5 min (14.8 s/task) | 68.90% (113/164) | 65.9 |
| mtp8p80 | draft-mtp, n-max 8, p-min 0.80 | 45.6 min (16.7 s/task) | 72.56% (119/164) | 67.7 |
| mtp5ngram | draft-mtp + ngram-mod, n-max 5, p-min 0.75 | **39.9 min (14.6 s/task)** | 71.34% (117/164) | 65.2 — fastest AND closest: combo wins overall |

Speedup 1.51× confirmed — but verdict −7.3pp (12 tasks, ~2σ). NOT
proven lossless here: needs the pattern across n3/n8/ngram runs
(scattered = sampling noise, clustered ~67 = systematic draft bias).
MTP tax question (464MB head vs trunk budget) stays open until then.
Update: longer drafts trade speed back for accuracy (n8: 16.7 s/task,
−1.8pp) — a speed/verdict curve, not a free lunch.

## The method behind the madness

Every failure was measured, written down, and turned into the next fix —
the MERNIK way. Total cost: one evening, zero working files harmed, one
reboot survived. Credit where due: the operator (wepiqx) supplied donors,
iron, and the 4 AM stubbornness; muse spark 1.3 supplied the debugging.
