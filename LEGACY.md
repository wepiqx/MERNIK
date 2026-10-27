# Legacy modes and removed allocation paths

Gnom is the active allocator. Everything in this file is frozen for
reproducing published artefacts, not for new work. Archived utilities and
their flags are hidden from `--help` and require an explicit:

```bash
python main.py ... --utility huber --legacy-modes
```

Without `--legacy-modes`, an archived utility or archived flag fails loudly
instead of silently taking a different path.

## Archived utility modes

| mode | belief | status / evidence |
|:---|:---|:---|
| `rmse` | perfectionist rescale of MSE | beats MSE on PPL twice; RINIQ-M2-6500-RMSE-Q38: PPL 7.5411, GPQA 50.00%, HE 88.41% — direction, not rank |
| `srmse` | SMAPE-relative × compressed absolute | demoted 2026-09-28; MiMo-5100 duel 0/3 |
| `huber` | fat middle: push Q4/Q6 stragglers into Q5_K | best of six at FrogNano-3500 (122/164); Mellum-7500 123 vs SMSE 125, duel 0/3 |
| `logcosh` | smooth gain reshaper | historical lens fixture |
| `logratio` | log(ec/en), extreme barbell | 110/164 on FrogNano-3500 |
| `powalpha` + `--pow-alpha A` | g = ec^A − en^A | non-monotonic axis; pure-absolute endpoint is the worst seat |
| `recovery` | (ec−en)/ec, fraction recovered | ties SMSE at Mellum-6500 (123 vs 126) |
| `mix` + `--mix-base` / `--mix-top-frac` | kings by MSE, rest by another base | `_scale()` normalizes formulas; verdict queued |
| `ssim`, `smape_ssim`, `smape_frag` | SSIM-flavoured variants | table-driven (`--ssim-table`, `--frag-w`) |
| `balance` | historical balance lens | kept only for reproduction |

`netpure` and `netfd` are deprecated aliases for `gnom`:

```bash
--utility netpure   # bottom-up Gnom (predicted damage is the gain)
--utility netfd     # top-down Gnom (auto-enables --top-down)
```

## Archived flags

These are accepted only with `--legacy-modes`:

- `--legacy-1d` — price 1D tensors and `ssm_conv1d` as quantizable F16.
  Reproduces pre-2026-09-28 budgets; the binary still writes F32.
- `--squeeze`, `--invert`, `--invert-keep` — adversarial/binary experiments.
- `--relief`, `--relief-thr` — measured relief ceilings.
- `--hinge-w`, `--huber-delta`, `--pow-alpha`, `--ssim-table`, `--frag-w`,
  `--mix-base`, `--mix-top-frac` — formula parameters for archived modes.
- `--pin-small MIB` — experimental: pin every sub-MIB tensor at F16, outside
  the budget (bottom-up only).
- `--max-tier T` — experimental runtime override of `CLASS_MAX_TIER`
  (bottom-up only; top-down starts at F16 by construction).
- `--free-pins` — experimental: let output/token_embd/MTP/routers compete for
  the budget instead of fixed pins.
- `--imatrix-legacy-combine` — with several `--imatrix` files, allocate on the
  combined lens but hand `llama-quantize` only the first one.

## Removed dead code

Deleted because no decision or published artefact depends on them:

- `experimental.py` (old `netpure`/`netdmg`/`netfd` helpers; `gnom.py` is the
  single source of truth now).
- `hybrid`, `pw_ssim`, `netdmg` utility modes and `--cv-w`, `--linf-w`,
  `--ptable`, `--netdmg-w`.
- `--aggro` (deprecated alias for `--size`).
- `--pin-norms` — proven no-op: the binary writes norms F32 regardless, so
  the shield changed 0 tensors.
- `--verbose` — parsed but never read; no output changed.
- `--show-floors` / `CLASS_HARD_FLOORS` — display-only values that the
  greedy never enforced.
- `QUANT_RANK`, `TOXICITY_MODEL`, `toxicity_multiplier` — unused.

## Toxicity note (measured 2026-09-28)

`TOXICITY_SUB4 = 2.0` is the only surviving sub-4 constant. The original
step model was tested against a tail model on 112 paired Spark-1.7B units
(Q5_K→Q4_K vs Q5_K→Q3_K):

```
damage @Q4:  mean +0.120  median +0.042  std 0.773  P(>0)=60.7%  P(>1)= 7.1%
damage @Q3:  mean +0.090  median +0.192  std 1.521  P(>0)=61.6%  P(>1)=19.6%
paired diff: mean -0.031  median +0.093  std 1.605  P(>0)=59.8%
```

Crossing into sub-4 does not inflate the mean; it changes the tail
(P(>1) 7.1% → 19.6%). A per-group inflation constant cannot capture that,
so the model was retired. The step constant remains for the legacy
bottom-up path.
