#!/bin/bash
# Stock control: Ornith-1.5-9B plain Q4_K_M (no imatrix, no allocator) through
# the full shelf — the row the top never had. Protocol identical to shelf_9b.sh:
# HE (temp 1.0, top_p .95, top_k 20, 164 tasks) -> HE+ (evalplus) ->
# PPL (wiki, -c 1024 -b 128, seed 7) -> KLD on wiki vs the BF16 ref.
set -u
cd /mnt/Vsio/MERNIK || exit 1
PY=/home/wepiqx/.venvs/ar/bin/python
B=/home/wepiqx/llama.cpp/build/bin/llama-perplexity
M=/mnt/Vsio/assets/models
R=/mnt/Vsio/assets/refs
E=/mnt/Vsio/MERNIK/eval_results
LOG=/mnt/Vsio/assets/labels/shelf_9b_stockq4.log
WIKI=/mnt/Vsio/wikitext-2-raw/wiki.test.raw
REF14=$R/ref_ornithbf16_wiki.dat
Q4KM=$M/Ornith-1.5-9B-Q4_K_M.gguf
TAG=ornithstockq4km

export HE_TEMP=1.0 HE_TOP_P=0.95 HE_TOP_K=20 HE_MIN_P=0.0
export HE_PRESENCE=0.0 HE_MAX_TOKENS=2048
export HUMANEVAL_SERVER_ARGS="--cache-ram 0"

step(){ echo "=== $* $(date '+%F %T') ===" >> "$LOG"; }

# strict queue: wait for the ongoing quantize, then own the GPU alone
while pgrep -f "llama-quantize" >/dev/null; do sleep 30; done
sleep 10

if [ ! -s "$Q4KM" ]; then step "Q4_K_M MISSING — ABORT"; exit 1; fi
step "stock Q4_K_M: $(stat -c%s "$Q4KM") bytes"

# --- 1. HumanEval (164 tasks), resume-safe ---
out="$E/humaneval_$TAG.jsonl"
if [ -f "$out" ] && [ "$(wc -l < "$out")" -ge 164 ]; then
    step "HE $TAG: already done"
else
    step "HE $TAG start"
    HUMANEVAL_SERVE_MODEL="$Q4KM" HUMANEVAL_OUT="$out" \
        $PY -u scripts/run_humaneval.py >> "$LOG" 2>&1
    step "HE $TAG done ($(wc -l < "$out") rows)"
fi

# --- 2. HumanEval+ (evalplus) ---
if [ -x /home/wepiqx/.venvs/ar/bin/evalplus.evaluate ]; then
    step "HE+ $TAG"
    /home/wepiqx/.venvs/ar/bin/evalplus.evaluate humaneval --samples "$out" >> "$LOG" 2>&1
fi

# --- 3. PPL on wiki (canary) ---
pp=$R/ppl_Ornith-1.5-9B-Q4_K_M.txt
if [ -s "$pp" ] && grep -qa "Final estimate" "$pp"; then
    step "PPL already done"
else
    step "PPL start"
    $B -m "$Q4KM" -f "$WIKI" -c 1024 -b 128 -ngl 99 --load-mode mmap --seed 7 > "$pp" 2>&1
    grep -a "Final estimate" "$pp" >> "$LOG" || step "PPL: no Final estimate, inspect $pp"
fi

# --- 4. KLD on wiki vs the BF16 reference (the fidelity row) ---
kd=$R/kld_Ornith-1.5-9B-Q4_K_M_bf16ref.txt
if [ -s "$kd" ] && grep -qa "Mean    KLD" "$kd"; then
    step "KLD already done"
else
    step "KLD start"
    $B -m "$Q4KM" -f "$WIKI" --kl-divergence --kl-divergence-base "$REF14" \
       -c 1024 -b 128 -ngl 99 --load-mode mmap --seed 7 > "$kd" 2>&1
    grep -aE "Mean    KLD|Maximum KLD|Median  KLD|RMS Δp|Same top p" "$kd" >> "$LOG" \
        || step "KLD: no metrics, inspect $kd"
fi

step "stock Q4_K_M shelf DONE"
