#!/bin/bash
# 9B wiki-fidelity chain (2026-10-23):
#   ref on WIKI from the Q8 base (deviation from the 4B BF16-ref protocol,
#   README note pending — Q8-vs-BF16 adds a ~constant offset to every arm,
#   so arms-vs-arms stays paired and clean; computing/wiki protocol per §1).
# Then: KLD on wiki for the 4 arms (NETPURE/SMSE duel arms + LAW/DEEP controls),
# then the LAW PPL hole.
# Ref size lands ~76.7 GB (302 chunks × 511 positions × 2×nv + tokens),
# therefore the >70 GB size check. Strict GPU queue: waits for any running
# llama-quantize first (the Q8 base IS the ref model).
set -u
B=/home/wepiqx/llama.cpp/build/bin/llama-perplexity
M=/mnt/Vsio/assets/models
R=/mnt/Vsio/assets/refs
WIKI=/mnt/Vsio/wikitext-2-raw/wiki.test.raw
LOG=/mnt/Vsio/assets/labels/shelf_9b_kld_wiki.log
REFQ8=$M/Ornith-1.5-9B-Q8_0.gguf
REFDAT=$R/ref_ornithq8_wiki.dat

step(){ echo "=== $* $(date '+%T') ===" >> "$LOG"; }

# strict GPU/CPU queue: the Q8 base is being quantized by llama-quantize
while pgrep -f "llama-quantize" >/dev/null; do sleep 60; done
sleep 5

# --- 1. wiki reference from the Q8 base (one big forward pass, ~76.7 GB) ---
if [ ! -s "$REFQ8" ]; then
  step "REF Q8 BASE MISSING ($REFQ8) — ABORT"
  exit 1
fi
if [ -s "$REFDAT" ] && [ "$(stat -c%s "$REFDAT")" -gt 70000000000 ]; then
  step "REF: already done"
else
  step "REF dump start (ngl 26, seed 7)"
  rm -f "$REFDAT"   # never resume a partial logits dump
  $B -m "$REFQ8" -f "$WIKI" --save-all-logits "$REFDAT" \
     -c 1024 -b 128 -ngl 26 --load-mode mmap --seed 7 > $R/ref_ornithq8_wiki.txt 2>&1
  S=$(stat -c%s "$REFDAT" 2>/dev/null || echo 0)
  if [ "$S" -gt 70000000000 ]; then
    step "REF dump done ($S bytes)"
  else
    step "REF dump FAILED ($S bytes) — ABORT, inspect $R/ref_ornithq8_wiki.txt"
    exit 1
  fi
fi

# --- 2. KLD on wiki for the 4 arms (tokens/chunks come from the ref file) ---
for A in NETPURE SMSE LAW DEEP; do
  F=Ornith-1.5-MERNIK-5300-$A
  O=$R/kld_${F}_wiki.txt
  if [ -s "$O" ] && grep -qa "Mean    KLD" "$O"; then
    step "KLD $F: already done"
  else
    step "KLD $F start"
    $B -m "$M/$F.gguf" -f "$WIKI" --kl-divergence --kl-divergence-base "$REFDAT" \
       -c 1024 -b 128 -ngl 99 --load-mode mmap --seed 7 > "$O" 2>&1
    grep -aE "Mean    KLD|Maximum KLD|Median  KLD|Mean    Δp|RMS Δp|Same top p" \
       "$O" >> "$LOG" || step "KLD $F: no metrics in output, inspect $O"
  fi
done

# --- 3. LAW PPL hole on wiki (NETPURE/SMSE/DEEP PPL already valid) ---
F=Ornith-1.5-MERNIK-5300-LAW
if [ -s $R/ppl_$F.txt ] && grep -qa "Final estimate" $R/ppl_$F.txt; then
  step "PPL LAW: already done"
else
  step "PPL LAW start"
  $B -m "$M/$F.gguf" -f "$WIKI" -c 1024 -b 128 -ngl 99 --load-mode mmap --seed 7 > $R/ppl_$F.txt 2>&1
  grep -a "Final estimate" $R/ppl_$F.txt >> "$LOG" || \
    step "PPL LAW: no Final estimate, inspect $R/ppl_$F.txt"
fi

step "9B wiki chain DONE"
