#!/bin/bash
# 9B ref-tier validation leg (2026-10-23, operator request):
# does the reference tier (Q8 vs BF16) shift the arms' KLD on the SAME
# 302-chunk wiki protocol? Runs AFTER the Q8 chain closes:
#   rm the Q8 ref (battery closed; re-dumpable in ~20 min if ever needed)
#   -> BF16 ref dump (ngl 10: 8 GB card cannot hold 18.4 GB; CPU-bound, slow)
#   -> KLD for the 4 arms vs the BF16 ref (kld_*_wiki_bf16ref.txt)
# The paired delta per arm answers "on which ref tier to run from now on".
# Disk law: only ONE ref exists at a time (76.7 GB each, 155 GB free).
set -u
B=/home/wepiqx/llama.cpp/build/bin/llama-perplexity
M=/mnt/Vsio/assets/models
R=/mnt/Vsio/assets/refs
WIKI=/mnt/Vsio/wikitext-2-raw/wiki.test.raw
LOG=/mnt/Vsio/assets/labels/shelf_9b_ref_ab.log
REFBF16=$R/ref_ornithbf16_wiki.dat
MODELB=$M/Ornith-1.5-9B-BF16.gguf

step(){ echo "=== $* $(date '+%T') ===" >> "$LOG"; }

# strict queue: the Q8 chain owns the GPU until DONE
while pgrep -f "shelf_9b_kld_wiki.sh" >/dev/null; do sleep 60; done
sleep 5

if [ ! -s "$MODELB" ]; then
  step "BF16 BASE MISSING ($MODELB) — ABORT"
  exit 1
fi

# free the Q8 ref BEFORE the BF16 dump: one ref at a time
rm -f $R/ref_ornithq8_wiki.dat
step "Q8 ref freed (battery closed, re-dumpable)"

if [ -s "$REFBF16" ] && [ "$(stat -c%s "$REFBF16")" -gt 70000000000 ]; then
  step "BF16 ref: already done"
else
  step "BF16 ref dump start (ngl 10, seed 7)"
  rm -f "$REFBF16"
  $B -m "$MODELB" -f "$WIKI" --save-all-logits "$REFBF16" \
     -c 1024 -b 128 -ngl 10 --load-mode mmap --seed 7 > $R/ref_ornithbf16_wiki.txt 2>&1
  S=$(stat -c%s "$REFBF16" 2>/dev/null || echo 0)
  if [ "$S" -gt 70000000000 ]; then
    step "BF16 ref dump done ($S bytes)"
  else
    step "BF16 ref dump FAILED ($S bytes) — ABORT, inspect $R/ref_ornithbf16_wiki.txt"
    exit 1
  fi
fi

for A in NETPURE SMSE LAW DEEP; do
  F=Ornith-1.5-MERNIK-5300-$A
  O=$R/kld_${F}_wiki_bf16ref.txt
  if [ -s "$O" ] && grep -qa "Mean    KLD" "$O"; then
    step "KLD $F: already done"
  else
    step "KLD $F start"
    $B -m "$M/$F.gguf" -f "$WIKI" --kl-divergence --kl-divergence-base "$REFBF16" \
       -c 1024 -b 128 -ngl 99 --load-mode mmap --seed 7 > "$O" 2>&1
    grep -aE "Mean    KLD|Maximum KLD|Median  KLD|Mean    Δp|RMS Δp|Same top p" \
       "$O" >> "$LOG" || step "KLD $F: no metrics in output, inspect $O"
  fi
done

step "9B ref-tier comparison DONE (Q8-ref battery + BF16-ref battery)"
