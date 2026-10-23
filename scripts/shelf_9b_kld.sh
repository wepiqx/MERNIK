#!/bin/bash
# 9B fidelity chain: KLD + PPL for EVERY arm, split per step.
# Arms: NETPURE, SMSE (duel) + LAWONLY, DEEP (falsification controls).
# HE only runs for NETPURE/SMSE (shelf_9b.sh); this script owns the rest.
# Every block is its own armored unit with a done-marker in its own file:
# a crashed step never takes down the rest — rerun resumes at the gaps.
# Reference: the forge's Q8 Ornith rlmini logit dump (26.9 GB), same base for
# all arms, so comparisons are paired; absolute values carry the
# CUDA-vs-Vulkan forge offset (only the relative arms-vs-arms is clean).
set -u
cd /mnt/Vsio/MERNIK || exit 1
B=/home/wepiqx/llama.cpp/build/bin/llama-perplexity
M=/mnt/Vsio/assets/models
R=/mnt/Vsio/assets/refs
WIKI=/mnt/Vsio/wikitext-2-raw/wiki.test.raw
LOG=/mnt/Vsio/assets/labels/shelf_9b_kld.log

# strict GPU queue: wait for the HE shelf (and its HE+ rescoring) to exit
while pgrep -f "shelf_9b.sh run" >/dev/null; do sleep 30; done
sleep 5

for A in NETPURE SMSE LAWONLY DEEP; do
  F=Ornith-1.5-MERNIK-5300-$A

  # --- KLD on rlmini vs the forge Q8 reference (5-column fidelity family) ---
  if [ -s "$R/kld_$F.txt" ] && grep -qa "Mean    KLD" "$R/kld_$F.txt"; then
    echo "=== KLD $F: already done ===" >> "$LOG"
  else
    echo "=== KLD $F start $(date '+%T') ===" >> "$LOG"
    $B -m "$M/$F.gguf" -f /mnt/Vsio/rl_reasoning_mini.raw --kl-divergence \
       --kl-divergence-base "$R/ref_ornithq8_rlmini.dat" \
       -c 1024 -b 128 -ngl 99 --load-mode mmap --seed 7 > "$R/kld_$F.txt" 2>&1
    grep -aE "Mean    KLD|Maximum KLD|Median  KLD|Mean    Δp|RMS Δp|Same top p" \
       "$R/kld_$F.txt" >> "$LOG" || echo "KLD $F: no metrics in output, inspect $R/kld_$F.txt" >> "$LOG"
  fi

  # --- PPL on wiki (fast-ring canary) ---
  if [ -s "$R/ppl_$F.txt" ] && grep -qa "Final estimate" "$R/ppl_$F.txt"; then
    echo "=== PPL $F: already done ===" >> "$LOG"
  else
    echo "=== PPL $F start $(date '+%T') ===" >> "$LOG"
    $B -m "$M/$F.gguf" -f $WIKI -c 1024 -b 128 -ngl 99 \
       --load-mode mmap --seed 7 > "$R/ppl_$F.txt" 2>&1
    grep -a "Final estimate" "$R/ppl_$F.txt" >> "$LOG" || \
       echo "PPL $F: no Final estimate, inspect $R/ppl_$F.txt" >> "$LOG"
  fi
done
echo "=== 9B KLD+PPL (4 arms) DONE $(date '+%T') ===" >> "$LOG"
