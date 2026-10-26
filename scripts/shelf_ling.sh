#!/bin/bash
# Ling-3.0-tiny-moe leg (2026-10-23): first MoE family in the Fracture pool.
# arch bailingmoe3 (MoE + KDA/SSM + MLA), 8B total / A1B active, 128 experts
# top-8 (+1 shared), vocab 157184, 526 tensors, pure BF16/F16/F32 base.
# Chain: BF16 ref on rlmini -> 1-unit smoke -> full per-tensor sweep Q5_K->Q4_K.
# Ref via --n-cpu-moe (experts CPU, attn GPU) — 8 GB card never holds the BF16.
# Note: MoE routers are F16-PINNED by constants.py — the router 'verified
# disaster' row is deliberately NOT in this leg (separate one-shot after).
set -u
B=/home/wepiqx/llama.cpp/build/bin
M=/mnt/Vsio/assets/models
LM=/mnt/Vsio/MERNIK/gnom
CORPUS=/mnt/Vsio/rl_reasoning_mini.raw
IMAT=/mnt/Vsio/shq-program/models/Ling-3.0-tiny-imatrix.gguf
MODEL=$M/Ling-3.0-tiny-bf16.gguf
REFDAT=$LM/ref_ling_rlmini.dat
OUT=$LM/damage_ling_moe_q4.jsonl
LOG=/mnt/Vsio/assets/labels/ling_leg.log
SWEEP=/mnt/Vsio/MERNIK/scripts/group_damage_sweep.py
PY=/home/wepiqx/.venvs/ar/bin/python

step(){ echo "=== $* $(date '+%T') ===" >> "$LOG"; }

# strict queue: any running local llama-perplexity owns the GPU first
while pgrep -f "llama-perplexity" >/dev/null; do sleep 60; done
sleep 5

if [ ! -s "$MODEL" ]; then step "LING BF16 MISSING — ABORT"; exit 1; fi
if [ ! -s "$IMAT" ]; then step "LING IMATRIX MISSING — ABORT"; exit 1; fi

# --- 1. ref dump (BF16, experts on CPU) ---
if [ -s "$REFDAT" ] && [ "$(stat -c%s "$REFDAT")" -gt 15000000000 ]; then
  step "ref: already exists ($(stat -c%s "$REFDAT") bytes)"
else
  # operator ladder: MORE GPU, fewer CPU expert layers — start 6, +2 per OOM
  for NM in 6 8 10 12 14 16 18 20 22 24; do
    step "ref dump try (n-cpu-moe $NM, seed 7)"
    rm -f "$REFDAT"
    $B/llama-perplexity -m "$MODEL" -f "$CORPUS" \
       --save-all-logits "$REFDAT" \
       -c 1024 -b 512 -ngl 99 --n-cpu-moe $NM --load-mode mmap --seed 7 \
       > $LM/ref_ling_rlmini.txt 2>&1
    S=$(stat -c%s "$REFDAT" 2>/dev/null || echo 0)
    if [ "$S" -gt 15000000000 ]; then step "ref dump done ($S bytes, n-cpu-moe $NM)"; break; fi
    step "ref try n-cpu-moe $NM FAILED ($S bytes) — next rung"
  done
  [ "$S" -gt 15000000000 ] || { step "ref dump FAILED on all rungs — ABORT"; exit 1; }
fi

# --- 2. 1-unit smoke (forge discipline) ---
if ! grep -qa "SMOKE DONE" "$LOG"; then
  step "smoke: 1 unit (attn_output@2)"
  $PY "$SWEEP" --model "$MODEL" --imatrix "$IMAT" --out "$OUT" \
     --base-tier Q5_K --drop-tier Q4_K --per-tensor \
     --kld-base "$REFDAT" --kld-only \
     --wiki-file "$CORPUS" --timeout 3600 --ngl 99 \
     --units "attn_output@2" >> "$LOG" 2>&1
  N=$(grep -ac '"unit"' "$OUT" 2>/dev/null || echo 0)
  if [ "$N" -ge 2 ]; then step "SMOKE DONE ($N rows) — full sweep next"
  else step "SMOKE FAILED — ABORT, inspect $LOG"; exit 1; fi
fi

# --- 3. full sweep (resume-safe) ---
step "full sweep start"
$PY "$SWEEP" --model "$MODEL" --imatrix "$IMAT" --out "$OUT" \
   --base-tier Q5_K --drop-tier Q4_K --per-tensor \
   --kld-base "$REFDAT" --kld-only \
   --wiki-file "$CORPUS" --timeout 3600 --ngl 99 --resume >> "$LOG" 2>&1
step "Ling MoE leg DONE ($(grep -ac '"unit"' "$OUT" 2>/dev/null) rows)"
