#!/bin/bash
# Deep-tier label chain — flexible start/stop.
# Jobs, in order: finish 4B-Q4 -> 1.7B-Q3 -> 4B-Q3.
# Every job is --resume, so killing and restarting is loss-free.
#
#   ./chain_deep.sh start     # launch in background
#   ./chain_deep.sh stop      # stop the current job (resume-safe)
#   ./chain_deep.sh status    # show progress + label counts
#   ./chain_deep.sh log       # tail the log
set -u
cd /mnt/Vsio/MERNIK || exit 1
export PYTHONPATH=/home/wepiqx/llama.cpp/gguf-py
export LLAMA_QUANTIZE=/home/wepiqx/llama.cpp/build/bin/llama-quantize
export LLAMA_PPL=/home/wepiqx/llama.cpp/build/bin/llama-perplexity

M4=/mnt/Vsio/assets/models/Spark-X2.5-4B.gguf
I4=/mnt/Vsio/assets/models/Spark-X2.5-4B-imatrix.gguf
R4=/mnt/Vsio/assets/refs/ref_4b_rlmini.dat
M17=/mnt/Vsio/assets/models/Spark-X2.5-1.7B-BF16.gguf
I17=/mnt/Vsio/assets/models/Spark-X2.5-1.7B-imatrix.gguf
R17=/mnt/Vsio/assets/refs/ref_17_rlmini.dat
CORPUS=/mnt/Vsio/rl_reasoning_mini.raw
L=/mnt/Vsio/assets/labels
PIDFILE=/tmp/chain_deep.pid
LOG="$L/chain_deep.log"
SWEEP="python3 -u scripts/group_damage_sweep.py"

job() {  # name model imatrix ref ngl out drop
    local name=$1 model=$2 im=$3 ref=$4 ngl=$5 out=$6 drop=$7
    echo "=== [$(date '+%F %T')] JOB $name (drop=$drop) -> $(basename "$out") ===" >> "$LOG"
    $SWEEP --model "$model" --imatrix "$im" --kld-base "$ref" --kld-only \
        --wiki-file "$CORPUS" --base-tier Q5_K --drop-tier "$drop" --ngl "$ngl" \
        --out "$out" --resume --timeout 3600 >> "$LOG" 2>&1
}

run() {
    while true; do
        job "4B-Q4"   "$M4"  "$I4"  "$R4"  31  "$L/damage_4b_rlmini_kldonly.jsonl"  Q4_K
        job "1.7B-Q3" "$M17" "$I17" "$R17" 99  "$L/damage_17b_rlmini_q3.jsonl"      Q3_K
        job "4B-Q3"   "$M4"  "$I4"  "$R4"  31  "$L/damage_4b_rlmini_q3.jsonl"       Q3_K
        echo "=== chain COMPLETE [$(date '+%F %T')] ===" >> "$LOG"
        exit 0
    done
}

stop() {
    [ -f "$PIDFILE" ] && kill -TERM "$(cat "$PIDFILE")" 2>/dev/null
    sleep 2
    [ -f "$PIDFILE" ] && kill -KILL "$(cat "$PIDFILE")" 2>/dev/null
    pkill -TERM -f "scripts/group_damage_sweep" 2>/dev/null
    sleep 1
    pkill -KILL -f "scripts/group_damage_sweep" 2>/dev/null
    pkill -KILL -f "bin/llama-quantize" 2>/dev/null
    pkill -KILL -f "bin/llama-perplexity" 2>/dev/null
    rm -f "$PIDFILE"
    echo "chain stopped (resume-safe)"
}

status() {
    if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
        echo "chain RUNNING (pid $(cat "$PIDFILE"))"
    else
        echo "chain not running"
    fi
    echo "--- jobs done ---"
    for f in damage_4b_rlmini_kldonly damage_17b_rlmini_q3 damage_4b_rlmini_q3; do
        [ -f "$L/$f.jsonl" ] && echo "  $f: $(wc -l < "$L/$f.jsonl") rows" || echo "  $f: -"
    done
    echo "--- last log ---"; tail -3 "$LOG" 2>/dev/null
}

case "${1:-}" in
    start)  nohup setsid "$0" run >/dev/null 2>&1 & echo $! > "$PIDFILE"; echo "chain started (pid $(cat "$PIDFILE"))" ;;
    stop)   stop ;;
    status) status ;;
    log)    tail -f "$LOG" ;;
    run)    run ;;
    *) echo "usage: $0 {start|stop|status|log}" ;;
esac
