#!/bin/bash
# 9B duel at the Q4 floor: Gnom (netpure) vs SMSE, both @5300, same protocol.
# Ordinary protocol (the Ornith family default) + --cache-ram 0 (RAM hygiene,
# does not change answers). One GPU job at a time; HE+ after each arm.
set -u
cd /mnt/Vsio/MERNIK || exit 1
PY=/home/wepiqx/.venvs/ar/bin/python
M=/mnt/Vsio/assets/models
E=/mnt/Vsio/MERNIK/eval_results
LOG=/mnt/Vsio/assets/labels/shelf_9b.log
PIDFILE=/tmp/shelf_9b.pid

export HE_TEMP=1.0 HE_TOP_P=0.95 HE_TOP_K=20 HE_MIN_P=0.0
export HE_PRESENCE=0.0 HE_MAX_TOKENS=2048
export HUMANEVAL_SERVER_ARGS="--cache-ram 0"

done_rows() { [ -f "$1" ] && [ "$(wc -l < "$1")" -ge 164 ] && echo yes || echo no; }

he_arm() {   # tag model
    local tag=$1 model=$2
    local out="$E/humaneval_$tag.jsonl"
    if [ "$(done_rows "$out")" = "yes" ]; then
        echo "=== HE $tag: already done, skipping ===" >> "$LOG"; return 0
    fi
    echo "=== HE $tag start $(date '+%F %T') ===" >> "$LOG"
    HUMANEVAL_SERVE_MODEL="$model" HUMANEVAL_OUT="$out" \
        $PY -u scripts/run_humaneval.py >> "$LOG" 2>&1
    echo "=== HE $tag done $(date '+%F %T') ===" >> "$LOG"
    if [ -x /home/wepiqx/.venvs/ar/bin/evalplus.evaluate ]; then
        echo "=== HE+ $tag $(date '+%F %T') ===" >> "$LOG"
        /home/wepiqx/.venvs/ar/bin/evalplus.evaluate humaneval --samples "$out" >> "$LOG" 2>&1
    fi
}

run() {
    echo "=== 9B shelf start $(date '+%F %T') ===" >> "$LOG"
    # wait until the Gnom build is finished (size stable)
    while pgrep -f "main.py.*ornith" >/dev/null; do sleep 20; done
    sleep 10
    he_arm ornith5300netpure "$M/Ornith-1.5-MERNIK-5300-NETPURE.gguf"
    he_arm ornith5300smse    "$M/Ornith-1.5-MERNIK-5300-SMSE.gguf"
    echo "=== 9B SHELF COMPLETE $(date '+%F %T') ===" >> "$LOG"
}

case "${1:-}" in
    start)  nohup setsid "$0" run >> "$LOG.err" 2>&1 & echo $! > "$PIDFILE"; echo "9B queue started (pid $(cat "$PIDFILE"))" ;;
    stop)   [ -f "$PIDFILE" ] && kill -TERM "$(cat "$PIDFILE")" 2>/dev/null; pkill -TERM -f "run_humaneval" 2>/dev/null; rm -f "$PIDFILE"; echo stopped ;;
    status) if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then echo "RUNNING (pid $(cat "$PIDFILE"))"; else echo "not running"; fi; tail -3 "$LOG" 2>/dev/null ;;
    run)    run ;;
    *) echo "usage: $0 {start|stop|status}" ;;
esac
