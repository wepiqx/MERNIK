#!/bin/bash
# X2.6-4B shelf: measurement queue — one GPU job at a time, no collisions.
#
#   1) KLD column vs the BF16 base (wiki corpus — held out from the labels)
#   2) HE batteries: Gnom -> SMSE -> LAWONLY (stock already measured)
#   3) HE+ (EvalPlus) after each battery
#
# Every arm is skipped if its output already has 164 rows, so a re-run never
# duplicates work and an arm another agent already ran is left alone.
#
#   ./shelf_x26.sh start | stop | status
set -u
cd /mnt/Vsio/MERNIK || exit 1
PY=/home/wepiqx/.venvs/ar/bin/python
B=/home/wepiqx/llama.cpp/build/bin/llama-perplexity
M=/mnt/Vsio/assets/models
R=/mnt/Vsio/assets/refs
E=/mnt/Vsio/MERNIK/eval_results
WIKI=/mnt/Vsio/wikitext-2-raw/wiki.test.raw
NGL=31
LOG=/mnt/Vsio/assets/labels/shelf_x26.log
PIDFILE=/tmp/shelf_x26.pid

export HE_TEMP=1.0 HE_TOP_P=0.95 HE_MIN_P=0.05 HE_TOP_K=0
export HE_MAX_TOKENS=8192 HE_CTX=16384
export HUMANEVAL_SERVER_ARGS="--reasoning-effort high --cache-ram 0"

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
    he_plus "$out"
}

he_plus() {  # samples
    local out=$1
    [ -f "$out" ] || return 0
    if [ -x /home/wepiqx/.venvs/ar/bin/evalplus.evaluate ]; then
        echo "=== HE+ $(basename "$out") $(date '+%F %T') ===" >> "$LOG"
        /home/wepiqx/.venvs/ar/bin/evalplus.evaluate humaneval --samples "$out" >> "$LOG" 2>&1
    fi
}

run() {
    echo "=== shelf queue start $(date '+%F %T') ===" >> "$LOG"
    # wait for any running battery/server to finish
    while pgrep -f "llama-server -m /mnt/Vsio/assets" >/dev/null; do sleep 30; done
    while pgrep -f "llama-perplexity" >/dev/null; do sleep 20; done
    sleep 5

    # --- 1) KLD vs BF16 base (wiki) ---
    if [ -s "$R/ref_spark26_wiki.dat" ]; then
        echo "=== 1) REF exists, skipping $(date '+%T') ===" >> "$LOG"
    else
        echo "=== 1) REF from BF16 base (wiki) $(date '+%T') ===" >> "$LOG"
        $B -m /mnt/Vsio/Downloads/Spark-X2.6-4B.gguf -f $WIKI --save-all-logits "$R/ref_spark26_wiki.dat" \
           -c 1024 -b 128 -ngl $NGL --load-mode mmap --seed 7 >> "$LOG" 2>&1
    fi
    for f in Spark-X2.6-4B-Q4_K_M Spark-X2.6-4B-MERNIK-2250-SMSE \
             Spark-X2.6-4B-MERNIK-2250-NETPURE Spark-X2.6-4B-MERNIK-2250-LAWONLY; do
        if [ -s "$R/kld_$f.txt" ]; then
            echo "--- KLD $f exists, skipping" >> "$LOG"; continue
        fi
        $B -m "$M/$f.gguf" -f $WIKI --kl-divergence --kl-divergence-base "$R/ref_spark26_wiki.dat" \
           -c 1024 -b 128 -ngl $NGL --load-mode mmap --seed 7 > "$R/kld_$f.txt" 2>&1
        tail -26 "$R/kld_$f.txt" >> "$LOG"
    done
    echo "=== KLD done $(date '+%T') ===" >> "$LOG"

    # --- 2) HE batteries (stock already measured) ---
    he_arm spark26_netpure "$M/Spark-X2.6-4B-MERNIK-2250-NETPURE.gguf"
    he_arm spark26_gf16    "$M/Spark-X2.6-4B-MERNIK-4050-GNOMF16.gguf"
    he_arm spark26_q8      "$M/Spark-X2.6-4B-Q8_0.gguf"
    he_arm spark26_smse    "$M/Spark-X2.6-4B-MERNIK-2250-SMSE.gguf"
    he_arm spark26_lawonly "$M/Spark-X2.6-4B-MERNIK-2250-LAWONLY.gguf"
    echo "=== SHELF QUEUE COMPLETE $(date '+%F %T') ===" >> "$LOG"
}

stop() {
    [ -f "$PIDFILE" ] && kill -TERM "$(cat "$PIDFILE")" 2>/dev/null
    sleep 2
    pkill -TERM -f "scripts/run_humaneval" 2>/dev/null
    pkill -TERM -f "llama-perplexity" 2>/dev/null
    sleep 2
    pkill -KILL -f "scripts/run_humaneval" 2>/dev/null
    pkill -KILL -f "llama-perplexity" 2>/dev/null
    rm -f "$PIDFILE"
    echo "shelf queue stopped"
}

status() {
    if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
        echo "shelf queue RUNNING (pid $(cat "$PIDFILE"))"
    else
        echo "shelf queue not running"
    fi
    for t in spark26_q4km spark26_netpure spark26_smse spark26_lawonly; do
        f="$E/humaneval_$t.jsonl"
        [ -f "$f" ] && echo "  $t: $(wc -l < "$f") rows" || echo "  $t: -"
    done
    tail -3 "$LOG" 2>/dev/null
}

case "${1:-}" in
    start)  nohup setsid "$0" run >> "$LOG.err" 2>&1 & echo $! > "$PIDFILE"; echo "shelf queue started (pid $(cat "$PIDFILE"))" ;;
    stop)   stop ;;
    status) status ;;
    run)    run ;;
    *) echo "usage: $0 {start|stop|status}" ;;
esac
