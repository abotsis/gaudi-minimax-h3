#!/usr/bin/env bash
# graphs_r768_watchdog.sh — run the r768 graphs-ON bench under an RSS watchdog.
#
# v121 post-mortem: r768 graph capture (host IR ~linear in seq 60144) ballooned
# the worker to 86.6 GB RSS and the KERNEL global-OOM-killed it — taking the
# whole box down with it. This harness runs the same config SOLO and kills the
# worker itself at a high-water mark, so the box survives even if capture
# can't finish. Threshold deliberately generous (110 GB) per user: let the
# capture finish if it possibly can.
#
# v131: also probes per-block capture mode (--per-block) and accepts
# PT_HPUGRAPH_DISABLE_TENSOR_CACHE via the environment (pass through env).
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KILL_GB="${WATCHDOG_KILL_GB:-110}"
KILL_KB=$((KILL_GB * 1048576))
LOG="$HERE/logs/graphs_r768_watchdog.log"
PER_BLOCK=0
[[ "${1:-}" == "--per-block" ]] && PER_BLOCK=1

echo "[wd] starting r768 graphs-ON solo run (kill at ${KILL_GB} GB RSS, per_block=$PER_BLOCK)" | tee -a "$LOG"

H3_ALLOW_HPU=1 H3_FUSED_DECODE=1 \
H3_DEVICE_MAP="te=stream,dit=hpu,vae=hpu,audio=hpu" \
H3_PER_BLOCK_GRAPHS=$PER_BLOCK \
    "$HERE/launch_h3.sh" \
    --prompt "a paper boat drifting down a rain-swollen gutter, macro lens" \
    --resolution 1344x768 --duration 8 --steps 8 --seed 777 \
    --cp 1 --te-threads 16 \
    > "$HERE/logs/graphs_r768_solo.log" 2>&1 &
RUN_PID=$!
echo "[wd] runner pid $RUN_PID" | tee -a "$LOG"

# Track the actual python worker (launch_h3.sh execs run_t2va.py in-place).
while kill -0 "$RUN_PID" 2>/dev/null; do
    # Highest-RSS descendant = the worker (parent holds ~nothing).
    PID=$(ps -o pid=,rss=,ppid= --ppid "$RUN_PID" -Ao pid=,rss=,ppid= 2>/dev/null \
        | awk -v me="$RUN_PID" '$3==me || $1==me {if ($2>m){m=$2;p=$1}} END{print p+0, m+0}' | awk '{print $1}')
    RSS_KB=$(ps -o rss= -p "$PID" 2>/dev/null | tr -d ' ')
    TS=$(date +%H:%M:%S)
    if [[ -n "$RSS_KB" ]]; then
        echo "[wd] $TS pid=$PID rss=$((RSS_KB / 1048576)) GB" | tee -a "$LOG"
        if (( RSS_KB > KILL_KB )); then
            echo "[wd] $TS RSS $((RSS_KB / 1048576)) GB > ${KILL_GB} GB -> killing worker $PID" | tee -a "$LOG"
            kill -9 "$PID" "$RUN_PID" 2>/dev/null
            sleep 2
            echo "[wd] watchdog fired; box preserved. See logs/graphs_r768_solo.log tail for capture progress." | tee -a "$LOG"
            exit 3
        fi
    fi
    sleep 2
done

wait "$RUN_PID"
RC=$?
echo "[wd] runner exited rc=$RC" | tee -a "$LOG"
grep -E "timings:|synStatus|FAILED|Error" "$HERE/logs/graphs_r768_solo.log" | tail -10 | tee -a "$LOG"
exit $RC
