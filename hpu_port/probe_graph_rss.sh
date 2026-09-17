#!/usr/bin/env bash
# probe_graph_rss.sh — measure graph-capture peak host RSS for one bucket.
# Usage: probe_graph_rss.sh <resolution WxH> <seconds> <label>
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RES="$1"; DUR="$2"; LABEL="$3"
export H3_RECIPES_DIR="$(mktemp -d /tmp/h3probe_recipes.XXXXXX)"
echo "$LABEL: fresh recipe cache $H3_RECIPES_DIR"
KILL_KB=$((110 * 1048576))
SOLO="$HERE/logs/probe_graph_rss_${LABEL}.log"
PEAK=0

H3_ALLOW_HPU=1 H3_FUSED_DECODE=1 \
H3_DEVICE_MAP="te=stream,dit=hpu,vae=hpu,audio=hpu" \
    "$HERE/launch_h3.sh" \
    --prompt "probe capture" --resolution "$RES" --duration "$DUR" \
    --steps 2 --seed 9 --cp 1 --te-threads 16 \
    > "$SOLO" 2>&1 &
RUN_PID=$!

while kill -0 "$RUN_PID" 2>/dev/null; do
    PID=$(pgrep -P "$RUN_PID" 2>/dev/null | head -1); PID=${PID:-$RUN_PID}
    RSS=$(ps -o rss= -p "$PID" 2>/dev/null | tr -d ' ')
    [[ -n "$RSS" && "$RSS" -gt "$PEAK" ]] && PEAK=$RSS
    if [[ -n "$RSS" && "$RSS" -gt "$KILL_KB" ]]; then
        kill -9 "$PID" "$RUN_PID" 2>/dev/null
        echo "$LABEL PEAK_RSS_GB=$((PEAK / 1048576)) RESULT=watchdog_killed"
        exit 3
    fi
    sleep 2
done
wait "$RUN_PID"; RC=$?
echo "$LABEL PEAK_RSS_GB=$((PEAK / 1048576)) RESULT=$RC"
grep -E "timings:|synStatus|FAILED" "$SOLO" | tail -3
