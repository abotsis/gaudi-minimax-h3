#!/usr/bin/env bash
# bench_r480_graphs_eager.sh — SAME-SAME CP=1 comparison: HPU graphs vs eager.
#
# Both arms use identical config: r480 (864x480), 5 s (124 frames), steps 8,
# seed 42, same prompt, same device map (te=stream,dit=hpu,vae=hpu,audio=hpu),
# server mode so weights stay resident and request 2 measures pure warm replay.
# Each arm gets a FRESH recipe cache so "cold" is a true compile/capture.
#
#   graphs arm: default H3_DIT_GRAPHS=1
#   eager arm:  --no-graphs (per-block mark_steps bound the lazy IR)
#
# Usage: ./bench_r480_graphs_eager.sh [graphs|eager]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" /dev/stdout 2>/dev/null || pwd)"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARM="${1:-}"
PORT_GRAPHS=8031
PORT_EAGER=8032
PROMPT="A red fox trotting through a snowy pine forest at dawn, volumetric light, shallow depth of field"
SEED=42
STEPS=8

post_and_wait() {  # post_and_wait <port> <label>
    local port="$1" label="$2"
    local resp id
    resp=$(curl -s -X POST "localhost:${port}/sdcpp/v1/vid_gen" \
        -H 'Content-Type: application/json' \
        -d "{\"prompt\":\"${PROMPT}\",\"sample_params\":{\"sample_steps\":${STEPS}},\"seed\":${SEED}}")
    id=$(echo "$resp" | python3 -c 'import json,sys; print(json.load(sys.stdin)["id"])')
    echo "[$label] job $id submitted"
    local t0=$(date +%s)
    for _ in $(seq 1 720); do
        local st
        st=$(curl -s "localhost:${port}/sdcpp/v1/jobs/${id}" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("status","?"))')
        case "$st" in
            completed)
                local t1=$(date +%s)
                echo "[$label] COMPLETED in $((t1 - t0)) s"
                grep -E 'timings:' "$LOG" | tail -1
                return 0 ;;
            failed|stale)
                echo "[$label] FAILED status=$st"; tail -20 "$LOG"; return 1 ;;
        esac
        sleep 5
    done
    echo "[$label] TIMED OUT"; return 1
}

run_arm() {  # run_arm <graphs|eager>
    local mode="$1"
    local extra=() port
    if [[ "$mode" == "eager" ]]; then extra=(--no-graphs); port=$PORT_EAGER; else port=$PORT_GRAPHS; fi
    local cache; cache=$(mktemp -d /tmp/h3bench_${mode}_recipes.XXXXXX)
    export H3_RECIPES_DIR="$cache"
    LOG="$HERE/logs/bench_r480_${mode}_s${SEED}.log"
    echo "=== arm: $mode (fresh recipes: $cache, log: $LOG) ==="

    H3_ALLOW_HPU=1 H3_FUSED_DECODE=1 \
    H3_DEVICE_MAP="te=stream,dit=hpu,vae=hpu,audio=hpu" \
        "$HERE/launch_h3.sh" \
        --server --port "$port" \
        --resolution 864x480 --duration 5 \
        --steps "$STEPS" --seed "$SEED" \
        --cp 1 --te-threads 32 \
        "${extra[@]}" \
        > "$LOG" 2>&1 &
    local srv=$!
    trap "kill -9 $srv 2>/dev/null || true" EXIT

    # wait for HTTP liveness
    for _ in $(seq 1 240); do
        curl -s -m 2 "localhost:${port}/health" | grep -q '"ok": true' && break
        kill -0 "$srv" 2>/dev/null || { echo "[$mode] server died at boot"; tail -30 "$LOG"; return 1; }
        sleep 5
    done
    curl -s -m 2 "localhost:${port}/health" | grep -q '"ok": true' || { echo "[$mode] server never became healthy"; tail -30 "$LOG"; return 1; }
    echo "[$mode] server healthy"

    echo "[$mode] --- COLD request (compile/capture) ---"
    post_and_wait "$port" "$mode:cold" || return 1
    echo "[$mode] --- WARM request (replay) ---"
    post_and_wait "$port" "$mode:warm" || return 1

    kill -9 "$srv" 2>/dev/null || true
    wait "$srv" 2>/dev/null || true
    trap - EXIT
    echo "=== arm $mode done ==="
}

case "$ARM" in
    graphs) run_arm graphs ;;
    eager)  run_arm eager ;;
    "")
        run_arm graphs
        run_arm eager
        echo "=== BOTH ARMS COMPLETE — summary ==="
        grep -H 'timings:' "$HERE/logs/bench_r480_graphs_s${SEED}.log" "$HERE/logs/bench_r480_eager_s${SEED}.log"
        ;;
    *) echo "usage: $0 [graphs|eager]" >&2; exit 2 ;;
esac
