#!/usr/bin/env bash
# bench_r768.sh — benchmark: 8 s video @ 1344x768 (r768), random seed, CP=1.
#
# Why --no-graphs: under CP=1 with HPU graphs, the r768/192-frame shape
# (packed seq 60144) OOM-killed the server during graph capture (v121: host
# RSS 86.6 GB -> kernel SIGKILL). Graphs OFF uses per-block mark_steps which
# bound the lazy IR.
#
# Usage:
#   ./bench_r768.sh            # cold (compiles eager recipes) then warm run
#   ./bench_r768.sh --warm-only
#   ./bench_r768.sh --steps 20 --prompt "..."
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STEPS="${H3_BENCH_STEPS:-20}"
PROMPT="${H3_BENCH_PROMPT:-A red fox trotting through a snowy pine forest at dawn, volumetric light, shallow depth of field}"
WARM_ONLY=0
[[ "${1:-}" == "--warm-only" ]] && WARM_ONLY=1

run_one() {
    local tag="$1"
    local seed="$2"
    local log="$HERE/logs/bench_r768_${tag}_s${seed}.log"
    echo "=== bench run: tag=$tag seed=$seed steps=$STEPS log=$log ==="
    local t0 t1
    t0=$(date +%s)
    H3_ALLOW_HPU=1 H3_FUSED_DECODE=1 \
    H3_DEVICE_MAP="te=stream,dit=hpu,vae=hpu,audio=hpu" \
        "$HERE/launch_h3.sh" \
        --prompt "$PROMPT" \
        --resolution 1344x768 \
        --duration 8 \
        --steps "$STEPS" \
        --seed "$seed" \
        --no-graphs \
        --cp 1 \
        --te-threads 32 \
        2>&1 | tee "$log"
    local rc=${PIPESTATUS[0]}
    t1=$(date +%s)
    if [[ $rc -ne 0 ]]; then
        echo "bench run $tag FAILED rc=$rc (see $log)" >&2
        return $rc
    fi
    echo "--- results [$tag seed=$seed] wall $((t1 - t0)) s ---"
    grep -E "timings:|denoise complete|video decode complete|pipeline complete|CP run complete" "$log" || true
    grep -oE "out/t2va_h3_[a-f0-9]+\.mp4" "$log" | tail -1 || true
}

SEED_COLD=$(( (RANDOM << 15) | RANDOM ))
SEED_WARM=$(( (RANDOM << 15) | RANDOM ))

if [[ $WARM_ONLY -eq 0 ]]; then
    run_one cold "$SEED_COLD"
fi
run_one warm "$SEED_WARM"
echo "=== bench complete ==="
