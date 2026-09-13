#!/usr/bin/env bash
# launch_h3.sh — launcher for the MiniMax-H3 HPU-port runner (run_t2va.py).
#
# Responsibilities (design 4.6):
#   * pin the interpreter env: /root/venv-gaudi2/bin/python needs the uv-cpython
#     lib dir on LD_LIBRARY_PATH or `import torch` fails on libpython3.12.so.1.0;
#     the Habana torch plugin autoloads torch-wide on import (harmless core:
#     it never init's a device) but is skipped entirely on non-approved runs;
#   * default to a pure-CPU session: the 8x HL-225 cards are production
#     (GLM-5.3-Flash vLLM server) and every HPU-touching run requires the
#     explicit card-window approval H3_ALLOW_HPU=1 in the environment;
#   * exec the runner with thread defaults suited to the CPU-pinned text
#     encoder (Qwen3-VL-32B, 65 GB bf16 resident on host).
#
# Usage:
#   ./launch_h3.sh --prompt "..."                        # CPU mode (default)
#   H3_ALLOW_HPU=1 ./launch_h3.sh --prompt "..." --bucket 480p-5s
#   ./launch_h3.sh --report-only --bucket 768p-16x9-5s   # plan without torch
#   H3_ALLOW_HPU=1 ./launch_h3.sh --prompt "..." --cp 2  # context parallel
#     (PLAN_CP.md: spawns N worker processes, one card each, HCCL PG; per-rank
#      recipe caches land in .recipe_cache_rankN; HPU graphs default OFF under
#      CP until collectives-in-graph is validated -- H3_CP_GRAPHS=1 forces)
#   H3_DECODE_SUBPROCESS=1 ./launch_h3.sh ...            # decode in a fresh
#     eager-mode subprocess (D2H launch-thread wedge escape hatch; the parent
#     hands latents to decode_subproc.py which writes frames + wav directly)
#
# All run_t2va.py flags pass through unchanged.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${H3_PYTHON:-$HERE/../.venv/bin/python}"
UVCPYHOME_LIB="/root/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/lib"

if [[ ! -x "$PYTHON" ]]; then
    echo "launch_h3.sh: interpreter not executable: $PYTHON" >&2
    exit 2
fi

# Standard Habana runtime env (mirrors /etc/profile.d/habanalabs.sh). H3 runs
# launched from non-login shells (rmux/agent) do NOT get profile.d, and a
# wrong or missing HABANA_PLUGINS_LIB_PATH (e.g. /usr/lib/habanatools/habana_plugins,
# which ships NO libGraphCompilerPlugin.so) makes every memcpy-node recipe —
# H2D and D2H — fail GC compile (memcpy_engine_manager selectEngine /
# REPLACE_FAILED_INVALID_NEW_NODES or generic recipe_manager "Can not compile
# graph") while pure-compute recipes still compile. That exact split cost the
# v14-v21 bring-up runs. Only set what's missing so an intentionally
# overridden env still wins.
export HABANA_PLUGINS_LIB_PATH="${HABANA_PLUGINS_LIB_PATH:-/opt/habanalabs/habana_plugins}"
if [[ ! -e "$HABANA_PLUGINS_LIB_PATH/libGraphCompilerPlugin.so" ]]; then
    echo "launch_h3.sh: GC plugin missing in $HABANA_PLUGINS_LIB_PATH; using /opt/habanalabs/habana_plugins" >&2
    export HABANA_PLUGINS_LIB_PATH=/opt/habanalabs/habana_plugins
fi
export HABANA_SCAL_BIN_PATH="${HABANA_SCAL_BIN_PATH:-/opt/habanalabs/engines_fw}"
export HABANA_LOGS="${HABANA_LOGS:-/var/log/habana_logs/}"

# GC_KERNEL_PATH hygiene (both plugin modes): inherited shells on this box
# export "/root/tpc_dev/glm52_engine/build/libglm52_kernels.so:..." with the
# GLM-5.2 engine TPC lib FIRST (first-in-path wins), which poisons stock ops;
# and an EMPTY GC_KERNEL_PATH (an earlier `unset` here) makes the GC fail
# recipe compilation for most nontrivial graphs in a dozen disguises (memcpy
# engine replace, range_f32 node create, allocateTensors "Failed to allocate
# DRAM", generic synStatus 26) — the v14-v49 failure saga. The H3 pipeline
# uses only stock SynapseAI kernels, so pin the stock lib unconditionally.
export GC_KERNEL_PATH=/usr/lib/habanalabs/libtpc_kernels.so

# D2H copies must NOT go through the plugin's async launch thread: the final
# post-decode D2H (616 MB video tensor, postprocess_video -> frame.cpu())
# deadlocks there — main thread waits forever in
# habana_lazy::copy_hpu_lazy_D2H -> JoinPendingLaunchThread while the launch
# thread idles (v50, and the same silent stall that ended the previous
# session at the decode phase). Small D2Hs worked, big ones hung, with no
# error in any log. Synchronous D2H avoids the launch-thread handoff.
export PT_HPU_ENABLE_D2H_ASYNC_THREAD=0

export LD_LIBRARY_PATH="${UVCPYHOME_LIB}:${LD_LIBRARY_PATH:-}"

# Default is the pure-CPU session: without the card-window approval the Habana
# torch plugin is not even autoloaded (TORCH_DEVICE_BACKEND_AUTOLOAD=0 keeps
# `import torch` device-backend-free). With H3_ALLOW_HPU=1 the plugin loads
# torch-wide (no device init, harmless prints) and the runner's own gate
# (--allow-hpu / H3_ALLOW_HPU) decides whether anything is placed on an HPU.
if [[ "${H3_ALLOW_HPU:-}" != "1" ]]; then
    export TORCH_DEVICE_BACKEND_AUTOLOAD=0
    notice="launch_h3.sh: no H3_ALLOW_HPU=1 -> pure-CPU session (backend autoload disabled)"
    echo "$notice" >&2
else
    # Lazy-mode requirement for HPU graphs (diffusers' denoiser loop and
    # wrap_in_hpu_graph both raise "available in lazy mode only" otherwise).
    # PAIR rule (verified empirically on this stack):
    #   * PT_HPU_LAZY_MODE=1 alone DISABLES the device-backend autoload
    #     (habana autoload module: "if PT_HPU_AUTOLOAD isn't set, disable
    #     autoload for lazy mode") -> torch never gains the hpu attr.
    #   * PT_HPU_AUTOLOAD=1 must be set alongside it so the LAZY plugin lib
    #     loads at import time. Setting LAZY only AFTER import aborts device
    #     init: "Wrong PT plugin library loaded. Expected LAZY, got EAGER."
    # Mode selection: lazy (default) enables HPU graphs; eager replicates the
    # v12 bring-up recipe (DearAPI graphs only, no fused lazy graphs) as a
    # fallback when a lazy-mode GC compile bug bites. H3_HPU_MODE=eager opts out.
    if [ "${H3_HPU_MODE:-lazy}" = "eager" ]; then
        echo "launch_h3.sh: H3_HPU_MODE=eager -> eager plugin, no lazy-mode exports" >&2
    else
        export PT_HPU_LAZY_MODE=1
        export PT_HPU_AUTOLOAD=1
    fi

    # Persistent on-disk compilation cache: compiled recipes (DiT per bucket,
    # VAE decoders) are written once and replayed across launches instead of
    # paying the multi-minute graph-compile tax every process start.
    #   param1 = cache dir (empty = disk cache DISABLED — the silent default)
    #   param2 = false: CLEAR-ON-INIT must be false or the cache wipes itself
    #            every launch, which silently restores the compile tax
    #   param3 = 8192 MiB retention budget (this model's recipes exceed 1 GiB)
    #   param4 = false: local disk, not NFS
    export H3_RECIPES_DIR="${H3_RECIPES_DIR:-$HERE/.recipe_cache}"
    mkdir -p "$H3_RECIPES_DIR"
    export PT_HPU_RECIPE_CACHE_CONFIG="$H3_RECIPES_DIR,false,8192,false"
fi

# CPU text-encoder thread budget (65 GB bf16 Qwen3-VL-32B runs on host cores):
# default 8, overridable, and never overrides an explicit --te-threads.
TE_THREADS="${H3_TE_THREADS:-8}"
have_te=0
for arg in "$@"; do
    if [[ "$arg" == "--te-threads" || "$arg" == --te-threads=* ]]; then
        have_te=1
        break
    fi
done
if [[ "$have_te" == 0 ]]; then
    exec "$PYTHON" "$HERE/run_t2va.py" --te-threads "$TE_THREADS" "$@"
else
    exec "$PYTHON" "$HERE/run_t2va.py" "$@"
fi
