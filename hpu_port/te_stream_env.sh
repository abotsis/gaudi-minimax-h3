#!/bin/bash
# te_stream_env.sh — env preamble for single-card HPU probes (no graphs).
# Derived from launch_h3.sh's verified exports (see its comments for the
# v14-v49 GC_KERNEL_PATH saga and the v50 D2H async-thread wedge).
# Usage: source this, then exec python <script>
export HABANA_PLUGINS_LIB_PATH=/opt/habanalabs/habana_plugins
if [[ ! -e "$HABANA_PLUGINS_LIB_PATH/libGraphCompilerPlugin.so" ]]; then
    export HABANA_PLUGINS_LIB_PATH=/opt/habanalabs/habana_plugins
fi
export HABANA_SCAL_BIN_PATH=/opt/habanalabs/engines_fw
export HABANA_LOGS=/var/log/habana_logs/
export GC_KERNEL_PATH=/usr/lib/habanalabs/libtpc_kernels.so
# v50 wedge: D2H via the async launch thread deadlocks on big copies.
export PT_HPU_ENABLE_D2H_ASYNC_THREAD=0
# PAIR rule: LAZY_MODE=1 alone disables backend autoload; PT_HPU_AUTOLOAD=1
# must accompany it so the LAZY plugin lib loads at import time.
export PT_HPU_LAZY_MODE=1
export PT_HPU_AUTOLOAD=1
export LD_LIBRARY_PATH="/root/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/lib:${LD_LIBRARY_PATH:-}"
