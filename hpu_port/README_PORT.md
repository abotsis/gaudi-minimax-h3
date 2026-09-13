# MiniMax-H3 HPU port (Intel Gaudi2)

Runtime port of the diffusers 0.40.0 **MiniMax-H3** modular pipeline
(`MiniMaxH3ModularPipeline`, transformer `MiniMaxH3Transformer3DModel`, video VAE
`AutoencoderKLMiniMaxH3`, audio VAE `AutoencoderKLMiniMaxH3Audio`, scheduler
`MiniMaxH3Scheduler`) to Intel Gaudi2 (HL-225, SynapseAI 1.24.1) via a
monkeypatch layer. No file inside site-packages (or this checkpoint tree) is
edited; every change hangs off class attributes / module namespaces at patch
time and is idempotent.

Files:

| File | Role |
| --- | --- |
| `hpu_patches.py` | The patch module: attention backend (incl. context-parallel routing), rope cache, autocast guard, HPU-graph helpers, mark_step strategy, entry points, CPU parity self-test. |
| `run_t2va.py` | Runner (`t2va` / `fl2va` / `ref2va` workflows, bucket tables, device map, CP parent/worker orchestration). Attention backend is imported from `hpu_patches.py` (single source of truth); the rope cache copy remains an inline fallback — see "Known overlap" below. |
| `decode_subproc.py` | Decode-subprocess escape hatch (`H3_DECODE_SUBPROCESS=1`): fresh eager-mode process loads the VAEs from the workdir, decodes video + audio, writes frames + wav + meta json. |
| `tests/cp_parity_cpu.py` | CPU-only CP tests: 2-rank gloo parity (backend routing + the model's `_cp_plan` through the real `enable_parallelism`), parent/worker spawn plumbing smoke, decode-subprocess CPU eager test. See PLAN_CP.md phase 1.5. |

## Device-guard policy

Every HPU branch keys off **the tensor's own device** (`x.device.type ==
"hpu"`), plus a one-time availability probe (`_probe_hpu()`). Importing and
exercising this module on a CPU-only host never imports
`habana_frameworks.torch`, never initializes an HPU device and never compiles
a graph. `H3_DISABLE_HPU=1` additionally reverts the whole module to the pure
CPU code path even on a card host (fallback / A-B testing). `apply_patches()`
is safe on CPU (only device-independent rebinds: autocast-guard subclass and
the backend registration).

On the production box the 8 Gaudi2 cards are fully occupied by the
GLM-5.3-Flash vLLM server. **No HPU battery runs without first clearing the
card-window protocol (`H3_ALLOW_HPU=1` / `--allow-hpu`).**

## What is patched, and why (design 3.2-3.5 mapping)

1. **Attention backend (3.2)** — `habana_fused_sdpa`, Habana's
   `hpex.kernels.FusedSDPA` autograd kernel, registered into diffusers'
   attention registry and forced on every H3 processor class
   (`MiniMaxH3AttnProcessor`, `MiniMaxH3VideoAttnProcessor`). diffusers'
   `AttentionBackendName` enum is closed, so the install also short-circuits
   the per-module `dispatch_attention_fn` bindings for the two consuming
   modules (transformer + video VAE). On non-HPU tensors the backend is a
   byte-exact mirror of the stock `native` SDPA sequence, so CPU smoke tests
   assert the two paths agree **bit-exactly** (`validate_cpu_equivalence()`).
   Sequences above `H3_SDPA_TILE_THRESHOLD` q-tile the query with
   `mark_step()` every `H3_SDPA_MARK_EVERY` tiles (the
   `VLLM_HPU_FSDPA_Q_TILE_ENABLE` pattern from vLLM-Gaudi's qwen2_5_vl).
   `enable_gqa=True` fails loudly (H3 never uses GQA).
   Softmax mode: `"fast"` for bf16 q (with `FLASH_ATTENTION_FAST_SOFTMAX=1`),
   otherwise the conservative default (`"None"`) — FusedSDPA's explicit fast
   mode is bf16-input-only and the video-VAE ViT runs fp32 q/k/v.

2. **Rope cos/sin cache (3.3)** — `MiniMaxH3RotaryPosEmbed.forward` is
   class-patched to serve cached fp32 cos/sin per **bucket** key, hoisted out
   of the per-step HPU graph. The cache fills lazily from the stock
   computation (so it is bit-exact vs the uncached path) and lives on the
   fill-up device. The fp64 grid produced by `build_packed_sequence` stays on
   the CPU host side; only the fp32 cos/sin cross to the device (deviation
   3.7 (ii) — Gaudi2 has no useful fp64 compute; the stock forward's
   `position_ids.to(torch.float32)` keeps the reduction fp32).

3. **Video-VAE decode autocast guard (3.5 #1)** — stock
   `MiniMaxH3VideoDecodeStep.__call__` gates the verified recipe (fp16
   autocast over fp32 weights) on `device.type == "cuda"`, so on HPU the VAE
   would decode in full precision. A subclass with the identical body except
   `enabled=device.type in ("cuda", "hpu")` replaces step 0 of
   `MiniMaxH3DecodeStep.block_classes`. A drift check against an audited
   diffusers revision fails loudly at patch time if upstream changes the
   guard.

4. **HPU graphs (3.4)** — `wrap_module_in_hpu_graph()` applies SynapseAI
   `wrap_in_hpu_graph` when the module has HPU-resident parameters, identity
   otherwise. The DiT is wrapped per bucket (shapes are static per request
   bucket, so one graph per bucket replays every diffusion step); VAE
   **decoders** are wrapped at the inner `*.decoder` submodule
   (opt-in `H3_VAE_GRAPHS=1`): `vae.decode(z)` never calls `vae.forward`, so
   wrapping the whole VAE would capture nothing, and the tiled/chunked decode
   counts are data-dependent — default OFF, iterate eagerly first.

5. **mark_step strategy (3.5 #4)** — eager mode installs an
   `nn.Module.register_forward_hook` that closes the lazy frontier with
   `htcore.mark_step()` after every forward of an HPU-resident DiT/VAE
   component. Under graphs the hook is redundant (the graph wrap owns the
   dispatch boundary) and is skipped. Inside the q-tile fallback, mark_step
   fires every `H3_SDPA_MARK_EVERY` tiles so eager tile loops interleave.

6. **AdaLN (3.5 #5) — deliberately not patched in v1.** The 50 per-block
   `adaln_proj` projections and their row `index_select`s execute inside the
   per-step graph with fixed op ordering; the optional batched-GEMM hoist is
   reserved behind `H3_BATCH_ADALN=1` and is a no-op until first-card
   profiling shows the 50 GEMVs as the decode bottleneck.

## Env flags

Patch module (`hpu_patches.py`):

| Flag | Default | Effect |
| --- | --- | --- |
| `H3_DISABLE_HPU` | `0` | Force the pure-CPU path even on a card host. |
| `H3_DIT_GRAPHS` | `1` | Wrap the DiT in an HPU graph (per bucket). |
| `H3_VAE_GRAPHS` | `0` | Also graph the VAE inner decoders (opt-in; chunked decode). |
| `H3_ROPE_CACHE` | `1` | Per-bucket rope cos/sin cache. |
| `H3_FAST_SOFTMAX` | `1` | bf16 q selects FusedSDPA `softmax_mode="fast"`. |
| `H3_SDPA_TILE_THRESHOLD` | `65536` | q rows above which attention q-tiles. |
| `H3_SDPA_TILE_SIZE` | `64` | q-tile row count. |
| `H3_SDPA_MARK_EVERY` | `75` | mark_step cadence inside the tile loop. |
| `H3_BATCH_ADALN` | `0` | Reserved; no-op in v1 (see #6 above). |
| `H3_DECODE_SUBPROCESS` | `0` | Run the VAE decode in a fresh eager-mode subprocess (`decode_subproc.py`) instead of in-process — escape hatch for the D2H launch-thread wedge (v50/51/52/58). Parent saves latents to a .pt post-sync/post-DiT-free; the subprocess writes frames + wav + meta json directly. |

Runner (`run_t2va.py`; more detail in its own docstring):

| Flag | Default | Effect |
| --- | --- | --- |
| `H3_ALLOW_HPU` / `--allow-hpu` | missing | Card-window approval gate; without it HPU collapses to CPU. |
| `H3_DEVICE_MAP` / `--device-map` | `te=cpu,dit=hpu,vae=hpu,audio=hpu` | Per-component placement (`te`,`dit`,`vae`,`audio` = `cpu\|hpu`). |
| `H3_FSDPA_TILE_SEQ` | `0` | Runner-side override for the tile threshold. |
| `H3_FORCE_ATTN_BACKEND` | `0` | Force the Habana attention backend even on a non-approved host (smoke). |
| `H3_AUDIO_VAE_DEVICE` | `hpu` | Kill-switch for the audio VAE (set `cpu` if it misbehaves on card). |

## Context parallelism (`--cp N`, PLAN_CP.md)

Sequence parallelism over the packed ~15.9k-row sequence via diffusers'
experimental CP machinery and the model's own `_cp_plan`: weights stay
replicated (62 GiB/card), activations split along the sequence, one Ulysses
all-to-all per attention. `--cp 1` (default) is the single-process path,
unchanged.

* `--cp N` (N in 1/2/4/8): the parent computes the plan, pads the text budget
  so the packed seq divides by N (printed as `cp_seq_pad`), spawns N workers
  with the **spawn** context (fork is unsafe with Habana threads), and merges
  the per-rank timing JSONs into a combined table at the end. Each worker gets
  `HABANA_VISIBLE_DEVICES=<rank>`, RANK/WORLD_SIZE/MASTER_ADDR/MASTER_PORT.
* Process group: `hccl` under an approved HPU run (1 proc : 1 HPU), `gloo`
  fallback when HPU is unavailable **and** `H3_CP_TEST=1` (or
  `H3_ALLOW_CPU_FALLBACK=1`) — the whole CP code path stays exercisable on CPU.
* Under CP the device map pins dit/vae/audio onto the SAME per-rank device
  set (replicated); the text encoder stays on the host. HPU graphs default
  OFF under CP (`H3_CP_GRAPHS=1` forces) until collectives-in-graph-capture
  are validated; first bring-up is eager lazy mode with per-block mark_step.
* Noise determinism: every rank draws the identical full CPU noise from the
  seeded generator (`patch_prepare_latents_cpu`); a digest all_gather asserts
  identity across ranks; the `_cp_plan` shards inside diffusers — nothing
  hand-slices latents.
* Rank-aware logging: every `[h3:*]` line from a worker carries `[h3:rN]`;
  the parent prints the merged phase/block timing table.
* Recipe cache is per rank: with `H3_RECIPES_DIR` set, workers use
  `<dir>_rank{r}` (e.g. `.recipe_cache_rank0`), because
  `PT_HPU_RECIPE_CACHE_CONFIG` is a per-process setting.
* Known upstream landmine (handled): `enable_parallelism` stamps
  `_parallel_config` on every attention processor including the token
  refiner's, but the refiner runs pre-packing on the FULL text run — the
  runner un-stamps exactly those processors (`_cp_enable_parallelism`).

Usage (card window required for the HPU path):

```sh
H3_ALLOW_HPU=1 ./launch_h3.sh --prompt "..." --bucket 480p-5s --cp 2
```

CPU-only verification (no cards touched):

```sh
LD_LIBRARY_PATH=/root/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/lib \
TORCH_DEVICE_BACKEND_AUTOLOAD=0 .venv/bin/python hpu_port/tests/cp_parity_cpu.py
# -> PASS 20 / 20 (gloo parity, spawn plumbing, decode hatch)
```

## Usage

```python
import torch, sys
sys.path.insert(0, "/root/src/h3/hpu_port")
import hpu_patches as hp

hpu_patches = hp.apply_patches()   # device-independent rebinds first
from diffusers import ModularPipeline
from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3ModularPipeline  # noqa: F401

pipe = ModularPipeline.from_pretrained("<model_root>", dtype=torch.bfloat16)
pipe.to("hpu")                                      # or H3_DEVICE_MAP-managed in the runner
report = hp.patch_pipeline(pipe)                    # backend force + rope cache + graphs
# report == {"attention_backend": "habana_fused_sdpa", "graphs": {...}, ...}
```

In production use the runner, which wires all of this plus the bucket tables and
the approval gate:

```sh
# user-approved card window only
H3_ALLOW_HPU=1 /root/venv-gaudi2/bin/python /root/src/h3/hpu_port/run_t2va.py \
    --device-map te=cpu,dit=hpu,vae=hpu,audio=hpu --prompt "..." --frames 124
```

Before any card run, this CPU smoke must be green:

```sh
cd /root/src/h3/hpu_port
LD_LIBRARY_PATH=/root/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/lib:$LD_LIBRARY_PATH \
  TORCH_DEVICE_BACKEND_AUTOLOAD=0 H3_DISABLE_HPU=1 \
  /root/venv-gaudi2/bin/python -c "
import sys; sys.path.insert(0,'.')
import hpu_patches; print(hpu_patches.validate_cpu_equivalence())"
# -> {'parity_ok': True, 'max_delta_video': 0.0, 'max_delta_audio': 0.0, ...}
```

## First-card pass order (when a window is granted)

1. One card, `H3_DISABLE_HPU=0`, CPU path on (`...--device-map te=cpu,dit=cpu,vae=cpu,audio=cpu`) — import, backend registration, no device init.
2. Single card, DiT graphs OFF (`H3_DIT_GRAPHS=0`), eager: confirm the mixed-precision fp32 modules (`proj_in`, `audio_proj_in`, `time_embedder`, `proj_out`, `audio_proj_out`, `rope`) load and the fp64->fp32 rope handoff compiles.
3. Run `validate_cpu_equivalence()`-style parity **on card**: same seed/weights, CPU vs HPU outputs within tolerance (bf16 tolerance, not bit-exact).
4. Turn on `H3_DIT_GRAPHS=1` (bucket capture; first 2-3 iterations of a bucket are compile/warmup that launches on the card).
5. Only then `H3_VAE_GRAPHS=1`, `H3_FAST_SOFTMAX` and tile-threshold tuning; audio VAE stays strictly fp32.
6. Clear with the card coordinator (gaudirpc window protocol) before and after; check `H3_ALLOW_HPU` gates.

## Pre-GPU caveats (verify on the first card window)

- These are CPU-validated only. Nothing here has executed on an HPU device yet,
  by the port's hard rules; items below are the failure modes to expect first.
- **Mixed-precision load**: load with `torch_dtype` in `from_pretrained` /
  `load_components(dtype=...)` (per the diffusers warning) rather than
  `.to(torch.bfloat16)` after the fact, so `_keep_in_fp32_modules` entries
  (`proj_in`, `audio_proj_in`, `time_embedder`, `proj_out`, `audio_proj_out`,
  `rope`) stay float32.
- **fp64 tensors on-device**: the runner holds `position_ids` on the host; if
  any non-patched block still moves an fp64 tensor to HPU, Gaudi2 will not
  compute it — expect "unsupported dtype", not silent wrongness.
- **`index_copy(1, ...)`**: upstream assembles the packed buffer with dim-1
  index_copy. If the PT bridge or graph capture rejects it, the fallback shape
  is a leading-dim scatter (mathematically identical) — patch point noted in
  the module docstring (scattered through a transposed buffer).
- **`nn.RMSNorm`** qualifies for Habana's fast RMSNorm only in supported
  layouts; otherwise it lowers through the generic path (correct, slower).
- **FusedSDPA**: `softmax_mode="fast"` is bf16-input-only per Habana docs; the fp32 video-VAE ViT calls go through mode `"None"` automatically.
  `attn_mask=None` is the only form H3 uses (one packed document, no mask).
- **HPU graph first-run**: 2-3 iterations per bucket are compile/warmup;
  time the bucket at 10+ steps, not the first three.
- **Audio VAE** remains fp32 end-to-end (documented -20 dB bf16 hazard); only
  its graphs flag is optional, never its dtype.
- **Memory**: with a 96 GB card and the standard 768x1344 canvas, video+
  audio latents plus DiT bf16 weights fit single-card; keep `H3_DEVICE_MAP` at
  one card per phase until the twin-tiled decode memory is measured.

## Known overlap

`run_t2va.py` still duplicates the rope cache because it was developed
self-contained. Its formerly duplicated FusedSDPA attention backend was an
audit blocker: the inline copy picked softmax_mode="fp32" for fp32 q/k/v,
which Habana asserts is BF16-input-only, so the fp32 video-VAE ViT decode
would raise AssertionError on card. The runner now always defers to
`hpu_patches.habana_fused_sdpa` (correct `_softmax_mode`: "fast" for bf16
under `H3_FAST_SOFTMAX=1`, "None" for fp32) and name-drift between the two
modules fails loudly at registration. If both files are edited later, keep
`hpu_patches.py` the single source of truth and import it from the runner.

## Validation status (this session, CPU only)

- `py_compile` clean on all files.
- `validate_cpu_equivalence()`: `parity_ok=True`, deltas `0.0` (bit-exact CPU
  parity between the stock native dispatch and the Habana backend passthrough).
- Rope cache: fill / hit / bucket-switch / `None`-fallback all bit-identical
  to the stock forward.
- `patch_video_vae_autocast_guard()`: drift check passes against diffusers
  0.40.0; `MiniMaxH3DecodeStep.block_classes[0]` rebind verified.
- `apply_patches()` / `apply_hpu_patches()` no-op correctly on CPU
  (`hpu_available=False`, graphs + mark_step hooks skipped).
