# MiniMax-H3 on Intel Gaudi2 (HPU port)

## Introduction

This project runs the diffusers 0.40.0 **MiniMax-H3** text-to-video and audio
pipeline on **Intel Gaudi2** cards (HL-225, SynapseAI 1.24.1). The pipeline is
the modular diffusers pipeline: `MiniMaxH3ModularPipeline`, transformer
`MiniMaxH3Transformer3DModel`, video VAE `AutoencoderKLMiniMaxH3`, audio VAE
`AutoencoderKLMiniMaxH3Audio`, and `MiniMaxH3Scheduler`.

The port is a monkeypatch layer. It does not change files in site-packages or
in the checkpoint tree. It attaches changes to class attributes and module
namespaces at patch time, and the changes are idempotent. On a host without
cards, the module imports without the Habana framework and the pipeline runs
the stock CPU path. CPU tests check that the patched attention backend agrees
with the stock SDPA path bit-for-bit.

The project gives you these parts:

- **Single-shot CLI** (`run_t2va.py`): t2va and fl2va workflows. It snaps
  resolution and duration to canonical buckets. It muxes frames and audio to
  an mp4 file with ffmpeg.
- **HTTP server** (`--server` with `h3_server.py`): a stable-diffusion.cpp
  compatible API. Weights stay on the card between requests. Requests flow to
  the worker pool through a file spool.
- **Context parallelism** (`--cp 1/2/4/8`): sequence parallelism over the
  packed sequence. Weights are replicated on each card (62 GiB), activations
  are split along the sequence, and each attention does one all-to-all.
- **HPU graphs** with static shapes per bucket, a streamed text encoder
  (`te=stream`), a fused decode configuration, and a decode subprocess escape
  hatch.

For patch internals and all environment flags, read `hpu_port/README_PORT.md`.
For the parallelism design and test log, read `hpu_port/PLAN_CP.md`.

## Features

**Generation**

- Text-to-video with synchronized audio (t2va workflow).
- First and last image anchors (fl2va workflow): drive a clip from a start
  image to an end image. Anchor geometry fit: `stretch` (stock), `crop`
  (cover-crop), or `pad` (letterbox).
- Per-request prompt, steps, seed, resolution, and duration. The runner snaps
  resolution to 32-aligned canvases and duration to the 17n+5 frame grid.
- Canonical buckets (preview, 480p-5s/10s/15s, 576p, 768p) in
  `hpu_port/buckets.json`.
- Output: mp4 (H.264 + AAC through ffmpeg), 24 fps.

**Serving**

- stable-diffusion.cpp compatible HTTP API: submit, poll, cancel, and
  capabilities endpoints. Jobs carry the result as base64 mp4.
- Weight-resident serving: the DiT stays on the card between requests.
- Embed cache: the text-encoder prefill runs one time per prompt+shape and is
  cached on disk. Repeated prompts start instantly.

**Placement and parallelism**

- Device map per component (`te`, `dit`, `vae`, `audio` = `cpu|hpu|stream`).
- Context parallelism at CP=1/2/4/8 (diffusers CP machinery, HCCL process
  group, one card per rank). CPU-only parity tests exercise the same code
  path with gloo.
- Streamed text-encoder weights (`te=stream`): the 51-layer Qwen3-VL
  conditioner streams its decoder layers H2D from host memory, layer by
  layer, instead of residing on the card. Text prefill drops from 48–89 s on
  CPU to a steady 4.1–4.5 s (12–20x), is bit-exact vs the CPU path for text,
  and keeps the card free of the 65 GB encoder weights.

**Execution modes**

- HPU graphs with static shapes per bucket (one graph per resolution+length
  shape; step count is a loop over the same graph).
- Per-block graph capture (`H3_PER_BLOCK_GRAPHS=1`): one graph per
  transformer block. Bounds the capture host-memory cost.
- Eager lazy mode with per-block mark_step (the default recommendation).
- Fused in-process decode (`H3_FUSED_DECODE=1`): decode enqueued while the
  denoise queue drains, with on-device quantize and contiguous D2H pulls.
- Decode alternatives: temporal-sharded decode farm
  (`H3_DECODE_SHARDED=1`) and fresh eager subprocess decode
  (`H3_DECODE_SUBPROCESS=1`) as escape hatches for launch-thread wedges.
- Habana fused SDPA attention backend (bit-exact vs stock SDPA on CPU),
  with q-tiling above a sequence threshold.
- Per-bucket rope cos/sin cache hoisted out of the per-step graph.

**Diagnostics**

- Per-block timing logs (`H3_LOG_BLOCKS=1`), op tracing (`H3_TRACE_OPS=1`,
  `H3_TRACE_DIT=1`), and a CPU smoke/parity test suite under
  `hpu_port/tests/`.

**Not yet (see caveats)**: LoRA adapters, reference images (ref2va), and
HPU graphs under context parallelism.

## Performance

**Graphs vs no-graphs, same conditions.** CP=1, r480 (864×480), 124 frames,
8 steps, seed 42, same prompt and device map, and a fresh recipe cache for
each arm. Rerun the comparison with `hpu_port/bench_r480_graphs_eager.sh`.
The comparison uses r480 because graph capture at r768 does not fit in host
memory (see caveats).

| | Graphs | Eager (`--no-graphs`) |
| --- | --- | --- |
| Cold wall time (first request, fresh cache) | **746 s** | **313 s** |
| Warm generate (second request) | **68.2 s** | 71.9 s |
| Warm denoise (host loop + device drain) | ≈51 s (loop 0.2 s + drain 50.9 s) | ≈51 s (loop 29.3 s + drain 21.8 s) |
| Warm video-VAE decode (device time) | **16.8 s** | 20.5 s |

Reading the numbers (corrected: the per-step bars are not comparable — see
below):

- **The device compute is the same in both modes.** The graph bar measures
  async enqueue (the host hands the replay to the card in milliseconds), and
  the eager bar measures host time per step. Add the host loop time and the
  device drain at decode entry: warm denoise is about 51 s of card time in
  both modes (about 7.3 s/step).
- **Warm end-to-end: graphs win by about 4 s** (68.2 s vs 71.9 s generate).
  The saving is host-side launch overhead, not device compute. Text-encoder
  prefill, VAE decode, export, and mux dominate the wall and are identical
  in both modes.
- **Graph decode is slightly faster** (16.8 s vs 20.5 s device time). The
  reason is not checked (better recipe scheduling or graph overlap). This is
  a small effect.
- **The first request is much faster without graphs** (313 s vs 746 s).
  Capture costs about 430 s one time per bucket. The recipe cache stays on
  disk across server restarts. Thus the capture cost disappears for a
  long-lived server.
- **Conclusion (as implemented):** graphs do not make the model faster on
  the card at this shape. They only remove host launch overhead. Use eager
  unless a server serves many requests from a warm cache, where the small
  per-request saving adds up. **The research is not exhausted:** the eager
  r768 configuration measures 74.9 s/step against about 7.3 s/step at r480.
  Quadratic attention scaling predicts about 30 s/step. The gap points to
  eager host overhead that graphs would remove. But graph capture at r768
  does not fit in host memory (see caveats). A capture strategy with a
  smaller host-side IR (per-block capture, or capture of attention regions
  only) could turn this gap into the true graphs payoff.

## Requirements

- **Hardware:** 1 to 8 Intel Gaudi2 cards (HL-225). Context parallelism with
  N ranks needs N cards, with one process per card.
- **Software:** Python 3.12, SynapseAI 1.24.1 (see the pins in
  `hpu_port/requirements-pinned.txt`), diffusers 0.40.0, and ffmpeg on PATH.
- **Model:** the MiniMax-H3 checkpoint under `MiniMax-H3/` in the repository
  root (gitignored). The weights use about 62 GiB per card. The Qwen3-VL text
  encoder stays on the host CPU (65 GB in bf16) or streams to the card with
  `te=stream`. All tests and benchmarks use the released checkpoint
  `MiniMaxAI/MiniMax-H3` at the default revision. Other revisions are not
  checked.

## Getting started

To install and run the pipeline, do these steps:

1. Make a Python 3.12 environment. Then install the pinned packages:
   `pip install -r hpu_port/requirements-pinned.txt`
2. Put the MiniMax-H3 checkpoint in `./MiniMax-H3/`.
3. Run the CPU check. It does not touch the cards:
   `hpu_port/launch_h3.sh --report-only --bucket 768p-16x9-5s`
4. Generate one video:
   `H3_ALLOW_HPU=1 ./hpu_port/launch_h3.sh --prompt "..." --bucket 480p-5s`
5. Start the HTTP server with resident weights. Run in eager mode (`--no-graphs`):
   `H3_ALLOW_HPU=1 H3_DEVICE_MAP=te=stream,dit=hpu,vae=hpu,audio=hpu ./hpu_port/launch_h3.sh --server --no-graphs --bucket 480p-5s --port 8021`

Start with eager mode. Graph mode gives a small warm-path saving only (see
performance), and the first graph build is slow.

**About first-run graph builds.** HPU graph mode builds a graph for each
shape. The shape is the combination of resolution and clip length (the step
count is a loop count over the same graph and does not change it). A new
shape triggers a fresh capture during the first request at that shape.
The capture is slow (hundreds of seconds) and uses a large amount of host
memory. The built graphs are stored in the recipe cache and survive server
restarts. Thus the cost is one time per shape. Do not turn graphs on for a
driver with many different shapes. Turn graphs on for a server with a fixed
bucket and a warm cache.

NOTE: `H3_ALLOW_HPU=1` turns on the HPU path. Without the flag, the runner
uses the CPU path, even on a host with cards.

Key runner flags (full table in `hpu_port/README_PORT.md`):

| Flag | Default | Effect |
| --- | --- | --- |
| `--cp N` | 1 | Context parallelism (1/2/4/8), one card per rank. |
| `--bucket NAME` | — | Canonical canvas, duration, and steps (see `hpu_port/buckets.json`). |
| `--resolution WxH` / `--duration S` | — | Alternative to `--bucket`. Snaps to 32-aligned canvases and the 17n+5 frame grid. |
| `--server` / `--port` | off / 8021 | stable-diffusion.cpp compatible HTTP API. |
| `--device-map` | `te=cpu,dit=hpu,vae=hpu,audio=hpu` | Placement per component. `te=stream` turns on the streamed text encoder. |
| `H3_DIT_GRAPHS=1` | 1 | Wrap the DiT in HPU graphs, one per bucket. |
| `H3_CP_GRAPHS=1` | off | Force graphs under context parallelism (see caveats). |
| `H3_PER_BLOCK_GRAPHS=1` | off | Wrap each transformer block separately, not the whole DiT. |
| `H3_DECODE_SUBPROCESS=1` | off | Decode the VAE in a fresh eager subprocess. Escape hatch for the D2H launch-thread wedge. |

## Examples

**First and last image (fl2va) with anchor geometry fit:**

```sh
H3_ALLOW_HPU=1 ./hpu_port/launch_h3.sh --workflow fl2va \
    --init_image start.png --last_image end.png \
    --fit crop --prompt "..." --bucket 480p-5s
```

`--fit stretch|crop|pad` sets the anchor geometry. Stock fl2va stretches the
anchor onto the canvas. A 4:3 photo then comes out 31% horizontally
squeezed. `crop` cover-crops. `pad` letterboxes and keeps subjects
undistorted. With `pad`, frame 0 matches the padded anchor.

**HTTP API (server mode):**

```sh
curl -s localhost:8021/sdcpp/v1/capabilities
JOB=$(curl -s -X POST localhost:8021/sdcpp/v1/vid_gen \
    -H 'Content-Type: application/json' \
    -d '{"prompt":"a red fox in snow","sample_params":{"sample_steps":30},"seed":-1}' \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["id"])')

curl -s localhost:8021/sdcpp/v1/jobs/$JOB          # poll
curl -s -X POST localhost:8021/sdcpp/v1/jobs/$JOB/cancel   # cancel pre-claim
# completed jobs carry result.b64_json (mp4)
```

These request fields are per request: prompt, steps, seed. These are
server-wide and fixed at boot: duration, resolution, fps. A per-request
duration would change the DiT graph shape and cause a recompile. The server
also accepts `init_image` and `end_image` per request for fl2va, with `fit`
(`stretch|crop|pad`).

**Context parallelism (8 cards):**

```sh
H3_ALLOW_HPU=1 ./hpu_port/launch_h3.sh --prompt "..." --bucket 480p-5s --cp 8
```

The runner spawns N workers with the spawn context, one card per worker, and
an HCCL process group. Noise is deterministic across ranks: each rank draws
the identical full noise, and an all-gather digest checks identity. Each rank
uses its own recipe cache in `.recipe_cache_rankN`.

**Benchmark harnesses** (under `hpu_port/`): `bench_r768.sh` (r768 bench
without graphs), `graphs_r768_watchdog.sh` (graphs bench under an RSS
watchdog, so a runaway capture cannot stop the host), and
`probe_graph_rss.sh` (peak host RSS for one bucket).

## TODO / caveats

### HPU graph limitations

CAUTION: HPU GRAPHS DO NOT WORK IN ALL CONFIGURATIONS. THE CHECKED
CONFIGURATIONS ARE: GRAPHS AT CP=1 WITHIN THE SEQUENCE CAP, AND EAGER IN ALL
OTHER CONFIGURATIONS. OTHER CONFIGURATIONS CAN STOP THE SERVER PROCESS OR
GIVE WRONG RESULTS.

- **CP=1 with graphs has a sequence cap** (`H3_MAX_GRAPH_SEQ`, default
  20000). A bucket that is not in the recipe cache starts a full graph
  capture. The host-side IR of the capture grows about linearly with the
  packed sequence length. r480 with 124 frames (sequence about 17.4k) fits.
  r768 with 192 frames (sequence about 60k) increased the host memory of the
  process to 86.6 GB, and the operating system stopped the server during
  capture. A ghost pool of 96 GiB stayed on the card. The runner now stops
  oversized jobs before publish and shows a message with remedies: use
  `--cp 4` or `--cp 8`, restart with `--no-graphs`, or set
  `H3_MAX_GRAPH_SEQ` higher (at your own risk).
- **Graphs under CP=2 and CP=8 do not work.** Runs with graphs at CP=2
  failed with `PT_DEVMEM` allocation errors during pipeline capture. The
  graphs run at CP=8 (r768) ended with a Synapse device critical error.
  Static equal-split all-to-all operations (kill switch
  `H3_CP_A2A_STATIC=0`) fixed the collective-in-capture abort. But capture
  under context parallelism is still not checked. Treat `H3_CP_GRAPHS=1` as
  experimental.
- Synapse emits a warning about `index_select` in captured graphs: "might
  result in accuracy issues". Runs completed. But no audit checked the
  accuracy impact.

### Feature gaps

- **LoRA is not supported.** The patch layer has no adapter loading.
- **Reference images (ref2va) are not supported yet.** The design names the
  workflow, but `run_t2va.py` implements only t2va and fl2va.
- **First and last image anchors (fl2va) work.** Use one or two anchors
  (`init_image`, `end_image`). In an acceptance run, frame 0 agreed with the
  start anchor at cosine 0.9997, and the last frame agreed with the end
  anchor at cosine 0.9995. fl2va needs the fused decode configuration
  (`vae=hpu` with `H3_FUSED_DECODE=1`) because the keyframe VAE encoder
  needs HPU tensors. A configuration with the VAE on the CPU is for t2va
  only.

### Server contract deviations

- `output_format` accepts only `mp4` (H.264 + AAC through ffmpeg). Other
  values give a 400 response.
- The server accepts `negative_prompt` and ignores it.
- `video_frames`, `fps`, and size are server-wide, not per request. This
  keeps the DiT graph shape stable.

### Operational caveats

- After a large lazy graph, the lazy bridge can stop new enqueues
  in-process. The engines then idle without a message. Smaller per-rank
  graphs make this rare. If it happens, use the decode subprocess
  (`H3_DECODE_SUBPROCESS=1`).
- `te=stream` gives bit-exact text presentations vs the CPU path. Image
  presentations can flip one knife-edge channel at layer 43, depending on
  the boot. An fp32 adjudication showed both branches equally close to the
  fp32 truth. Within one process, runs are deterministic.
- Under context parallelism, each rank uses its own recipe cache. Set
  `H3_RECIPES_DIR`, and rank N uses `<dir>_rankN`. The reason is that the
  recipe cache configuration is per process.
- `requirements-pinned.txt` references some Habana wheels by absolute local
  `file://` paths. On another machine, put the wheels at these paths or
  change these lines.
