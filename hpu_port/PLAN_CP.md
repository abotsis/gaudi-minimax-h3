# H3 multi-card plan: sequence/context parallelism (SP/CP) on 8× HL-225

Status: **rev 3** (2026-09-12) — supersedes rev 2. Adds the Phase 1
implementation record (CPU-only verification; no card window used).

## Ground truth (verified)

| Fact | Value | Source |
|---|---|---|
| Attention heads | **56**, head_dim 128, hidden 5376, ffn 14336, 50 blocks | `transformer/config.json` |
| Head-shard sizes | 1/2/4/8 → 56/28/14/7 per rank — exact | arith |
| Hidden/FFN shards | 5376 → 2688/1344/672; 14336 → 7168/3584/1792 — exact | arith |
| Native CP plan | `_cp_plan` on the H3 model: rope outputs dim-0, `hidden_states` dim-1 (seq), `adaln_indices`/`timestep_indices` dim-0, heads gather via sequence-wide indices; supports `ulysses_anything`/`ring_anything` | `transformer_minimax_h3.py:463` |
| Activation API | `model.enable_parallelism(config=ContextParallelConfig(...))`; mesh via `torch.distributed.device_mesh` | `modeling_utils.py:1607` |
| Backend gate | CP allowed only for backends in `_AttentionBackendRegistry._supports_context_parallel`; our custom `habana_fused_sdpa` must be added | `modeling_utils.py:638`, `attention_dispatch.py:261` |
| Seq @480p-5s | 15911 = 512 text + 37·405 video + 414 audio; text budget pads → divisibility knob | `run_t2va.py` |
| Weights under CP | **Replicated per rank** (62 GiB/card stays); activations+seq split | definition |
| Proc:model | 1 process : 1 HPU, N workers (spawn), HCCL PG — same as vLLM-Gaudi | user directive |
| **Measured single-card denoise** | **174.3 s true device time / 29 steps ≈ 6.0 s/step** (decode-entry sync, v58). Host enqueue is 8.2 s with cached recipes — do NOT trust host-side block times | v58 timed sync |
| **DiT-free at decode entry** | works: `update_components(transformer=None)` + `del` + `gc.collect()`; 62 GiB functionally returned (62 GiB ballast allocates after). `torch.hpu.mem_get_info` LAGS one allocation cycle — never trust a single read | freeprobe 2026-09-11 |
| Baseline artifact | `/root/src/h3/out/t2va_h3_bc680ca43f.mp4` (v55: CPU VAE decode 1804 s) | v55 |

## Fault map (what CP changes, what it doesn't)

| Failure | Root cause | Does CP=N relieve it? |
|---|---|---|
| v14–v21 memcpy recipe compile fails | wrong `HABANA_PLUGINS_LIB_PATH` (no GC plugin) | n/a (fixed in launcher) |
| v14–v49 assorted synStatus 26 | `GC_KERNEL_PATH` unset by launcher → GC compiles with no kernel lib | n/a (fixed, pinned to stock lib) |
| `SYN_MEM_ALLOC` host-pool errors | `vm.nr_hugepages=0` | n/a (fixed, 4096 persisted) |
| `range_f32` node-create fail | `aten.arange` inside giant fused lazy graph | **YES** — per-rank graphs are smaller (host-basis patch also fixes it outright) |
| 3-way broadcast mul `ival` IR failure | flaky PT-bridge lowering (per-axis rope patch avoids it) | reduces exposure (smaller tensors), not a guarantee |
| `synFailedSectionValidation` at decode | GC section placement under memory pressure | **YES** — CP=8 activations ≈ activations/8; VAE back on-card plausible |
| v50/51/52/58 **D2H launch-thread wedge** (`JoinPendingLaunchThread`) | lazy bridge work-queue stops accepting new enqueues after (giant graph → full drain) in-process; silent, engines idle, copy never reaches CCB. NOT memory (v58 had ~76 GiB free at the wedge) | **probable YES** — per-rank denoise graphs are seq/N-sized and far smaller; the drain+re-enqueue pattern may simply never wedge. Not guaranteed → keep the decode-subprocess escape hatch |
| v54 "Compute or dma timeout" kill | device critical error during full-frame VAE decode | likely relieved (VAE on-card with 60+ GiB free after DiT-free; CP shrinks nothing in VAE — VAE is post-loop and sequence-parallel-friendly anyway) |

Escape hatch (implement regardless): **decode subprocess** — after the decode-entry
sync, rank0 writes latents to a temp file, a fresh eager-mode worker
(`PT_HPU_LAZY_MODE=0`) does VAE decode + D2H + export. Small D2Hs are reliable in
fresh processes; the wedge only appears after the big lazy graph in the same
process.

## Architecture (target)

```
run_t2va.py --cp N
  parent: parse args, build plan (bucket/seed/cache key), spawn N workers
          (multiprocessing spawn — fork is unsafe with Habana threads),
          gather artifacts, mux mp4
  worker r:
    HABANA_VISIBLE_DEVICES=r               (1:1 proc:HPU)
    torch.distributed.init_process_group("hccl", world=N, rank=r)
    load FULL DiT (62 GiB) -> hpu          (replicated)
    enable_parallelism(ContextParallelConfig(ulysses_degree=N))   # ring A/B later
    run the same modular pipeline — diffusers _cp_plan shards seq
    embed cache: rank0 loads/creates in shared .ecache, barrier, others read
    latents: every rank draws IDENTICAL CPU noise (same seed via
             patch_prepare_latents_cpu), local shard = slice — no broadcast
    per-rank logs: logs/h3_vX_rank{r}.log; parent merges timings
```

Ulysses vs ring: ulysses splits heads (7–28/rank) + one all-to-all per attention;
ring splits seq + N ring passes, supports arbitrary seq via `ring_anything`.
Start ulysses; A/B ring for 768p / long durations.

## Phases

### Phase 0 — baseline (done)
- [x] v55 end-to-end artifact (CPU VAE); v50–52 HPU DiT denoise proven
- [x] Timed syncs at decode entry / audio entry / export
- [x] DiT freed at decode entry (62 GiB returned)
- [x] True denoise device time measured: 6.0 s/step @480p-5s

### Phase 1 — CP=2 bring-up (first multi-card window)
1. [x] Backend whitelist: `habana_fused_sdpa` added to
   `_supports_context_parallel` (both `hpu_patches._install_attention_backend`
   and the runner's `_register_attention_backend`; idempotent, drift-checked).
   CPU world=1 parity re-verified bit-exact after (cpu_smoke 39/39).
2. [x] Seq divisibility: text budget padded upward so seq ≡ 0 mod N
   (`--cp` pad computed in main(); rides apply_text_budget's zero-embed rows;
   over-budget prompts raise loudly under CP). Verified: 15911 -> 15912 @ CP=2.
3. [x] FusedSDPA under CP: the backend routes `_parallel_config` calls through
   diffusers' `TemplatedUlyssesAttention` with the one-rank body as
   `forward_op` (rank-shape assert q-seq == kv-seq). CPU gloo parity: delta 0.0
   (backend level), ≤ 1.2e-6 (tiny-DiT-through-enable_parallelism level).
4. [x] Launcher/runner: `--cp N`, spawn-context workers, per-rank env
   (`HABANA_VISIBLE_DEVICES`, RANK/WORLD_SIZE/MASTER_*), hccl/gloo selection,
   rank-tagged logs (`[h3:rN]`), per-rank recipe cache (`_rank{r}` suffix),
   parent timing merge.
5. [x] CPU 2-rank gloo parity harness: `tests/cp_parity_cpu.py` — PASS 20/20
   (backend+registry, templated-Ulysses sharded attention vs unsharded, real
   tiny DiT through the actual `enable_parallelism` API vs unsharded reference,
   spawn plumbing smoke, decode-subprocess CPU eager hatch).
6. [x] Acceptance on card (v59–v67, 2026-09-12 00:40): CP=2 end-to-end — HCCL PG
   init OK on both ranks, `DeviceMesh((ring=1, ulysses=2), 'hpu')`, refiner
   un-stamped, noise determinism sha256-verified across ranks via hccl
   all_gather, artifact muxed (`out/t2va_h3_cp2.mp4`, 124 frames + wav; frame
   content plausible vs the CP=1 baseline — 2-step run is under-converged as
   expected). Card fixes landed during bring-up: hccl all_gather needs HPU
   tensors (digest moved to device), plan-derived width/height/steps in
   workers, decode blob `weights_only=False`, `no_grad` in the subprocess
   (autograd-on OOM'd), per-frame D2H indexing (`[:, :, i]` not `[..., i]`),
   subprocess gets its OWN card (parent's freed pages don't cross processes),
   per-frame D2H indexing (`[:, :, i]` not `[..., i]`),
   wav adopted from the subprocess meta. Warm-step scaling number: v67
   30-step run vs the 174 s CP=1 baseline.

#### CP=8 acceptance (v69/v70, 2026-09-12 01:25)

* 19 evals @ **0.853 s/step** steady-state vs CP=1's 6.0 s/step -> **7.0x
  per-step speedup, 88% scaling efficiency**; denoise true wall ~17.3 s vs
  182 s. Drain at decode entry 1.08 s. 15912 seq % 8 == 0, no extra pad.
* v69 lesson (card handoff): ranks 1..7 finishing denoise do NOT free their
  cards while they sit in destroy_process_group waiting for rank0's long
  decode. Fixes: (a) hard exit (`os._exit(0)`) for ALL ranks after their
  work is on disk — skips HCCL teardown AND the rank0 interpreter-teardown
  wedge (237 idle threads at exit); (b) `_pick_free_hpu_card` scans all 8
  cards via `hl-smi -Q index,memory.used`, waits up to 300 s for a card to
  drain to <=2 GiB, and NEVER returns an out-of-range card.
* v69 also showed the CP=2-era 315 s stall was picker timeout + rc=1
  subprocess on nonexistent card 8; not a Synapse wedge.
* Known cost: a NEW prompt makes all 8 ranks re-run the CPU text encoder
  redundantly (v69: 452 s). Cache hit makes it instant (v70). Future: encode
  once in the parent and pass embeddings via the payload.

#### Implementation record (2026-09-12, CPU-only)

* **Actual `enable_parallelism` API** (diffusers 0.40.0, differs from rev 2's
  guess): keyword-only `enable_parallelism(*, config: ParallelConfig |
  ContextParallelConfig | TensorParallelConfig, cp_plan: dict | None = None)`.
  It derives device_type from `torch._C._get_accelerator().type` internally and
  creates its own mesh unless `ContextParallelConfig.mesh` is provided — the
  runner always provides the mesh (`_cp_enable_parallelism`), built for the
  backend actually in use (hccl -> "hpu", gloo -> "cpu"). `config.setup()`
  records rank/world/mesh; collectives use the ulysses mesh group only.
* **Discovery — token refiner must stay replicated**: `enable_parallelism`
  stamps `_parallel_config` on EVERY attention processor in the model,
  including the token refiner's — but H3 runs the refiner on the unsharded
  text run BEFORE the packed buffer is built (`_cp_plan` splits only at
  `transformer_blocks.0`). CP-wrapping that replicated full-seq input through
  the Ulysses all-to-all expands the sequence (ws×S with repeated blocks) and
  shrinks the head dim — wrong math and broken shapes, with the stock native
  backend too. `_cp_enable_parallelism` un-stamps exactly the refiner's
  processors after enable_parallelism (pinned by the parity test).
* **Backend CP routing**: `habana_fused_sdpa(q, k, v, _parallel_config=cfg)`
  wraps the one-rank computation (`_habana_local_sdpa`, shared verbatim with
  the unsharded path) in `_templated_context_parallel_attention` with a
  native-style `forward_op`/stub `backward_op` — same template the stock
  backends use, so correctness holds on gloo (CPU parity) and hccl alike.
  The explicit `softmax_mode`/`recompute_mode` overrides are not propagated
  through the templated forward_op signature (H3 never passes them).
* **gloo confirmed**: `all_to_all_single`, `funcol.all_gather_tensor`, and
  `init_device_mesh("cpu", (1, N), ("ring", "ulysses"))` all work on this
  torch 2.11.0a0+habana build — the CPU harness exercises the real collective
  patterns (probe + parity test).
* **Noise determinism**: all ranks draw the IDENTICAL full CPU noise via the
  seeded CPU generator (patch_prepare_latents_cpu); a one-time sha256 digest
  all_gather asserts identity per rank pair before any H2D. No hand-slicing:
  the `_cp_plan` splits inside diffusers.
* **HPU graphs default OFF under CP** (`H3_CP_GRAPHS=1` forces): collectives
  inside `wrap_in_hpu_graph` capture are untested on this stack; first CP
  window should run eager lazy mode with the per-block mark_step hooks.
* **Decode-subprocess hatch shipped** (`H3_DECODE_SUBPROCESS=1`,
  `decode_subproc.py`): post-sync, post-DiT-free, rank0 saves latents +
  audio latents + pixel mean/std to a .pt, a fresh eager-mode process
  (`PT_HPU_LAZY_MODE=0`, same `HABANA_VISIBLE_DEVICES`) loads the VAEs from
  the workdir, decodes, writes frames + wav + meta json; the parent muxes.
  CPU eager mode is the tested path (tiny-VAE test in cp_parity_cpu.py); the
  hpu branch reuses the identical math and the per-frame chunked D2H.
* Embed-cache tmp files are now PID-unique (`store_cached_embeds`) — under CP
  every rank stores the same key and a shared `.tmp` would race.

### Phase 2 — CP=4/8 + VAE back on-card
- [ ] CP=4/8 runs; recipe cache per-rank dirs (PT_HPU_RECIPE_CACHE_CONFIG is
      per-process); parallel cold-compile is fine (separate cards)
- [ ] Restore `vae=hpu,audio=hpu` + chunked-D2H decode under CP=8
- [ ] 768p-5s, 480p-10s buckets (seq ~2×; CP pays most there)
- [ ] Validate the D2H wedge is gone at CP=8 (per-rank graphs small); if not,
      decode-subprocess path ships instead

### Phase 3 — performance pass (profiler-gated)
- [ ] HABANA_PROFILE=1 on CP=8: comm/compute overlap, allreduce cost vs the
      6 s/step MME-bound floor
- [ ] `H3_BATCH_ADALN=1` if the 50 GEMVs show
- [ ] FP8 (INC) — orthogonal; halves per-rank weight traffic

### Phase 4 (optional) — TP if we ever need >62 GiB/card headroom
Write a TensorParallel plan for MiniMaxH3 (column qkv/mlp-in, row
proj-out/mlp-out; 56 heads exact). ~a focused port week; do on demand only.

## Risks (rank-ordered)
1. FusedSDPA × CP correctness under all-to-all'd KV → CPU gloo parity first.
2. HCCL PG init from our venv (`habana_frameworks.torch.distributed.hccl`) —
   verify early, cheap.
3. D2H wedge may persist even with small graphs → decode-subprocess hatch.
4. `enable_parallelism` may rewrite forwards → re-run the port's `getsource`
   drift checks (patch_decode_steps etc.) after it applies.
5. Recipe cache divergence between ranks (same graph, different rank shapes? —
   under ulysses, per-rank shapes are IDENTICAL; under ring, seq differs per
   rank — one dir per rank regardless).
6. Embed-cache cross-rank race → rank0-writes + barrier.

## Definition of done
480p-5s CP=8: denoise wall ≤ ~45 s (≥60% scaling vs 174 s single-card true
device time); VAEs on-card; no phase hangs; artifact parity vs v55 baseline
(SSIM on frames, waveform RMSE on audio).

## Implementation assignment (2026-09-11)
Subagent task: Phases 1.1–1.5 + the decode-subprocess hatch (Phase 2 item) —
CPU-only verification (py_compile, gloo 2-rank parity, CPU pipeline smoke).
NO card runs without explicit user approval (H3_ALLOW_HPU protocol).
**Status: DONE 2026-09-12** (see Implementation record above); Phase 1 item 6
(card acceptance) remains, gated on the next card window.

## Phase 3 — Temporal-sharded decode farm (v71–v77, 2026-09-12)

GOAL: keep the decode off the denoise phase's critical path and stop paying
the 112 s single-process decode at CP=8 (denoise is only 15 s there).

### Why in-process sharded decode is impossible (v71/v72/v73c)
The internal `_decode` loop distributes exactly: chunk i -> rank i%world, the
only cross-chunk state is a 5-frame pixel tail, so clips decode concurrently
and only the blend needs neighbor data. THREE attempts to run this inside the
denoise parent's process all wedged:
* v71: broadcast-before-decode serialized the clips (watchdog killed waiters).
* v72: decode-first order; rank0 completed clip0 + 17 frame D2Hs, then the
  cast feeding the overlap broadcast stalled (enqueued-but-stalled work trips
  the Synapse no-progress watchdog; idle cards are safe).
* v73c: file-based handoff (NO collectives); rank0 wedged in the first tail
  D2H. Conclusion: post-drain device work in the process that ran the giant
  denoise graph wedges at a VARIABLE point — the lazy launch-thread bug from
  v50-58, now precisely characterized. Fresh processes are immune (8/8).

### The farm architecture (v74-v77)
* Parent rank0 (post-denoise, DiT freed): saves the latents blob (the one
  post-drain D2H that is reliably small+immediate) and Popens 7 video workers
  (chunk i -> card i+1) + 1 audio worker (card 0, ~5 GiB, fits beside rank0's
  residue).
* Workers: fresh EAGER processes running decode_subproc.py --shard i, which
  call the same `_decode_sharded_core` (file-based tail handoff + done
  markers under frames_dir/.shard_tmp, per-frame D2Hs). No collectives.
* Export phase polls the 8 done markers (host-side, 900 s timeout), reaps
  workers, muxes. All ranks os._exit(0) hard — no HCCL teardown, no atexit
  (rank0's interpreter teardown wedges otherwise, v70).
* Card handoff: the driver lags MINUTES behind os._exit on memory release, so
  the spawn gate is "enough free" (<=78 GiB used => ~16 GiB free), not idle.
* v74/v75 workers OOM'd (394 MB PT_DEVMEM) because `_decode_sharded_core` ran
  with AUTOGRAD ON outside the block's @torch.no_grad: 36 decoder layers of
  saved activations filled the card. The core now forces grad off.

### Results (v77, CP=8, 20 steps, 5 s 480p)
* denoise 15.2 s; decode farm markers complete 35.1 s after spawn (vs 112 s
  single-process subprocess; 88 s in-process eager); mux 2.2 s.
* Chunk-boundary frame deltas 2.4-4.2 (same as natural motion; no seams).
* Frame content bit-identical to the v70 run (determinism across the farm).
* CPU parity: tests/decode_shard_cpu.py — byte-identical vs stock decode at
  world=3 (chunkless rank) and world=2 (cross-rank tails), plus a farm-level
  CPU test of concurrent workers.

## Phase 4 — Weight-resident serving at CP=8: in-process FUSED decode (v93–v98)

### The corrected root cause (v97, decisive)

The v72/v73c "post-drain wedge" model was WRONG. The user's cross-run audit
established the real invariant: **lazy-mode video-VAE decode has NEVER
completed inside the DiT process, drained or not** — v91/v92 removed the
boundary sync/free (decode enqueued while the denoise queue drained) and
still hung at the first video-VAE device work. The v78/v79 "resident serve
works" baselines were hollow (their device map was vae=cpu — those 85 s
decodes were CPU decodes).

Actual mechanism: `install_block_mark_steps` only targeted the DiT. The
video VAE decoder is the same graph class (36-layer ViT + atDecoder3d) and
got NO frontier breaks: the post-call mark_step lives on `vae.decode`, which
the sharded/fused cores bypass (`vae._decode_clip` directly), so each clip
compiled as ONE unbroken giant frontier (tiling off, compound size
unlimited). A graph compile that never returns looks exactly like idle
engines + free memory — which is why it read as a "wedge".

Fix (v97): `install_block_mark_steps(pipe.vae.decoder)` (36 blocks) +
`_flush_lazy_frontier()` after EVERY `_decode_clip` in both cores. One
10-line change made the whole subprocess/farm machine unnecessary.

### Synapse device-acquire rules (probe-proven, v82–v85)

* `HABANA_VISIBLE_DEVICES` and `HABANA_VISIBLE_MODULES` are IGNORED by device
  acquire: every process first-free-scans all 8 cards and refuses held ones
  (synStatus=8, "already acquired by PID"). No card sharing, no pinning.
* 8 resident CP parents ⇒ ZERO cards for decode workers. The v80/v81/v85
  farm-worker massacres were this, not host hugepage exhaustion (eager
  workers consume ZERO hugepages — measured; the HP bump to 12288 was
  harmless but not the fix).
* Consequence: with resident DiTs, NO subprocess can decode — decode must be
  in-process. Which requires the frontier-break fix above.

### The fused path (H3_FUSED_DECODE=1, v98 production)

Per rank, ONE decode sequence enqueued while the denoise queue still drains
(NO boundary sync, NO DiT free — decode activations ≈ +0.1 GiB, fits beside
the resident 62 GiB DiT; 62+9.7+0.1 < 94.6):

1. owned clips (`vae._decode_clip`, bf16 autocast) — cached recipes, per-block
   frontier breaks + per-clip flush;
2. tail slots (n_chunks,3,fov,Hp,Wp) + `all_gather` (collectives arrive in a
   clean flushed queue — the denoise-step discipline; an all_gather in an OPEN
   frontier deadlocks in work.wait(), v95);
3. blend + denorm + on-device uint8 quantize + frames-first contiguous
   permute, IN-FRONTIER;
4. flush + `torch.hpu.synchronize()`;
5. post-sync: PURE contiguous D2H frame pulls (~0.8 MB each — the 8/8-reliable
   small-copy pattern) + png writes.

Audio: last rank decodes AFTER the sync (separate small graph, proven to
complete in every run); its `done_r{world-1}` marker is written after the wav
so the export's marker poll also guarantees the wav exists.

Audio-flag fixes (v97b): `_decode_sharded_audio` sets
`sharded_audio_done=True` itself (the fused/queue branches forgot, so rank0
re-decoded audio post-drain); the serve-sync timeout now RAISES instead of
warn-and-continue (a stuck export used to send ranks 1..N-1 into collectives
without rank0 — v85's massacre).

### v98 acceptance (CP=8, serve=2, 5 steps, seed 42)

* All 8 ranks fused-decoded IN-PROCESS with resident DiTs: 17 frames/rank
  (rank6: 22 = tail chunk, rank7: 0 + audio), wall 11–13 s (vs farm 35 s —
  no worker VAE loads, no extra cards).
* Request 2: generate 9.7 s total (5-step denoise replay ~4.3 s + decode
  replay) — the resident-serve steady state.
* req0.mp4 md5 == req1.mp4 md5 (determinism through the fused path).
* ZERO watchdog/DFA events, zero card-wait timeouts, no extra processes.
* Artifacts: `out/t2va_h3_v98_cp8serve2_req{0,1}.mp4`.
* CPU parity: `tests/decode_fused_cpu.py` — fused core byte-identical to
  stock decode at world=2 and world=3 (gloo all_gather, real processes).

### Status

* Goal 2 (weight-resident serving) COMPLETE at CP=1, CP=2, CP=8.
* Remaining for goal 3 (stable server): wrap serve mode in an HTTP/socket
  surface (prompt+params → video); fresh-prompt TE tax (~450 s) still
  applies — encode-once-broadcast or cached-embed distribution is the next
  lever; embed cache makes repeat prompts instant.

## Phase 5 — Stable server, sdcpp-compatible (v99–v102) [goal 3]

### v99: encode-once-broadcast + TE threading
* Under CP, a missing --embed-cache-dir now defaults to a shared
  `workdir/.ecache`; rank0 encodes + stores (atomic PID-unique tmp +
  replace), ranks 1..N-1 wait (fail-open 660 s) for the cache file.
* `--te-threads 32` (128-core host): rank0 TE 452 s -> 89 s.
* Fresh-prompt CP=2 request: 663 s (v84) -> 117 s end-to-end.

### v100–v102: the server (hpu_port/h3_server.py)
* Parent-process HTTP front-end (ThreadingHTTPServer, no HPU devices);
  workers receive requests via a FILE SPOOL — no collectives added:
  - POST /sdcpp/v1/vid_gen -> 202 {id, poll_url} (spool req_{seq}_{id}.json)
  - GET /sdcpp/v1/jobs/{id} -> queued/running/completed/failed/cancelled;
    completed carries result.b64_json (whole mp4, base64), mime video/mp4,
    fps, frame_count
  - POST /sdcpp/v1/jobs/{id}/cancel (spool {id}.cancel; honoured at claim)
  - GET /sdcpp/v1/capabilities, GET /health
* Worker side: H3_SERVER_MODE=1 makes the --serve loop a claim loop
  (`_server_claim_request`): rank0 claims (atomic rename), broadcasts via
  serve_sync/current.json; ranks follow on seq change; SHUTDOWN file ends the
  loop cleanly on all ranks. Per-request: prompt/steps/seed + job-named
  artifacts; duration/resolution server-wide (graph-shape stability).
* v101 bug (fixed v101b): request seq was generated from spool glob —
  claimed files rename away, so request 2 got seq 1 again; rank1 never
  reclaimed and rank0 ran solo -> denoise collectives stalled -> Synapse
  watchdog killed rank0 (loud, correct failure). Fixed with a monotonic
  counter file + rank1 acceptance on seq != last.

### Server acceptance (CP=2, live)
* Puppy request: 22.8 s generate; fox (fresh prompt): 189 s (TE 163 s @8thr
  + resident replay); fox repeat: 21.4 s generate, artifact md5 IDENTICAL to
  first fox (deterministic end-to-end through HTTP).
* b64 retrieval verified: 933,758 bytes mp4, 5.17 s duration, md5 match.
* Cancel-before-claim verified live (status cancelled).
* frame_count fix (png count fallback) lands next boot.

### Steady-state budget (CP=2, cached-embed prompt)
TE ~0 s | denoise replay ~4 s (2 steps) | fused decode ~22 s wall |
export+mux ~4 s => ~26-30 s per request, weights NEVER leave HBM.

## Phase 6 — Per-request frames/resolution + i2va (v103–v108) [goals 3+4 complete]

### v103/v104: per-request frames + resolution
* `_server_apply_params` (closure in run_pipeline): video_frames -> nearest
  supported duration bucket (4n+1 normalization upstream), WxH -> bucket snap
  via resolve_resolution; recomputes the CP divisibility pad for the new
  packed sequence and re-keys the embed cache on prompt+shape+workflow.
* Validated live: 175-frame job (7 s, T_lat 52, new DiT shape compile 213 s
  then replays), r576 1024x576 job (221 s first / 28.6 s repeat), portrait
  request snaps to the canonical landscape bucket, r720 (gated) REJECTED
  cleanly pre-publish (job failed, pool unharmed — v103b rank0-validates-
  before-publish; ranks re-apply the same deterministic params on receive).

### v105–v108: i2va (fl2va) through the server
* v105 standalone fl2va probe at CP=1: keyframe VAE encoder (conv stack, 6
  down_blocks) got per-block frontier breaks (v105 insurance); HPU graphs ON
  at CP=1 for the whole fl2va chain; anchor fidelity frame0 diff 4.0/255.
* v106: dual-pipe server — the fl2va ModularPipeline shares EVERY component
  instance with t2va (transformer/vae/audio_vae/TE shard-identical, md5
  verified); only the Python block graph is separate. Per-request workflow:
  init_image present -> fl2va.
* v107 crash (user-visible): 1-anchor fl2va packed seq = 16317 (ODD) —
  EquipartitionSharder asserted on hidden_states dim 1. Root cause: the t2va
  boot CP pad doesn't cover fl2va's anchors*rows rows (1 anchor = 405, odd).
* v108 fix: _server_apply_params ALWAYS recomputes seq = budget +
  anchors*rows + 2*A + T_lat*rows and repads. 1-anchor i2va at CP=2 now
  completes; end_image optional (2nd anchor = end-frame conditioning).

### v108 acceptance
* i2va over HTTP: 124 frames + audio, anchor fidelity 4.7/255, async job
  contract, b64 retrieval verified (1.56 MB mp4).
* t2va regression after an fl2va request: md5 a38da26e... == v99/v103
  artifact (cross-workflow determinism intact), 24.4 s generate.
* Server steady state: ~21-30 s/request (cached embeds); fresh prompt ~90 s
  TE (32 threads) + compile-once for new shapes.

### All four goals: DELIVERED
1. parallel decode: farm (v77) -> superseded by in-process fused decode (v98)
2. weight-resident serving: CP=1/2/8, weights never leave HBM (v97/v98)
3. sdcpp-compatible server: async jobs, b64 results, cancel, per-request
   frames/resolution/workflow (v100-v108)
4. i2va: init_image (+ optional end_image) + prompt -> video with audio (v105-v108)

## Phase 7 — CP=8 server production boot (v109–v112)

* v109 crash: 3+ ranks died at placement (synStatus=8) — boot-time acquire
  RACE. All 8 ranks first-free-scan simultaneously during `.to()`; two ranks
  scanning in the same instant can target the same card; loser dies. Earlier
  CP=8 boots were lucky on import stagger.
* v110 fix: BOOT CHAIN — rank r waits for `.bootchain/r{r-1}.acquired`
  before its first placement, publishes `r{rank}.acquired` after; placement
  retries 6x with 10-50 s backoff on synStatus=8. (v111 NameError from a
  dropped tip_to_component dict — re-added.)
* v111: ranks placed but rank2 exhausted retries on GHOST pools (74 GiB on
  cards 4/6 that hl-smi showed but no process held — driver-held residue
  from the earlier crashes). Targeted driver module cycle (user preference):
  stop hl_traf, sync + drop_caches, modprobe -r {ib,en,habanalabs}, modprobe
  back. All 8 cards back to 768 MiB, firmware clean.
* v112 ACCEPTANCE (CP=8 server): all 8 ranks dual-pipe ready, first HTTP
  request 124 frames in 9.86 s generate (resident replay + fused decode),
  repeat request md5-identical (b4a35e6d...), 8.97 s generate.
  NOTE: CP=8 artifacts differ from CP=2 by design (different cp_pad ->
  different packed seq -> different content); determinism is per-CP-degree.
* Production server: H3_ALLOW_HPU=1 H3_FUSED_DECODE=1 launch_h3.sh --cp 8
  --server --port 8032 --te-threads 32. Steady state ~9-10 s/request
  (cached embeds), weights never leave HBM.

## Phase 8 — Streamed text encoder (te=stream, v115+, 2026-09-13)

### Motivation
Fresh-prompt TE tax was the last big server cost (~48-89 s CPU Qwen3-VL-32B
forward vs ~9-10 s steady-state request). Full TE on-card is impossible
alongside the DiT (63 GB vs 62 GB replicated) AND triggers the giant-lazy-
graph wedge class. Hybrid: stream decoder-layer weights H2D per prompt.

### Design (te_stream.py, StreamedQwen3VL)
* LM decoder layers stay CPU-resident (mmap params, identical footprint to
  te=cpu); ONE scratch Qwen3VLTextDecoderLayer on-card; per layer: 11
  unpinned per-param copy_ H2D (~235 ms) + stock eager forward (5.5 ms,
  ~105 TFLOPS) + mark_step. NO HPU graphs (avoids the wedge class).
* MiniMax-H3 conditions on hidden_states[50] -> layers 51..63 (14 GB) and
  lm_head dropped at activate.
* Vision tower + embed_tokens + mrope + masked_scatter stay ON CPU
  (bit-exact): on-card vision runs imprecisely (pool cos 0.92 — tanh-gelu/
  conv3d/interp divergences) and the 9% error amplifies through 51 streamed
  layers into garbage. Rotary emb on-card (safe).
* Deepstack injection: full-sequence zero-padded dense adds (no advanced
  indexing in lazy IR); numerically identical to the stock indexed scatter.
* Rank0-only activation (pinning would make weights private x8 ranks);
  ranks 1..7 keep mmap CPU TE as the encode-once-broadcast fail-open path.
* Bridge: patch_text_encoder_device routes get_qwen3vl_prompt_embeds to
  stream.encode when text_encoder._h3_stream exists. Embed-cache
  fingerprint includes the te device (the two paths are DIFFERENT bf16
  branches — see below).

### Synapse host-memory facts (probe-proven, /tmp/pin_bisect*)
* torch pin_memory caps at ~24-25 GB total in mixed/sub-1GB allocations:
  Synapse backs host allocs with the machine's 2 MB huge-page pool
  (12288 HPs stock; vm.nr_hugepages to grow) and its regular-page fallback
  fails with ENOMEM under pressure (dmesg "Failed to pin host memory").
  Uniform >=1 GB allocations pin past 80 GB (driver path, no HP pool).
* H2D from an allocation's BASE pointer: 24 GB/s. From any offset: ~2-6
  GB/s (runtime staging). Flat per-layer buffers exploit (a)+(b).
* CPU->HPU module .to() and param copy_ stage through synHostMalloc ->
  build scratch with torch.device("hpu") context instead; never rebind
  Parameter objects post-construction (Module._apply rebinds silenty —
  parity-debug: captured id != live id).
* aten::view.dtype (uint8->bf16) is a REAL lazy graph node: precomputed
  dtype-views of a mirror snapshot the EMPTY tensor at activate ->
  zero-weight identity layers (bit-identical garbage across runs). Use
  plain narrow() strided views in bf16 element space.

### Numerics: the sink-channel branch flip (img_bisect6/7)
* Streamed hidden states track the CPU bf16 reference within 1 ulp through
  layer 42 (maxdiff 64 = 1 bf16 ulp at the ~16k massive-activation
  channels), then layer 43 flips a sink-channel cancellation: output cos
  0.69 vs the CPU branch.
* NOT an HPU bug: CPU bf16 layer 43 on the SAME streamed input flips
  identically (cos 0.695 vs CPU branch, 0.99996 vs HPU output). Given the
  EXACT CPU input, HPU layer 43 matches the CPU branch (cos 1.0005). The
  layer sits on a knife-edge cancellation; 1-ulp input noise decides the
  branch on either device.
* fp32 replay of layer 43 on fp32(ref[43]) shows the CPU bf16 branch
  itself deviates from fp32 truth by 681 (vs HPU-streamed 18367) — bf16
  has no "correct" branch here.
* Downstream A/B (r384, 124 frames, 4 steps, seed 7; vae=cpu both sides):
  video per-frame RGB cos 0.995-0.997 (PSNR 30.1 dB) — same content,
  detail divergence; audio waveform cos 0.927. Per-path determinism
  unaffected (same path + seed = same bits).

### Results
* Card test (tests/te_stream_card_test.py, real weights, seq 415 image
  presentation): text min per-token cos 0.99978 rel_fro 0.0043 PASS;
  steady 3.7-4.1 s for the full 51-layer prefill vs 48-89 s CPU (12-20x).
* CPU parity (tests/te_stream_cpu_parity.py): BIT-EXACT vs stock forward
  on tiny-config Qwen3-VL (text-only, deepstack image, long-seq cases).
* End-to-end A/B through run_t2va.py: both paths complete; artifacts
  differ at the level documented above.
* Production server fresh-prompt: ~90 s -> ~10 s projected (TE ~4 s +
  resident replay); embed cache unchanged for repeats (~0 s).

### Usage
H3_ALLOW_HPU=1 ... --device-map te=stream,dit=hpu,vae=hpu,audio=hpu
(pin_weights=False default = v1 unpinned; pin_weights=True is the v2 flat-
buffer fast path, blocked on the HP-pool cap above).

### Goal acceptance (2026-09-13, tests/te_stream_goal_probe.py + end-to-end)
* Varying text sizes (10/40/146/510 tokens): streamed vs CPU-bf16 rel_fro
  0.00000 (bit-identical ALL sizes), within-process deterministic, steady
  4.1-4.5 s (copy-bound; seq-independent as designed).
* Image presentation (405-token anchor @ r480 grid): within-process
  deterministic; BOOT-DEPENDENT branch: some boots bit-match CPU
  (rel_fro 0.000), others flip the layer-43 sink channel (rel_fro 0.24-
  0.35). Text presentations are boot-stable; image presentations have
  knife-edge layers that resolve per-process. Adjudication (localized
  fp32 layer-43 replay, CAUSAL mask — a maskless replay is non-causal and
  invalid: goal_probe5's triangle-inequality violation): both branches sit
  0.00284 from fp32 truth — peers, no truth violation. Acceptance gates
  are therefore: within-process determinism + truth-faithfulness, NOT
  cross-path bit-identity.
* fl2va end-to-end, 2 anchors (start+end), te=stream + fused decode
  (H3_FUSED_DECODE=1, vae=hpu — the v98 production config): COMPLETED,
  124 frames + wav, generate 197 s. Anchor fidelity frame0-vs-start
  cos 0.9997 / frameN-vs-end cos 0.9995 (cross ~0.74: distinct anchors,
  real interpolation). NOTE: keyframe VAE encoder requires HPU tensors ->
  fl2va with vae=cpu fails pre-TE ("Got a non-HPU tensor") — fl2va needs
  the fused-decode config, CPU-VAE decode config is t2va-only.
* te=stream produced ZERO errors across all runs; the two pipeline
  failures observed (VAE decode dma-timeout at CP=1 in-process; keyframe
  encoder on CPU) are pre-existing device-map issues unrelated to the TE.
