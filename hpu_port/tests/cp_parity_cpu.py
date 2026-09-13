#!/usr/bin/env python
"""CPU-only context-parallelism tests for the H3 HPU port (PLAN_CP.md phase 1.5).

Hard guarantee: NO HPU work. H3_DISABLE_HPU=1 and TORCH_DEVICE_BACKEND_AUTOLOAD=0
are set before anything imports torch; the process group is gloo over CPU
tensors; habana_frameworks is never imported. This validates:

  test 1 (parity, 2-rank gloo via mp.spawn):
    a. the registry edit: `habana_fused_sdpa` is registered AND listed in
       `_AttentionBackendRegistry._supports_context_parallel` (the gate
       `enable_parallelism` consults);
    b. the CP math diffusers uses: a DeviceMesh over gloo, seq-sharded
       q/k/v through the port's backend in its CPU passthrough mode routed via
       the SAME TemplatedUlyssesAttention template the stock backends use, and
       the gather per the `_cp_plan` semantics (equipartitioned dim-1 split,
       all-gather on the output) -- allclose vs the unsharded stock computation;
    c. a real tiny `MiniMaxH3Transformer3DModel` (the model whose class-level
       `_cp_plan` production uses) put into CP through the actual
       `enable_parallelism(ContextParallelConfig(ulysses_degree=2))` API, with
       identical inputs on every rank, compared against the unsharded
       single-process reference. This is NOT a full-model test: it validates the
       registry edit and our understanding of `_cp_plan` end to end on CPU.

  test 2 (spawn plumbing smoke, H3_CP_TEST=1):
    run_t2va.main() as CP parent with N=2 -- ranks initialize the gloo process
    group, barrier, verify the padded-seq plan accounting, emit rank-tagged
    logs and per-rank timing files; the parent merges them and exits 0.
    Tiny fake components; no model load.

  test 3 (decode-subprocess hatch, CPU):
    tiny video/audio VAEs saved into a fake workdir; a blob of random latents
    is decoded by decode_subproc.py in a FRESH eager-mode process (pure CPU);
    frames/wav/meta are verified against the in-process decode step math
    (uint8-exact frame 0, wav geometry).

Run:
  cd /root/src/h3/hpu_port
  LD_LIBRARY_PATH=/root/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/lib \
  TORCH_DEVICE_BACKEND_AUTOLOAD=0 ../../.venv/bin/python tests/cp_parity_cpu.py
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("H3_DISABLE_HPU", "1")  # hard CPU guarantee, BEFORE torch
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

HPU_PORT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HPU_PORT))

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import hpu_patches as hp
import run_t2va

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, cond: bool, detail: str = "") -> bool:
    ok = bool(cond)
    RESULTS.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    return ok


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


# ---------------------------------------------------------------------------
# Tiny-model fixtures (same geometry as tests/cpu_smoke.py; SEQ=198 % 2 == 0)
# ---------------------------------------------------------------------------

DIT_CFG = {
    "num_attention_heads": 2,
    "attention_head_dim": 96,  # 6 * rope_freq_dim(16) = 96 <= head_dim
    "hidden_size": 192,
    "num_layers": 2,
    "num_refiner_layers": 1,
    "ffn_dim": 256,
    "in_channels": 24,
    "audio_in_channels": 32,
    "patch_size": (1, 2, 2),
    "text_dim": 64,
    "freq_dim": 32,
    "time_embed_hidden_dim": 64,
    "time_embed_dim": 64,
    "rope_freq_dim": 16,
}
T_LAT = 7
LAT_H = LAT_W = 8
N_TEXT = 6
AUDIO_LATENTS = 40
AUDIO_CHANNELS = 2
ROWS_PER_FRAME = (LAT_H // 2) * (LAT_W // 2)
PATCH_DIM = DIT_CFG["in_channels"] * 4
N_VIDEO_ROWS = T_LAT * ROWS_PER_FRAME
N_AUDIO_ROWS = AUDIO_LATENTS * AUDIO_CHANNELS
SEQ = N_TEXT + N_AUDIO_ROWS + N_VIDEO_ROWS  # 198, divisible by 2
PIX_FRAMES_EXPECTED = 22  # 17n + 5 with n = (T_LAT - 2) // 5 = 1


def _build_reference(workdir: Path) -> dict:
    """Single-process, unsharded: seed everything, run the tiny DiT once, save
    weights + inputs + reference outputs for the CP workers to load."""
    torch.manual_seed(0)
    from diffusers.models.transformers.transformer_minimax_h3 import (
        MiniMaxH3Transformer3DModel,
    )
    from diffusers.modular_pipelines.minimax_h3.before_denoise import (
        MiniMaxH3PrepareLayoutStep,
        MiniMaxH3SetTimestepsStep,
    )
    from diffusers.schedulers.scheduling_minimax_h3 import MiniMaxH3Scheduler

    model = MiniMaxH3Transformer3DModel(**DIT_CFG).eval()

    text_token_tags = torch.full((N_TEXT,), 1, dtype=torch.long)
    position_ids, token_tags, video_indices, audio_indices, text_indices, _, _ = (
        MiniMaxH3PrepareLayoutStep.build_packed_sequence(
            text_token_tags=text_token_tags,
            num_latent_frames=T_LAT,
            latent_height=LAT_H,
            latent_width=LAT_W,
            num_audio_latents=AUDIO_LATENTS,
            patch_size=DIT_CFG["patch_size"],
            audio_channels=AUDIO_CHANNELS,
            audio_tag=2,
            video_tag=0,
        )
    )
    sched_v = MiniMaxH3Scheduler(shift=12.0)
    sched_a = MiniMaxH3Scheduler(shift=3.0)
    sched_v.set_timesteps(2, device="cpu")
    sched_a.set_timesteps(2, device="cpu")
    unique_timesteps, timestep_indices = MiniMaxH3SetTimestepsStep.build_row_timesteps(
        video_indices,
        audio_indices,
        num_condition_video_rows=0,
        num_condition_audio_rows=0,
        num_text_tokens=N_TEXT,
        video_timestep=float(sched_v.timesteps[0]),
        audio_timestep=float(sched_a.timesteps[0]),
        condition_video_timestep=max(float(sched_v.timesteps[0]), 0.999),
        condition_audio_timestep=1.0,
    )
    torch.manual_seed(1234)
    video_rows = torch.randn(1, N_VIDEO_ROWS, PATCH_DIM)
    audio_rows = torch.randn(1, N_AUDIO_ROWS, DIT_CFG["audio_in_channels"])
    text_embeds = torch.randn(1, N_TEXT, DIT_CFG["text_dim"])

    inputs = dict(
        hidden_states=video_rows,
        audio_hidden_states=audio_rows,
        encoder_hidden_states=text_embeds,
        timestep=unique_timesteps,
        timestep_indices=timestep_indices,
        token_tags=token_tags,
        position_ids=position_ids,
        video_indices=video_indices,
        audio_indices=audio_indices,
        text_indices=text_indices,
        return_dict=False,
    )
    with torch.no_grad():
        ref_v, ref_a = model(**inputs)

    # raw backend-level fixture: (1, SEQ, H, D) q/k/v and its unsharded output
    torch.manual_seed(4321)
    q = torch.randn(1, SEQ, 4, 8)
    k = torch.randn(1, SEQ, 4, 8)
    v = torch.randn(1, SEQ, 4, 8)
    ref_attn = hp.habana_fused_sdpa(q, k, v)

    blob = {
        "state_dict": model.state_dict(),
        "inputs": inputs,
        "ref_v": ref_v,
        "ref_a": ref_a,
        "q": q,
        "k": k,
        "v": v,
        "ref_attn": ref_attn,
    }
    torch.save(blob, workdir / "cp_ref.pt")
    return blob


def _parity_worker(rank: int, world: int, port: int, tmpdir: str):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world)
    dist.init_process_group("gloo", rank=rank, world_size=world)
    workdir = Path(tmpdir)
    report: dict = {"rank": rank, "checks": {}}

    try:
        from diffusers.models import attention_dispatch as ad
        from diffusers.models._modeling_parallel import (
            ContextParallelConfig,
            ParallelConfig,
        )
        from diffusers.hooks.context_parallel import EquipartitionSharder
        from diffusers.models.transformers.transformer_minimax_h3 import (
            MiniMaxH3Transformer3DModel,
        )
        from torch.distributed.device_mesh import init_device_mesh

        # (a) registry edit effective in a FRESH process. NOTE: the runner's
        # registration imports `hpu_port.hpu_patches` while this test imports
        # `hpu_patches` directly -- two module objects, so compare by
        # name/qualifications, not `is`.
        run_t2va._register_attention_backend()
        reg = ad._AttentionBackendRegistry
        backend_fn = reg._backends.get(hp.BACKEND_NAME)
        registered = (
            callable(backend_fn)
            and getattr(backend_fn, "__name__", "") == "habana_fused_sdpa"
        )
        cp_listed = hp.BACKEND_NAME in reg._supports_context_parallel
        report["checks"]["backend_registered"] = registered
        report["checks"]["backend_in_cp_set"] = cp_listed

        blob = torch.load(workdir / "cp_ref.pt", map_location="cpu", weights_only=True)

        mesh = init_device_mesh(
            "cpu", mesh_shape=(1, world), mesh_dim_names=("ring", "ulysses")
        )
        cp_cfg = ContextParallelConfig(ulysses_degree=world, mesh=mesh)
        cp_cfg.setup(rank, world, torch.device("cpu"), mesh)
        parallel_config = ParallelConfig(context_parallel_config=cp_cfg)

        # (b) backend-level CP math: seq-sharded q/k/v, Ulysses all-to-all
        # through the port backend's CPU passthrough, gather per _cp_plan
        # semantics (equipartition dim-1 split; all-gather the output).
        q, k, v = blob["q"], blob["k"], blob["v"]
        parts = [t.chunk(world, dim=1) for t in (q, k, v)]
        q_l, k_l, v_l = (parts[i][rank].contiguous() for i in range(3))
        out_local = hp.habana_fused_sdpa(q_l, k_l, v_l, _parallel_config=parallel_config)
        out_gathered = EquipartitionSharder.unshard(
            out_local, 1, cp_cfg._flattened_mesh
        )
        ref_attn = blob["ref_attn"]
        attn_delta = (out_gathered.float() - ref_attn.float()).abs().max().item()
        report["checks"]["attn_cp_delta"] = attn_delta
        report["checks"]["attn_cp_shape_ok"] = tuple(out_gathered.shape) == tuple(
            ref_attn.shape
        )

        # (c) the real model through the runner's actual CP setup helper
        # (mesh + enable_parallelism + the token-refiner un-stamp)
        model = MiniMaxH3Transformer3DModel(**DIT_CFG).eval()
        model.load_state_dict(blob["state_dict"])
        model.set_attention_backend(hp.BACKEND_NAME)  # validates enum + CP set
        run_t2va._cp_enable_parallelism(model, world, "gloo")
        refiner_unstamped = all(
            getattr(m.processor, "_parallel_config", None) is None
            for m in model.token_refiner.modules()
            if getattr(m, "processor", None) is not None
        )
        report["checks"]["refiner_unstamped"] = refiner_unstamped
        with torch.no_grad():
            out_v, out_a = model(**blob["inputs"])
        ref_v, ref_a = blob["ref_v"], blob["ref_a"]
        report["checks"]["model_delta_v"] = (out_v.float() - ref_v.float()).abs().max().item()
        report["checks"]["model_delta_a"] = (out_a.float() - ref_a.float()).abs().max().item()
    except Exception as exc:  # surface the failure into the parent's summary
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        (workdir / f"cp_parity_rank{rank}.json").write_text(json.dumps(report))
        dist.barrier()
        dist.destroy_process_group()


def test_cp_parity(tmp: Path) -> None:
    print("\n=== test 1: 2-rank gloo CP parity (backend routing + _cp_plan) ===")
    world = 2
    run_t2va._register_attention_backend()
    _build_reference(tmp)
    mp.spawn(
        _parity_worker,
        args=(world, _free_port(), str(tmp)),
        nprocs=world,
        join=True,
    )
    for rank in range(world):
        report = json.loads((tmp / f"cp_parity_rank{rank}.json").read_text())
        c = report.get("checks", {})
        if "error" in report:
            check(f"rank {rank} completed without error", False, report["error"])
            continue
        check(
            f"rank {rank}: backend registered + in _supports_context_parallel",
            c.get("backend_registered") and c.get("backend_in_cp_set"),
        )
        check(
            f"rank {rank}: token refiner un-stamped (replicated pre-packing compute)",
            c.get("refiner_unstamped", False),
        )
        check(
            f"rank {rank}: sharded-attention gather matches unsharded (allclose 1e-2)",
            c.get("attn_cp_shape_ok", False)
            and c.get("attn_cp_delta", 1e9) <= 1e-2,
            f"max_delta={c.get('attn_cp_delta')}",
        )
        check(
            f"rank {rank}: tiny DiT forward under enable_parallelism matches "
            "unsharded reference (allclose 1e-2)",
            c.get("model_delta_v", 1e9) <= 1e-2 and c.get("model_delta_a", 1e9) <= 1e-2,
            f"delta_v={c.get('model_delta_v')} delta_a={c.get('model_delta_a')}",
        )


def test_cp_spawn_plumbing(tmp: Path) -> None:
    print("\n=== test 2: parent/worker spawn plumbing (H3_CP_TEST=1, N=2) ===")
    out = tmp / "plumbing" / "h3_smoke.mp4"
    # The mp.spawn workers print to the REAL stdout (separate processes), so
    # the whole plumbing run -- parent + workers -- is captured by executing
    # this file's --plumbing-only branch as its own subprocess.
    proc = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--plumbing-only",
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "H3_DISABLE_HPU": "1", "TORCH_DEVICE_BACKEND_AUTOLOAD": "0"},
        cwd=str(HPU_PORT),
        timeout=300,
    )
    captured = proc.stdout + proc.stderr
    check(
        "plumbing subprocess (parent + 2 workers) exits 0",
        proc.returncode == 0,
        captured[-1500:] if proc.returncode else f"{len(captured)} chars",
    )
    check(
        "plan print carries the CP keys (cp / cp_seq_pad)",
        '"cp": 2' in captured and '"cp_seq_pad"' in captured,
    )
    for rank in range(2):
        path = out.parent / f"{out.stem}_cp_timings_rank{rank}.json"
        exists = path.exists()
        check(f"rank {rank} timing file written", exists, str(path))
        if exists:
            blob = json.loads(path.read_text())
            check(
                f"rank {rank} recorded plumbing timings + block records",
                blob.get("rank") == rank and blob.get("blocks"),
            )
    check(
        "parent merged the per-rank timings into a combined table",
        "CP timing merge" in captured and "FakeStep" in captured,
    )
    check(
        "worker logs carry the rank tag ([h3:rN]) on the combined console",
        captured.count("[h3:r0]") >= 1 and captured.count("[h3:r1]") >= 1,
    )


def _plumbing_only(out: Path) -> int:
    """--plumbing-only branch: run the CP parent/worker plumbing once, in this
    process as the parent (workers are mp.spawn children printing to the real
    stdout). Returns 0/1; the outer test asserts on this process's console."""
    os.environ["H3_CP_TEST"] = "1"
    rc = 1
    try:
        rc = run_t2va.main(
            ["--prompt", "cp plumbing smoke", "--cp", "2", "--out", str(out)]
        )
    finally:
        os.environ.pop("H3_CP_TEST", None)
    return 0 if rc == 0 else 1


def test_decode_subprocess(tmp: Path) -> None:
    print("\n=== test 3: decode-subprocess hatch (CPU eager decode) ===")
    torch.manual_seed(7)
    from diffusers import AutoencoderKLMiniMaxH3, AutoencoderKLMiniMaxH3Audio

    workdir = tmp / "hatch_workdir"
    (workdir / "vae").mkdir(parents=True)
    (workdir / "audio_vae").mkdir(parents=True)
    video_vae = AutoencoderKLMiniMaxH3(
        in_channels=3,
        out_channels=3,
        latent_channels=24,
        block_out_channels=(32, 32, 32, 32, 32, 32),
        layers_per_block=1,
        spatial_downsample_factors=(2, 2, 2, 2, 1, 1),
        temporal_downsample_factors=(1, 2, 2, 1, 1, 1),
        decoder_num_layers=2,
        decoder_num_attention_heads=2,
        decoder_attention_head_dim=32,
        decoder_num_register_tokens=2,
        decoder_ffn_mult=2,
    ).eval()
    audio_vae = AutoencoderKLMiniMaxH3Audio(
        encoder_dim=32,
        encoder_rates=(2, 4, 4, 5, 5),
        latent_dim=64,
        latent_channels=32,
        num_attention_heads=2,
        decoder_dim=128,
        decoder_rates=(5, 5, 2, 2, 2, 2, 2),
        sampling_rate=32000,
    ).eval()
    video_vae.save_pretrained(workdir / "vae", safe_serialization=True)
    audio_vae.save_pretrained(workdir / "audio_vae", safe_serialization=True)

    latents = torch.randn(1, 24, T_LAT, LAT_H, LAT_W)
    audio_latents = torch.randn(2, 32, AUDIO_LATENTS)
    blob_path = tmp / "hatch_blob.pt"
    frames_dir = tmp / "hatch_frames"
    wav_path = tmp / "hatch.wav"
    meta_path = tmp / "hatch_meta.json"
    torch.save(
        {
            "latents": latents,
            "audio_latents": audio_latents,
            "pixel_mean": [0.0, 0.0, 0.0],
            "pixel_std": [1.0, 1.0, 1.0],
        },
        blob_path,
    )
    proc = subprocess.run(
        [
            sys.executable,
            str(HPU_PORT / "decode_subproc.py"),
            "--blob",
            str(blob_path),
            "--workdir",
            str(workdir),
            "--frames-dir",
            str(frames_dir),
            "--wav",
            str(wav_path),
            "--meta",
            str(meta_path),
            "--device",
            "cpu",
        ],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "H3_DISABLE_HPU": "1",
            "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
            "PYTHONPATH": str(HPU_PORT.parent),
        },
        timeout=600,
    )
    check(
        "decode_subproc.py exits 0 on the CPU eager path",
        proc.returncode == 0,
        (proc.stdout + proc.stderr)[-1200:] if proc.returncode else proc.stdout.strip()[-160:],
    )
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    n_frames = len(list(frames_dir.glob("frame_*.png")))
    check(
        "frames + wav + meta written (22 frames = 17n+5, 32 kHz meta)",
        n_frames == PIX_FRAMES_EXPECTED
        and wav_path.exists()
        and meta.get("sampling_rate") == 32000
        and meta.get("num_frames") == PIX_FRAMES_EXPECTED,
        f"frames={n_frames} meta={meta}",
    )
    # parity: the subprocess frames must match the in-process decode step math
    import wave

    with torch.no_grad():
        video = video_vae.decode(latents, return_dict=False)[0].clamp(0, 1)
        audio = audio_vae.decode(audio_latents, return_dict=False)[0].float().permute(1, 0, 2)
    import numpy as np
    from PIL import Image

    frame_ref = (
        (video[0].permute(1, 2, 3, 0).numpy() * 255.0)
        .clip(0, 255)
        .astype("uint8")
    )
    saved = np.asarray(Image.open(frames_dir / "frame_000000.png"))
    check(
        "subprocess frame 0 matches the in-process decode math (uint8 exact)",
        saved.shape == frame_ref[0].shape and bool((saved == frame_ref[0]).all()),
        f"saved {saved.shape} ref {frame_ref[0].shape}",
    )
    with wave.open(str(wav_path), "rb") as wf:
        wav_frames = wf.getnframes()
        wav_rate = wf.getframerate()
    check(
        "wav geometry matches the audio decode (32 kHz, 40 latents x hop 800)",
        wav_rate == 32000 and wav_frames == AUDIO_LATENTS * 800,
        f"frames={wav_frames} rate={wav_rate}",
    )


def main() -> int:
    if "--plumbing-only" in sys.argv:
        raw = sys.argv[sys.argv.index("--plumbing-only") :]
        out_arg = raw[raw.index("--out") + 1] if "--out" in raw else "h3_smoke.mp4"
        out = Path(out_arg)
        out.parent.mkdir(parents=True, exist_ok=True)
        return _plumbing_only(out)
    with tempfile.TemporaryDirectory(prefix="h3_cp_") as td:
        tmp = Path(td)
        test_cp_parity(tmp)
        test_cp_spawn_plumbing(tmp)
        test_decode_subprocess(tmp)

    passed = sum(1 for _, ok, _ in RESULTS if ok)
    failed = len(RESULTS) - passed
    print("\n=== SUMMARY ===")
    print(f"PASS {passed} / {len(RESULTS)} (FAIL {failed})")
    if failed:
        print("Failures:")
        for name, ok, detail in RESULTS:
            if not ok:
                print(f"  - {name}" + (f" [{detail}]" if detail else ""))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
