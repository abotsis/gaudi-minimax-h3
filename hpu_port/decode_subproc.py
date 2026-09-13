#!/usr/bin/env python
"""Decode subprocess for the H3 HPU port (D2H-wedge escape hatch, PLAN_CP.md).

The lazy-mode launch thread can stop accepting enqueues after the giant
denoise graph drains (JoinPendingLaunchThread wedge, v50/51/52/58) -- the big
post-decode D2H never completes, silently, with plenty of free memory. Small
D2Hs and FRESH PROCESSES are reliable; the wedge only appears in the process
that ran the big lazy graph. So the parent saves the final latents to disk and
THIS script -- a fresh interpreter running the VAEs in EAGER mode
(PT_HPU_LAZY_MODE=0), or pure CPU -- decodes video + audio, writes
frames/frame_*.png + audio.wav + a small meta json, and exits. The parent then
muxes as usual. With --device cpu this is a plain eager torch decode, which is
the mode the CPU test exercises.

The decode math mirrors run_t2va.patch_decode_steps' two decode steps: fp16
autocast over the fp32 video VAE on hpu (strict fp32 audio), per-frame chunked
D2H on hpu, frames written straight from the clamped [0,1] tensor -- the
pipeline's `VideoProcessor` is created with do_normalize=False (decoders.py
0.40.0 "the processor must not denormalize a second time"), so the uint8
mapping is linear and `run_t2va.save_frames` produces the same PIL frames as
the in-process `postprocess_video(output_type="pil")` path.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="H3 decode subprocess (eager VAE decode).")
    p.add_argument("--blob", required=True, help=".pt with latents/audio_latents/pixel_mean/pixel_std")
    p.add_argument("--workdir", required=True, help="runner workdir (vae/ + audio_vae/ components)")
    p.add_argument("--frames-dir", required=True)
    p.add_argument("--wav", required=True)
    p.add_argument("--meta", required=True, help="output json {sampling_rate, num_frames}")
    p.add_argument("--device", choices=["cpu", "hpu"], default="cpu")
    p.add_argument(
        "--shard",
        type=int,
        default=None,
        help="sharded-farm mode: decode chunk i%%N of the temporal chunk loop "
        "(ownership/blend/tail handoff identical to the in-process sharded "
        "decode, which post-drain device work wedges -- v72/v73c).",
    )
    p.add_argument("--nshards", type=int, default=None, help="number of video shard workers (with --shard)")
    p.add_argument("--audio-shard", action="store_true", help="decode ONLY the audio (farm audio worker)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.device == "hpu":
        # Eager plugin pairing rule (see launch_h3.sh): LAZY/AUTOLOAD are read
        # at torch import; the parent ran lazy, so force eager BEFORE importing
        # torch. HABANA_VISIBLE_DEVICES is inherited from the parent rank.
        os.environ["PT_HPU_LAZY_MODE"] = "0"

    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    blob = torch.load(args.blob, map_location="cpu", weights_only=False)  # trusted local blob (own writer)
    from diffusers import AutoencoderKLMiniMaxH3, AutoencoderKLMiniMaxH3Audio

    workdir = Path(args.workdir)
    device = torch.device(args.device)
    frames_dir = Path(args.frames_dir)
    meta_path = Path(args.meta)

    if args.audio_shard:
        # Farm audio worker: decode audio only, write wav + meta + done marker
        # (the parent's export phase polls markers under .shard_tmp/).
        audio_vae = (
            AutoencoderKLMiniMaxH3Audio.from_pretrained(
                workdir / "audio_vae",
                dtype=torch.float32,
                low_cpu_mem_usage=True,
                local_files_only=True,
            )
            .to(device)
            .eval()
        )
        audio_latents = blob.get("audio_latents")
        sampling_rate = 32000
        if audio_latents is not None:
            audio_latents = audio_latents.to(device)
            a_mean = getattr(audio_vae.config, "latents_mean", None)
            if a_mean is not None:
                a_mean = torch.tensor(a_mean, device=device).view(1, -1, 1)
                a_std = torch.tensor(audio_vae.config.latents_std, device=device).view(1, -1, 1)
                audio_latents = audio_latents * a_std + a_mean
            with torch.no_grad():
                audio = audio_vae.decode(audio_latents, return_dict=False)[0]
            if audio.device.type == "hpu":
                audio = audio.float().cpu()
            audio = audio.float().permute(1, 0, 2)  # (1, 2, S), stereo as channels
            sampling_rate = int(audio_vae.config.sampling_rate)
            from hpu_port.run_t2va import save_audio_wav

            save_audio_wav(audio, sampling_rate, Path(args.wav))
        shard = args.shard if args.shard is not None else 7
        marker = frames_dir / ".shard_tmp" / f"done_r{shard}.json"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps({"rank": shard, "chunks": [], "audio": True}))
        meta_path.write_text(json.dumps({"sampling_rate": sampling_rate}))
        print(f"[h3] audio farm worker r{shard} done: wav -> {args.wav}", flush=True)
        return 0

    if args.shard is not None:
        # Farm video worker: decode chunk i (mod nshards) of the temporal
        # chunk loop with the SAME file-handoff protocol the in-process
        # sharded decode uses (tail files + done markers under
        # frames_dir/.shard_tmp). Fresh eager process => no post-drain wedge.
        import run_t2va

        run_t2va._ensure_torch()
        vae = (
            AutoencoderKLMiniMaxH3.from_pretrained(
                workdir / "vae", dtype=torch.float32, low_cpu_mem_usage=True, local_files_only=True
            )
            .to(device)
            .eval()
        )
        try:
            frames_saved, clip_s, total_s = run_t2va._decode_sharded_core(
                vae,
                blob["latents"],
                blob["pixel_mean"],
                blob["pixel_std"],
                frames_dir,
                args.shard,
                args.nshards or 1,
                device,
            )
        except Exception:
            # Allocator forensics: the 394 MB PT_DEVMEM failures (v74/v75)
            # need the pool state at the failure point.
            if device.type == "hpu":
                try:
                    f, t = torch.hpu.mem_get_info(0)
                    print(
                        f"[h3] FAIL forensics: free={f / 2 ** 30:.1f} GiB "
                        f"total={t / 2 ** 30:.1f} use_tiling={vae.use_tiling} "
                        f"tile_min={vae.tile_sample_min_height}x{vae.tile_sample_min_width} "
                        f"overlap={vae.tile_sample_min_overlap_height}/{vae.tile_sample_min_overlap_width} "
                        f"inputs={vae.decoder.num_register_tokens if hasattr(vae.decoder, 'num_register_tokens') else '?'}",
                        flush=True,
                    )
                except Exception as e:
                    print(f"[h3] forensics failed: {e}", flush=True)
            raise
        print(
            f"[h3] video farm worker r{args.shard}: {frames_saved} frames, "
            f"clip {clip_s:.1f}s, phase {total_s:.1f}s",
            flush=True,
        )
        return 0

    from hpu_port.run_t2va import save_audio_wav, save_frames

    vae = (
        AutoencoderKLMiniMaxH3.from_pretrained(
            workdir / "vae", dtype=torch.float32, low_cpu_mem_usage=True, local_files_only=True
        )
        .to(device)
        .eval()
    )
    audio_vae = (
        AutoencoderKLMiniMaxH3Audio.from_pretrained(
            workdir / "audio_vae",
            dtype=torch.float32,
            low_cpu_mem_usage=True,
            local_files_only=True,
        )
        .to(device)
        .eval()
    )

    # --- video decode (mirrors MiniMaxH3VideoDecodeStep) ------------------
    latents = blob["latents"].to(device)
    latents_mean = getattr(vae.config, "latents_mean", None)
    if latents_mean is not None:
        latents_mean = torch.tensor(latents_mean, device=device).view(1, -1, 1, 1, 1)
        latents_std = torch.tensor(vae.config.latents_std, device=device).view(1, -1, 1, 1, 1)
        z = latents * latents_std + latents_mean
    else:
        # Synthetic configs (CPU tests) may omit the normalization constants;
        # the real checkpoint always carries them.
        z = latents
    with torch.no_grad(), torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16 if device.type == "hpu" else torch.float16,
        enabled=device.type in ("cuda", "hpu"),
    ):
        video = vae.decode(z, return_dict=False)[0]
    pixel_mean = torch.tensor(blob["pixel_mean"], device=device).view(1, -1, 1, 1, 1)
    pixel_std = torch.tensor(blob["pixel_std"], device=device).view(1, -1, 1, 1, 1)
    video = (video.float() * pixel_std + pixel_mean).clamp(0, 1)
    # Chunked per-frame D2H on hpu (the one pattern that reliably completes);
    # a no-op passthrough on cpu.
    if video.device.type == "hpu" and video.ndim == 5:
        # video is (B, C, T, H, W): select the time index directly.
        t_dim = video.shape[2]
        video = torch.stack(
            [video[:, :, i].contiguous().cpu() for i in range(t_dim)], dim=2
        )
    if video.device.type == "hpu":
        torch.hpu.synchronize()
    save_frames(video[0], Path(args.frames_dir))
    num_frames = len(list(Path(args.frames_dir).glob("frame_*.png")))

    # --- audio decode (mirrors MiniMaxH3AudioDecodeStep, strict fp32) ------
    sampling_rate = 32000
    audio_latents = blob.get("audio_latents")
    if audio_latents is not None:
        audio_latents = audio_latents.to(device)
        a_mean = getattr(audio_vae.config, "latents_mean", None)
        if a_mean is not None:
            a_mean = torch.tensor(a_mean, device=device).view(1, -1, 1)
            a_std = torch.tensor(audio_vae.config.latents_std, device=device).view(1, -1, 1)
            audio_latents = audio_latents * a_std + a_mean
        with torch.no_grad():
            audio = audio_vae.decode(audio_latents, return_dict=False)[0]
        if audio.device.type == "hpu":
            audio = audio.float().cpu()
        audio = audio.float().permute(1, 0, 2)  # (1, 2, S), stereo as channels
        sampling_rate = int(audio_vae.config.sampling_rate)
        save_audio_wav(audio, sampling_rate, Path(args.wav))

    Path(args.meta).write_text(
        json.dumps({"sampling_rate": sampling_rate, "num_frames": num_frames})
    )
    print(
        f"[h3] decode subprocess done: {num_frames} frames -> {args.frames_dir}, "
        f"wav -> {args.wav} (device {device.type})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception:
        import traceback

        traceback.print_exc()
        sys.exit(1)
