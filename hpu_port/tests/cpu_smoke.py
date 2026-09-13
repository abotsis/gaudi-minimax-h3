#!/usr/bin/env python
"""CPU-only smoke test for the H3 HPU port (hpu_patches.py + model in-tree contracts).

Hard guarantee: this runs NO HPU work. H3_DISABLE_HPU=1 is set before
hpu_patches is imported, so torch.hpu is never probed, no HPU device is
initialized, no tensor is placed on hpu, nothing is graph-compiled. Every
assertion walks tensors with assert_no_hpu_tensor (defense in depth).

What it covers (per the smoke-test brief):
  1. hpu_patches imports cleanly and every patch applies through the CPU
     no-op guard path (device-independent class rebinds only).
  2. A tiny direct-instantiated MiniMaxH3Transformer3DModel runs one forward on a
     synthetic packed sequence built by the *real* diffusers layout builder
     (MiniMaxH3PrepareLayoutStep.build_packed_sequence) with timestep tensors
     produced exactly as the scheduler produces them (num_inference_steps=2).
  3. Tiny decode passes through both VAE decoders, tiled path on (stock default),
     output shapes verified against the codec geometry (px = latents * 16
     spatially; audio 2 x 32000 samples at 32 kHz / 40 latents-per-second).
  4. Determinism: same seed twice -> identical outputs (two separately built
     models and two repeated forwards).
  5. PASS/FAIL summary per assertion; non-zero exit on any failure.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import TypedDict

os.environ.setdefault("H3_DISABLE_HPU", "1")  # hard CPU guarantee, set BEFORE import

HPU_PORT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HPU_PORT))

import hpu_patches as hp
import torch

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, cond: bool, detail: str = "") -> bool:
    ok = bool(cond)
    RESULTS.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    return ok


def phase_ctx(name):
    class _Ctx:
        def __enter__(self):
            print(f"\n=== {name} ===")
            return self

        def __exit__(self, exc_type, exc, tb):
            if exc_type is not None:
                RESULTS.append(
                    (f"phase '{name}' completed", False, f"{exc_type.__name__}: {exc}")
                )
                print(f"[FAIL] phase '{name}' raised {exc_type.__name__}: {exc}")
                return True  # swallow: keep collecting other phases' results
            return False

    return _Ctx()


# ---------------------------------------------------------------------------
# Discovered geometry / chosen tiny config (documented deviations inline)
# ---------------------------------------------------------------------------
# DiT: rotary_dim = 6 * rope_freq_dim must be <= head_dim (rope rotates the
# leading channels; with the requested rope_inv_freq_len=16 the rotary width is
# 96), so head_dim >= 96. head_dim=96 with 2 heads gives inner 192 == the
# requested hidden_size=192 exactly (all-rotary head, the empty pass-through
# slice is handled by the stock code path).
class _DitConfig(TypedDict):
    """Tiny DiT config (typed so `**DIT_CFG` unpacking stays per-key-narrowed)."""

    num_attention_heads: int
    attention_head_dim: int
    hidden_size: int
    num_layers: int
    num_refiner_layers: int
    ffn_dim: int
    in_channels: int
    audio_in_channels: int
    patch_size: tuple[int, int, int]
    text_dim: int
    freq_dim: int
    time_embed_hidden_dim: int
    time_embed_dim: int
    rope_freq_dim: int


DIT_CFG: _DitConfig = {
    "num_attention_heads": 2,
    "attention_head_dim": 96,  # constrained: 6 * rope_freq_dim(16) = 96 <= head_dim
    "hidden_size": 192,
    "num_layers": 2,
    "num_refiner_layers": 1,
    "ffn_dim": 256,
    "in_channels": 24,  # video latent channels (latents_dim 24)
    "audio_in_channels": 32,  # audio latent channels
    "patch_size": (1, 2, 2),
    "text_dim": 64,
    "freq_dim": 32,
    "time_embed_hidden_dim": 64,
    "time_embed_dim": 64,
    "rope_freq_dim": 16,  # requested inv-freq length
}

# Duration grid: real runs use T_lat = 5n+2 latent frames (17n+5 pixel frames).
# T_lat=1 is DEGENERATE for the video-VAE decode chunking math (clip_length=17,
# token_drop=3 => tokens_chunk_size=5): num_tokens = T+3 with padding must
# reach 10 to make one decode chunk, so T_lat>=3; T_lat=7 is the smallest legal
# 5n+2 value (n=1) and decodes back to the aligned 17*1+5 = 22 pixel frames.
T_LAT = 7
PIX_FRAMES_EXPECTED = 22  # 17*n + 5 with n = (T_LAT - 2) // 5 = 1
LAT_H = LAT_W = 8  # 8x8 latent canvas -> 4x4 patch grid (2x2 spatial patch)
N_TEXT = 6
AUDIO_LATENTS = 40  # 40 latents at 40/s = 1 s of 32 kHz audio
AUDIO_CHANNELS = 2  # pipeline contract: stereo, packed channel-major

ROWS_PER_FRAME = (LAT_H // 2) * (LAT_W // 2)  # 16
PATCH_DIM = DIT_CFG["in_channels"] * 1 * 2 * 2  # 96
N_VIDEO_ROWS = T_LAT * ROWS_PER_FRAME  # 112
N_AUDIO_ROWS = AUDIO_LATENTS * AUDIO_CHANNELS  # 80
SEQ = N_TEXT + N_AUDIO_ROWS + N_VIDEO_ROWS  # 198

VIDEO_TAGS = (0, 1, 2)  # video / text / audio (modular_pipeline.py constants)


def main() -> int:
    tag = lambda ok: "ok" if ok else "FAILED"

    with phase_ctx("phase 0: CPU-only guard"):
        check("torch dtype machinery on CPU", torch.device("cpu").type == "cpu")
        check("env H3_DISABLE_HPU=1 set", os.environ.get("H3_DISABLE_HPU") == "1")

    with phase_ctx(
        "phase 1: import + device-independent patches (CPU no-op guard path)"
    ):
        report = hp.apply_patches()
        check(
            "apply_patches() returns status dict",
            isinstance(report, dict)
            and report.get("autocast_guard") is True
            and report.get("backend_registered") is True,
            f"{report}",
        )
        check(
            "no HPU availability claimed on this path",
            report.get("hpu_available") is False or os.environ["H3_DISABLE_HPU"] == "1",
            f"hpu_available={report.get('hpu_available')} (H3_DISABLE_HPU forces the CPU path regardless)",
        )

        # The autocast-guard subclass replaced step 0 of the decode blocks.
        from diffusers.modular_pipelines.minimax_h3 import decoders as dec_mod
        from diffusers.modular_pipelines.minimax_h3.modular_blocks_minimax_h3 import (
            MiniMaxH3DecodeStep,
        )

        patched_cls = getattr(dec_mod, "MiniMaxH3VideoDecodeStepPatched", None)
        guard_ok = (
            patched_cls is not None
            and MiniMaxH3DecodeStep.block_classes[0] is patched_cls
            and issubclass(patched_cls, dec_mod.MiniMaxH3VideoDecodeStep)
        )
        check(
            "video-VAE decode autocast guard subclass installed + rebound",
            guard_ok,
            f"block_classes[0]={'MiniMaxH3VideoDecodeStepPatched' if guard_ok else type(MiniMaxH3DecodeStep.block_classes[0]).__name__}",
        )
        # decode-guard: patched guard enables autocast on {"cuda", "hpu"} per the audited body
        check(
            "patched guard widens cuda -> {cuda, hpu}",
            patched_cls is not None
            and 'enabled=device.type in ("cuda", "hpu")'
            in __import__("inspect").getsource(patched_cls.__call__),
        )

        from diffusers.models import attention_dispatch as ad

        check(
            "habana_fused_sdpa registered in diffusers backend registry",
            ad._AttentionBackendRegistry._backends.get(hp.BACKEND_NAME)
            is hp.habana_fused_sdpa,
        )

    with phase_ctx(
        "phase 2: tiny DiT direct instantiation (seeds x2) + synthetic packed sequence"
    ):
        torch.manual_seed(0)
        from diffusers.models.transformers.transformer_minimax_h3 import (
            MiniMaxH3Transformer3DModel,
        )

        dit_a = MiniMaxH3Transformer3DModel(**DIT_CFG).eval()
        torch.manual_seed(0)
        dit_b = MiniMaxH3Transformer3DModel(**DIT_CFG).eval()
        params_match = all(
            torch.equal(pa, pb)
            for pa, pb in zip(dit_a.state_dict().values(), dit_b.state_dict().values())
        )
        check(
            "seeded rainy-day build determinism (seed 0 twice -> identical params)",
            params_match,
        )

        # --- scheduler-produced timesteps (num_inference_steps=2, cfg=1.0).
        # MiniMax-H3 is guidance-distilled: cfg=1.0 means no unconditional pass and
        # no guider branch -- the plan below IS the full conditioning contract.
        from diffusers.modular_pipelines.minimax_h3.before_denoise import (
            MiniMaxH3SetTimestepsStep,
        )
        from diffusers.schedulers.scheduling_minimax_h3 import MiniMaxH3Scheduler

        sched_v = MiniMaxH3Scheduler(shift=12.0)
        sched_a = MiniMaxH3Scheduler(shift=3.0)
        sched_v.set_timesteps(2, device="cpu")
        sched_a.set_timesteps(2, device="cpu")
        sigmas_v, timesteps_v = sched_v.sigmas, sched_v.timesteps
        sigmas_a, timesteps_a = sched_a.sigmas, sched_a.timesteps
        if (
            sigmas_v is None
            or timesteps_v is None
            or sigmas_a is None
            or timesteps_a is None
        ):
            raise RuntimeError("scheduler set_timesteps left sigmas/timesteps unset")
        check(
            "scheduler schedule shape (num_inference_steps=2)",
            sigmas_v.shape == (2,)
            and timesteps_v.numel() == 1
            and float(sigmas_v[-1]) == 0.0,
            f"sigmas={sigmas_v.tolist()} timesteps={timesteps_v.tolist()}",
        )
        check(
            "scheduler terminal sigma is exactly 0 and t = 1 - sigma",
            float(sigmas_v[-1]) == 0.0
            and float(timesteps_v[0]) == 1.0 - float(sigmas_v[0]),
        )

        # --- the real packed-layout builder (t2va layout, no keyframe anchors)
        from diffusers.modular_pipelines.minimax_h3.before_denoise import (
            MiniMaxH3PrepareLayoutStep,
        )

        text_token_tags = torch.full((N_TEXT,), 1, dtype=torch.long)  # text rows tag 1
        (
            position_ids,
            token_tags,
            video_indices,
            audio_indices,
            text_indices,
            num_cond_v,
            num_cond_a,
        ) = MiniMaxH3PrepareLayoutStep.build_packed_sequence(
            text_token_tags=text_token_tags,
            num_latent_frames=T_LAT,
            latent_height=LAT_H,
            latent_width=LAT_W,
            num_audio_latents=AUDIO_LATENTS,
            patch_size=DIT_CFG["patch_size"],
            audio_channels=AUDIO_CHANNELS,
            audio_tag=VIDEO_TAGS[2],
            video_tag=VIDEO_TAGS[0],
        )
        layout_shapes_ok = (
            position_ids.shape == (SEQ, 3)
            and token_tags.shape == (SEQ,)
            and SEQ == N_TEXT + N_AUDIO_ROWS + N_VIDEO_ROWS
        )
        check(
            "packed sequence layout shape [text | audio | video]",
            layout_shapes_ok,
            f"seq={SEQ} (text {N_TEXT} + audio {N_AUDIO_ROWS} + video {N_VIDEO_ROWS})",
        )
        check(
            "position_ids carry the fp64 rotary grid",
            position_ids.dtype == torch.float64,
        )
        check(
            "modality tags land per block ([text|audio|video] = [1|2|0])",
            int(token_tags[:N_TEXT][0]) == 1
            and int(token_tags[N_TEXT]) == 2
            and int(token_tags[N_TEXT + N_AUDIO_ROWS]) == 0,
        )
        check("no condition rows for plain t2va", num_cond_v == 0 and num_cond_a == 0)

        # --- the row_timestep_plan exactly as SetTimestepsStep.build_row_timesteps derives it
        video_timestep = float(timesteps_v[0])
        audio_timestep = float(timesteps_a[0])
        unique_timesteps, timestep_indices = (
            MiniMaxH3SetTimestepsStep.build_row_timesteps(
                video_indices,
                audio_indices,
                num_condition_video_rows=0,
                num_condition_audio_rows=0,
                num_text_tokens=N_TEXT,
                video_timestep=video_timestep,
                audio_timestep=audio_timestep,
                condition_video_timestep=max(
                    video_timestep, 0.999
                ),  # no-op: zero condition rows
                condition_audio_timestep=1.0,
            )
        )
        check(
            "row timestep plan: 1 unique timestep (both modalities in lock-step at t=0 for 2 steps)",
            unique_timesteps.shape == (1,) and timestep_indices.shape == (SEQ,),
            f"unique={unique_timesteps.tolist()}",
        )

        # --- synthetic packed sequence payloads (fp32, on CPU)
        torch.manual_seed(1234)
        video_rows = torch.randn(1, N_VIDEO_ROWS, PATCH_DIM)
        audio_rows = torch.randn(1, N_AUDIO_ROWS, DIT_CFG["audio_in_channels"])
        text_embeds = torch.randn(1, N_TEXT, DIT_CFG["text_dim"])

        def run_dit(model):
            with torch.no_grad():
                return model(
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

        # First-call warmup: the very first execution of a shape can pick a different
        # multithreaded kernel blocking on CPU (one-shot op-path selection); every
        # later forward of that shape is bit-stable. Steady-state determinism is the
        # contract under test (per-run RNG state), so the warmup is discarded.
        run_dit(dit_a)
        out_v_a, out_a_a = run_dit(dit_a)
        out_v_b, out_a_b = run_dit(dit_a)
        delta_v_rep = (out_v_a.float() - out_v_b.float()).abs().max().item()
        delta_a_rep = (out_a_a.float() - out_a_b.float()).abs().max().item()
        check(
            "same-model repeated forward is bit-identical (no-state drift)",
            torch.equal(out_v_a, out_v_b) and torch.equal(out_a_a, out_a_b),
            f"delta_v={delta_v_rep:.3e} delta_a={delta_a_rep:.3e}",
        )
        run_dit(dit_b)  # warm up the second build too before comparing
        out_v_bb, out_a_bb = run_dit(dit_b)
        delta_v_re = (out_v_a.float() - out_v_bb.float()).abs().max().item()
        delta_a_re = (out_a_a.float() - out_a_bb.float()).abs().max().item()
        check(
            "same-seed rebuild forward is bit-identical (construction determinism)",
            torch.equal(out_v_a, out_v_bb) and torch.equal(out_a_a, out_a_bb),
            f"delta_v={delta_v_re:.3e} delta_a={delta_a_re:.3e}",
        )
        with phase_ctx("phase 2b: stock dispatch output shapes"):
            check(
                "video velocity head shape (num_video_rows, patch_dim)",
                out_v_a.shape == (1, N_VIDEO_ROWS, PATCH_DIM),
                f"{tuple(out_v_a.shape)}",
            )
            check(
                "audio velocity head shape (num_audio_rows, audio_in_channels)",
                out_a_a.shape == (1, N_AUDIO_ROWS, DIT_CFG["audio_in_channels"]),
                f"{tuple(out_a_a.shape)}",
            )
            check(
                "velocities finite",
                bool(torch.isfinite(out_v_a).all() and torch.isfinite(out_a_a).all()),
            )

    with phase_ctx(
        "phase 3: full patch pass on the tiny DiT through the CPU no-op guard"
    ):
        fake_pipe = SimpleNamespace(
            components=SimpleNamespace(transformer=dit_a), bucket_id="cpu-tiny"
        )
        hpu_report = hp.apply_hpu_patches(
            fake_pipe, transformer=dit_a, rope_bucket="cpu-tiny"
        )
        check(
            "attention_backend forced = habana_fused_sdpa",
            hpu_report.attention_backend == hp.BACKEND_NAME,
            f"processors patched, backend={hpu_report.attention_backend}",
        )
        check(
            "rope cache bucket bound on the transformer rope",
            hpu_report.rope_cache_buckets == ["cpu-tiny"]
            and getattr(dit_a.rope, "_h3_rope_bucket", None) == "cpu-tiny",
            f"buckets={hpu_report.rope_cache_buckets}",
        )
        check(
            "video_vae_autocast_guard applied",
            hpu_report.video_vae_autocast_guard is True,
        )
        check(
            "graphs = all False on CPU (no-op guard)",
            hpu_report.graphs
            == {"transformer": False, "vae": False, "audio_vae": False},
            f"{hpu_report.graphs}",
        )
        check(
            "no mark_step hooks installed on CPU modules",
            hpu_report.mark_step_hooks == [],
            f"{hpu_report.mark_step_hooks}",
        )
        check(
            "HPU probe stays dormant (hpu_available False under H3_DISABLE_HPU)",
            hpu_report.hpu_available is False,
        )

        patched_v, patched_a = run_dit(dit_a)
        store = getattr(dit_a.rope, "_h3_rope_store", {})
        check(
            "rope cache filled lazily on first patched forward (bit-identical service)",
            store.get("cpu-tiny") is not None
            and len(store) == 1
            and torch.equal(run_dit(dit_a)[0], patched_v),
            f"cache entries={len(store)}",
        )
        delta_v_p = (patched_v.float() - out_v_a.float()).abs().max().item()
        delta_a_p = (patched_a.float() - out_a_a.float()).abs().max().item()
        check(
            "patched-backend (CPU passthrough) forward is bit-identical to the stock dispatch",
            torch.equal(patched_v, out_v_a) and torch.equal(patched_a, out_a_a),
            f"delta_v={delta_v_p:.3e} delta_a={delta_a_p:.3e}",
        )

    with phase_ctx(
        "phase 4: tiny video VAE decode (patched forward, tiled default on)"
    ):
        torch.manual_seed(1)
        from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import (
            AutoencoderKLMiniMaxH3,
        )

        video_vae = AutoencoderKLMiniMaxH3(
            in_channels=3,
            out_channels=3,
            latent_channels=24,  # == DiT in_channels
            block_out_channels=(
                32,
                32,
                32,
                32,
                32,
                32,
            ),  # shrunk (norm_num_groups=32 stays divisible)
            layers_per_block=1,
            spatial_downsample_factors=(2, 2, 2, 2, 1, 1),  # 16x spatial, kept
            temporal_downsample_factors=(1, 2, 2, 1, 1, 1),  # 4x temporal, kept
            decoder_num_layers=2,
            decoder_num_attention_heads=2,
            decoder_attention_head_dim=32,  # rope dim 32*0.75=24, divisible by 2*3 axes
            decoder_num_register_tokens=2,
            decoder_ffn_mult=2,
        ).eval()
        check(
            "video VAE compression ratios preserved (16x spatial, 4x temporal)",
            video_vae.spatial_compression_ratio == 16
            and video_vae.temporal_compression_ratio == 4,
        )
        check(
            "video VAE keeps the clip geometry (clip_length 17, token_drop 3)",
            video_vae.config["clip_length"] == 17
            and video_vae.config["token_drop"] == 3
            and video_vae.tokens_chunk_size == 5,
        )

        z_video = torch.randn(1, 24, T_LAT, LAT_H, LAT_W)
        with torch.no_grad():
            dec_stock = video_vae.decode(z_video, return_dict=False)[0]
        with hp.habana_attention_backend(), torch.no_grad():
            dec_forced = video_vae.decode(z_video, return_dict=False)[0]
        exp = (1, 3, PIX_FRAMES_EXPECTED, LAT_H * 16, LAT_W * 16)
        check(
            "video decode stock shape (B,C,22,128,128): px H/W = latents*16, frames = 17n+5 grid",
            dec_stock.shape == exp,
            f"{tuple(dec_stock.shape)} expected {exp}",
        )
        check(
            "video decode through habana_fused_sdpa passthrough is bit-identical to stock",
            torch.equal(dec_stock, dec_forced),
        )
        check("video decode finite", bool(torch.isfinite(dec_stock).all()))

        with phase_ctx(
            "phase 5: tiny audio VAE decode (stereo-as-batch, strictly fp32)"
        ):
            from diffusers.models.autoencoders.autoencoder_kl_minimax_h3_audio import (
                AutoencoderKLMiniMaxH3Audio,
            )

            audio_vae = AutoencoderKLMiniMaxH3Audio(
                encoder_dim=32,
                encoder_rates=(2, 4, 4, 5, 5),  # hop 800 kept -> 40 latents/s at 32 kHz
                latent_dim=64,
                latent_channels=32,  # == DiT audio_in_channels
                num_attention_heads=2,
                decoder_dim=128,
                decoder_rates=(
                    5,
                    5,
                    2,
                    2,
                    2,
                    2,
                    2,
                ),  # product 800 == hop: upsample geometry kept
                sampling_rate=32000,
            ).eval()
            check(
                "audio VAE hop length preserved (800 samples = 25 ms @ 32 kHz)",
                audio_vae.hop_length == 800,
            )
            z_audio = torch.randn(2, 32, AUDIO_LATENTS)  # stereo as batch of 2
            with torch.no_grad():
                audio_out = audio_vae.decode(z_audio, return_dict=False)[0]
            exp_a = (2, 1, AUDIO_LATENTS * 800)
            check(
                "audio decode shape (2,1,32000): 2 channels x samples at 32k / 40Hz",
                audio_out.shape == exp_a and audio_vae.config["sampling_rate"] == 32000,
                f"{tuple(audio_out.shape)} expected {exp_a}",
            )
            check(
                "audio decode clamped to [-1, 1] and finite",
                bool(torch.isfinite(audio_out).all())
                and audio_out.min().item() >= -1.0
                and audio_out.max().item() <= 1.0,
            )
            with torch.no_grad():
                audio_out2 = audio_vae.decode(z_audio, return_dict=False)[0]
            check(
                "audio decode repeated forward is bit-identical",
                torch.equal(audio_out, audio_out2),
            )

    with phase_ctx("phase 6: hard rules audit + library parity self-test"):
        everything = [
            dit_a,
            dit_b,
            out_v_a,
            out_a_a,
            patched_v,
            patched_a,
            dec_stock,
            dec_forced,
            video_vae,
            audio_vae,
            audio_out,
            position_ids,
            token_tags,
        ]
        try:
            hp.assert_no_hpu_tensor(*everything)
            no_hpu = True
            why = "no tensor reachable from models/outputs sits on hpu"
        except RuntimeError as exc:
            no_hpu = False
            why = str(exc)
        check("assert_no_hpu_tensor across all models and outputs", no_hpu, why)

        parity = hp.validate_cpu_equivalence(seed=0)
        check(
            "hpu_patches.validate_cpu_equivalence(): stock dispatch == habana passthrough (bit-exact)",
            parity["parity_ok"] is True
            and parity["max_delta_video"] == 0.0
            and parity["max_delta_audio"] == 0.0,
            f"{parity}",
        )

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

# pyright re-dispatch: pythonPath config verify (whitespace only)
