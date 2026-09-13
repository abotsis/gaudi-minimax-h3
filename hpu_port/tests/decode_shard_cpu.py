#!/usr/bin/env python
"""CPU-only tests for the temporal-sharded VAE decode (_decode_sharded_core).

Hard guarantee: NO HPU work (H3_DISABLE_HPU=1, TORCH_DEVICE_BACKEND_AUTOLOAD=0
before torch imports; gloo over CPU tensors). Validates that the sharded
decode replicates `AutoencoderKLMiniMaxH3._decode`'s internal chunk loop
exactly with the FILE-BASED overlap handoff (no collectives -- post-drain
HCCL stalls on HPU, v72):

  test 1 (world=3, n_chunks=2): chunk 0 -> rank0, chunk 1 -> rank1, rank2
    chunkless (the CP=8 audio-rank case). All ranks share ONE frames_dir;
    frame PNGs written by the ranks must be BYTE-IDENTICAL to the stock
    single-process `vae.decode` of the same latents (fp32 CPU, same ops same
    order). Per-rank ownership verified via the done markers.
  test 2 (world=2): each rank owns one chunk AND consumes the other's tail
    through the shared scratch dir (both directions exercised).

Run:
  cd /root/src/h3/hpu_port
  LD_LIBRARY_PATH=/root/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/lib \
  TORCH_DEVICE_BACKEND_AUTOLOAD=0 ../../.venv/bin/python tests/decode_shard_cpu.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("H3_DISABLE_HPU", "1")  # hard CPU guarantee, BEFORE torch
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

HPU_PORT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HPU_PORT))

import torch
import torch.multiprocessing as mp

T_LAT = 12  # n_chunks = (12+3)//5 - 1 = 2 internal chunks
LAT_H = LAT_W = 4  # -> 64x64 pixels via the 16x decoder unpatchify


def _tiny_vae():
    from diffusers import AutoencoderKLMiniMaxH3

    return AutoencoderKLMiniMaxH3(
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


def _shard_worker(rank: int, world: int, tmp_str: str, z_path: str) -> None:
    try:
        torch.manual_seed(123)  # identical weights on every rank
        vae = _tiny_vae()
        import run_t2va

        run_t2va._ensure_torch()  # bind run_t2va's module-global torch (CPU)
        z_raw = torch.load(z_path, weights_only=True)
        pm = [0.485, 0.456, 0.406]
        ps = [0.229, 0.224, 0.225]
        frames_dir = Path(tmp_str) / f"shard_w{world}" / "frames"
        saved, clip_s, total_s = run_t2va._decode_sharded_core(
            vae, z_raw, pm, ps, frames_dir, rank, world, torch.device("cpu")
        )
        print(f"[w{world} r{rank}] saved {saved} frames in {total_s:.2f}s", flush=True)
    except BaseException:
        import traceback

        traceback.print_exc()
        raise


def _stock_reference_frames(vae, z_raw: torch.Tensor) -> torch.Tensor:
    """Stock full decode -> (3, T, H, W) uint8, the same math as the runner."""
    cfg = vae.config
    lm = getattr(cfg, "latents_mean", None)
    if lm is not None:
        z = z_raw * torch.tensor(cfg.latents_std).view(1, -1, 1, 1, 1) + torch.tensor(
            lm
        ).view(1, -1, 1, 1, 1)
    else:
        z = z_raw
    pm = torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1, 1)
    ps = torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1, 1)
    with torch.no_grad():
        video = vae.decode(z, return_dict=False)[0]
    return ((video.float() * ps + pm).clamp(0, 1) * 255.0).to(torch.uint8)[0]


def _compare(frames_dir: Path, ref: torch.Tensor, tag: str) -> bool:
    import numpy as np
    from PIL import Image

    paths = sorted(frames_dir.glob("frame_*.png"))
    if len(paths) != ref.shape[1]:
        print(f"FAIL {tag}: {len(paths)} frames on disk vs {ref.shape[1]} expected")
        return False
    for i, p in enumerate(paths):
        arr = np.asarray(Image.open(p)).transpose(2, 0, 1)  # HWC -> CHW
        if not (arr == ref[:, i].numpy()).all():
            d = np.abs(arr.astype(int) - ref[:, i].numpy().astype(int)).max()
            print(f"FAIL {tag}: frame {i} differs (max delta {d})")
            return False
    print(f"PASS {tag}: {len(paths)} frames byte-identical to stock decode")
    return True


def main() -> int:
    torch.manual_seed(123)
    failures = 0
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        vae_ref = _tiny_vae()
        z_raw = torch.randn(1, 24, T_LAT, LAT_H, LAT_W)
        z_path = tmp / "z_raw.pt"
        torch.save(z_raw, z_path)
        ref = _stock_reference_frames(vae_ref, z_raw)
        print(f"stock reference: {tuple(ref.shape)} (expect T = 2*17+5 = 39)")

        for world in (3, 2):
            print(f"\n=== sharded decode, world={world}, n_chunks=2 ===")
            mp.spawn(
                _shard_worker,
                args=(world, str(tmp), str(z_path)),
                nprocs=world,
                join=True,
            )
            # All ranks wrote into ONE shared frames dir (production layout);
            # the done markers record per-rank ownership.
            frames_dir = tmp / f"shard_w{world}" / "frames"
            owners = {}
            for r in range(world):
                m = frames_dir / ".shard_tmp" / f"done_r{r}.json"
                if m.exists():
                    owners[r] = json.loads(m.read_text())
            print(f"markers: {owners}")
            total = sum(v["frames"] for v in owners.values())
            if total != 39:
                print(f"FAIL world={world}: marker frames sum {total} != 39")
                failures += 1
                continue
            # Ownership: chunks i%world==r; chunk0 owner 17 frames, last-chunk
            # owner 17+5=22, chunkless 0.
            ok = all(
                v["chunks"] == [i for i in range(2) if i % world == r]
                for r, v in owners.items()
            )
            if not ok or not _compare(frames_dir, ref, f"world={world}"):
                failures += 1

    print(f"\n{'PASS' if failures == 0 else 'FAIL'} ({failures} failing configs)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
