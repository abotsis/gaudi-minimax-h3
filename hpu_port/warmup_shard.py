#!/usr/bin/env python
"""One-time recipe warmup for the temporal-sharded VAE decode (per rank).

Decodes one REAL-SHAPE video clip (ts+token_overlap latent frames at the
target resolution) and one real-shape audio latent blob on this rank's card,
under LAZY mode with the rank's recipe cache dir -- exactly the shapes and
mode the in-process sharded decode (H3_DECODE_SHARDED=1) will use. Run all 8
ranks concurrently once; afterwards the sharded decode phase costs ~one clip
instead of ~8 cold compiles, and no rank starves past the Synapse
no-progress watchdog while waiting for chunk 0 (v71 lesson).

Usage (from hpu_port, with the launcher env already sourced):
  HABANA_VISIBLE_DEVICES=$r PT_HPU_RECIPE_CACHE_CONFIG=.recipe_cache_rank$r,... \
      python warmup_shard.py --rank $r --workdir workdir/t2va --lat-h 30 --lat-w 54
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--rank", type=int, required=True)
    p.add_argument("--workdir", required=True)
    p.add_argument("--lat-h", type=int, default=30)
    p.add_argument("--lat-w", type=int, default=54)
    p.add_argument("--audio-len", type=int, default=207)
    args = p.parse_args()

    rank = args.rank
    # Per-rank recipe cache dir -- same derivation as _cp_worker_entry.
    recipes = os.environ.get("H3_RECIPES_DIR")
    if recipes:
        rank_dir = str(Path(recipes).with_name(Path(recipes).name + f"_rank{rank}"))
        os.environ["H3_RECIPES_DIR"] = rank_dir
        if "PT_HPU_RECIPE_CACHE_CONFIG" in os.environ:
            parts = os.environ["PT_HPU_RECIPE_CACHE_CONFIG"].split(",")
            parts[0] = rank_dir
            os.environ["PT_HPU_RECIPE_CACHE_CONFIG"] = ",".join(parts)

    import torch

    from diffusers import AutoencoderKLMiniMaxH3, AutoencoderKLMiniMaxH3Audio

    workdir = Path(args.workdir)
    dev = torch.device("hpu")
    vae = (
        AutoencoderKLMiniMaxH3.from_pretrained(
            workdir / "vae", dtype=torch.float32, low_cpu_mem_usage=True, local_files_only=True
        )
        .to(dev)
        .eval()
    )
    ts = int(vae.tokens_chunk_size)
    tok_ov = int(vae.token_overlap)
    z = torch.zeros(1, 24, ts + tok_ov, args.lat_h, args.lat_w, device=dev)
    with torch.no_grad(), torch.autocast(device_type="hpu", dtype=torch.bfloat16):
        clip = vae._decode_clip(z)
    torch.hpu.synchronize()
    print(f"[warmup r{rank}] video clip OK {tuple(clip.shape)}", flush=True)

    audio_vae = (
        AutoencoderKLMiniMaxH3Audio.from_pretrained(
            workdir / "audio_vae",
            dtype=torch.float32,
            low_cpu_mem_usage=True,
            local_files_only=True,
        )
        .to(dev)
        .eval()
    )
    a = torch.zeros(2, 32, args.audio_len, device=dev)
    with torch.no_grad():
        out = audio_vae.decode(a, return_dict=False)[0]
    torch.hpu.synchronize()
    print(f"[warmup r{rank}] audio OK {tuple(out.shape)}", flush=True)
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
