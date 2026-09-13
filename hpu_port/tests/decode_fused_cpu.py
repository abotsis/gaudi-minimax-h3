"""CPU parity test for _decode_fused_core (v93): the fused in-process decode
must be byte-identical to the stock AutoencoderKLMiniMaxH3._decode at any
world size.

Runs the fused core under torch.distributed gloo with multiple REAL processes
(the all_gather is a collective; threads are not enough to trust it).

Usage: TORCH_DEVICE_BACKEND_AUTOLOAD=0 ../.venv/bin/python tests/decode_fused_cpu.py
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # hpu_port
sys.path.insert(0, str(Path(__file__).resolve().parent))  # tests

import torch
import torch.multiprocessing as mp

import decode_shard_cpu as base


def _fused_worker(rank: int, world: int, tmp_str: str, z_path: str) -> None:
    try:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29611"
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world)
        import torch.distributed as dist

        dist.init_process_group("gloo", rank=rank, world_size=world)
        torch.manual_seed(123)  # identical weights on every rank
        vae = base._tiny_vae()
        import run_t2va

        run_t2va._ensure_torch()
        z_raw = torch.load(z_path, weights_only=True)
        pm = [0.485, 0.456, 0.406]
        ps = [0.229, 0.224, 0.225]
        frames_dir = Path(tmp_str) / f"fused_w{world}" / "frames"  # shared, production layout
        saved, clip_s, total_s = run_t2va._decode_fused_core(
            vae, z_raw, pm, ps, frames_dir, rank, world, torch.device("cpu")
        )
        print(f"[fused w{world} r{rank}] saved {saved} frames in {total_s:.2f}s", flush=True)
        dist.destroy_process_group()
    except BaseException:
        import traceback

        traceback.print_exc()
        raise


def main() -> int:
    torch.manual_seed(7)
    T_LAT, LAT_H, LAT_W = base.T_LAT, base.LAT_H, base.LAT_W
    z = torch.randn(1, 24, T_LAT, LAT_H, LAT_W)
    # Workers seed(123) BEFORE building the VAE; the reference must do the
    # same (z is passed by file, so the RNG order is vae-after-seed-123).
    torch.manual_seed(123)
    with tempfile.TemporaryDirectory() as tmpdir:
        z_path = Path(tmpdir) / "z.pt"
        torch.save(z, z_path)
        vae = base._tiny_vae()
        ref = base._stock_reference_frames(vae, z)  # (3, T_px, H, W) uint8
        ok = True
        for world in (2, 3):
            ctx = mp.get_context("spawn")
            procs = []
            for r in range(world):
                p = ctx.Process(target=_fused_worker, args=(r, world, str(tmpdir), str(z_path)))
                p.start()
                procs.append(p)
            for p in procs:
                p.join(timeout=180)
            if any(p.exitcode != 0 for p in procs):
                print(f"FAIL world={world}: worker exit codes {[p.exitcode for p in procs]}")
                ok = False
                continue
            ok = base._compare(Path(tmpdir) / f"fused_w{world}" / "frames", ref, f"fused world={world}") and ok
    print("FUSED-CPU PASS" if ok else "FUSED-CPU FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
