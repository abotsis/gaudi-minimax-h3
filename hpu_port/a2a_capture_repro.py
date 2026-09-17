"""Minimal repro: all_to_all inside HPU graph capture, 2 ranks (h3expert matrix).

Isolates the v122 abort ("Split sizes doesn't match total dim 0 size" from
torch's checkSplitSizes on the HCCL lazy-collective job thread). Predictions:
  (a) fresh 1-D, funcol(None,None)            -> pass
  (b) 5-D base dim0=world, x=base.flatten()   -> ABORT (diffusers path)
  (c) x=base.flatten().clone()                -> pass (materialized base)
  (d) classic dist.all_to_all_single(out,base) -> pass (no split lists anywhere)
  (b) in plain eager (no capture)             -> ABORT pins lazy-collectives, not capture
"""
import argparse
import os

import torch
import torch.distributed as dist
from torch.distributed import _functional_collectives as funcol


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rank", type=int, required=True)
    p.add_argument("--case", required=True)  # a | b | c | d | b_eager
    args = p.parse_args()

    os.environ["HABANA_VISIBLE_MODULES"] = str(args.rank + 4)  # cards 4,5 free
    dist.init_process_group("hccl", rank=args.rank, world_size=2)
    torch.hpu.set_device(0)
    import habana_frameworks.torch.core as htcore

    world, S, B, HL, D = 2, 128, 1, 8, 64
    N = world * S

    def a2a_funcol(x):
        out = funcol.all_to_all_single(x, None, None, dist.group.WORLD)
        return out

    def a2a_classic(x):
        out = torch.empty_like(x)
        dist.all_to_all_single(out, x, group=dist.group.WORLD)  # no split lists
        return out

    if args.case == "a":
        x = torch.arange(N, dtype=torch.bfloat16, device="hpu")
        op = a2a_funcol
    elif args.case in ("b", "c"):
        base = torch.arange(world * S * B * HL * D, dtype=torch.bfloat16, device="hpu").reshape(world, S, B, HL, D)
        x = base.flatten(0, 1)  # VIEW, dim0 = N
        if args.case == "c":
            x = x.clone()
        op = a2a_funcol
    elif args.case == "d":
        base = torch.arange(world * S * B * HL * D, dtype=torch.bfloat16, device="hpu").reshape(world, S, B, HL, D)
        x = base  # dim0 = world, contiguous base
        op = a2a_classic
    else:  # e/f/g: input DERIVED inside the capture, real pipeline style
        base = torch.arange(world * S * B * HL * D, dtype=torch.bfloat16, device="hpu").reshape(B, S, world, HL, D)
        op_pre = None
        if args.case == "e":
            op_pre = lambda t: t.reshape(B, S, world, HL, D).permute(2, 1, 0, 3, 4).contiguous()  # (world,S,B,HL,D) in-capture
        elif args.case == "f":
            op_pre = lambda t: t.reshape(B, S, world, HL, D).permute(2, 1, 0, 3, 4).contiguous()
        else:  # g: big real-size tensor, in-capture derived
            S = 8724
            base = torch.zeros(B, S, world, HL, D, dtype=torch.bfloat16, device="hpu")
            op_pre = lambda t: t.reshape(B, S, world, HL, D).permute(2, 1, 0, 3, 4).contiguous()
        x = base  # op applied inside run()

        def run():
            t = op_pre(x) if op_pre is not None else x
            if args.case in ("f", "g"):
                htcore.mark_step()  # vLLM discipline: mark_step before collective
            return a2a_funcol(t.flatten(0, 1))  # flatten INSIDE capture, diffusers-style

    def run_outer():
        return run()

    if args.case == "b_eager":
        y = run_outer()
        htcore.mark_step()
        torch.hpu.synchronize()
    else:
        for _ in range(2):
            run_outer()
        htcore.mark_step()
        torch.hpu.synchronize()
        g = torch.hpu.HPUGraph()
        with torch.hpu.graph(g):
            y = run_outer()
        torch.hpu.synchronize()  # capture output lands in y's storage

    host = y.cpu()
    print(f"[rank{args.rank}] case={args.case} OK; out {tuple(y.shape)} first={host.flatten()[0].item()} last={host.flatten()[-1].item()}")


if __name__ == "__main__":
    main()
