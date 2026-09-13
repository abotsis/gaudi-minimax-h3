#!/usr/bin/env python
"""te_stream_probe.py — hybrid TE streaming probe (NO HPU graphs).

Question: can the Qwen3-VL-32B text encoder prefill run on one Gaudi2 card
eagerly (lazy PT bridge, no HPU graph capture) with layer weights STREAMED
H2D instead of kept resident (~63 GB bf16), and what would the full-prefill
time be vs the CPU TE baseline (~89 s @ 32 threads)?

Stages:
  0. device sanity
  1. H2D bandwidth: pinned 1 GB copy, steady state
  2. real-weight page-in: safetensors shard get_tensor for one layer (~1 GB)
  3. eager layer forward: hand-rolled Qwen3-VL decoder layer (exact config
     dims), bf16, seq=512, lazy mark_step — NO wrap_in_hpu_graph anywhere
  4. streaming loop: copy_ real layer weights into the resident module,
     forward, per-layer (copy+compute) timing over N layers
  5. CPU baseline: identical layer math @ torch threads
  6. projection: 64 layers + embedding overhead estimate

Weight names verified against MiniMax-H3/FL2VA/text_encoder
(model.language_model.layers.{i}.*).
"""

import json
import os
import sys
import time
from pathlib import Path

import torch

TE_DIR = Path(os.environ.get(
    "TE_DIR",
    "/root/src/h3/MiniMax-H3/FL2VA/text_encoder",
))
SEQ = int(os.environ.get("TE_PROBE_SEQ", "512"))
LAYERS_PROBED = int(os.environ.get("TE_PROBE_LAYERS", "8"))
ITERS = int(os.environ.get("TE_PROBE_ITERS", "5"))
CPU_THREADS = int(os.environ.get("TE_PROBE_CPU_THREADS", "32"))

H = 5120
NQ = 64 * 128  # q width: 64 heads x 128 head_dim
NKV = 8 * 128  # kv width: 8 kv heads x 128
FFN = 25600
HDIM = 128
N_LAYERS = 64

# CPU baseline documented in PLAN_CP.md Phase 5/6 (fresh prompt, 32 threads)
CPU_TE_BASELINE_S = float(os.environ.get("TE_CPU_BASELINE_S", "89"))


def log(msg: str) -> None:
    print(f"[probe] {msg}", flush=True)


def gb(x: float) -> str:
    return f"{x / 1e9:.2f} GB"


# ---------------------------------------------------------------- layer math
def rope_cos_sin(seq: int, device, dtype):
    pos = torch.arange(seq, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (
        10000 ** (torch.arange(0, HDIM, 2, device=device, dtype=torch.float32) / HDIM)
    )
    angles = torch.outer(pos, inv_freq)  # [seq, HDIM/2]
    cos = angles.cos().to(dtype)
    sin = angles.sin().to(dtype)
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [1, heads, seq, hdim]; cos/sin: [seq, hdim/2]
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    c = cos[None, None]  # [1,1,seq,hdim/2]
    s = sin[None, None]
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)


class DecoderLayer(torch.nn.Module):
    """Qwen3-VL text decoder layer (bias-free, GQA, per-head q/k rmsnorm)."""

    def __init__(self):
        super().__init__()
        self.q_proj = torch.nn.Linear(H, NQ, bias=False)
        self.k_proj = torch.nn.Linear(H, NKV, bias=False)
        self.v_proj = torch.nn.Linear(H, NKV, bias=False)
        self.o_proj = torch.nn.Linear(NQ, H, bias=False)
        self.q_norm = torch.nn.RMSNorm(HDIM, eps=1e-6)
        self.k_norm = torch.nn.RMSNorm(HDIM, eps=1e-6)
        self.input_layernorm = torch.nn.RMSNorm(H, eps=1e-6)
        self.post_attention_layernorm = torch.nn.RMSNorm(H, eps=1e-6)
        self.gate_proj = torch.nn.Linear(H, FFN, bias=False)
        self.up_proj = torch.nn.Linear(H, FFN, bias=False)
        self.down_proj = torch.nn.Linear(FFN, H, bias=False)

    def forward(self, x, cos, sin):
        r = x
        h = self.input_layernorm(x)
        q = self.q_proj(h).view(1, SEQ, 64, HDIM).transpose(1, 2)
        k = self.k_proj(h).view(1, SEQ, 8, HDIM).transpose(1, 2)
        v = self.v_proj(h).view(1, SEQ, 8, HDIM).transpose(1, 2)
        q = apply_rope(self.q_norm(q), cos, sin)
        k = apply_rope(self.k_norm(k), cos, sin)
        k_full = k.repeat_interleave(64 // 8, dim=1)
        v_full = v.repeat_interleave(64 // 8, dim=1)
        a = torch.nn.functional.scaled_dot_product_attention(q, k_full, v_full, is_causal=True)
        a = a.transpose(1, 2).reshape(1, SEQ, NQ)
        x = r + self.o_proj(a)
        r = x
        x = self.post_attention_layernorm(x)
        x = self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))
        return r + x


def make_layer(device, dtype=torch.bfloat16) -> DecoderLayer:
    torch.manual_seed(0)
    layer = DecoderLayer().to(dtype)
    for p in layer.parameters():
        torch.nn.init.normal_(p, std=0.02)
    return layer.to(device)


SUFFIXES = [
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "self_attn.k_norm.weight",
    "self_attn.k_proj.weight",
    "self_attn.o_proj.weight",
    "self_attn.q_norm.weight",
    "self_attn.q_proj.weight",
    "self_attn.v_proj.weight",
    "mlp.down_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
]


INDEX = json.load(open(TE_DIR / "model.safetensors.index.json"))


def load_real_layer_cpu(layer_idx: int, pinned_bufs: dict | None = None) -> dict:
    from safetensors import safe_open

    prefix = f"model.language_model.layers.{layer_idx}."
    wm = INDEX["weight_map"]
    # a layer's tensors may span multiple shards — group suffixes per shard
    by_shard: dict[str, list[str]] = {}
    for suffix in SUFFIXES:
        by_shard.setdefault(wm[prefix + suffix], []).append(suffix)
    tensors = {}
    for shard_file, suffixes in by_shard.items():
        with safe_open(TE_DIR / shard_file, framework="pt", device="cpu") as f:
            for suffix in suffixes:
                t = f.get_tensor(prefix + suffix)
                if pinned_bufs is not None:
                    buf = pinned_bufs.get(suffix)
                    if buf is None:
                        buf = torch.empty_like(t).pin_memory()
                        pinned_bufs[suffix] = buf
                    buf.copy_(t)  # page-cache -> pinned staging
                    t = buf
                tensors[suffix] = t
    return tensors


def copy_into_module(layer: DecoderLayer, tensors: dict) -> None:
    with torch.no_grad():
        getattr(layer.input_layernorm, "weight").copy_(tensors["input_layernorm.weight"])
        getattr(layer.post_attention_layernorm, "weight").copy_(
            tensors["post_attention_layernorm.weight"]
        )
        getattr(layer.q_norm, "weight").copy_(tensors["self_attn.q_norm.weight"])
        getattr(layer.k_norm, "weight").copy_(tensors["self_attn.k_norm.weight"])
        layer.q_proj.weight.copy_(tensors["self_attn.q_proj.weight"])
        layer.k_proj.weight.copy_(tensors["self_attn.k_proj.weight"])
        layer.v_proj.weight.copy_(tensors["self_attn.v_proj.weight"])
        layer.o_proj.weight.copy_(tensors["self_attn.o_proj.weight"])
        layer.gate_proj.weight.copy_(tensors["mlp.gate_proj.weight"])
        layer.up_proj.weight.copy_(tensors["mlp.up_proj.weight"])
        layer.down_proj.weight.copy_(tensors["mlp.down_proj.weight"])


def layer_bytes() -> int:
    return (
        2 * H * NQ + 2 * H * NKV + NQ * H + FFN * H + H * FFN + FFN * H
        + HDIM * 2 + H * 2
    ) * 2  # bf16


def main() -> None:
    import habana_frameworks.torch as ht  # noqa: F401  (backend registration)

    log(f"torch={torch.__version__} seq={SEQ} layers_probed={LAYERS_PROBED}")
    log(f"hpu available={torch.hpu.is_available()} count={torch.hpu.device_count()}")

    # ---- stage 0: sanity ---------------------------------------------------
    free, total = torch.hpu.mem_get_info()
    log(f"mem: free={gb(free)} total={gb(total)}")

    idx = json.load(open(TE_DIR / "model.safetensors.index.json"))
    shard0 = idx["weight_map"]["model.language_model.layers.0.self_attn.q_proj.weight"]
    log(f"layer0 shard: {shard0}")
    # ---- stage 1: H2D bandwidth --------------------------------------------
    n = 536870912  # ~1 GB bf16
    src = torch.randn(n, dtype=torch.bfloat16, device="cpu")
    src = src.pin_memory()
    dst = torch.empty_like(src, device="hpu")
    torch.hpu.synchronize()
    times = []
    for _ in range(4):
        t0 = time.perf_counter()
        dst.copy_(src, non_blocking=True)
        torch.hpu.synchronize()
        times.append(time.perf_counter() - t0)
    steady = times[-1]
    nbytes = n * 2
    log(f"H2D pinned {gb(nbytes)}: {steady*1e3:.1f} ms -> {nbytes/steady/1e9:.1f} GB/s (first={times[0]*1e3:.1f} ms)")
    del src, dst

    # ---- stage 2: real-weight page-in --------------------------------------
    t0 = time.perf_counter()
    tensors0 = load_real_layer_cpu(0)
    page_in = time.perf_counter() - t0
    nbytes = sum(t.numel() * t.element_size() for t in tensors0.values())
    log(f"real layer-0 page-in from mmap shard: {page_in*1e3:.1f} ms ({gb(nbytes)}, expected ~{gb(layer_bytes())})")

    # ---- stage 3: eager layer forward (NO graphs) ---------------------------
    layer = make_layer("hpu")
    cos, sin = rope_cos_sin(SEQ, "hpu", torch.bfloat16)
    x = torch.randn(1, SEQ, H, device="hpu", dtype=torch.bfloat16)
    torch.hpu.synchronize()
    fw_times = []
    for i in range(3 + ITERS):
        torch.hpu.synchronize()
        t0 = time.perf_counter()
        y = layer(x, cos, sin)
        torch.hpu.synchronize()
        dt = time.perf_counter() - t0
        if i >= 3:
            fw_times.append(dt)
        else:
            log(f"layer forward warmup {i}: {dt*1e3:.1f} ms (first includes per-op lazy compile)")
    fw = sum(fw_times) / len(fw_times)
    log(f"layer forward steady: {fw*1e3:.2f} ms (min {min(fw_times)*1e3:.2f} max {max(fw_times)*1e3:.2f})")
    log(f"layer forward FLOPs ~ {2*SEQ*(2*H*NQ+2*H*NKV+2*NQ*H+3*FFN*H)/1e12:.1f} TFLOP -> {2*SEQ*(2*H*NQ+2*H*NKV+2*NQ*H+3*FFN*H)/fw/1e12:.0f} TFLOPS eff.")

    # ---- stage 4: streaming loop (real weights, resident=1 layer) ----------
    # pinned staging: page-cache -> pinned memcpy, then pinned -> HPU
    pinned: dict = {}
    stream_times = []
    for i in range(LAYERS_PROBED):
        # fresh page-in each layer, index-resolved shard (page-cache warm):
        t0 = time.perf_counter()
        tensors = load_real_layer_cpu(i, pinned)
        page = time.perf_counter() - t0
        t1 = time.perf_counter()
        copy_into_module(layer, tensors)
        torch.hpu.synchronize()
        copy_t = time.perf_counter() - t1
        t2 = time.perf_counter()
        y = layer(x, cos, sin)
        torch.hpu.synchronize()
        comp = time.perf_counter() - t2
        del tensors
        stream_times.append(page + copy_t + comp)
        log(f"stream layer {i}: page-in {page*1e3:.0f} ms | H2D copy {copy_t*1e3:.0f} ms | compute {comp*1e3:.0f} ms | total {(page+copy_t+comp)*1e3:.0f} ms")

    avg_stream = sum(stream_times) / len(stream_times)
    log(f"stream+compute avg/layer (incl. cold page-in): {avg_stream*1e3:.0f} ms")

    # ---- stage 4b: STEADY-STATE streaming (weights pinned in host RAM) ------
    # This models the real architecture: TE weights live pinned in host RAM
    # permanently (as te=cpu already does), only the H2D stream is per-prompt.
    resident: dict[int, dict] = {}
    pin_load0 = time.perf_counter()
    for i in range(LAYERS_PROBED):
        resident[i] = load_real_layer_cpu(i, pinned)  # keeps tensors in pinned bufs
    log(f"pinned preload of {LAYERS_PROBED} layers: {time.perf_counter()-pin_load0:.1f} s "
        f"({gb(LAYERS_PROBED*layer_bytes())} host RAM)")
    steady_times = []
    for rep in range(2):
        for i in range(LAYERS_PROBED):
            t0 = time.perf_counter()
            copy_into_module(layer, resident[i])
            torch.hpu.synchronize()
            copy_t = time.perf_counter() - t0
            t1 = time.perf_counter()
            y = layer(x, cos, sin)
            torch.hpu.synchronize()
            comp = time.perf_counter() - t1
            if rep == 1:
                steady_times.append((copy_t, comp))
    avg_copy = sum(c for c, _ in steady_times) / len(steady_times)
    avg_comp = sum(k for _, k in steady_times) / len(steady_times)
    avg_steady = avg_copy + avg_comp
    log(f"STEADY stream+compute avg/layer: {avg_steady*1e3:.0f} ms "
        f"(copy {avg_copy*1e3:.0f} + compute {avg_comp*1e3:.0f}; 1st pass absorbed warmup)")

    # ---- stage 5: CPU baseline ---------------------------------------------
    cpu_layer = make_layer("cpu")
    torch.set_num_threads(CPU_THREADS)
    cos_c, sin_c = rope_cos_sin(SEQ, "cpu", torch.bfloat16)
    x_c = x.cpu()
    cpu_layer(x_c, cos_c, sin_c)  # warm
    t0 = time.perf_counter()
    for _ in range(3):
        cpu_layer(x_c, cos_c, sin_c)
    cpu_fw = (time.perf_counter() - t0) / 3
    log(f"CPU baseline layer forward ({CPU_THREADS} threads): {cpu_fw*1e3:.1f} ms")

    # ---- stage 6: projection -----------------------------------------------
    hpu_full_cold = 64 * avg_stream + 64 * fw  # incl. disk page-in
    hpu_full_steady = 64 * avg_steady          # RAM-resident pinned weights
    cpu_full = 64 * cpu_fw
    log("=== projection (64 layers, seq 512, bf16) ===")
    log(f"HPU streamed eager, cold-disk weights: {hpu_full_cold:.1f} s (disk-bound, ~1 GB/s ZFS)")
    log(f"HPU streamed eager, RAM-resident:      {hpu_full_steady:.1f} s  <-- the design")
    log(f"CPU baseline layer math:               {cpu_full:.1f} s")
    log(f"documented CPU TE end-to-end:          {CPU_TE_BASELINE_S:.1f} s")
    log(f"speedup steady vs CPU layer math:      {cpu_full/hpu_full_steady:.1f}x")
    log("note: excludes embedding lookup (gather, negligible); double-buffering")
    log("the H2D copy under compute would hide the copy behind the 5.5 ms compute")
    log("only partially (copy >> compute), so the copy IS the steady-state floor.")

    # safety: flush pending CS before exit (probe discipline)
    del layer, x, y, cpu_layer, x_c, cos, sin, cos_c, sin_c, resident, tensors0
    torch.hpu.synchronize()
    log("done.")


if __name__ == "__main__":
    sys.exit(main())
