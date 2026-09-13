"""CPU-only feasibility benchmark for H3 Qwen3-VL-32B text encoder on host."""

import gc
import os
import time

import torch

THREADS = int(os.environ.get("BENCH_THREADS", "32"))
torch.set_num_threads(THREADS)
print(f"threads={torch.get_num_threads()}", flush=True)
print(f"torch={torch.__version__}", flush=True)


# ---- 1. bf16 vs fp32 GEMM: 5120-dim linear, batched tokens ----
def bench_linear(num_tokens, n, k, dtype, iters, warmup):
    weights = torch.randn(n, k).to(dtype)
    x = torch.randn(num_tokens, k).to(dtype)
    for _ in range(warmup):
        _ = x @ weights.T
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = x @ weights.T
    elapsed = (time.perf_counter() - t0) / iters
    flops = 2 * num_tokens * n * k
    return elapsed, flops / elapsed / 1e9


for token_count in (256, 1024, 4096):
    for dt in (torch.float32, torch.bfloat16):
        seconds, gflops = bench_linear(token_count, 5120, 5120, dt, 10, 3)
        print(
            f"linear {token_count}x5120x5120 {str(dt):15s}: {seconds * 1e3:8.1f} ms  {gflops:8.1f} GFLOPS",
            flush=True,
        )


# ---- 2. attention SDPA bf16 on CPU at realistic seq ----
def bench_sdpa(seq, nheads, headdim, dtype):
    q = torch.randn(1, nheads, seq, headdim).to(dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    for _ in range(2):
        torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    t0 = time.perf_counter()
    for _ in range(3):
        torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    return (time.perf_counter() - t0) / 3


HEADS, HDIM = 64, 128
for seq in (2048, 8192):
    for dt in (torch.float32, torch.bfloat16):
        seconds = bench_sdpa(seq, HEADS, HDIM, dt)
        print(
            f"sdpa seq={seq} heads=64 hd=128 {str(dt):15s}: {seconds * 1e3:9.1f} ms",
            flush=True,
        )

gc.collect()

# ---- 3. reduced-layer Qwen3-VL text stack forward with output_hidden_states ----
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration  # noqa: E402

NLAYERS, SEQ = 2, 1024

cfg = Qwen3VLConfig()
cfg.text_config.num_hidden_layers = NLAYERS
cfg.text_config.vocab_size = 1000  # shrink init; not perf-relevant
cfg.vision_config.depth = 2
cfg.vision_config.hidden_size = 1152
cfg.vision_config.intermediate_size = 256
cfg.vision_config.num_position_embeddings = 900

t0 = time.perf_counter()
model = Qwen3VLForConditionalGeneration(cfg).to(torch.bfloat16)
print(f"init time {time.perf_counter() - t0:.1f}s", flush=True)
model.eval()

ids = torch.randint(0, 999, (1, SEQ))
two_layer_ms = None
for _ in range(3):
    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            use_cache=False,
            output_hidden_states=True,
        )
    elapsed_ms = (time.perf_counter() - t0) * 1e3
    two_layer_ms = elapsed_ms
    print(
        f"qwen3vl text-only fwd layers={NLAYERS} seq={SEQ}: {elapsed_ms:9.1f} ms "
        f"hs={len(outputs.hidden_states)} shape={outputs.hidden_states[0].shape} dtype={outputs.hidden_states[0].dtype}",
        flush=True,
    )

# ---- extrapolation to the real 32B conditioner ----
HIDDEN, INTER, HD, QH, KVH = 5120, 25600, 128, 64, 8
attn_params = HIDDEN * (QH * HD) + HIDDEN * (KVH * HD) * 2 + (QH * HD) * HIDDEN
ffn_params = 3 * HIDDEN * INTER
gflop_per_token_layer = (attn_params + ffn_params) * 2 / 1e9
for seq in (1024, 4096, 8192):
    for n_layers in (50, 51, 64):
        linear_tflop = seq * n_layers * gflop_per_token_layer / 1e3
        attention_tflop = n_layers * 2 * seq * seq * (QH * HD) * 2 / 1e12
        print(
            f"extrapolate layers={n_layers} seq={seq}: linear~{linear_tflop:8.1f} TFLOP"
            f" attention~{attention_tflop:7.1f} TFLOP total~{linear_tflop + attention_tflop:8.1f} TFLOP",
            flush=True,
        )

print(
    f"measured two-layer {SEQ}-token forward: {two_layer_ms:.1f} ms -> extrapolate by layer count & seq",
    flush=True,
)
