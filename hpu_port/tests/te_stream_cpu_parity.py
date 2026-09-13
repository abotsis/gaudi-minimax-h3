#!/usr/bin/env python
"""te_stream_cpu_parity.py — CPU bit-parity: streamed vs stock Qwen3-VL forward.

Uses a TINY randomly-initialized Qwen3VLForConditionalGeneration with the same
structural features MiniMax-H3 exercises (GQA, per-head q/k norms, mrope
interleaved, deepstack vision injection). Verifies, on CPU, that the streamed
path (pinned weights -> scratch layer copy_ -> stock layer math, hidden states
recorded pre-norm, stack truncated at the condition layer) reproduces the
stock forward's hidden_states[k] EXACTLY (same device, same dtype, same op
order -> bit-identical).

Real-weights + on-card tolerance test: tests/te_stream_card_test.py.
Run: TORCH_DEVICE_BACKEND_AUTOLOAD=0 .venv/bin/python hpu_port/tests/te_stream_cpu_parity.py
"""

import os
import sys
import types

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from te_stream import StreamedQwen3VL  # noqa: E402

from transformers.models.qwen3_vl.configuration_qwen3_vl import (  # noqa: E402
    Qwen3VLConfig,
)
from transformers.models.qwen3_vl.modeling_qwen3_vl import (  # noqa: E402
    Qwen3VLForConditionalGeneration,
)

torch.manual_seed(0)

COND_LAYER = 5  # condition on hidden_states[5] of an 8-layer stack

cfg = Qwen3VLConfig(
    text_config={
        "hidden_size": 128,
        "num_hidden_layers": 8,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "intermediate_size": 256,
        "vocab_size": 100,
        "rms_norm_eps": 1e-6,
        "max_position_embeddings": 512,
        "rope_scaling": {"mrope_section": [8, 4, 4], "mrope_interleaved": True, "rope_type": "default"},
        "rope_theta": 10000.0,
    },
    vision_config={
        "hidden_size": 64,
        "num_hidden_layers": 4,
        "num_attention_heads": 2,
        "intermediate_size": 128,
        "patch_size": 4,
        "spatial_merge_size": 2,
        "deepstack_visual_indexes": [0, 1, 2],
        "num_position_embeddings": 64,
        "out_hidden_size": 128,  # must equal text hidden (merger fc2 out)
    },
    image_token_id=90,
    video_token_id=91,
)
cfg.vocab_size = 100

model = Qwen3VLForConditionalGeneration(cfg)
model.eval()
# controlled comparison: eager attention both sides (on-card the streamed
# path forces eager for the scratch layer; sdpa-vs-eager mixes reduction
# order and shows ~1e-7 fp32 noise)
model.config.text_config._attn_implementation = "eager"
dtype = torch.float32  # CPU parity in fp32 -> bit-exactness is meaningful

model = model.to(dtype)


def run_stock(model, input_ids, mm_token_type_ids, vision_kwargs):
    """The stock get_qwen3vl_prompt_embeds call path, verbatim."""
    outputs = model.model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        mm_token_type_ids=mm_token_type_ids,
        use_cache=False,
        output_hidden_states=True,
        **vision_kwargs,
    )
    return outputs.hidden_states[COND_LAYER]


def run_streamed(stream, input_ids, mm_token_type_ids, vision_kwargs):
    outputs = stream.text_encoder.model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        mm_token_type_ids=mm_token_type_ids,
        use_cache=False,
        output_hidden_states=True,
        **vision_kwargs,
    )
    return outputs.hidden_states[COND_LAYER]


def check(name, stock, streamed):
    same_shape = stock.shape == streamed.shape
    if same_shape:
        diff = (stock - streamed).abs().max().item()
        bitexact = torch.equal(stock, streamed)
    else:
        diff, bitexact = float("nan"), False
    status = "PASS" if (same_shape and bitexact) else "FAIL"
    print(
        f"[{status}] {name}: shape stock={tuple(stock.shape)} streamed={tuple(streamed.shape)} "
        f"maxdiff={diff:.3e} bit-exact={bitexact}"
    )
    return status == "PASS"


ok = True

# ---- case 1: text-only presentation ---------------------------------------
input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]])
mm = torch.zeros_like(input_ids)
vision_kwargs = {}
stock = run_stock(model, input_ids, mm, vision_kwargs)

stream = StreamedQwen3VL(
    model,
    processor=None,
    exec_device="cpu",
    text_encoder_layer=COND_LAYER,
    pin_weights=False,  # CPU: pinning unnecessary
    drop_unused_layers=True,
)
stream.activate()
streamed = run_streamed(stream, input_ids, mm, vision_kwargs)
ok &= check("text-only hidden[5]", stock, streamed)

# state-after check: stock forward must still work identically after shutdown
stream.shutdown()
stock2 = run_stock(model, input_ids, mm, vision_kwargs)
print(f"[{'PASS' if torch.equal(stock, stock2) else 'FAIL'}] shutdown restores stock (bit-exact={torch.equal(stock, stock2)})")
ok &= torch.equal(stock, stock2)

# ---- case 2: image presentation (deepstack injection) ----------------------
# grid 1x4x4 patches, patch 4, merge 2 -> 2x2 merged = 4 image tokens
# (image convention: each patch row carries the temporal axis duplicated,
# 3*2*patch*patch floats, grid t=1 — matches the real processor)
image_tokens = [cfg.image_token_id] * 4
input_ids_img = torch.tensor([[1, 2, 3, *image_tokens, 4, 5, 6, 7, 8]])
mm_img = torch.tensor([[0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0]])
n_patches = 4 * 4  # h*w (grid t=1)
patch_row = torch.randn(1, 3 * 1 * 4 * 4).repeat(1, 2)  # 3*2*4*4 = 96, T duplicated
pixel_values = patch_row.repeat(n_patches, 1)
image_grid_thw = torch.tensor([[1, 4, 4]])
vision_kwargs_img = {
    "pixel_values": pixel_values.to(dtype),
    "image_grid_thw": image_grid_thw,
}
stock_img = run_stock(model, input_ids_img, mm_img, vision_kwargs_img)

stream2 = StreamedQwen3VL(
    model,
    processor=None,
    exec_device="cpu",
    text_encoder_layer=COND_LAYER,
    pin_weights=False,
    drop_unused_layers=True,
)
stream2.activate()
streamed_img = run_streamed(stream2, input_ids_img, mm_img, vision_kwargs_img)
ok &= check("image deepstack hidden[5]", stock_img, streamed_img)
stream2.shutdown()

# ---- case 3: longer text (loop/replay sanity at larger seq) ----------------
input_ids_long = torch.tensor([list(range(1, 65))])
mm_long = torch.zeros_like(input_ids_long)
stock_long = run_stock(model, input_ids_long, mm_long, {})
stream3 = StreamedQwen3VL(
    model,
    processor=None,
    exec_device="cpu",
    text_encoder_layer=COND_LAYER,
    pin_weights=False,
    drop_unused_layers=True,
)
stream3.activate()
streamed_long = run_streamed(stream3, input_ids_long, mm_long, {})
ok &= check("long text hidden[5]", stock_long, streamed_long)
stream3.shutdown()

print("=" * 60)
print("ALL PASS" if ok else "FAILURES PRESENT")
sys.exit(0 if ok else 1)
