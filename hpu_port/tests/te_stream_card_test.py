#!/usr/bin/env python
"""te_stream_card_test.py — real-weights streamed TE vs stock CPU TE.

One card, no graphs. Sequence:
  1. load Qwen3-VL-32B (bf16, mmap'd shards)
  2. stock CPU forward -> hidden[50] (the ~75-90 s reference)
  3. activate StreamedQwen3VL on the HPU (pin layers 0..50, scratch layer)
  4. streamed HPU forward x2 (first = recipe compiles, second = steady)
  5. compare within bf16 tolerance; report timings

Env: source te_stream_env.sh; HABANA_VISIBLE_DEVICES=<free card>.
"""

import os
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent.parent
TE_DIR = Path(os.environ.get("TE_DIR", HERE.parent / "MiniMax-H3/FL2VA/text_encoder"))
PROMPT = os.environ.get("TE_TEST_PROMPT", "a red fox trotting through snowy woods at dusk")
COND_LAYER = 50

sys.path.insert(0, str(HERE))
from te_stream import StreamedQwen3VL  # noqa: E402


def log(m):
    print(f"[card-test] {m}", flush=True)


def main():
    torch.set_num_threads(int(os.environ.get("TE_TEST_THREADS", "32")))
    # Open the Synapse device BEFORE the big TE load: the runtime sizes its
    # host pool at device-open time from free RAM — opening after a 63 GB
    # page-cache fill leaves free=2 GB and synHostMalloc OOMs (test2/3).
    import habana_frameworks.torch as ht  # noqa: F401

    assert torch.hpu.is_available(), "HPU not available"
    torch.zeros(8, device="hpu")
    torch.hpu.synchronize()
    log("Synapse device opened (pre-load)")

    from transformers import AutoProcessor, AutoTokenizer
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLForConditionalGeneration,
    )

    log(f"loading TE from {TE_DIR} (bf16, mmap)")
    t0 = time.perf_counter()
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        TE_DIR, dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    model.eval()
    log(f"load: {time.perf_counter() - t0:.1f} s")

    tokenizer = AutoTokenizer.from_pretrained(TE_DIR)
    processor = AutoProcessor.from_pretrained(TE_DIR)

    ids = tokenizer(PROMPT, add_special_tokens=False)["input_ids"]
    log(f"prompt tokens: {len(ids)}")

    # ---- stock CPU reference ------------------------------------------------
    t0 = time.perf_counter()
    input_ids = torch.tensor([ids], dtype=torch.long)
    mm = torch.tensor(processor.create_mm_token_type_ids([ids]), dtype=torch.long)
    with torch.no_grad():
        out = model.model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            mm_token_type_ids=mm,
            use_cache=False,
            output_hidden_states=True,
        )
    ref = out.hidden_states[COND_LAYER].detach().clone()
    cpu_s = time.perf_counter() - t0
    log(f"stock CPU forward: {cpu_s:.1f} s -> hidden[{COND_LAYER}] {tuple(ref.shape)}")

    # ---- image reference (fl2va-style anchor) — BEFORE activate -------------
    try:
        from PIL import Image

        import numpy as np

        arr = (np.random.default_rng(0).integers(0, 255, (480, 864, 3))).astype("uint8")
        img = Image.fromarray(arr)
        proc = processor(images=img, return_tensors="pt")
        pv = proc["pixel_values"]
        grid = proc["image_grid_thw"]
        log(f"image: pixel_values {tuple(pv.shape)} grid {grid.tolist()}")
        vision_kwargs = {"pixel_values": pv, "image_grid_thw": grid}

        # the presentation must carry the image placeholder tokens (405 for
        # 864x480: 30x54 patches merged 2x2) — insert after the text tokens
        img_tok = model.config.image_token_id
        n_img = int(grid[0].prod() // (model.config.vision_config.spatial_merge_size ** 2))
        ids_img = list(ids) + [img_tok] * n_img
        log(f"image presentation: {len(ids)} text + {n_img} image tokens")

        t0 = time.perf_counter()
        with torch.no_grad():
            out_img = model.model(
                input_ids=torch.tensor([ids_img], dtype=torch.long),
                attention_mask=torch.ones(len(ids_img)).long().unsqueeze(0),
                mm_token_type_ids=torch.tensor(
                    processor.create_mm_token_type_ids([ids_img]), dtype=torch.long
                ),
                use_cache=False,
                output_hidden_states=True,
                **{k: (v.to(model.dtype) if k.startswith("pixel_") else v) for k, v in vision_kwargs.items()},
            )
        ref_img = out_img.hidden_states[COND_LAYER].detach().clone()
        log(f"stock CPU image forward: {time.perf_counter() - t0:.1f} s")
    except Exception as e:  # noqa: BLE001
        log(f"image prep/reference FAILED: {type(e).__name__}: {e}")
        ref_img = None
        vision_kwargs = None

    # ---- streamed HPU -------------------------------------------------------
    stream = StreamedQwen3VL(model, processor, exec_device="hpu", text_encoder_layer=COND_LAYER)
    t0 = time.perf_counter()
    stream.activate()
    log(f"activate (pin+scratch): {time.perf_counter() - t0:.1f} s")

    times = []
    hpu_out = None
    for rep in range(2):
        t0 = time.perf_counter()
        hpu_out = stream.encode(ids, device=torch.device("hpu"), dtype=torch.bfloat16)
        torch.hpu.synchronize()
        times.append(time.perf_counter() - t0)
        log(f"streamed HPU encode rep{rep}: {times[-1]:.1f} s")

    # ---- compare -------------------------------------------------------------
    a = ref.float()
    b = hpu_out.detach().to("cpu").float()
    # bf16-correct gates (h3help review): per-token cosine + relative
    # Frobenius error. mad/rel-vs-1e-6 are meaningless here — the reference
    # has massive-activation channels at ~1.5e4 where one bf16 ulp = 128.
    cos_tok = torch.nn.functional.cosine_similarity(a, b, dim=-1)
    min_cos = cos_tok.min().item()
    rel_fro = ((a - b).norm() / a.norm()).item()
    tol_ok = min_cos > 0.999 and rel_fro < 0.01
    log(
        f"compare: min per-token cos={min_cos:.7f} rel_fro={rel_fro:.5f} -> "
        f"{'PASS' if tol_ok else 'FAIL'} (gates: cos>0.999, rel_fro<0.01)"
    )
    log(f"speedup steady: {cpu_s / min(times[1:]):.1f}x ({cpu_s:.1f}s -> {min(times[1:]):.2f}s)")

    # ---- image: streamed HPU vs the pre-activation stock reference ----------
    if ref_img is not None:
        try:
            t0 = time.perf_counter()
            hpu_img = stream.encode(
                ids_img, vision_inputs=vision_kwargs, device=torch.device("hpu"), dtype=torch.bfloat16
            )
            torch.hpu.synchronize()
            log(f"streamed HPU image encode: {time.perf_counter() - t0:.1f} s")

            b_img = hpu_img.detach().to("cpu").float()
            a_img = ref_img.float()
            cos_tok_i = torch.nn.functional.cosine_similarity(a_img, b_img, dim=-1)
            min_cos_i = cos_tok_i.min().item()
            rel_fro_i = ((a_img - b_img).norm() / a_img.norm()).item()
            img_ok = min_cos_i > 0.999 and rel_fro_i < 0.01
            log(
                f"image compare: min per-token cos={min_cos_i:.7f} rel_fro={rel_fro_i:.5f} -> "
                f"{'PASS' if img_ok else 'FAIL'}"
            )
        except Exception as e:  # noqa: BLE001
            log(f"streamed image encode FAILED: {type(e).__name__}: {e}")
            img_ok = False
    else:
        img_ok = False
        log("image path SKIPPED (no stock reference)")

    ok = tol_ok and img_ok

    # flush pending CS before exit (probe discipline)
    torch.hpu.synchronize()
    log("ALL PASS" if ok else "FAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
