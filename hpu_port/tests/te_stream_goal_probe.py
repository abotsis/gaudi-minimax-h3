#!/usr/bin/env python
"""te_stream_goal_probe.py — goal acceptance for the streamed TE.

A. Varying text prompt sizes: streamed encode at several seq lengths vs
   CPU bf16 reference + two streamed runs (per-path determinism).
B. Image presentation vs fp32 TRUTH (fp32 LM forward, per-layer weight
   casts): streamed and CPU-bf16 must sit at the same distance from truth
   (the sink-channel branch flip makes bf16-vs-bf16 gates meaningless —
   PLAN_CP.md Phase 8, img_bisect7).
C. Timings per length (the production claim: ~4-12 s per fresh prompt).

Env: source te_stream_env.sh; HABANA_VISIBLE_DEVICES=<free card>.
"""

import os
import sys
import time
import types
from pathlib import Path

import torch

import te_stream  # noqa: E402  (module ref needed for _layer_forward_streamed)

HERE = Path(__file__).resolve().parent.parent
TE_DIR = Path(os.environ.get("TE_DIR", HERE.parent / "MiniMax-H3/FL2VA/text_encoder"))
COND_LAYER = 50

sys.path.insert(0, str(HERE))
from te_stream import StreamedQwen3VL  # noqa: E402


def log(m):
    print(f"[goal-probe] {m}", flush=True)


def rel_fro(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a - b).norm() / a.norm().clamp_min(1e-9)).item()


def main():
    torch.set_num_threads(int(os.environ.get("TE_TEST_THREADS", "32")))
    import habana_frameworks.torch as ht  # noqa: F401

    assert torch.hpu.is_available()
    torch.zeros(8, device="hpu")
    torch.hpu.synchronize()

    from transformers import AutoProcessor, AutoTokenizer
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLForConditionalGeneration,
    )

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        TE_DIR, dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(TE_DIR)
    processor = AutoProcessor.from_pretrained(TE_DIR)
    log("model + processor loaded")

    # varying-length prompts (text-only presentations)
    base = "a red fox trotting through snowy woods at dusk"
    prompts = [
        base,
        base * 4,
        " ".join([f"word{i} {base}" for i in range(12)]),
        " ".join([f"word{i} {base}" for i in range(40)]),
    ]

    stream = StreamedQwen3VL(model, processor, exec_device="hpu", text_encoder_layer=COND_LAYER)
    stream.activate()
    log("streamed conditioner active")

    ok = True
    for pi, prompt in enumerate(prompts):
        ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        # CPU bf16 reference
        mm = torch.tensor(processor.create_mm_token_type_ids([ids]), dtype=torch.long)
        with torch.no_grad():
            ref = model.model(
                input_ids=torch.tensor([ids], dtype=torch.long),
                attention_mask=torch.ones(1, len(ids)).long(),
                mm_token_type_ids=mm,
                use_cache=False,
                output_hidden_states=True,
            ).hidden_states[COND_LAYER].float()
        # streamed x2
        outs = []
        times = []
        for rep in range(2):
            t0 = time.perf_counter()
            o = stream.encode(ids, dtype=torch.bfloat16)
            torch.hpu.synchronize()
            times.append(time.perf_counter() - t0)
            outs.append(o.detach().float().cpu())
        det = torch.equal(outs[0], outs[1])
        rf = rel_fro(outs[0], ref)
        case_ok = det and rf < 0.02
        ok &= case_ok
        log(
            f"seq {len(ids):5d}: rel_fro(vs cpu bf16)={rf:.5f} det={det} "
            f"t0={times[0]:.1f}s steady={times[1]:.2f}s -> {'PASS' if case_ok else 'FAIL'}"
        )

    # ---- B: image presentation vs fp32 truth --------------------------------
    from PIL import Image
    import numpy as np

    img = Image.fromarray(np.random.default_rng(0).integers(0, 255, (480, 864, 3)).astype("uint8"))
    proc = processor(images=img, return_tensors="pt")
    pv, grid = proc["pixel_values"], proc["image_grid_thw"]
    img_tok = model.config.image_token_id
    n_img = int(grid[0].prod() // (model.config.vision_config.spatial_merge_size**2))
    ids_img = list(range(100, 110)) + [img_tok] * n_img
    vision_kwargs = {"pixel_values": pv.to(model.dtype), "image_grid_thw": grid}
    mm = torch.tensor(processor.create_mm_token_type_ids([ids_img]), dtype=torch.long)
    input_ids_img = torch.tensor([ids_img], dtype=torch.long)

    # CPU bf16 reference (stock forward)
    with torch.no_grad():
        ref_bf16 = model.model(
            input_ids=input_ids_img,
            attention_mask=torch.ones(1, len(ids_img)).long(),
            mm_token_type_ids=mm,
            use_cache=False,
            output_hidden_states=True,
            **vision_kwargs,
        ).hidden_states[COND_LAYER].float()
    log(f"image presentation seq {len(ids_img)}: cpu bf16 ref done")

    # streamed x2 (determinism)
    t0 = time.perf_counter()
    s1 = stream.encode(ids_img, vision_inputs=vision_kwargs, dtype=torch.bfloat16)
    torch.hpu.synchronize()
    t1 = time.perf_counter() - t0
    s2 = stream.encode(ids_img, vision_inputs=vision_kwargs, dtype=torch.bfloat16)
    torch.hpu.synchronize()
    s1f, s2f = s1.detach().float().cpu(), s2.detach().float().cpu()
    det_img = torch.equal(s1f, s2f)
    rf_bf16 = rel_fro(s1f, ref_bf16)
    log(f"streamed image: t0={t1:.1f}s det={det_img} rel_fro(vs cpu bf16)={rf_bf16:.5f}")

    # Localized fp32 adjudication (full-model fp32 truth OOMs the host at
    # 63 GB resident + fp32 casts — goal_probe2/3 exit 137). The branch flip
    # lives at layer 43 (PLAN_CP.md Phase 8); replay ONLY that layer at fp32
    # on (a) the CPU reference input and (b) the streamed run's own input.
    # If the streamed output matches fp32(streamed-input) as well as CPU
    # matches fp32(cpu-input), the streamed path is truth-faithful to its
    # own (1-ulp-different) input and the divergence is upstream bf16 noise.
    log("localized fp32 adjudication at layer 43...")
    lang = model.model.language_model
    layer43 = lang.layers[43]
    pe = None

    # capture the streamed run's layer-43 input + the CPU bf16 ref[43]
    caps = {}

    def spy43(self, hidden_states, **kw):
        out = te_stream._layer_forward_streamed(self, hidden_states, **kw)
        if "x" not in caps:
            p_emb = kw.get("position_embeddings")
            caps["x"] = hidden_states.detach().clone()
            caps["y"] = out.detach().clone()
            caps["pe"] = tuple(t.detach().clone() for t in p_emb) if p_emb is not None else None
        return out

    stream.lang.layers[43].forward = types.MethodType(spy43, stream.lang.layers[43])
    _ = stream.encode(ids_img, vision_inputs=vision_kwargs, dtype=torch.bfloat16)
    torch.hpu.synchronize()
    stream.lang.layers[43].forward = types.MethodType(
        te_stream._layer_forward_streamed, stream.lang.layers[43]
    )
    x_stream, y_stream_hpu, pe_hpu = caps["x"], caps["y"], caps["pe"]
    pe_cpu = tuple(t.float().cpu() for t in pe_hpu)

    # CPU bf16 reference through layer 42 (stock forward)
    with torch.no_grad():
        ref_all = model.model(
            input_ids=input_ids_img,
            attention_mask=torch.ones(1, len(ids_img)).long(),
            mm_token_type_ids=mm,
            use_cache=False,
            output_hidden_states=True,
            **vision_kwargs,
        ).hidden_states
    ref43_bf16 = ref_all[43].float()
    ref44_bf16 = ref_all[44].float()

    def fp32_layer43(x_in):
        saved = {}
        with torch.no_grad():
            for n, p in layer43.named_parameters(recurse=True):
                saved[n] = p.data
                p.data = p.data.float()
            # CAUSAL mask — must match the streamed/stock path (mask=None in
            # an earlier adjudicator made the replay non-causal and the
            # distances triangle-inequality-inconsistent: probe5 FAIL)
            from transformers.masking_utils import create_causal_mask

            x_f = x_in.float()
            causal = create_causal_mask(
                config=lang.config,
                inputs_embeds=x_f,
                attention_mask=None,
                past_key_values=None,
                position_ids=None,
            )
            y = stream._orig_layer_forward(
                layer43,
                x_f,
                position_embeddings=(pe_cpu[0], pe_cpu[1]),
                attention_mask=causal,
                position_ids=None,
                past_key_values=None,
                use_cache=False,
            ).float().cpu()
            for n, p in layer43.named_parameters(recurse=True):
                p.data = saved[n]
        return y

    y32_on_ref = fp32_layer43(ref43_bf16.to(torch.bfloat16))
    y32_on_stream = fp32_layer43(x_stream.cpu().to(torch.bfloat16))
    # distances
    d = lambda a, b: rel_fro(a, b)
    log(f"cpu-bf16 out vs fp32(cpu-input):     rel_fro={d(ref44_bf16, y32_on_ref):.5f}")
    log(f"streamed out vs fp32(streamed-input): rel_fro={d(y_stream_hpu.float().cpu(), y32_on_stream):.5f}")
    log(f"cross: cpu-bf16 vs fp32(streamed-input) rel_fro={d(ref44_bf16, y32_on_stream):.5f}; "
        f"streamed vs fp32(cpu-input) rel_fro={d(y_stream_hpu.float().cpu(), y32_on_ref):.5f}")
    # acceptance: each path is truth-faithful to its own input within bf16 noise
    a_ok = d(ref44_bf16, y32_on_ref) < 0.02
    b_ok = d(y_stream_hpu.float().cpu(), y32_on_stream) < 0.02
    img_ok = det_img and a_ok and b_ok
    ok &= img_ok
    log(
        f"adjudication: cpu-truth-faithful={a_ok} streamed-truth-faithful={b_ok} "
        f"det={det_img} -> {'PASS' if img_ok else 'FAIL'}"
    )

    torch.hpu.synchronize()
    log("ALL PASS" if ok else "FAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
