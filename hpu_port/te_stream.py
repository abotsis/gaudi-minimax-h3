#!/usr/bin/env python
"""te_stream.py — hybrid streamed Qwen3-VL text encoder for MiniMax-H3 on HPU.

Design (probed on card1, logs/te_stream_probe5.log — 47 ms/layer steady):
  * The LM decoder-layer weights (~1.05 GB bf16/layer) stay in PINNED host RAM.
  * ONE scratch decoder layer lives on the HPU; before each layer's forward
    its real weights are copy_'d pinned->HPU on the compute stream (42 ms),
    then the stock layer math runs eagerly (5.5 ms @ ~105 TFLOPS). NO HPU
    graphs — per-region lazy recipes only, mark_step between regions.
  * MiniMax-H3 conditions on hidden_states[50] (encoders.get_qwen3vl_prompt_
    embeds), so layers 51..63 and lm_head are NEVER executed and are freed.
  * The vision tower (1.19 GB) + embed_tokens (1.56 GB) + rotary are small
    enough to stay RESIDENT on the HPU.
  * Everything else (masked_scatter, mrope 3D position ids, deepstack
    injection after LM layers 0-2, causal mask) is the STOCK transformers
    5.16.1 Qwen3-VL code path — only weight residency and the layer loop are
    replaced, so t2va / fl2va / ref2va presentations all work unchanged.

Parity discipline: the streamed path must reproduce the stock CPU TE. Run
tests/te_stream_cpu_parity.py (CPU, bit-exact expected) before trusting any
on-card run (bf16 tolerance there).
"""

from __future__ import annotations

import gc
import os
import types

import torch

from typing import Any

NEEDED_DEFAULT = 50  # conditions on hidden_states[50] -> layers 0..50 run


def _htcore():
    from habana_frameworks.torch import core as htcore

    return htcore


def _is_hpu(device) -> bool:
    return torch.device(device).type == "hpu"


def _mark_step(exec_device) -> None:
    if _is_hpu(exec_device):
        _htcore().mark_step()


class StreamedQwen3VL:
    """Wraps a loaded Qwen3VLForConditionalGeneration for streamed prefill."""

    def __init__(
        self,
        text_encoder,
        processor,
        exec_device="hpu",
        text_encoder_layer: int = NEEDED_DEFAULT,
        pin_weights: bool = False,  # v1 default: unpinned streaming (HP-pool-proof)
        drop_unused_layers: bool = True,
    ):
        self.text_encoder = text_encoder
        self.processor = processor
        self.exec_device = torch.device(exec_device)
        self.text_encoder_layer = int(text_encoder_layer)
        self.pin_weights = bool(pin_weights)
        self.drop_unused_layers = bool(drop_unused_layers)
        self.config = text_encoder.config
        self.text_config = self.config.text_config
        self.lm = text_encoder.model
        self.lang = self.lm.language_model
        self._orig_lm_forward = None
        self._orig_lang_forward = None
        self._orig_layer_forward = None
        self._needed_layers = self.text_encoder_layer + 1  # 0..50 inclusive
        self._pinned_param_names: list[str] = []
        self._activated = False

    # ------------------------------------------------------------------ setup
    def activate(self) -> None:
        if self._activated:
            return
        dev = self.exec_device
        is_hpu = dev.type == "hpu"
        if is_hpu:
            # Defensive: the Synapse runtime sizes its host (pinned-memory)
            # pool at device-open time from free RAM. If this process hasn't
            # opened the device yet, do it NOW — opening it after a 63 GB
            # page-cache fill leaves free~2 GB and synHostMalloc OOMs at the
            # first ~250 MB pin (card-test2/3; pin_repro2 proved the order).
            if not torch.hpu.is_available():
                import habana_frameworks.torch  # noqa: F401
            torch.zeros(8, device=dev)
            torch.hpu.synchronize()

        # Attention dispatch: force the stock eager path (explicit repeat_kv,
        # no enable_gqa kwarg) for the scratch layer — proven on HPU; sdpa's
        # enable_gqa=True kwarg is not exercised by Habana's aten SDPA.
        self._prev_text_attn_impl = getattr(self.text_config, "_attn_implementation", None)
        self.text_config._attn_implementation = "eager"

        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            Qwen3VLTextDecoderLayer,
        )

        # 1. free the never-executed tail: layers 51..63 (~14 GB) and lm_head
        #    (tied to embed_tokens; deleting it releases the CPU alias once
        #    embed_tokens is moved on-card).
        if self.drop_unused_layers and len(self.lang.layers) > self._needed_layers:
            freed = len(self.lang.layers) - self._needed_layers
            self.lang.layers = torch.nn.ModuleList(
                list(self.lang.layers)[: self._needed_layers]
            )
            print(f"[te-stream] dropped {freed} never-executed decoder layers")
        if getattr(self.text_encoder, "lm_head", None) is not None:
            self.text_encoder.lm_head = None
            del self.text_encoder.lm_head
            print("[te-stream] dropped lm_head (never used; embeds are tied)")

        # 2. resident components: NONE beyond the scratch layer. The vision
        #    tower (1.19 GB) and embed_tokens (1.56 GB) STAY ON CPU — on-card
        #    they run imprecisely (vision pool cos 0.92 vs CPU: tanh-gelu /
        #    conv3d / bilinear-interpolate divergences, vision_diag) and that
        #    9% error amplifies through 51 streamed layers into garbage
        #    hidden[50] (card-test12 image cos 0.57). On CPU they are bit-
        #    exact vs the stock reference and cheap (~1-3 s). The streamed LM
        #    receives CPU-computed inputs_embeds + deepstack tensors via H2D.
        self.lang.rotary_emb.to(dev)  # cos/sin tables only, HPU-safe
        print(f"[te-stream] rotary on {dev}; vision tower + embed stay on CPU (bit-exact)")

        # 3. scratch layer on-card (constructed directly on-device — a
        #    CPU->HPU .to() here would stage through synHostMalloc, whose
        #    regular-page fallback fails with ENOMEM once the pinned buffers
        #    sit in front of it; card-test6-8). NO init fill: every streamed
        #    forward copies the real layer weights before running.
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            Qwen3VLTextDecoderLayer,
        )

        # construct directly on-device AND in the target dtype: a post-hoc
        # .to(dtype) rebinds the module's Parameter objects (Module._apply),
        # which silently invalidated the activate-time param capture
        # (parity-debug: captured id != live id => copies hit dead objects).
        _prev_dtype = torch.get_default_dtype()
        torch.set_default_dtype(next(iter(self.lang.layers[0].parameters())).dtype)
        try:
            with torch.device(dev):
                self.scratch = Qwen3VLTextDecoderLayer(self.text_config, layer_idx=0)
        finally:
            torch.set_default_dtype(_prev_dtype)

        # 4. weight sources.
        #    pin_weights=True (v2, experimental): FLAT per-layer PINNED
        #    buffers + ONE flat HPU mirror. Constraint discovered by pin-
        #    trace (card-test14): Synapse backs every host alloc with the
        #    machine's 2 MB huge-page pool (12288 HPs = ~24 GB stock), so
        #    synHostMalloc pinning caps out at the pool size and its
        #    regular-page fallback fails under page pressure (dmesg "Failed
        #    to pin host memory"). Only ~24 layers fit; full-TE pinning is
        #    impossible without growing vm.nr_hugepages.
        #    pin_weights=False (v1, robust): copy straight from the model's
        #    own CPU-resident params per layer per prompt — NO pinning, NO
        #    HP pool, NO extra host RAM (identical footprint to te=cpu).
        #    Unpinned H2D stages at ~4.5 GB/s => ~235 ms + 5.5 ms compute
        #    per layer => ~12 s per fresh prompt (vs ~89 s CPU TE).
        flat_params = list(self.lang.layers[0].named_parameters(recurse=True))
        assert all(
            [n for n, _ in flat_params]
            == [n for n, _ in layer.named_parameters(recurse=True)]
            for layer in self.lang.layers
        ), "layer param orders differ"
        if self.pin_weights:
            _dt = flat_params[0][1].dtype
            _align = 128  # elements (256 B for bf16)
            offsets, off = [], 0
            for _, p in flat_params:
                n = p.numel()
                off += (-off) % _align
                offsets.append(off)
                off += n
            FLAT_ELEMS = off
            self._flat_offsets = offsets
            self._flat_pinned: list[torch.Tensor] = []
            _pin_trace = os.environ.get("H3_TE_PIN_TRACE", "") == "1"
            for li, layer in enumerate(self.lang.layers):
                buf = torch.empty(FLAT_ELEMS, dtype=_dt, device="cpu")
                if _pin_trace and li % 8 == 0:
                    mi = {
                        ln.split(":")[0]: ln.split(":")[1].strip().split()[0]
                        for ln in open("/proc/meminfo")
                        if ln.startswith(("MemFree", "MemAvailable", "Cached:", "HugePages_Free", "HugePages_Total"))
                    }
                    print(
                        f"[te-stream] pin layer {li}: {mi}",
                        flush=True,
                    )
                buf = buf.pin_memory()
                with torch.no_grad():
                    for (_, p), o in zip(layer.named_parameters(recurse=True), offsets):
                        buf[o : o + p.numel()].copy_(p.data.reshape(-1))
                self._flat_pinned.append(buf)
            print(
                f"[te-stream] {len(self._flat_pinned)} flat pinned layer buffers "
                f"({FLAT_ELEMS * 2 / 1e9:.2f} GB each)"
            )

            # the HPU mirror: ONE flat bf16 tensor on-device; per-param source
            # views are plain NARROW strided views (tracked by the bridge, no
            # view.dtype node) with fixed addresses, precomputed once.
            self._flat_hpu = torch.empty(FLAT_ELEMS, dtype=_dt, device=dev)
            self._flat_src_views = []
            for (_, p), o in zip(flat_params, offsets):
                self._flat_src_views.append(
                    self._flat_hpu.narrow(0, o, p.numel()).view(p.shape)
                )

        # per-layer source bookkeeping
        for idx, layer in enumerate(self.lang.layers):
            layer._h3_flat_idx = idx
            if self.pin_weights:
                layer._h3_flat_src = self._flat_pinned[idx]

        # 5. patch forwards (per-instance; stock class methods untouched).
        self._orig_lm_forward = self.lm.forward
        self._orig_lang_forward = self.lang.forward
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            Qwen3VLTextDecoderLayer as _LayerCls,
        )

        self._orig_layer_forward = _LayerCls.forward
        self.lm.forward = types.MethodType(_lm_forward_streamed, self.lm)
        self.lang.forward = types.MethodType(_lang_forward_streamed, self.lang)
        # the streamed forwards reach the ctx through _h3_stream_ctx on the
        # modules they are bound to (lm, lang, layers)
        self.lm._h3_stream_ctx = self
        self.lang._h3_stream_ctx = self
        for layer in self.lang.layers:
            layer.forward = types.MethodType(_layer_forward_streamed, layer)
            layer._h3_stream_ctx = self
        self.scratch._h3_stream_ctx = self

        self._activated = True
        print(f"[te-stream] activated on {dev} (graphs OFF, eager recipes per region)")

    def shutdown(self) -> None:
        """Restore stock forwards (CPU parity tests / clean teardown)."""
        if not self._activated:
            return
        self.lm.forward = self._orig_lm_forward
        self.lang.forward = self._orig_lang_forward
        if self._prev_text_attn_impl is not None:
            self.text_config._attn_implementation = self._prev_text_attn_impl
        if hasattr(self.lm, "_h3_stream_ctx"):
            del self.lm._h3_stream_ctx
        if hasattr(self.lang, "_h3_stream_ctx"):
            del self.lang._h3_stream_ctx
        for layer in self.lang.layers:
            del layer.forward
            del layer._h3_stream_ctx
            del layer._h3_flat_idx
            if self.pin_weights:
                del layer._h3_flat_src
        self._flat_pinned = []
        self._flat_hpu = None
        self._flat_src_views = []
        self._activated = False
        gc.collect()

    # ----------------------------------------------------------------- encode
    def encode(
        self,
        token_ids: list[int],
        vision_inputs: dict | None = None,
        text_encoder_layer: int | None = None,
        device=None,
        dtype=None,
    ) -> torch.Tensor:
        """Drop-in for get_qwen3vl_prompt_embeds's model call, streamed."""
        if not self._activated:
            self.activate()
        layer_idx = self.text_encoder_layer if text_encoder_layer is None else int(text_encoder_layer)
        if layer_idx >= len(self.lang.layers):
            raise ValueError(
                f"[te-stream] condition layer {layer_idx} >= streamed stack "
                f"({len(self.lang.layers)} layers)"
            )
        # ids/mm/vision tensors stay on CPU (vision device): embedding,
        # scatter and mrope run there bit-exactly; the streamed LM gets one
        # H2D of the assembled tensors (see _lm_forward_streamed).
        input_ids = torch.tensor([token_ids], dtype=torch.long)
        mm_token_type_ids = torch.tensor(
            self.processor.create_mm_token_type_ids([token_ids]),
            dtype=torch.long,
        )
        vision_kwargs = {}
        for name, value in (vision_inputs or {}).items():
            vision_kwargs[name] = (
                value.to(torch.device("cpu"), self.text_encoder.dtype)
                if name.startswith("pixel_")
                else value.to(torch.device("cpu"))
            )
        outputs = self.text_encoder.model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            mm_token_type_ids=mm_token_type_ids,
            use_cache=False,
            output_hidden_states=True,
            **vision_kwargs,
        )
        # hidden[50] lives on the exec device (HPU); `device` is accepted for
        # signature compatibility and ignored (the caller's scatter bridges
        # devices if it wants them elsewhere).
        return outputs.hidden_states[layer_idx].to(dtype=dtype) if dtype is not None else outputs.hidden_states[layer_idx]

    def warmup(self, token_count: int = 16) -> None:
        """One dummy text encode to compile the per-region recipes (persisted
        via PT_HPU_RECIPE_CACHE_CONFIG). fl2va's vision region compiles on its
        first real request (pixel shapes are bucket-static)."""
        ids = [0] * token_count
        self.encode(ids)


# --------------------------------------------------------------------------
# forward replacements (bound per-instance with types.MethodType)
# --------------------------------------------------------------------------
def _lm_forward_streamed(
    self,
    input_ids=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    pixel_values=None,
    pixel_values_videos=None,
    image_grid_thw=None,
    video_grid_thw=None,
    mm_token_type_ids=None,
    **kwargs,
):
    """Verbatim stock Qwen3VLModel.forward (transformers 5.16.1), except the
    final language_model call lands in the streamed TextModel forward and the
    result keeps .hidden_states."""
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    # Vision tower, embedding table, masked_scatter and mrope position ids
    # run on the VISION device (cpu for te=stream — bit-exact, see activate
    # note). One H2D of the assembled tensors follows.
    vd = getattr(self, "_h3_vision_device", None)
    if vd is None:
        vd = next(self.parameters()).device  # stock behaviour (all-CPU)
    if inputs_embeds is None:
        if input_ids.device != vd:
            input_ids = input_ids.to(vd)
        inputs_embeds = self.get_input_embeddings()(input_ids)

    image_mask = None
    video_mask = None

    def _on_vd(t, want_dtype=False):
        if want_dtype:
            return t.to(vd, self.visual.dtype if hasattr(self, "visual") else None)
        return t.to(vd)

    if pixel_values is not None:
        image_outputs = self.get_image_features(
            _on_vd(pixel_values, True), image_grid_thw, return_dict=True, **kwargs
        )
        image_embeds = image_outputs.pooler_output
        deepstack_image_embeds = image_outputs.deepstack_features
        image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        image_mask, _ = self.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    if pixel_values_videos is not None:
        video_outputs = self.get_video_features(
            _on_vd(pixel_values_videos, True), video_grid_thw, return_dict=True, **kwargs
        )
        video_embeds = video_outputs.pooler_output
        deepstack_video_embeds = video_outputs.deepstack_features
        video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        _, video_mask = self.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

    # deepstack: resolve (img|video) features into FULL-SEQUENCE zero-padded
    # tensors on the vision device. The streamed LM consumes them as dense
    # adds (hidden += full), exactly equivalent to the stock indexed scatter
    # add but free of advanced-indexing ops in lazy mode.
    deepstack_scattered = None
    if image_mask is not None or video_mask is not None:
        if image_mask is not None and video_mask is not None:
            image_mask = image_mask[..., 0]
            video_mask = video_mask[..., 0]
            visual_pos_masks = image_mask | video_mask
            joint = []
            image_mask_joint = image_mask[visual_pos_masks]
            video_mask_joint = video_mask[visual_pos_masks]
            for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
                embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1]).to(img_embed.device)
                embed_joint[image_mask_joint, :] = img_embed
                embed_joint[video_mask_joint, :] = vid_embed
                joint.append(embed_joint)
            src_embeds = joint
        elif image_mask is not None:
            image_mask = image_mask[..., 0]
            visual_pos_masks = image_mask
            src_embeds = deepstack_image_embeds
        else:
            video_mask = video_mask[..., 0]
            visual_pos_masks = video_mask
            src_embeds = deepstack_video_embeds
        deepstack_scattered = []
        seq = inputs_embeds.shape[1]
        mask_1d = visual_pos_masks.reshape(-1)
        for emb in src_embeds:
            full = emb.new_zeros(seq, emb.shape[-1])
            full[mask_1d, :] = emb
            deepstack_scattered.append(full)

    if position_ids is None:
        position_ids = self.compute_3d_position_ids(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            mm_token_type_ids=mm_token_type_ids,
        )
    _mark_step_stream_ctx(self)

    # single H2D of the assembled prefill tensors (all small: seq x 5120)
    ed = torch.device(_mark_step_exec_device(self))
    inputs_embeds = inputs_embeds.to(ed)
    position_ids = position_ids.to(ed) if position_ids is not None else None
    if attention_mask is not None:
        attention_mask = attention_mask.to(ed)
    if deepstack_scattered is not None:
        deepstack_scattered = [d.to(ed) for d in deepstack_scattered]

    outputs = self.language_model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        visual_pos_masks=None,
        deepstack_visual_embeds=None,
        deepstack_scattered=deepstack_scattered,
        **kwargs,
    )
    return types.SimpleNamespace(
        hidden_states=outputs.hidden_states,
        last_hidden_state=outputs.last_hidden_state,
        rope_deltas=self.rope_deltas,
    )


def _lang_forward_streamed(
    self,
    input_ids=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    use_cache=None,
    visual_pos_masks=None,
    deepstack_visual_embeds=None,
    deepstack_scattered=None,
    **kwargs,
):
    """Stock Qwen3VLTextModel.forward truncated at the condition layer:
    identical body, layers list already truncated, hidden states recorded
    per layer (pre-norm, matching stock output_hidden_states semantics), and
    the final RMSNorm SKIPPED (hidden_states[k<last] is pre-norm; MiniMax-H3
    reads hidden_states[50] which must NOT be post-norm)."""
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
    if past_key_values is not None or use_cache:
        raise NotImplementedError("[te-stream] prefill only (no cache)")

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if position_ids is None:
        past_seen_tokens = 0
        position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
        position_ids = position_ids.view(1, 1, -1).expand(4, inputs_embeds.shape[0], -1)
    elif position_ids.ndim == 2:
        position_ids = position_ids[None, ...].expand(4, position_ids.shape[0], -1)

    if position_ids.ndim == 3 and position_ids.shape[0] == 4:
        text_position_ids = position_ids[0]
        position_ids = position_ids[1:]
    else:
        text_position_ids = None

    from transformers.masking_utils import create_causal_mask

    attention_mask_out = create_causal_mask(
        config=self.config,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        past_key_values=None,
        position_ids=text_position_ids,
    )

    hidden_states = inputs_embeds
    position_embeddings = self.rotary_emb(hidden_states, position_ids)
    _mark_step_stream_ctx(self)

    recorded = [hidden_states]
    for layer_idx, decoder_layer in enumerate(self.layers):
        hidden_states = decoder_layer(
            hidden_states,
            attention_mask=attention_mask_out,
            position_ids=text_position_ids,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        recorded.append(hidden_states)
        if deepstack_scattered is not None and layer_idx < len(deepstack_scattered):
            # dense zero-padded add == the stock indexed scatter-add
            # (zeros contribute +0.0 only); no advanced indexing in lazy IR
            hidden_states = hidden_states + deepstack_scattered[layer_idx]

    _mark_step_stream_ctx(self)
    return types.SimpleNamespace(
        last_hidden_state=hidden_states,
        hidden_states=recorded,
        past_key_values=None,
    )


def _layer_forward_streamed(
    self,
    hidden_states: torch.Tensor,
    position_embeddings=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    use_cache: bool = False,
    **kwargs,
):
    """Streamed decoder layer: stage this layer's weights into the resident
    scratch layer (v1: unpinned H2D from the layer's own CPU params; v2:
    base-pointer pinned->mirror + H2H), then the stock forward. Copies
    enqueue on the compute stream before the layer ops (ordered, no sync);
    each layer region becomes one small lazy recipe (compiled once per
    shape, replayed after)."""
    ctx = self._h3_stream_ctx
    if past_key_values is not None or use_cache:
        raise NotImplementedError("[te-stream] prefill only (no cache)")
    with torch.no_grad():
        if ctx.pin_weights:
            # v2 fast path: ONE base-pointer pinned->mirror copy, then
            # mirror->param H2H copies (no host staging). H2H goes through
            # the PARAM OBJECT (not the .data alias): on HPU the params are
            # HabanaParameterWrapper lazy tensors, and a copy via the .data
            # alias lands in a different lazy node than the one the forward
            # reads (card-test10: correct on CPU, garbage on HPU).
            ctx._flat_hpu.copy_(self._h3_flat_src, non_blocking=True)
            for (_, s_param), src_view in zip(
                ctx.scratch.named_parameters(recurse=True), ctx._flat_src_views
            ):
                s_param.copy_(src_view, non_blocking=True)
        else:
            # v1: unpinned H2D straight from this layer's own CPU params
            # (~4.5 GB/s staging-bound; still ~7x faster than the CPU TE).
            for (_, l_param), (_, s_param) in zip(
                self.named_parameters(recurse=True),
                ctx.scratch.named_parameters(recurse=True),
            ):
                s_param.copy_(l_param.data, non_blocking=True)
            if os.environ.get("H3_TE_COPY_SYNC", "") == "1":
                # diagnostic/experimental: force the staged weights to land
                # before the forward reads them (cross-stream visibility)
                torch.hpu.synchronize()
    out = ctx._orig_layer_forward(
        ctx.scratch,
        hidden_states,
        position_embeddings=position_embeddings,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=None,
        use_cache=False,
        **kwargs,
    )
    _mark_step_stream_ctx(ctx)
    return out


def _mark_step_exec_device(obj) -> str:
    """The streamed LM's exec device ('hpu' or 'cpu') from the ctx."""
    ctx = obj if isinstance(obj, StreamedQwen3VL) else getattr(obj, "_h3_stream_ctx", None)
    return ctx.exec_device.type if ctx is not None else "cpu"


def _mark_step_stream_ctx(obj) -> None:
    """mark_step for streamed regions; obj is a module with _h3_stream_ctx or
    the ctx itself. No-op on CPU."""
    ctx = obj if isinstance(obj, StreamedQwen3VL) else getattr(obj, "_h3_stream_ctx", None)
    if ctx is None:
        return
    if ctx.exec_device.type == "hpu":
        _htcore().mark_step()
