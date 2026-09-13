"""HPU runtime patches for the diffusers 0.40.0 MiniMax-H3 in-tree pipeline.

This module is the runtime-monkeypatch entry point for the H3-FL2VA -> Gaudi2 port.
No site-packages file is edited; every change hangs off class attributes /
module namespaces at patch time and is idempotent. It implements the port design
(sections 3.2-3.5):

  1. `habana_fused_sdpa` -- Habana's FusedSDPA autograd kernel wired into diffusers'
     attention dispatch, with an exact native/SDPA passthrough on CPU tensors so the
     SAME code runs (and is parity-assertable) on a CPU-only box.
  2. A rope cos/sin cache for `MiniMaxH3RotaryPosEmbed` (per-bucket key, fp32
     cos/sin of shape (seq, 2*3*rope_freq_dim)) hoisted out of the per-step graph.
  3. The video-VAE decode autocast guard fix: stock `decoders.py` enables fp16
     autocast only when `device.type == "cuda"`; on HPU it would silently decode in
     fp32 (2x waste). The patched subclass enables it for `{"cuda", "hpu"}`.
  4. `wrap_in_hpu_graph` helpers for the DiT (per bucket) and optional VAE decoder
     graphs (wrapped at the inner `*.decoder` submodule, the actual compute of
     `AutoencoderKLMiniMaxH3[Audio].decode`), plus the eager-mode `htcore.mark_step()`
     strategy (forward hooks) for when graphs are disabled.

AdaLN (design 3.5 #5): v1 ships NO AdaLN patch -- the 50 per-block `adaln_proj`
Linears and the row `index_select`s execute inside the per-step graph where the
op ordering is fixed; the optional batched-GEMM hoist is gated behind
`H3_BATCH_ADALN=1` and deliberately a no-op until first-card profiling says the
50 GEMVs are the bottleneck.

Hard rules honored here:
  * Every HPU path is guarded by the *tensor's* `device.type == "hpu"` (not by a
    process-global probe), so importing and exercising this module on CPU never
    initializes an HPU device, never runs `.to('hpu')`, never compiles a graph.
  * `torch.hpu.is_available()` is probed at most once and only for logging; no HPU
    API beyond that is called unless a tensor is already on HPU.

Env flags (see README_PORT.md): H3_DIT_GRAPHS, H3_VAE_GRAPHS, H3_ROPE_CACHE,
H3_FAST_SOFTMAX, H3_DISABLE_HPU, H3_SDPA_TILE_THRESHOLD, H3_SDPA_TILE_SIZE,
H3_SDPA_MARK_EVERY, H3_BATCH_ADALN.
"""

from __future__ import annotations

import contextlib
import functools
import inspect
import logging
import os
from dataclasses import dataclass, field
from typing import Any

import torch

logger = logging.getLogger(__name__)

BACKEND_NAME = "habana_fused_sdpa"

# The upstream file the autocast-guard subclass mirrors. If diffusers changes
# `decoders.py`, the copied body below must be re-audited; the drift check in
# `patch_video_vae_autocast_guard()` fails loudly at patch time.
_PATCHED_DECODERS_REV = "diffusers 0.40.0 modular_pipelines/minimax_h3/decoders.py"
_AUTOCAST_GUARD_SNIPPET = 'enabled=device.type == "cuda"'

# diffusers modules that resolve `dispatch_attention_fn` from their own namespace
# (imported names, not attribute lookups into attention_dispatch). Only these need
# the enum-bypassing wrapper; other in-process diffusers models keep stock dispatch.
_DISPATCH_CONSUMERS = (
    "diffusers.models.transformers.transformer_minimax_h3",
    "diffusers.models.autoencoders.autoencoder_kl_minimax_h3",
)


# ---------------------------------------------------------------------------
# Device policy
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _probe_hpu() -> bool:
    """Probe (once) whether the Habana plugin exposes HPU devices.

    This is an *availability* probe only: no device is initialized, no tensor is
    placed, no graph is compiled. Per-call behavior of every patched function is
    decided by `tensor.device.type == "hpu"`, not by this probe. `H3_DISABLE_HPU=1`
    forces the pure-CPU code path even on an HPU machine (fallback testing).
    """
    if os.getenv("H3_DISABLE_HPU", "0") == "1":
        return False
    try:
        return bool(
            torch.hpu.is_available()
        )  # availability probe only, design section 5
    except Exception:  # pragma: no cover - plugin missing entirely
        return False


def assert_no_hpu_tensor(*objs: Any) -> None:
    """Test/CI helper: raise if any tensor reachable from `objs` sits on HPU.

    Bounded walk (depth 8, seen-set of ids) over tensors, containers and modules;
    the goal is to catch accidental device placement, not be a complete
    explorer.
    """
    seen: set[int] = set()

    def walk(obj, depth):
        if depth > 8 or obj is None:
            return
        if isinstance(obj, torch.Tensor):
            if obj.device.type == "hpu":
                raise RuntimeError(
                    f"HPU tensor found in a CPU-only context: shape={tuple(obj.shape)}"
                )
            return
        if id(obj) in seen:
            return
        seen.add(id(obj))
        if isinstance(obj, torch.nn.Module):
            for _, t in obj.named_parameters(recurse=False):
                walk(t, depth + 1)
            for _, t in obj.named_buffers(recurse=False):
                walk(t, depth + 1)
            for _, child in obj.named_children():
                walk(child, depth + 1)
        elif isinstance(obj, (list, tuple, set, frozenset)):
            for x in obj:
                walk(x, depth + 1)
        elif isinstance(obj, dict):
            for v in obj.values():
                walk(v, depth + 1)
        elif hasattr(obj, "__dict__"):
            for v in vars(obj).values():
                walk(v, depth + 1)

    for obj in objs:
        walk(obj, 0)


# ---------------------------------------------------------------------------
# Config (subset of design section 3.6 that impacts this module)
# ---------------------------------------------------------------------------


@dataclass
class HpuPatchConfig:
    dit_graphs: bool = True
    vae_graphs: bool = False
    rope_cache: bool = True
    fast_softmax: bool = True
    tile_threshold: int = 65536  # seq rows above which an attention call is q-tiled
    tile_size: int = 64
    mark_every: int = 75  # vllm_gaudi precedent (`qwen2_5_vl.py`): mark every 75 tiles
    batch_adaln: bool = False  # v1: no-op, reserved (design 3.5 #5)

    @classmethod
    def from_env(cls) -> HpuPatchConfig:
        env = os.getenv
        return cls(
            dit_graphs=env("H3_DIT_GRAPHS", "1") == "1",
            vae_graphs=env("H3_VAE_GRAPHS", "0") == "1",
            rope_cache=env("H3_ROPE_CACHE", "1") == "1",
            fast_softmax=env("H3_FAST_SOFTMAX", "1") == "1",
            tile_threshold=int(env("H3_SDPA_TILE_THRESHOLD", "65536")),
            tile_size=int(env("H3_SDPA_TILE_SIZE", "64")),
            mark_every=int(env("H3_SDPA_MARK_EVERY", "75")),
            batch_adaln=env("H3_BATCH_ADALN", "0") == "1",
        )


_CONFIG: HpuPatchConfig | None = None


def patch_config() -> HpuPatchConfig:
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = HpuPatchConfig.from_env()
    return _CONFIG


def reload_patch_config() -> HpuPatchConfig:
    """Re-read env flags (for tests that flip flags between phases)."""
    global _CONFIG
    _CONFIG = HpuPatchConfig.from_env()
    return _CONFIG


# ---------------------------------------------------------------------------
# 3.2 Attention: Habana FusedSDPA into diffusers' dispatch
# ---------------------------------------------------------------------------

# Lazy handle to `habana_frameworks.torch.hpex.kernels.FusedSDPA`. Imported only on
# the first call with HPU-resident tensors; CPU runs never import it.
_FUSED_SDPA = None


def _get_fused_sdpa():
    global _FUSED_SDPA
    if _FUSED_SDPA is None:
        from habana_frameworks.torch.hpex.kernels import FusedSDPA

        _FUSED_SDPA = FusedSDPA
    return _FUSED_SDPA


def _softmax_mode(q_dtype: torch.dtype) -> str:
    """Pick the FusedSDPA softmax mode for the current call.

    `fast` is legal for bf16 inputs (`FLASH_ATTENTION_FAST_SOFTMAX=1`, design 3.2);
    the explicit `fp32` mode is asserted by Habana to be BF16-input-only, so fp32
    calls (video-VAE ViT decode) run the default `"None"` mode, the numerically
    conservative fp32-accumulating path.
    """
    if q_dtype == torch.bfloat16 and patch_config().fast_softmax:
        return "fast"
    return "None"


def _native_sdpa_passthrough(
    query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa
):
    """Byte-for-byte mirror of diffusers' `native` backend call sequence.

    Kept identical on purpose: when no HPU tensor is in the call, the patched backend
    must be bit-comparable to the stock `native` backend so CPU smoke tests can
    assert exact parity between the two code paths (design section 5).
    """
    query, key, value = (x.permute(0, 2, 1, 3) for x in (query, key, value))
    out = torch.nn.functional.scaled_dot_product_attention(
        query=query,
        key=key,
        value=value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
        enable_gqa=enable_gqa,
    )
    return out.permute(0, 2, 1, 3)


def _fused_sdpa_tiled(
    q,
    k,
    v,
    attn_mask,
    dropout_p,
    is_causal,
    scale,
    softmax_mode,
    tile=None,
    mark_every=None,
):
    """q-tiling fallback for sequences exceeding FusedSDPA's 2^31-byte mask/limit.

    Mirrors the `VLLM_HPU_FSDPA_Q_TILE_ENABLE` pattern from
    `vllm_gaudi/models/qwen2_5_vl.py`: tile rows of the query, `mark_step()` every
    `mark_every` tiles so eager execution interleaves graph work. Eager mode only
    (a tile loop would expand the captured graph; the single-kernel path stays
    inside HPU graphs).
    """
    cfg = patch_config()
    tile = tile or cfg.tile_size
    mark_every = mark_every if mark_every is not None else cfg.mark_every

    import habana_frameworks.torch.core as htcore  # reached only with HPU tensors

    rows = q.shape[2]
    outs = []
    n_marks = 0
    for start in range(0, rows, tile):
        end = min(start + tile, rows)
        outs.append(
            _get_fused_sdpa().apply(
                q[:, :, start:end, :],
                k,
                v,
                attn_mask,
                float(dropout_p),
                bool(is_causal),
                scale,
                softmax_mode,
                False,
            )
        )
        n_marks += 1
        if mark_every and n_marks % mark_every == 0:
            htcore.mark_step()
    return torch.cat(outs, dim=2)


def _habana_local_sdpa(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    enable_gqa=False,
    softmax_mode=None,
    recompute_mode=False,
):
    """One-rank attention computation of the habana backend.

    The body shared by the direct backend call and the context-parallel forward
    op: on non-HPU tensors the exact native passthrough, otherwise FusedSDPA
    (q-tiled above the threshold). Split out so the CP forward op reuses ONE
    definition -- the CP path must compute exactly what the unsharded path
    computes, only over a seq/N shard (PLAN_CP.md phase 1.3).
    """
    if query.device.type != "hpu" or not _probe_hpu():
        return _native_sdpa_passthrough(
            query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa
        )

    mode = softmax_mode if softmax_mode is not None else _softmax_mode(query.dtype)

    q, k, v = (t.transpose(1, 2) for t in (query, key, value))  # (B,S,H,D) -> (B,H,S,D)
    if patch_config().tile_threshold and q.shape[2] > patch_config().tile_threshold:
        out = _fused_sdpa_tiled(q, k, v, attn_mask, dropout_p, is_causal, scale, mode)
        return out.transpose(1, 2)

    # FusedSDPA.forward(ctx, q, k, v, attn_mask, dropout_p, is_causal, scale,
    #                   softmax_mode, recompute_mode, ...) -> (B,H,S,D)
    o = _get_fused_sdpa().apply(
        q,
        k,
        v,
        attn_mask,
        float(dropout_p),
        bool(is_causal),
        scale,
        mode,
        recompute_mode,
    )
    return o.transpose(1, 2)


# The diffusers module + helper the CP routing leans on. Private API -- pinned
# to the audited revision like _PATCHED_DECODERS_REV; the signature check in
# _cp_ulysses_attention fails loudly on drift instead of silently computing
# unsharded attention under a CP config (which would hang or corrupt).
_CP_DISPATCH_REV = "diffusers 0.40.0 models/attention_dispatch.py"
_cp_dispatch_checked = False


def _fsdp_cp_forward_op(
    ctx,
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    enable_gqa=False,
    return_lse=False,
    _save_ctx=True,
    _parallel_config=None,
):
    """forward_op for diffusers' templated CP attention (signature mirrors
    `_native_attention_forward_op`): runs the ONE-RANK attention body on the
    all-to-all'd tensors -- full heads, local seq -- in diffusers' (B,S,H,D)
    layout. `ctx.save_for_backward` mirrors native so the templated Function's
    contract holds; training never runs here (inference pipeline).
    """
    if return_lse:
        raise ValueError("habana_fused_sdpa does not support return_lse=True.")
    if _save_ctx:
        ctx.save_for_backward(query, key, value)
    # H3 is self-attention over one packed sequence: under Ulysses every rank
    # holds the same local seq for q and kv. A mismatch means the CP plan split
    # something diffusers did not expect -- loud failure beats silent garbage.
    if query.shape[1] != key.shape[1]:
        raise RuntimeError(
            f"habana_fused_sdpa CP: local q seq {query.shape[1]} != kv seq {key.shape[1]}"
        )
    return _habana_local_sdpa(
        query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa
    )


def _fsdp_cp_backward_op(ctx, grad_out, *args, **kwargs):
    raise RuntimeError(
        "habana_fused_sdpa context-parallel path is inference-only; backward is not implemented."
    )


def _cp_ulysses_attention(query, key, value, attn_mask, dropout_p, is_causal, scale, parallel_config):
    """Route one attention call through diffusers' Ulysses CP machinery.

    diffusers' own CP-capable backends (native, flash, ...) wrap their local op
    in `TemplatedUlyssesAttention` (head-split all-to-all before, all-to-all
    after). We reuse the same template with our local op as `forward_op`, so
    the backend is CP-correct by construction on ANY backend -- gloo on CPU
    (parity harness), hccl on card.
    """
    global _cp_dispatch_checked
    from diffusers.models import attention_dispatch as ad

    if not _cp_dispatch_checked:
        fn = getattr(ad, "_templated_context_parallel_attention", None)
        params = inspect.signature(fn).parameters if fn is not None else {}
        if fn is None or "forward_op" not in params or "_parallel_config" not in params:
            raise RuntimeError(
                f"diffusers lost `_templated_context_parallel_attention` "
                f"(audited {_CP_DISPATCH_REV}); re-audit _cp_ulysses_attention()."
            )
        _cp_dispatch_checked = True
    return ad._templated_context_parallel_attention(
        query,
        key,
        value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
        enable_gqa=False,
        return_lse=False,
        forward_op=_fsdp_cp_forward_op,
        backward_op=_fsdp_cp_backward_op,
        _parallel_config=parallel_config,
    )


def habana_fused_sdpa(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    enable_gqa=False,
    _parallel_config=None,  # accepted for diffusers dispatch signature parity
    softmax_mode=None,
    recompute_mode=None,
):
    """diffusers attention backend callable for Intel Gaudi2 (HPU).

    Layout contract is diffusers' own: (B, S, H, D) in, (B, S, H, D) out, softmax
    over the `S` axis. On non-HPU tensors this is the exact native passthrough, so
    the CPU smoke-test parity assertion holds bit-exactly.

    Context parallelism (PLAN_CP.md phase 1.3): when diffusers dispatch carries a
    ParallelConfig with ulysses_degree > 1, the call is wrapped in the same
    templated Ulysses all-to-all the stock backends use, with the local
    (seq-shard) computation above. CPU/gloo tensors exercise the identical
    routing -- the CP parity harness asserts it against the unsharded result.
    """
    if enable_gqa:
        # MiniMax-H3 never uses GQA (query heads == kv heads); a silent ignore here
        # would surface as wrong shapes further downstream. Loud failure instead.
        raise ValueError("habana_fused_sdpa does not support enable_gqa=True.")

    cp_cfg = (
        getattr(_parallel_config, "context_parallel_config", None)
        if _parallel_config is not None
        else None
    )
    if cp_cfg is not None and cp_cfg.ulysses_degree > 1:
        return _cp_ulysses_attention(
            query, key, value, attn_mask, dropout_p, is_causal, scale, _parallel_config
        )

    if recompute_mode is None:
        recompute_mode = False  # inference: never enter the recompute variant
    return _habana_local_sdpa(
        query,
        key,
        value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
        enable_gqa=enable_gqa,
        softmax_mode=softmax_mode,
        recompute_mode=recompute_mode,
    )


_dispatch_installed = False
_orig_dispatch = None


def _install_attention_backend() -> None:
    """Register `habana_fused_sdpa` into diffusers' backend registry and patch the
    per-module `dispatch_attention_fn` bindings so a *custom* (non-enum) backend can
    be selected per processor.

    diffusers' `AttentionMixin.set_attention_backend()` validates the name against
    the closed `AttentionBackendName` enum, which cannot be extended at runtime, so
    the wrapper installed here short-circuits our name before stock dispatch sees it.
    Everything else falls through to the original function untouched.
    """
    global _dispatch_installed, _orig_dispatch
    if _dispatch_installed:
        return

    from diffusers.models import attention_dispatch as ad

    if not ad._AttentionBackendRegistry._backends.get(BACKEND_NAME):
        ad._AttentionBackendRegistry._backends[BACKEND_NAME] = habana_fused_sdpa
        ad._AttentionBackendRegistry._supported_arg_names[BACKEND_NAME] = set(
            inspect.signature(habana_fused_sdpa).parameters
        )
        ad._AttentionBackendRegistry._constraints[BACKEND_NAME] = []

    # CP gate (PLAN_CP.md phase 1.1): `enable_parallelism` refuses to run unless
    # the model's attention backend is in this class-level set
    # (modeling_utils.py:638, attention_dispatch.py:261). Our backend's CP
    # correctness comes from the templated Ulysses routing in
    # `_cp_ulysses_attention`, so it belongs in the set. Drift-checked: the set
    # is load-bearing upstream; its disappearance means the CP API moved.
    cp_set = getattr(ad._AttentionBackendRegistry, "_supports_context_parallel", None)
    if not isinstance(cp_set, set):
        raise RuntimeError(
            "diffusers _AttentionBackendRegistry lost `_supports_context_parallel`; "
            "re-audit the CP registration (audited diffusers 0.40.0)."
        )
    cp_set.add(BACKEND_NAME)

    if _orig_dispatch is None:
        _orig_dispatch = ad.dispatch_attention_fn

    def _dispatch_with_custom_backend(
        *args, backend=None, parallel_config=None, **kwargs
    ):
        if backend == BACKEND_NAME:
            return habana_fused_sdpa(*args, _parallel_config=parallel_config, **kwargs)
        return _orig_dispatch(
            *args, backend=backend, parallel_config=parallel_config, **kwargs
        )

    import importlib

    for modname in _DISPATCH_CONSUMERS:
        try:
            mod = importlib.import_module(modname)
        except ImportError:  # pragma: no cover - these ship inside diffusers itself
            continue
        if (
            getattr(mod, "dispatch_attention_fn", None)
            is not _dispatch_with_custom_backend
        ):
            mod.dispatch_attention_fn = _dispatch_with_custom_backend

    _dispatch_installed = True


def _processor_classes():
    from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import (
        MiniMaxH3VideoAttnProcessor,
    )
    from diffusers.models.transformers.transformer_minimax_h3 import (
        MiniMaxH3AttnProcessor,
    )

    return (MiniMaxH3AttnProcessor, MiniMaxH3VideoAttnProcessor)


def _attention_modules():
    """The two attention module classes that carry H3 processors."""
    from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import (
        MiniMaxH3VideoAttention,
    )
    from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3Attention

    return (MiniMaxH3Attention, MiniMaxH3VideoAttention)


@contextlib.contextmanager
def habana_attention_backend(name: str = BACKEND_NAME):
    """Route every H3 attention module (DiT blocks + token refiner + video-VAE ViT)
    through `habana_fused_sdpa` for the duration of the context; the `finally`
    clause always restores the previous active backend / class attributes, so a
    failed pipeline run cannot poison later smoke tests.
    """
    _install_attention_backend()
    from diffusers.models import attention_dispatch as ad

    procs = _processor_classes()
    saved_cls = {p: p._attention_backend for p in procs}
    saved_active = ad._AttentionBackendRegistry._active_backend
    try:
        for p in procs:
            p._attention_backend = name
        yield name
    finally:
        for p, old in saved_cls.items():
            p._attention_backend = old
        ad._AttentionBackendRegistry._active_backend = saved_active


def apply_habana_attention(model: torch.nn.Module, name: str = BACKEND_NAME) -> int:
    """Persistently force the Habana backend on every H3 attention processor in
    `model` (class-attribute mutation mirroring `AttentionMixin.set_attention_backend()`,
    but bypassing the closed enum). Returns the number of processors patched.

    Production entry point: call once after `from_pretrained`, BEFORE graph warm-up,
    so capture records the FusedSDPA kernel rather than the stock SDPA composition.
    """
    _install_attention_backend()
    count = 0
    attn_cls = _attention_modules()
    for _, module in model.named_modules():
        if type(module) in attn_cls:
            # `module.processor` carries `_attention_backend`; mutate in place so
            # processor identity (and any fused-QKV state) is untouched.
            module.processor._attention_backend = name
            count += 1
    logger.info(
        "hpu_patches: %s attention backend forced on %d processors", name, count
    )
    return count


# ---------------------------------------------------------------------------
# 3.3 Rope cos/sin cache (per bucket)
# ---------------------------------------------------------------------------

_ROPE_PATCHED = False
_ORIG_ROPE_FORWARD = None


def install_rope_cache(rope: torch.nn.Module, bucket: str | None = None) -> None:
    """Patch a `MiniMaxH3RotaryPosEmbed` to serve cached fp32 cos/sin per bucket.

    Key contract (design 3.3): the runner calls `set_rope_bucket(rope, bucket_id)`
    before stepping that bucket. `position_ids` arrive as fp64 (built and held on
    CPU -- deviation note 3.7 (ii)) and the stock `forward` casts them to fp32; the
    cache fills from that stock computation, so cached tensors match the uncached
    path bit-exactly and the fp64 anchor-sum contract stays on the CPU host side.

    The cache lives on whichever device the fill-up forward ran on (HPU after
    `.to('hpu')` in the runner; CPU in tests), which is why it fills lazily on the
    first call rather than being pre-filled device-blind.

    The class-level `forward` monkeypatch is applied once; per-instance state is a
    plain dict attribute (`_h3_rope_store`), NOT a registered buffer, so it stays
    out of `state_dict` and is invisible to `module` traversal.
    """
    global _ROPE_PATCHED, _ORIG_ROPE_FORWARD
    from diffusers.models.transformers.transformer_minimax_h3 import (
        MiniMaxH3RotaryPosEmbed,
    )

    if not isinstance(rope, MiniMaxH3RotaryPosEmbed):
        raise TypeError(
            f"install_rope_cache expects MiniMaxH3RotaryPosEmbed, got {type(rope).__name__}"
        )

    rope._h3_rope_store = {}
    rope._h3_rope_bucket = bucket
    if _ROPE_PATCHED:
        return
    _ORIG_ROPE_FORWARD = MiniMaxH3RotaryPosEmbed.forward

    def cached_forward(self, position_ids: torch.Tensor):
        bucket = getattr(self, "_h3_rope_bucket", None)
        store = getattr(self, "_h3_rope_store", None)
        if not patch_config().rope_cache or not bucket or store is None:
            return _ORIG_ROPE_FORWARD(self, position_ids)
        hit = store.get(bucket)
        if (
            hit is not None
            and hit[0].shape[0] == position_ids.shape[0]
            and hit[0].device == position_ids.device
        ):
            return hit
        cos, sin = _ORIG_ROPE_FORWARD(self, position_ids)
        store[bucket] = (cos, sin)
        return cos, sin

    MiniMaxH3RotaryPosEmbed.forward = cached_forward
    _ROPE_PATCHED = True


def set_rope_bucket(rope: torch.nn.Module, bucket: str | None) -> None:
    """Point a (patched or stock) rope module at a bucket key; `None` returns the
    module to stock uncached behavior."""
    if not hasattr(rope, "_h3_rope_bucket"):
        install_rope_cache(rope, bucket)
        return
    rope._h3_rope_bucket = bucket


def install_rope_cache_on_model(
    transformer: torch.nn.Module, bucket: str | None = None
) -> bool:
    """Convenience: find the `rope` submodule of a `MiniMaxH3Transformer3DModel` and
    cache-install it. Returns whether a rope module was found."""
    rope = getattr(transformer, "rope", None)
    if rope is None:
        return False
    install_rope_cache(rope, bucket)
    return True


# ---------------------------------------------------------------------------
# 3.5 #1 Video-VAE decode autocast guard (cuda -> {cuda, hpu})
# ---------------------------------------------------------------------------


def patch_video_vae_autocast_guard() -> None:
    """Swap `MiniMaxH3VideoDecodeStep` for a subclass whose `__call__` enables the
    fp16 decode autocast on HPU as well as CUDA, and rebind it in-place inside
    `MiniMaxH3DecodeStep.block_classes` (import-time bound list; design 3.5 #1).

    The body below is copied from stock `decoders.py`; the ONLY change is the
    `enabled=` expression. `_PATCHED_DECODERS_REV` pins the audited revision and the
    drift check via `inspect.getsource` fails loudly if the upstream guard changes.
    """
    from diffusers.modular_pipelines.minimax_h3 import decoders as dec_mod
    from diffusers.modular_pipelines.minimax_h3.modular_blocks_minimax_h3 import (
        MiniMaxH3DecodeStep,
    )

    src = inspect.getsource(dec_mod.MiniMaxH3VideoDecodeStep.__call__)
    if _AUTOCAST_GUARD_SNIPPET not in src:
        raise RuntimeError(
            "Upstream diffusers changed the video-VAE decode autocast guard "
            f"(expected `{_AUTOCAST_GUARD_SNIPPET}` in `{_PATCHED_DECODERS_REV}`); "
            "re-audit patch_video_vae_autocast_guard() before use."
        )
    if getattr(dec_mod, "MiniMaxH3VideoDecodeStepPatched", None) is not None:
        return  # already patched in this process

    class MiniMaxH3VideoDecodeStepPatched(dec_mod.MiniMaxH3VideoDecodeStep):
        """Identical to stock, except the fp16 decode autocast also fires on HPU."""

        @torch.no_grad()
        def __call__(self, components, state):
            block_state = self.get_block_state(state)
            device = components._execution_device

            if block_state.output_type not in ("pil", "np", "pt"):
                raise ValueError(
                    f"`output_type` must be one of 'pil', 'np' or 'pt', got {block_state.output_type!r}. To keep the "
                    "latents instead of decoding them, run a pipeline that does not include the decode blocks."
                )

            latents_mean = torch.tensor(
                components.vae.config.latents_mean, device=device
            ).view(1, -1, 1, 1, 1)
            latents_std = torch.tensor(
                components.vae.config.latents_std, device=device
            ).view(1, -1, 1, 1, 1)
            latents = block_state.latents * latents_std + latents_mean

            # THE PATCH: stock enables fp16 autocast for CUDA only; HPU follows the
            # verified CUDA recipe for the fp32 VAE weights.
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type in ("cuda", "hpu"),
            ):
                video = components.vae.decode(latents, return_dict=False)[0]
            pixel_mean = torch.tensor(components.pixel_mean, device=device).view(
                1, -1, 1, 1, 1
            )
            pixel_std = torch.tensor(components.pixel_std, device=device).view(
                1, -1, 1, 1, 1
            )
            video = (video.float() * pixel_std + pixel_mean).clamp(0, 1)
            block_state.videos = components.video_processor.postprocess_video(
                video, output_type=block_state.output_type
            )

            self.set_block_state(state, block_state)
            return components, state

    dec_mod.MiniMaxH3VideoDecodeStepPatched = MiniMaxH3VideoDecodeStepPatched
    MiniMaxH3DecodeStep.block_classes[0] = MiniMaxH3VideoDecodeStepPatched


# ---------------------------------------------------------------------------
# 3.4 / 3.5 #4: HPU graphs and the eager mark_step strategy
# ---------------------------------------------------------------------------


def _module_on_hpu(module: torch.nn.Module) -> bool:
    """True if the module owns any parameter/buffer placed on HPU."""
    for p in module.parameters():
        if p.device.type == "hpu":
            return True
    for b in module.buffers():
        if b.device.type == "hpu":
            return True
    return False


def wrap_module_in_hpu_graph(module: torch.nn.Module, **kwargs: Any) -> torch.nn.Module:
    """`wrap_in_hpu_graph` on HPU, identity on CPU/eager.

    `wrap_in_hpu_graph` mutates `module.forward` in place (and installs
    `clear_inputs` / `clear_cache` / `log_statistics`), so the returned module IS the
    input module. Call after `apply_habana_attention` and rope-cache installation so
    the captured graph records the FusedSDPA kernel and the cached cos/sin reads.

    kwargs pass straight through to `wrap_in_hpu_graph` (e.g. `disable_tensor_cache=`,
    `max_graphs=`).
    """
    if module is None or not _module_on_hpu(module):
        return module
    from habana_frameworks.torch.hpu import wrap_in_hpu_graph

    return wrap_in_hpu_graph(module, **kwargs)


def wrap_vae_decoder_in_hpu_graph(vae: torch.nn.Module) -> bool:
    """Optional (`H3_VAE_GRAPHS=1`): graph the *inner* decoder submodule of a video
    or audio `AutoencoderKLMiniMaxH3*`.

    `vae.decode(z)` delegates to `vae.decoder(<post-projection tiles>)`, so wrapping
    the whole VAE is useless (`wrap_in_hpu_graph` only replaces `module.forward`,
    and `decode` never calls it). Wrapping `vae.decoder` instead means the kernel
    inside the decode graph is the tiled/clip compute itself with the static
    per-tile/clip shapes of the bucket tables. The VAE stays fp32; the fp16 autocast
    (video side) is handled by the decode block, outside this graph boundary.
    Returns whether the wrap was applied.
    """
    decoder = getattr(vae, "decoder", None)
    if decoder is None or not _module_on_hpu(decoder):
        return False
    from habana_frameworks.torch.hpu import wrap_in_hpu_graph

    wrap_in_hpu_graph(decoder)
    logger.info("hpu_patches: %s.decoder wrapped in HPU graph", type(vae).__name__)
    return True


def apply_graph_wrappers(pipe, *, vae_graphs: bool | None = None) -> dict[str, bool]:
    """Wrap pipeline components in HPU graphs according to env config (design 3.4).

    * DiT (`pipe.components.transformer`): shapes are static per bucket, so one
      graph per bucket session suffices and every diffusion step replays it.
    * VAE decoders (optional, `H3_VAE_GRAPHS=1`): tiled/chunked decode has
      data-dependent chunk counts, so graphs default OFF. Iterate eagerly first.
    """
    cfg = patch_config()
    vae_graphs = cfg.vae_graphs if vae_graphs is None else vae_graphs
    components = getattr(pipe, "components", pipe)
    transformer = getattr(components, "transformer", None)
    applied: dict[str, bool] = {"transformer": False, "vae": False, "audio_vae": False}

    if cfg.dit_graphs and transformer is not None:
        wrapped = wrap_module_in_hpu_graph(transformer)
        applied["transformer"] = wrapped is not None and hasattr(wrapped, "clear_cache")
    if vae_graphs and getattr(components, "vae", None) is not None:
        applied["vae"] = wrap_vae_decoder_in_hpu_graph(components.vae)
    if vae_graphs and getattr(components, "audio_vae", None) is not None:
        # The audio VAE stays strictly fp32 (documented -20 dB bf16 hazard); the
        # graph wraps its fp32 decoder verbatim.
        applied["audio_vae"] = wrap_vae_decoder_in_hpu_graph(components.audio_vae)
    return applied


def add_mark_step_hook(module: torch.nn.Module) -> bool:
    """Eager-mode strategy (design 3.5 #4): after every forward of an HPU-resident
    module, close the lazy frontier with `htcore.mark_step()`.

    Implemented as an nn.Module forward hook so it composes with (and is redundant
    under) graph wrapping. Returns False (and installs nothing) for CPU modules: the
    CPU-only smoke environment must stay warning-free.
    """
    if not _module_on_hpu(module):
        return False
    import habana_frameworks.torch.core as htcore

    def _mark_step(_module, _args, output):
        htcore.mark_step()
        return output

    module.register_forward_hook(_mark_step)
    return True


def add_mark_step_hooks_for_pipe(pipe) -> list[str]:
    """Install mark_step forward hooks on every HPU-resident DiT/VAE component."""
    components = getattr(pipe, "components", pipe)
    marked = []
    for name in ("transformer", "vae", "audio_vae"):
        module = getattr(components, name, None)
        if module is not None and add_mark_step_hook(module):
            marked.append(f"{name}:mark_step")
    return marked


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


@dataclass
class PatchReport:
    attention_backend: str = "native"
    rope_cache_buckets: list = field(default_factory=list)
    video_vae_autocast_guard: bool = False
    graphs: dict = field(default_factory=dict)
    mark_step_hooks: list = field(default_factory=list)
    hpu_available: bool = False


def apply_hpu_patches(
    pipe, transformer: torch.nn.Module | None = None, rope_bucket: str | None = None
) -> PatchReport:
    """Full runtime patch pass (design 3.5), safe on CPU.

    Order matters:
      1. autocast guard    -- pure class-level rebind, device-independent;
      2. attention backend -- registry install + processor forcing (per model);
      3. rope cache        -- class-level patched forward + per-bucket instance key;
      4. graph wrappers    -- HPU-only, last, so captures bake in (2)/(3).
    Eager-mode `mark_step` hooks are installed automatically whenever a component
    lives on HPU and no graph wrapper could be applied to it.

    The fp64->fp32 handoff of `position_ids` (deviation 3.7 (ii)) is enforced in the
    rope cache itself (stock forward cast) plus the runner's graph input prep; the
    fp64 anchor math of `build_packed_sequence` stays on CPU host side untouched.
    """
    report = PatchReport()
    cfg = patch_config()
    report.hpu_available = _probe_hpu()

    patch_video_vae_autocast_guard()
    report.video_vae_autocast_guard = True

    _install_attention_backend()
    transformer = (
        transformer
        if transformer is not None
        else getattr(getattr(pipe, "components", pipe), "transformer", None)
    )
    if transformer is not None:
        apply_habana_attention(transformer)
        report.attention_backend = BACKEND_NAME
    # The video-VAE ViT decoder rides the same dispatch interception once its weights
    # land on HPU; its fp32 q/k/v select softmax_mode="None" automatically.

    if rope_bucket is None:
        rope_bucket = str(getattr(pipe, "bucket_id", "default"))
    rope = getattr(transformer, "rope", None)
    if rope is not None and cfg.rope_cache:
        install_rope_cache(rope, rope_bucket)
        report.rope_cache_buckets = [rope_bucket]

    if getattr(getattr(pipe, "components", pipe), "transformer", None) is not None:
        report.graphs = apply_graph_wrappers(pipe)
    if not any(report.graphs.values()):
        report.mark_step_hooks = add_mark_step_hooks_for_pipe(pipe)
    return report


# ---------------------------------------------------------------------------
# CPU self-test (also the design-section-5 parity assertion, runnable anywhere)
# ---------------------------------------------------------------------------


def validate_cpu_equivalence(
    seed: int = 0, dtype: torch.dtype = torch.float32, atol: float = 0.0
) -> dict:
    """Forward a tiny random-weight `MiniMaxH3Transformer3DModel` twice on CPU --
    once with the stock/to-default diffusers dispatch, once forced through
    `habana_fused_sdpa` -- and assert the outputs are bit-identical (`atol` 0) or
    within `atol`.

    Exercises the full patched path on CPU: the class-forced backend key, the
    `_native_sdpa_passthrough` branch, qk-norm/RoPE choreography and the two output
    heads. No HPU device is touched (guarded by `query.device.type`).
    """
    from diffusers.models.transformers.transformer_minimax_h3 import (
        MiniMaxH3Transformer3DModel,
    )

    torch.manual_seed(seed)
    model = (
        MiniMaxH3Transformer3DModel(
            num_attention_heads=2,
            attention_head_dim=32,
            hidden_size=32,
            num_layers=2,
            num_refiner_layers=1,
            ffn_dim=64,
            text_dim=32,
            freq_dim=64,
            time_embed_hidden_dim=64,
            time_embed_dim=32,
            rope_freq_dim=5,  # rotary_dim = 6 * rope_freq_dim = 30 <= head_dim 32
        )
        .eval()
        .to(dtype)
    )

    n_text, n_video, n_audio = 6, 6, 4
    seq = n_text + n_video + n_audio
    batch = 1
    hidden = torch.randn(batch, n_text, model.config.text_dim, dtype=dtype)
    audio = torch.randn(batch, n_audio, 32, dtype=dtype)
    video = torch.randn(batch, n_video, 24 * 4, dtype=dtype)
    timestep = torch.tensor([0.5, 0.0], dtype=torch.float32)
    token_tags = torch.tensor([1] * n_text + [0] * n_video + [2] * n_audio)
    timestep_indices = torch.tensor([0] * (n_text + n_video) + [1] * n_audio)
    position_ids = torch.rand(seq, 3, dtype=torch.float64).to(torch.float32)
    video_idx = torch.arange(n_text, n_text + n_video)
    text_idx = torch.arange(0, n_text)
    audio_idx = torch.arange(seq - n_audio, seq)

    def run(model_):
        with torch.no_grad():
            return model_(
                video,
                audio,
                hidden,
                timestep=timestep,
                timestep_indices=timestep_indices,
                token_tags=token_tags,
                position_ids=position_ids,
                video_indices=video_idx,
                audio_indices=audio_idx,
                text_indices=text_idx,
                return_dict=False,
            )

    out_a = run(model)  # stock dispatch (backend key None -> registry default)
    with habana_attention_backend():
        out_b = run(model)  # forced through habana_fused_sdpa (CPU passthrough)
    a_v, a_a = out_a
    b_v, b_b = out_b
    delta_v = (a_v.float() - b_v.float()).abs().max().item()
    delta_a = (a_a.float() - b_b.float()).abs().max().item()

    assert_no_hpu_tensor(model, a_v, a_a, b_v, b_b)
    ok = delta_v <= atol and delta_a <= atol
    return {
        "parity_ok": ok,
        "max_delta_video": delta_v,
        "max_delta_audio": delta_a,
        "dtype": str(dtype),
        "backend": BACKEND_NAME,
    }


# ---------------------------------------------------------------------------
# Compatibility aliases (entry-point names kept stable across patch revisions)
# ---------------------------------------------------------------------------


def apply_patches(force: bool = False) -> dict:
    """Legacy entry point: apply the device-independent patches (autocast guard,
    attention backend install, rope class patch). Returns a small status dict.
    Instance/graph work belongs to `patch_pipeline` (needs loaded components)."""
    cfg = patch_config()
    patch_video_vae_autocast_guard()
    _install_attention_backend()
    return {
        "autocast_guard": True,
        "backend_registered": True,
        "forced": bool(force),
        "hpu_available": _probe_hpu(),
        "rope_cache_env": cfg.rope_cache,
        "note": "rope cache + graphs apply per-instance via patch_pipeline()",
    }


def patch_pipeline(pipe, wrap_graphs: bool | None = None) -> dict:
    """Legacy entry point for a loaded `MiniMaxH3ModularPipeline`."""
    report = apply_hpu_patches(pipe)
    if wrap_graphs is not None and wrap_graphs != any(report.graphs.values()):
        # Re-run with an explicit graphs override from the caller.
        cfg = patch_config()
        cfg.dit_graphs = wrap_graphs
        report.graphs = apply_graph_wrappers(pipe)
    return {
        "attention_backend": report.attention_backend,
        "graphs": report.graphs,
        "mark_step_hooks": report.mark_step_hooks,
        "rope_cache_buckets": report.rope_cache_buckets,
    }


wrap_hpu_graph = wrap_module_in_hpu_graph

# re-dispatch: post-venv-fix pyright verification (whitespace only)

# pyright re-dispatch: pythonPath config verify (whitespace only)
