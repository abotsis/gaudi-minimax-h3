#!/usr/bin/env python
"""H3-FL2VA -> Gaudi2 runner (t2va / fl2va CLI).

Runner + launcher side of the H3 HPU port (design section 4). Loads the
`MiniMaxH3ModularPipeline` from the local MiniMax-H3 checkout through a
symlink workdir that remaps the legacy-shipped component class names onto the
diffusers in-tree classes, applies the HPU port's runtime monkeypatches
(device map, packed-sequence attention backend, rope cache, fp16-autocast
video-VAE decode guard, zero-embed text budget), runs one generation and
exports `frames/*.png` + `audio.wav`, then muxes into an mp4 with ffmpeg.

Everything HPU-touching is behind two gates:

  * `H3_ALLOW_HPU=1` / `--allow-hpu` (set by launch_h3.sh once the user has
    approved a card), and
  * `torch.hpu.is_available()` actually reporting HPU.

Without both, this script is a pure-CPU program: identical code paths, `hpu`
device tips collapse to `cpu`. It must never touch HPU devices from a CPU-only
session (the production cards run the GLM-5.3-Flash vLLM server).

Design sections 3.1-3.5 expect sibling modules under this package
(load_shim.py, attention.py, rope.py, patch.py, graphs.py); where a sibling
exists it is imported and used, otherwise the inline fallbacks below implement
the same contracts. Reason: this agent's file ownership is limited to
run_t2va.py / launch_h3.sh / buckets.json, so the runner stays self-sufficient
when the modularized siblings are absent, and defers to them when present.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import importlib
import inspect
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import types
import wave
from contextlib import suppress
from pathlib import Path
from typing import Any

REPO_DEFAULT = "/root/src/h3/MiniMax-H3"
DEFAULT_DEVICE_MAP = "te=cpu,dit=hpu,vae=hpu,audio=hpu"
DEFAULT_EMBED_CACHE_DIR = "/root/src/h3/hpu_port/.ecache"

# ---------------------------------------------------------------------------
# Env flags (design 3.6). Read lazily so callers can monkeypatch os.environ.
# ---------------------------------------------------------------------------


def _env_flag(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


# torch is imported lazily (report-only runs must work without it); class-body
# decorators need the module global, hence `_ensure_torch()`.
torch: Any = None  # populated by _ensure_torch()


def _ensure_torch():
    global torch
    if torch is None:
        torch = importlib.import_module("torch")
    return torch


def _try_module(name: str):
    """Import hpu_port.<name> when a sibling implementation exists."""
    try:
        return importlib.import_module(f"hpu_port.{name}")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Bucket table (design 2 + 4.4)
# ---------------------------------------------------------------------------

BUCKETS_PATH = Path(__file__).resolve().parent / "buckets.json"


def load_buckets(path: Path | None = None) -> dict:
    with open(path or BUCKETS_PATH) as fh:
        return json.load(fh)


def resolve_resolution(table: dict, requested: str) -> tuple[str, int, int]:
    """Match a `WxH` request or a bucket id onto a table row.

    Portrait variants are the (h, w)-swapped tuple of the landscape row and map
    to the same shape count. Anything else snaps to the nearest valid bucket
    (aspect ratio first, then area difference), warning loudly; unsupported
    combos are never silently padded.
    """
    res = table["resolutions"]
    if requested in res:
        hw = res[requested]["hw"]
        return requested, hw[0], hw[1]
    if "x" not in requested:
        raise ValueError(
            f"--resolution {requested!r} is neither a bucket id nor a WxH string."
        )
    try:
        w, h = (int(part) for part in requested.lower().split("x"))
    except ValueError as exc:
        raise ValueError(
            f"--resolution {requested!r} does not parse as WxH: {exc}"
        ) from exc

    def sort_key(item):
        name, spec = item
        bh, bw = spec["hw"]
        if (w, h) in ((bw, bh), (bh, bw)):
            return (0.0, 0.0, name)
        aspect = (
            abs((w / h) - (bw / bh)) / max(bw / bh, bh / bw)
            if 0 not in (bw, bh)
            else 9e9
        )
        area = abs((w * h) - (bw * bh))
        return (aspect, area, name)

    best_name = sorted(sort_key(item) for item in res.items())[0][2]
    if best_name == "r720" and not _env_bool("H3_ALLOW_R720", False):
        raise ValueError(
            "r720 (1280x704 / 704x1280) is a LATE/optional bucket in this port; "
            "set H3_ALLOW_R720=1 to unlock it. Supported buckets: "
            + ", ".join(sorted(k for k in res if k != "r720"))
        )
    bw, bh = res[best_name]["hw"]
    if (w, h) not in ((bw, bh), (bh, bw)):
        print(
            f"[h3] WARNING: resolution {requested} is not a supported bucket; snapping to {best_name} "
            f"({bw}x{bh} / {bh}x{bw})."
        )
    return best_name, bw, bh


def resolve_duration(table: dict, requested: int) -> tuple[int, int]:
    """Nearest supported duration -> (frames, T_lat); warns on snap.

    >15 s is rejected outright (the model contract caps at 15 s = 345 frames;
    e.g. 360 cannot be satisfied by any 17n+5 alignment in range).
    """
    durations = {int(k): v for k, v in table["durations"].items()}
    if requested in durations:
        spec = durations[requested]
        return spec["frames"], spec["T_lat"]
    if requested <= 0:
        raise ValueError(f"--duration must be positive, got {requested}.")
    if requested > 15:
        raise ValueError(
            f"--duration {requested}s exceeds the MiniMax-H3 15 s contract; use one of {sorted(durations)}."
        )
    best = min(durations, key=lambda d: (abs(d - requested), d))
    spec = durations[best]
    print(
        f"[h3] WARNING: duration {requested}s is not a supported bucket; snapping to {best}s "
        f"({spec['frames']} frames, T_lat {spec['T_lat']})."
    )
    return spec["frames"], spec["T_lat"]


# ---------------------------------------------------------------------------
# Workdir builder (design 3.1): symlinks onto the legacy FL2VA partition plus a
# rewritten model_index.json remapping legacy class names -> diffusers in-tree.
# Entries are the len-2 `[library, class]` form so the blocks loader assigns
# `repo=workdir` and `subfolder=<component name>` itself.
# ---------------------------------------------------------------------------

CLASS_REMAP = {
    "MiniMaxH3Qwen3VLHFEncoder": ("transformers", "Qwen3VLForConditionalGeneration"),
    "MiniMaxH3AudioVAE": ("diffusers", "AutoencoderKLMiniMaxH3Audio"),
    "MiniMaxH3DiTModel": ("diffusers", "MiniMaxH3Transformer3DModel"),
    "MiniMaxH3VideoVAE": ("diffusers", "AutoencoderKLMiniMaxH3"),
}
# Block component name -> legacy FL2VA directory name when they disagree.
COMPONENT_DIRS = {
    "text_encoder": "FL2VA/text_encoder",
    "tokenizer": "FL2VA/tokenizer",
    "processor": "FL2VA/processor",
    # The diffusers-format weights live at the REPO TOP LEVEL (the FL2VA
    # partition is the SGLang/vLLM original checkpoint whose keys use the
    # `blocks.*` naming -- incompatible with the in-tree diffusers modules).
    "vae": "vae",
    "audio_vae": "audio_vae",
    "transformer": "transformer",
}
# The legacy partition ships the scheduler shifts in model_index `_minimax_h3`
# rather than as subfolder configs; the blocks expect both schedulers.
SCHEDULER_SHIFTS = {"scheduler": ("video", 12.0), "audio_scheduler": ("audio", 3.0)}
# Block name -> tip of the default device map (design 3.6). te/dit/vae/audio.
DEFAULT_TIP = {
    "text_encoder": "te",
    "transformer": "dit",
    "vae": "vae",
    "audio_vae": "audio",
}
COMPONENT_CLASS_INDEX = {
    "text_encoder": ["transformers", "Qwen3VLForConditionalGeneration"],
    "tokenizer": ["transformers", "Qwen2TokenizerFast"],
    "processor": ["transformers", "Qwen3VLProcessor"],
    "vae": ["diffusers", "AutoencoderKLMiniMaxH3"],
    "audio_vae": ["diffusers", "AutoencoderKLMiniMaxH3Audio"],
    "transformer": ["diffusers", "MiniMaxH3Transformer3DModel"],
    "scheduler": ["diffusers", "MiniMaxH3Scheduler"],
    "audio_scheduler": ["diffusers", "MiniMaxH3Scheduler"],
}


def build_workdir(
    repo: Path | str, workflow: str = "t2va", workdir: Path | None = None
) -> Path:
    """Materialize the loader workdir (symlinks + remapped model_index.json).

    Idempotent: reuses an existing workdir whose remap-spec hash matches.
    Reads only small json/config files; never opens `.safetensors` blobs.
    """
    shim = _try_module("load_shim")
    if shim is not None and hasattr(shim, "build_workdir"):
        return shim.build_workdir(repo=repo, workflow=workflow, workdir=workdir)

    repo = Path(repo)
    workdir = workdir or (Path(__file__).resolve().parent / "workdir" / workflow)
    fl2va = repo / "FL2VA"
    legacy_path = fl2va / "model_index.json"
    with open(legacy_path) as fh:
        legacy = json.load(fh)

    spec_hash = hashlib.sha256(
        (json.dumps(legacy, sort_keys=True) + "|remap-spec:v3").encode()
    ).hexdigest()[:16]
    marker = workdir / ".remap_spec.json"
    if marker.exists():
        try:
            spec = json.loads(marker.read_text())
            if spec.get("hash") == spec_hash and spec.get("spec_version") == 3:
                return workdir
        except (json.JSONDecodeError, OSError):
            pass
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)

    for block_name, rel_dir in COMPONENT_DIRS.items():
        src = repo / rel_dir
        if not src.is_dir():
            raise RuntimeError(f"H3 repo is missing component directory {src}")
        (workdir / block_name).symlink_to(src.resolve(), target_is_directory=True)

    # Scheduler configs (config-only components; never weight reads).
    sigma_shifts = (legacy.get("_minimax_h3") or {}).get("sigma_shift_scales", {})
    for name, (modality, default_shift) in SCHEDULER_SHIFTS.items():
        shift = float(sigma_shifts.get(modality, default_shift))
        sched_dir = workdir / name
        sched_dir.mkdir()
        with open(sched_dir / "scheduler_config.json", "w") as fh:
            json.dump(
                {"_class_name": "MiniMaxH3Scheduler", "shift": shift}, fh, indent=2
            )

    index: dict[str, Any] = {
        "_class_name": "MiniMaxH3ModularPipeline",
        "_diffusers_version": "0.40.0",
        "_blocks_class_name": "MiniMaxH3Blocks",
        "_remap_spec_version": 3,
    }
    if "_minimax_h3" in legacy:
        meta = dict(legacy["_minimax_h3"])
        meta.pop("partition", None)  # partition resolves through the workdir symlinks
        index["_minimax_h3"] = meta
    # Component entries MUST be [lib, cls, {pretrained_model_name_or_path,
    # subfolder, ...}]: load_components() silently skips any spec whose
    # `pretrained_model_name_or_path` is None, and bare [lib, cls] entries bind
    # no path. Bind every component to the workdir itself, subfolder = the
    # symlinked component dir (scheduler dirs are config-only, same scheme).
    for name in COMPONENT_CLASS_INDEX:
        lib, cls = COMPONENT_CLASS_INDEX[name]
        index[name] = [
            lib,
            cls,
            {
                "type_hint": [lib, cls],
                "pretrained_model_name_or_path": str(workdir),
                "subfolder": name,
                "variant": None,
                "revision": None,
            },
        ]
    with open(workdir / "model_index.json", "w") as fh:
        json.dump(index, fh, indent=2)
    # ModularPipeline.resolve reads `_blocks_class_name` from the *modular* config
    # only; the model_index.json fallback binds `config_dict`, which __init__
    # does not consult for blocks resolution. Write the modular filename too.
    with open(workdir / "modular_model_index.json", "w") as fh:
        json.dump(index, fh, indent=2)
    # --- diffusers-style weight-name normalization ------------------------
    # diffusers ModelMixin.from_pretrained resolves only
    # `diffusion_pytorch_model[.safetensors|.bin|-sharded + .index.json]`.
    # The FL2VA partition ships transformers-style names (`model.safetensors`,
    # sharded `model-000NN-of-000NN.safetensors`): normalize `transformer` and
    # `audio_vae` into real dirs of symlinks under diffusers names. The text
    # encoder needs no handling (transformers accepts `model.safetensors`) and
    # `vae` already uses diffusers naming at the repo root.
    shard_re = re.compile(r"^model-(\d+)-of-(\d+)")
    for comp in ("transformer", "audio_vae"):
        src = Path(os.path.realpath(workdir / comp))
        legacy_files = list(src.glob("*.safetensors")) + list(
            src.glob("*.safetensors.index.json")
        )
        if not legacy_files:
            raise RuntimeError(
                f"{src} has no .safetensors weights (download incomplete?)"
            )
        norm_dir = workdir / comp
        norm_dir.unlink()  # replace the plain symlink with a real dir
        norm_dir.mkdir()
        shard_new_names: dict[str, str] = {}
        for f in sorted(set(legacy_files), key=lambda p: p.name):
            new_name = shard_re.sub(r"diffusion_pytorch_model-\1-of-\2", f.name)
            if f.name == "model.safetensors":
                new_name = "diffusion_pytorch_model.safetensors"
            if f.name.endswith(".index.json") and f.name.startswith("model."):
                new_name = "diffusion_pytorch_model.safetensors.index.json"
            shard_new_names[f.name] = new_name
            (norm_dir / new_name).symlink_to(f.resolve())
        for f in src.iterdir():
            if f.suffix == ".safetensors" or f.name.endswith(".safetensors.index.json"):
                continue
            (norm_dir / f.name).symlink_to(f.resolve())
        if comp == "transformer":
            idx_src = src / "model.safetensors.index.json"
            if idx_src.exists():
                idx = json.load(open(idx_src))
                idx["weight_map"] = {
                    w: shard_new_names.get(sf, sf)
                    for w, sf in idx["weight_map"].items()
                }
                (
                    norm_dir / "diffusion_pytorch_model.safetensors.index.json"
                ).write_text(json.dumps(idx))
                legacy_idx = norm_dir / "model.safetensors.index.json"
                if legacy_idx.is_symlink():
                    legacy_idx.unlink()

    marker.write_text(
        json.dumps({"hash": spec_hash, "spec_version": 3, "workflow": workflow})
    )
    return workdir


# ---------------------------------------------------------------------------
# Attention backend (design 3.2)
# ---------------------------------------------------------------------------

_BACKEND_NAME = "habana_fused_sdpa"

# NOTE (audit fix): the runner previously kept its own inline copy of the
# FusedSDPA backend fn (`_make_habana_fused_sdpa` + `_fused_sdpa_tiled`). That
# copy selected softmax_mode="fp32" for fp32 q/k/v, but Habana's FusedSDPA
# asserts softmax_mode "fp32" is BF16-input-only -- so the fp32 video-VAE ViT
# decode raised AssertionError("softmax_mode = fp32 is supported only when q/k/v
# inputs are BF16") on card. The duplicated copy is deleted; `hpu_patches` is
# the single source of truth (its `_softmax_mode` maps fp32 inputs to mode
# "None", the numerically conservative path).


def _hpu_ok() -> bool:
    """Probe torch's HPU bridge WITHOUT touching a device (no placement, no
    tensor creation, no graph compile). Importing torch may autoload the
    Habana plugin (TORCH_DEVICE_BACKEND_AUTOLOAD=1) — that is torch-wide
    autoload, not device init."""
    try:
        import torch

        return bool(hasattr(torch, "hpu") and torch.hpu.is_available())
    except Exception:
        return False


def _register_attention_backend():
    """Register `habana_fused_sdpa` into diffusers' attention dispatch.

    The FusedSDPA call semantics (softmax-mode selection, q-tiling, CPU
    passthrough parity) are owned by `hpu_patches.habana_fused_sdpa` -- the
    single source of truth -- and the runner defers to it. In particular the
    fp32 video-VAE ViT decode must run with softmax_mode=None: Habana's
    FusedSDPA asserts softmax_mode="fp32" is BF16-input-only, and the runner's
    former inline copy got this wrong (it passed "fp32" for fp32 q/k/v).

    The registry is keyed by `AttentionBackendName`, a str-Enum, so the custom
    name must be added as a real (str-backed) enum member — then
    `model.set_attention_backend("habana_fused_sdpa")` validates cleanly.
    """
    shim = _try_module("attention")
    if shim is not None and hasattr(shim, "register_habana_backend"):
        return shim.register_habana_backend()

    from diffusers.models.attention_dispatch import (
        AttentionBackendName,
        _AttentionBackendRegistry,
    )

    patches = importlib.import_module("hpu_port.hpu_patches")
    backend_fn = patches.habana_fused_sdpa
    if _BACKEND_NAME != patches.BACKEND_NAME:
        raise RuntimeError(
            f"backend name drift: runner {_BACKEND_NAME!r} != hpu_patches {patches.BACKEND_NAME!r}"
        )
    try:
        member = AttentionBackendName(_BACKEND_NAME)
    except ValueError:
        member = str.__new__(AttentionBackendName, _BACKEND_NAME)
        member._name_ = _BACKEND_NAME.upper()
        member._value_ = _BACKEND_NAME
        AttentionBackendName._member_map_[member._name_] = member
        AttentionBackendName._value2member_map_.setdefault(_BACKEND_NAME, member)
        AttentionBackendName._member_names_.append(member._name_)
    _AttentionBackendRegistry._backends[member] = backend_fn
    _AttentionBackendRegistry._constraints[member] = []
    _AttentionBackendRegistry._supported_arg_names[member] = set(
        inspect.signature(backend_fn).parameters
    )
    # CP gate (PLAN_CP.md phase 1.1): `enable_parallelism` refuses to run unless
    # the model's attention backend is in this class-level set
    # (modeling_utils.py:638 / attention_dispatch.py:261 in diffusers 0.40.0).
    # The backend's CP correctness comes from the templated Ulysses routing
    # installed in hpu_patches; membership here is what unlocks
    # transformer.enable_parallelism(). Harmless (and unused) on non-HPU runs.
    cp_set = getattr(_AttentionBackendRegistry, "_supports_context_parallel", None)
    if not isinstance(cp_set, set):
        raise RuntimeError(
            "diffusers _AttentionBackendRegistry lost `_supports_context_parallel`; "
            "re-audit the CP registration (audited diffusers 0.40.0)."
        )
    cp_set.add(_BACKEND_NAME)
    return _BACKEND_NAME, backend_fn


def _force_backend(model, backend_name: str) -> bool:
    """`AttentionMixin.set_attention_backend` at the top-level module applies to
    every AttentionModuleMixin inside (DiT packed attention; VAE ViT decoder)."""
    if model is None or not hasattr(model, "set_attention_backend"):
        return False
    try:
        model.set_attention_backend(backend_name)
        return True
    except Exception as exc:
        print(
            f"[h3] WARNING: set_attention_backend({backend_name!r}) failed: {exc}; staying on the default backend"
        )
        return False


# ---------------------------------------------------------------------------
# Rope cache (design 3.3)
# ---------------------------------------------------------------------------


def install_rope_cache(transformer) -> bool:
    """Memoize `MiniMaxH3RotaryPosEmbed.forward` per (device, shape, content).

    The packed layout's `position_ids` are static across all steps of one run,
    so the fp32 cos/sin `(seq, 2*3*rope_freq_dim)` table is computed once per
    bucket and reused. The fp64->fp32 cast order and the fp64 anchor-sum
    contract are untouched — the cache wraps (not rewrites) the original op.
    """
    if not _env_bool("H3_ROPE_CACHE", True) or transformer is None:
        return False
    rope = getattr(transformer, "rope", None)
    if rope is None or getattr(rope, "_h3_cache_installed", False):
        return False
    orig_forward = rope.forward.__func__
    cache_store: dict = {}

    def cached_forward(self, position_ids):
        try:
            # Fingerprint WITHOUT device-side reduce: an fp64 multi-dim sum on
            # the HPU lowers to a reduce_sum_multi_dim complex GUID whose
            # CGUID graph fails translation on this stack
            # (complex_guid_extractor.cpp:314 -> synStatus 26). The key only
            # needs determinism, so pull the raw values to the host and reduce
            # there. The two scalar pulls are tiny sync points.
            flat = position_ids.detach().reshape(-1)
            fingerprint = (
                str(position_ids.device),
                tuple(position_ids.shape),
                float(flat[0].item()),
                float(flat[-1].item()),
                float(flat.cpu().double().sum().item()),
            )
        except Exception:
            return orig_forward(self, position_ids)
        if cache_store.get(fingerprint) is None:
            cache_store[fingerprint] = orig_forward(self, position_ids)
        cos, sin = cache_store[fingerprint]
        return cos, sin

    rope._h3_rope_cache_store = cache_store
    rope.forward = types.MethodType(cached_forward, rope)
    rope._h3_cache_installed = True
    return True


# ---------------------------------------------------------------------------
# Block-level runtime patches (design 3.5 + 3.7)
# ---------------------------------------------------------------------------


def _import_h3_module(name: str):
    return importlib.import_module(f"diffusers.modular_pipelines.minimax_h3.{name}")


def patch_execution_device(pipe_cls):
    """Runner-owned execution device (design 1): with a mixed text-encoder-on-
    CPU map, stock `_execution_device` guesses the first registered module and
    lands on the wrong device. A runner-set `_h3_exec_device` wins."""
    orig_getter = pipe_cls._execution_device.fget

    def _get(self):
        device = self.__dict__.get("_h3_exec_device")
        if device is not None:
            import torch

            return torch.device(device)
        return orig_getter(self)

    pipe_cls._execution_device = property(_get)


def patch_prepare_latents_cpu(cp_world_size: int = 1):
    _ensure_torch()
    """Run MiniMaxH3PrepareLatentsStep's noise draw + patchify on the CPU host.

    Card repro (v22-v27, localized by H3_SYNC_PER_BLOCK=1): the step's device
    side — randn draw, the 8-D `patchify_video_latents` permute/reshape/
    contiguous copy, audio row draw — enqueues a lazy fused graph whose GC
    lowering contains an internal Memcpy node the engine manager cannot
    replace (memcpy_engine_manager.cpp:77 selectEngine,
    REPLACE_FAILED_INVALID_NEW_NODES -> synStatus 26 graph-compile failure).
    The identical op sequence compiles fine standalone, so the lowering only
    breaks inside this block's fused graph.

    The fix keeps numerics bit-exact: the request generator is a CPU
    generator, and diffusers' randn_tensor already draws on the generator's
    device and copies to the target — so drawing on the CPU host and
    patchifying there produces the SAME tensors; only the transport changes
    (one contiguous rows H2D copy, a shape the GC compiles cleanly).

    Shows loudly (getsource drift check) if upstream changes the draw or
    patchify structure.
    """
    bd = _import_h3_module("before_denoise")
    cls = bd.MiniMaxH3PrepareLatentsStep
    if getattr(cls, "_h3_cpu_latents", False):
        cls._h3_cp_world = int(cp_world_size)
        return
    orig = cls.__call__
    src = inspect.getsource(orig)
    if "randn_tensor" not in src or "patchify_video_latents" not in src:
        raise RuntimeError(
            "upstream MiniMaxH3PrepareLatentsStep.__call__ changed; "
            "re-audit patch_prepare_latents_cpu()"
        )
    randn_tensor = inspect.getmodule(orig).randn_tensor
    patchify_video_latents = inspect.getmodule(orig).patchify_video_latents

    @torch.no_grad()
    @functools.wraps(orig)
    def cpu_latents_call(self, components, state):
        block_state = self.get_block_state(state)
        device = components._execution_device
        patch_size = components.patch_size

        latents = block_state.latents
        if latents is None:
            # Same draw order and values as upstream: the generator is a CPU
            # generator, so randn_tensor(device=hpu) already computed the noise
            # on the host and shipped it over. Stay on the host instead.
            latents = randn_tensor(
                (
                    1,
                    components.vae_latent_channels,
                    block_state.num_latent_frames,
                    block_state.latent_height,
                    block_state.latent_width,
                ),
                generator=block_state.generator,
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
        video_rows = patchify_video_latents(
            latents.detach().to("cpu", torch.float32), patch_size
        )

        if block_state.audio_latents is None:
            audio_rows = randn_tensor(
                (
                    block_state.num_audio_latents * components.audio_channels,
                    components.audio_latent_channels,
                ),
                generator=block_state.generator,
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
        else:
            audio_rows = (
                block_state.audio_latents.detach()
                .to("cpu", torch.float32)
                .permute(0, 2, 1)
                .reshape(-1, components.audio_latent_channels)
            )

        if getattr(self.__class__, "_h3_cp_world", 1) > 1:
            # CP determinism contract (PLAN_CP.md phase 1 item 8): every rank
            # draws the IDENTICAL full noise on the CPU host from the same
            # seeded generator; the model's _cp_plan does the seq slicing
            # internally, nothing here hand-slices. Verify the CPU rows agree
            # across ranks BEFORE any H2D copy (the verify never touches the
            # device, so it cannot fuse into a lazy frontier).
            _cp_verify_identical_cpu(video_rows, "video noise rows")
            _cp_verify_identical_cpu(audio_rows, "audio noise rows")

        block_state.latents = video_rows.to(device)
        block_state.audio_latents = audio_rows.to(device)
        self.set_block_state(state, block_state)
        return components, state

    cls.__call__ = cpu_latents_call
    cls._h3_cpu_latents = True
    cls._h3_cp_world = int(cp_world_size)


def _free_dit_from_hpu(components) -> None:
    """Release the DiT's ~62 GiB from the card before the decode phase.

    After the decode-entry boundary sync nothing reads the transformer again,
    but its resident weights shrink the card's free DRAM to ~33 GiB — the
    direct cause of the decode-phase failures (v53 synFailedSectionValidation,
    v54 compute/dma timeout, v50/51 D2H deadlocks): recipe sections and VAE
    workspace need placement room that the dead denoiser was holding.

    update_components(transformer=None) drops the pipeline's reference; gc +
    empty_cache returns the pool. The rope cache dies with the module (its
    per-bucket cos/sin tables are device tensors held by the module only).
    """
    if not _hpu_ok():
        return
    if _RUNTIME.get("serve_keep_dit"):
        # Weight-resident serving (--serve N): the DiT stays on the card for
        # the NEXT request; decode runs in a fresh subprocess on another card.
        print("[h3] serve mode: DiT kept resident for the next request", flush=True)
        return
    tr = getattr(components, "transformer", None)
    if tr is None:
        return
    try:
        components.update_components(transformer=None)
    except Exception as exc:
        print(f"[h3] WARNING: could not detach transformer from pipeline ({exc})", flush=True)
    try:
        free_before, total = torch.hpu.mem_get_info(0)
        del tr
        import gc as _gc
        _gc.collect()
        # This Habana build exposes no empty_cache()/release_cached() — the
        # caching allocator only shrinks via its own pressure heuristics.
        # Report what actually happened either way.
        free_after, _ = torch.hpu.mem_get_info(0)
        # NOTE: mem_get_info lags one allocation cycle (alloc shows the old
        # value, free shows the pre-free value), so this "after" read may
        # still show the pre-free number. The freeprobe (2026-09-11) proved
        # the release is functional: after del+gc, 62 GiB of touched ballast
        # allocates on a card that reported only 14.5 GiB free.
        print(
            f"[h3] DiT freed from HPU before decode: mem_get_info free "
            f"{free_before/2**30:.1f} -> {free_after/2**30:.1f} GiB of {total/2**30:.1f} "
            f"(accounting lags; release verified functional)",
            flush=True,
        )
    except Exception as exc:
        print(f"[h3] WARNING: DiT free incomplete ({exc})", flush=True)


def _inproc_decode_prewarm(components, latents, audio_latents) -> None:
    """H3_INPROC_DECODE support: compile (or cache-load) every recipe the
    in-process sharded decode needs, in EACH CP parent process, BEFORE the
    first denoise drains the device.

    Evidence chain: post-drain REPLAY of already-compiled recipes is safe
    (v78/v79: back-to-back serve requests replay the denoise graph exactly,
    bit-identical, 0.04 s host); post-drain COMPILATION of new recipes wedges
    at a variable point (v72: cast+broadcast stall after a clip decode;
    v73c: first tail D2H with zero collectives). The v72/v73c in-process
    decode attempts compiled the VAE recipes AFTER the drain -- hence the
    farm of fresh processes (v74-v77). This prewarm inverts that: compile
    early, replay late, and the decode needs NO extra process and NO extra
    card, so every CP rank keeps its 62 GiB DiT resident across serve
    requests.

    Per rank (shapes only; values are zeros -- recipes are shape-keyed):
    - standard clip (ts+tok_ov tokens) via vae._decode_clip under bf16
      autocast, exactly like _decode_sharded_core Phase A;
    - the pixel blend + denormalize + on-device uint8 quantize + the two
      small D2H patterns (uint8 frame ~1.5 MB, bf16 tail frame ~2.5 MB) + an
      H2D of an overlap tensor (Phase B's torch.load(map_location=device));
    - the SHORT final-clip shape (token tail) on the rank that owns chunk
      n_chunks-1;
    - the audio decode (fp32, strict) on rank world-1 (the chunkless rank
      that owns the audio in the in-proc split).
    """
    rank = _RUNTIME.get("rank", 0) or 0
    world = _RUNTIME.get("world_size", 1) or 1
    if world <= 1 or _RUNTIME.get("backend") is None:
        return
    vae = components.vae
    device = vae.device
    if device.type != "hpu":
        return False
    if latents is None or getattr(latents, "dim", lambda: 0)() not in (4, 5):
        print(f"[h3:r{rank}]] in-proc prewarm: no latents shape; skipping", flush=True)
        return False
    if latents.dim() == 4:
        # Loop-state latents are (C, T, H, W); the decode sees (1, C, T, H, W).
        C, T_lat, H, W = latents.shape
    else:
        _, C, T_lat, H, W = latents.shape
    t0 = time.perf_counter()
    cfg = vae.config
    ts = int(vae.tokens_chunk_size)
    tok_ov = int(vae.token_overlap)
    pre = int(vae.frame_pre_padding)
    fov = int(vae.frame_overlap)
    cnf = ts * int(vae.temporal_compression_ratio)
    token_drop = int(getattr(cfg, "token_drop", 0) or 0)
    dec_dtype = next(vae.decoder.parameters()).dtype

    was_grad = torch.is_grad_enabled()
    torch.set_grad_enabled(False)
    try:
        def _clip(tokens: int):
            z = torch.zeros(1, C, tokens, H, W, device=device, dtype=torch.bfloat16)
            zn = (z * ls + lm).to(dec_dtype)  # core's per-chunk normalization shape
            with torch.autocast(device_type="hpu", dtype=torch.bfloat16):
                return vae._decode_clip(zn)

        lm = torch.tensor(cfg.latents_mean, device=device).view(1, -1, 1, 1, 1)
        ls = torch.tensor(cfg.latents_std, device=device).view(1, -1, 1, 1, 1)
        pm = torch.tensor(components.pixel_mean, device=device).view(1, -1, 1, 1, 1)
        ps = torch.tensor(components.pixel_std, device=device).view(1, -1, 1, 1, 1)
        clip = _clip(ts + tok_ov)
        _flush_lazy_frontier()
        part = clip[:, :, :cnf][:, :, pre:]
        ov_out = clip[:, :, cnf:][:, :, pre:]
        # Phase B replay set: H2D overlap load, blend, denorm/quantize, the
        # two D2H frame-copy shapes.
        ov_cpu = torch.zeros(1, C, fov, H, W, dtype=torch.bfloat16, device="cpu")
        overlap_in = ov_cpu.to(device).to(part.dtype)
        blended = vae._blend(overlap_in, part, fov, dim=-3)
        pix = ((blended.float() * ps + pm).clamp(0, 1) * 255.0).to(torch.uint8)
        _ = pix[0, :, 0].contiguous().cpu()      # uint8 frame D2H (~1.5 MB)
        _ = ov_out[0, :, 0].contiguous().cpu()   # bf16 tail-frame D2H (~2.5 MB)
        _flush_lazy_frontier()
        del clip, part, ov_out, overlap_in, blended, pix
        # Short final-clip shape (token tail) for the rank that owns it.
        num_tokens = T_lat + token_drop
        pad = (-num_tokens) % ts
        n_chunks = (num_tokens + pad) // ts - int(token_drop > 0)
        if n_chunks > 1 and rank == (n_chunks - 1) % world:
            tail_tokens = (num_tokens + pad) - (n_chunks - 1) * ts
            clip_t = _clip(tail_tokens)
            _flush_lazy_frontier()
            del clip_t
        # Audio decode (fp32 strict) for the chunkless audio rank.
        if rank == world - 1 and audio_latents is not None:
            al_shape = tuple(audio_latents.shape)
            a_mean = torch.tensor(components.audio_vae.config.latents_mean, device=device).view(1, -1, 1)
            a_std = torch.tensor(components.audio_vae.config.latents_std, device=device).view(1, -1, 1)
            az = torch.zeros(*al_shape[-3:], device=device, dtype=torch.float32) if len(al_shape) == 3 else torch.zeros(1, *al_shape[-2:], device=device, dtype=torch.float32)
            _ = components.audio_vae.decode(az * a_std + a_mean, return_dict=False)[0]
            _flush_lazy_frontier()
            del az
        torch.hpu.synchronize()
        print(f"[h3:r{rank}]] in-proc decode prewarm done ({time.perf_counter() - t0:.1f}s)", flush=True)
        return True
    except Exception as e:
        print(f"[h3:r{rank}]] in-proc decode prewarm FAILED: {e}", flush=True)
        raise
    finally:
        torch.set_grad_enabled(was_grad)


def patch_inproc_decode_prewarm() -> None:
    """Run _inproc_decode_prewarm once per process, at the denoise LOOP's
    first step (pre-denoise = pre-drain = healthy compile window).

    v86 lesson: the DenoiseStep's block_state.latents is still None at
    __call__ entry (the loop step materializes it), so the hook must sit on
    loop_step's first call where the noise exists.
    """
    den = _import_h3_module("denoise")
    # v89: hook the LOOP DENOISER's __call__ (not DenoiseStep.loop_step) -- its
    # block_state already carries latents/audio_latents at entry, and the call
    # is still pre-drain (the denoise loop has not run yet). v86/v87/v88 showed
    # both the DenoiseStep.__call__ and loop_step hook points see no latents:
    # the loop sub-blocks materialize them per-call from the shared state.
    cls = den.MiniMaxH3LoopDenoiser
    if getattr(cls, "_h3_inproc_prewarm", False):
        return
    orig = cls.__call__

    @functools.wraps(orig)
    def prewarmed_call(self, components, block_state, *args, **kwargs):
        if (
            _env_bool("H3_INPROC_DECODE", False)
            and not _RUNTIME.get("_inproc_prewarmed")
        ):
            # v97b: the flag moves to AFTER a successful prewarm (it used to be
            # set before the call, so a 'no latents' skip left the guard inert
            # and the decode proceeded UNPREWARMED into the wedge). If the
            # prewarm skips, the flag stays unset and the decode branch raises
            # loudly instead.
            _ok = _inproc_decode_prewarm(
                components,
                getattr(block_state, "latents", None),
                getattr(block_state, "audio_latents", None),
            )
            if _ok:
                _RUNTIME["_inproc_prewarmed"] = True
        return orig(self, components, block_state, *args, **kwargs)

    cls.__call__ = prewarmed_call
    cls._h3_inproc_prewarm = True


def patch_decode_steps():
    _ensure_torch()
    """Rewrite the decode sub-blocks (design 3.5 #1'):

    * video: fp16-autocast guard extended `cuda -> cuda|hpu`, and the whole
      body pinned to the video VAE's device so a CPU-pinned VAE fallback stays
      coherent. Code is otherwise textually identical to decoders.py.
    * audio: body pinned to the audio VAE's device (fp32 strictly, never bf16).
    """
    blocks = _import_h3_module("modular_blocks_minimax_h3")
    decoders = _import_h3_module("decoders")

    video_base = decoders.MiniMaxH3VideoDecodeStep
    audio_base = decoders.MiniMaxH3AudioDecodeStep
    decode_cls = blocks.MiniMaxH3DecodeStep

    class _H3VideoDecodeStep(video_base):
        @torch.no_grad()
        def __call__(self, components, state):
            block_state = self.get_block_state(state)
            device = components.vae.device
            # Phase-boundary sync (denoise -> decode): drain + surface any
            # pending device error HERE, and guarantee the latents are final
            # before anything pulls them to the host. Per-step syncs inside
            # the denoise loop would serialize enqueue/execute; at the phase
            # boundary the one-time cost is noise.
            # v91/v93: H3_QUEUE_DECODE or H3_FUSED_DECODE skips the sync AND
            # the DiT free -- the decode is enqueued while the denoise queue
            # still drains (the wedge only fires on post-drain enqueues) and
            # the decode fits beside the resident DiT (~+0.1 GiB activations,
            # v74).
            _late_decode = (
                (_env_bool("H3_QUEUE_DECODE", False) or _env_bool("H3_FUSED_DECODE", False))
                and (_RUNTIME.get("world_size", 1) or 1) > 1
                and _RUNTIME.get("backend") is not None
            )
            if (
                block_state.latents is not None
                and block_state.latents.device.type == "hpu"
                and not _late_decode
            ):
                # Timed: everything the host "finished" without syncing —
                # including the whole denoise loop's async device work —
                # drains HERE. Without this timer the denoise device cost
                # hides inside the decode phase's wall clock.
                t_sync = time.perf_counter()
                torch.hpu.synchronize()
                print(f"[h3] denoise complete: device synced at decode entry "
                      f"({time.perf_counter() - t_sync:.2f}s)", flush=True)
                # The denoiser is dead weight from here on: release its
                # 62 GiB before the VAE phase needs section space.
                _free_dit_from_hpu(components)
            if (
                _env_bool("H3_DECODE_SHARDED", False)
                and (_RUNTIME.get("world_size", 1) or 1) > 1
                and _RUNTIME.get("backend") is not None
            ):
                # Temporal-sharded decode (H3_DECODE_SHARDED=1): FARM of fresh
                # eager subprocesses, one per temporal chunk, sharing the
                # file-based tail/marker protocol. In-process sharded decode
                # wedges on post-drain device work (v72/v73c); fresh processes
                # are immune and the clip recipes are cached.
                return _spawn_decode_farm(self, components, state, block_state)
            if (
                _env_bool("H3_FUSED_DECODE", False)
                and (_RUNTIME.get("world_size", 1) or 1) > 1
                and _RUNTIME.get("backend") is not None
            ):
                # v93: FUSED in-process decode (see _decode_fused_core). One
                # lazy frontier per rank: owned clip decodes + in-frontier
                # all_gather of the 5-frame tails + blend + on-device uint8
                # quantize; post-frontier work is PURE contiguous D2H pulls
                # (the 8/8-reliable pattern). No file tails, no H2D loads, no
                # post-drain device enqueues, no DiT free -- resident serve.
                rank = _RUNTIME.get("rank", 0) or 0
                world = _RUNTIME.get("world_size", 1) or 1
                frames_dir = Path(_RUNTIME["frames_dir"])
                tmp = frames_dir / ".shard_tmp"
                if rank == 0:
                    tmp.mkdir(parents=True, exist_ok=True)
                    for stale in tmp.glob("*"):
                        stale.unlink()
                else:
                    time.sleep(2.0)  # rank0 owns the stale wipe
                    tmp.mkdir(parents=True, exist_ok=True)
                # v97b: VIDEO first, then audio on the last rank — the fused
                # core ends in a full sync; the audio decode is a separate
                # small graph after that sync (the pattern that completes in
                # every hung run). Doing audio BEFORE the core made the audio
                # wav D2H drain the last rank's queue pre-collective while
                # rank0 waited eagerly in all_gather — the v96 stall shape.
                _owns_marker = rank != world - 1
                t_dec = time.perf_counter()
                frames_saved, clip_s, _phase = _decode_fused_core(
                    components.vae,
                    block_state.latents,
                    components.pixel_mean,
                    components.pixel_std,
                    frames_dir,
                    rank,
                    world,
                    device,
                    write_marker=_owns_marker,
                )
                print(
                    f"[h3:r{rank}]] fused decode: {frames_saved} frames, "
                    f"clip {clip_s:.1f}s, wall {time.perf_counter() - t_dec:.1f}s",
                    flush=True,
                )
                _RUNTIME["sharded_audio_rank"] = world - 1
                if rank == world - 1:
                    # Audio AFTER the sync (separate small graph; proven to
                    # complete post-drain in every run). Its marker is written
                    # here so the export phase's marker poll also guarantees
                    # the wav exists before muxing.
                    _decode_sharded_audio(components, state, block_state, rank)
                    (tmp / f"done_r{rank}.json").write_text(
                        json.dumps({"rank": rank, "chunks": "own+audio", "frames": frames_saved})
                    )
                else:
                    # Rank0 never decodes audio inline; the stub logic in the
                    # audio step must see the flag so it doesn't re-decode.
                    _RUNTIME["sharded_audio_done"] = True
                _RUNTIME["decode_sharded_done"] = True
                block_state.videos = None
                self.set_block_state(state, block_state)
                return components, state
            if (
                _env_bool("H3_QUEUE_DECODE", False)
                and (_RUNTIME.get("world_size", 1) or 1) > 1
                and _RUNTIME.get("backend") is not None
            ):
                # v91: NO-BOUNDARY-SYNC in-process decode. The wedge only fires
                # on work enqueued AFTER the device drains (v72/v73c); the
                # recipes are already cached per rank; and clip decode costs
                # only ~+0.1 GiB activations (v74) -- so the decode fits BESIDE
                # the resident DiT (62+9.7+0.1 < 94.6). Skip the boundary sync
                # and the DiT free entirely: enqueue the full decode chain
                # (every rank decodes ALL clips, replicated + deterministic;
                # no cross-rank tails, no collectives) while the denoise queue
                # is still draining. The device never goes idle mid-pipeline.
                rank = _RUNTIME.get("rank", 0) or 0
                world = _RUNTIME.get("world_size", 1) or 1
                frames_dir = Path(_RUNTIME["frames_dir"])
                tmp = frames_dir / ".shard_tmp"
                tmp.mkdir(parents=True, exist_ok=True)
                if rank == 0:
                    for stale in tmp.glob("*"):
                        stale.unlink()
                else:
                    time.sleep(2.0)  # rank0 owns the stale wipe
                # v91b: keep the SHARDED distribution (chunk i -> rank
                # i%world) with the file-tail backpressure -- v91 tried every
                # rank decoding all 7 clips queue-fed and the HOST OOM-killed
                # rank 1 (45 GB RSS: two giant lazy enqueues + D2H staging
                # buffers live concurrently, 125 GB box). One chunk per rank
                # is what the farm/v77 pacing always did.
                if rank == world - 1:
                    # Chunkless last rank owns the audio (fp32, resident VAE).
                    _decode_sharded_audio(components, state, block_state, rank)
                t_dec = time.perf_counter()
                frames_saved, clip_s, _phase = _decode_sharded_core(
                    components.vae,
                    block_state.latents,
                    components.pixel_mean,
                    components.pixel_std,
                    frames_dir,
                    rank,
                    world,
                    device,
                )
                print(
                    f"[h3:r{rank}]] queue-decode: {frames_saved} frames, "
                    f"clip {clip_s:.1f}s, wall {time.perf_counter() - t_dec:.1f}s",
                    flush=True,
                )
                _RUNTIME["decode_sharded_done"] = True
                block_state.videos = None
                self.set_block_state(state, block_state)
                return components, state
            if (
                _env_bool("H3_INPROC_DECODE", False)
                and (_RUNTIME.get("world_size", 1) or 1) > 1
                and _RUNTIME.get("backend") is not None
            ):
                # v86: IN-PROCESS sharded decode with PRE-WARMED recipes (see
                # patch_inproc_decode_prewarm). Every CP rank replays its own
                # chunk's decode on the resident VAE (no subprocess, no extra
                # card, no per-worker VAE load); the tail handoff stays
                # file-based and there are no collectives. The DiT weights
                # never leave HBM, which is the whole point of serve mode.
                # Requires the prewarm to have run (first denoise call); if it
                # did not, post-drain compilation wedges (v72/v73c) -- loud
                # failure is better than a silent hang.
                if not _RUNTIME.get("_inproc_prewarmed"):
                    raise RuntimeError(
                        "H3_INPROC_DECODE=1 but the decode prewarm never ran; "
                        "post-drain recipe compilation wedges on this stack"
                    )
                rank = _RUNTIME.get("rank", 0) or 0
                world = _RUNTIME.get("world_size", 1) or 1
                frames_dir = Path(_RUNTIME["frames_dir"])
                tmp = frames_dir / ".shard_tmp"
                if rank == 0:
                    tmp.mkdir(parents=True, exist_ok=True)
                    for stale in tmp.glob("*"):
                        stale.unlink()
                else:
                    # Rank0 owns the stale-marker wipe; give it a beat so no
                    # rank races ahead of the wipe into Phase B tail writes.
                    time.sleep(2.0)
                    tmp.mkdir(parents=True, exist_ok=True)
                if rank == world - 1:
                    # Chunkless rank owns the audio (fp32, resident audio VAE).
                    _decode_sharded_audio(components, state, block_state, rank)
                frames_saved, clip_s, _phase = _decode_sharded_core(
                    components.vae,
                    block_state.latents,
                    components.pixel_mean,
                    components.pixel_std,
                    frames_dir,
                    rank,
                    world,
                    device,
                )
                print(
                    f"[h3:r{rank}]] in-proc sharded decode: {frames_saved} frames, "
                    f"clip {clip_s:.1f}s",
                    flush=True,
                )
                _RUNTIME["decode_sharded_done"] = True
                block_state.videos = None
                self.set_block_state(state, block_state)
                return components, state
            if _cp_skip_decode():
                # CP with replicated weights: latents are identical on every
                # rank, so rank0 alone renders the artifact; the other ranks
                # stop at the decode boundary (their card is free for the next
                # request while rank0 decodes).
                block_state.videos = None
                self.set_block_state(state, block_state)
                return components, state
            if _env_bool("H3_DECODE_SUBPROCESS", False):
                # Escape hatch for the D2H launch-thread wedge (v50/51/52/58):
                # the decode itself runs in a FRESH eager-mode process which
                # writes frames + wav directly; this process never does the
                # big post-decode D2H. See decode_subproc.py.
                block_state = _decode_via_subprocess(self, components, state, block_state)
                self.set_block_state(state, block_state)
                return components, state
            block_state.latents = block_state.latents.to(device)

            latents_mean = torch.tensor(
                components.vae.config.latents_mean, device=device
            ).view(1, -1, 1, 1, 1)
            latents_std = torch.tensor(
                components.vae.config.latents_std, device=device
            ).view(1, -1, 1, 1, 1)
            latents = block_state.latents * latents_std + latents_mean

            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16 if device.type == "hpu" else torch.float16,
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
            # D2H in per-frame chunks: the single monolithic D2H of the full
            # video (616 MB at 480p-5s) deadlocks the lazy launch thread —
            # the main thread waits forever in
            # habana_lazy::copy_hpu_lazy_D2H -> JoinPendingLaunchThread
            # (v50 AND v51, silent, no log entry; also the stall that ended
            # the previous session). Small D2H copies complete reliably, so
            # move the frames over one at a time (≈5 MB each) and hand
            # postprocess_video an already-host tensor (its .cpu() no-ops).
            if video.device.type == "hpu" and video.ndim == 5:
                # video is (B, C, T, H, W): select the time index directly
                # (an `[..., i, :, :, :]` here is 5 index positions for 5 dims
                # -- the ellipsis collapses and `i` hits the channel dim).
                t_dim = video.shape[2]
                frames_host = [video[:, :, i].contiguous().cpu() for i in range(t_dim)]
                video = torch.stack(frames_host, dim=2)
            block_state.videos = components.video_processor.postprocess_video(
                video, output_type=block_state.output_type
            )

            self.set_block_state(state, block_state)
            return components, state

    class _H3AudioDecodeStep(audio_base):
        @torch.no_grad()
        def __call__(self, components, state):
            if _RUNTIME.get("decode_sharded_done"):
                # Sharded video decode wrote the frames. If the chunkless
                # audio rank already decoded audio inline (concurrent with
                # the clips), stub here; otherwise the audio rank decodes now
                # (it also owns video chunks). NO barrier: post-drain
                # collectives stall (v72); rank0's export phase polls the
                # marker files instead.
                block_state = self.get_block_state(state)
                my_rank = _RUNTIME.get("rank", 0) or 0
                audio_rank = _RUNTIME.get("sharded_audio_rank", 0)
                if not _RUNTIME.get("sharded_audio_done") and my_rank == audio_rank:
                    _decode_sharded_audio(components, state, block_state, my_rank)
                    _RUNTIME["sharded_audio_done"] = True
                block_state.audio = None
                block_state.sampling_rate = (
                    int(components.audio_sampling_rate)
                    if _RUNTIME.get("sharded_audio_done")
                    else components.audio_sampling_rate
                )
                self.set_block_state(state, block_state)
                return components, state
            if _RUNTIME.get("decode_subproc_done"):
                # The subprocess hatch decoded BOTH modalities (frames + wav
                # already on disk); this block only replays the bookkeeping so
                # the export phase sees a coherent state.
                block_state = self.get_block_state(state)
                meta = _RUNTIME.get("decode_meta") or {}
                block_state.audio = None
                block_state.sampling_rate = int(meta.get("sampling_rate", 0)) or (
                    components.audio_sampling_rate
                )
                self.set_block_state(state, block_state)
                return components, state
            if _cp_skip_decode():
                block_state = self.get_block_state(state)
                block_state.audio = None
                block_state.sampling_rate = components.audio_sampling_rate
                self.set_block_state(state, block_state)
                return components, state
            block_state = self.get_block_state(state)
            device = components.audio_vae.device
            # Same phase-boundary sync as the video decode step.
            if block_state.audio_latents is not None and block_state.audio_latents.device.type == "hpu":
                t_sync = time.perf_counter()
                torch.hpu.synchronize()
                print(f"[h3] video decode complete: device synced at audio entry "
                      f"({time.perf_counter() - t_sync:.2f}s)", flush=True)
            block_state.audio_latents = block_state.audio_latents.to(device)

            audio_latents_mean = torch.tensor(
                components.audio_vae.config.latents_mean, device=device
            ).view(1, -1, 1)
            audio_latents_std = torch.tensor(
                components.audio_vae.config.latents_std, device=device
            ).view(1, -1, 1)
            audio_latents = (
                block_state.audio_latents * audio_latents_std + audio_latents_mean
            )

            audio = components.audio_vae.decode(audio_latents, return_dict=False)[0]
            block_state.audio = audio.float().permute(1, 0, 2)
            block_state.sampling_rate = components.audio_sampling_rate

            self.set_block_state(state, block_state)
            return components, state

    decode_cls.block_classes = [_H3VideoDecodeStep, _H3AudioDecodeStep]
    return _H3VideoDecodeStep, _H3AudioDecodeStep


def patch_set_timesteps_step():
    """Mixed-device fix for `MiniMaxH3SetTimestepsStep.build_row_timesteps`.

    The prepare-layout steps build `video_indices` / `audio_indices` on the
    pipeline's `_execution_device` (HPU), but `build_row_timesteps` creates its
    `row_timesteps` base tensor on CPU (`torch.full` without a device), so the
    index-assign `row_timesteps[video/audio_indices...] = t` dispatches an HPU
    kernel against a CPU base and raises (HPU mixed-device kernel error).

    Fix: run the whole reduce on CPU (indices -> cpu) — internally consistent
    and bit-identical; the caller `.to(device)`s the returned
    (timesteps, indices) pair onto the exec device anyway.
    Shows loudly (getsource drift check) if upstream changes the base-tensor
    construction that motivates this patch.
    """
    bd = _import_h3_module("before_denoise")
    cls = bd.MiniMaxH3SetTimestepsStep
    if getattr(cls, "_h3_device_bridged", False):
        return
    orig = cls.build_row_timesteps  # staticmethod: class access yields the plain function
    src = inspect.getsource(orig)
    if "torch.full((sequence_length,), video_timestep, dtype=torch.float32)" not in src:
        raise RuntimeError(
            "upstream build_row_timesteps changed (expected CPU torch.full base "
            "tensor); re-audit patch_set_timesteps_step()"
        )

    @functools.wraps(orig)
    def bridged(*args, **kwargs):
        # First two positional args are the row-index tensors.
        args = list(args)
        for i in (0, 1):
            if i < len(args) and isinstance(args[i], torch.Tensor):
                args[i] = args[i].cpu()
        out = orig(*args, **kwargs)
        # Frontier break AFTER the CPU reduce: keep the caller's follow-up H2D
        # pull of the returned pair in its own clean unfused copy graph. The
        # pre-scalarize flush lives in the __call__ replacement below.
        _flush_lazy_frontier()
        return out

    cls.build_row_timesteps = staticmethod(bridged)
    cls._h3_device_bridged = True

    # --- 2. schedule on CPU + frontier flush before scalarizing  ---------
    # Card repro (v19/v20): upstream pins the scheduler timesteps to the exec
    # device, then the row-timestep comprehension's `float(timestep)`
    # (before_denoise.py ~:1237) forces a per-step D2H sync that fuses the
    # ENTIRE pending lazy frontier (prepare-layout kernels + scheduler sigma
    # kernels + duplicate cast Memcpy nodes) into one HabanaFusedOpLazy graph
    # the GC cannot compile — synStatus 26; the reported last error flips
    # between memcpy REPLACE_FAILED_INVALID_NEW_NODES and a generic
    # recipe_manager "Can not compile graph" from run to run. Every schedule
    # consumer is device-agnostic (`MiniMaxH3Scheduler.step` / `scale_noise`
    # do `.to(device=sample.device)` internally; the decode loop only floats
    # the scalars), so:
    #   (a) build the schedule on CPU — `float(timestep)` becomes host-only,
    #       the loop does no schedule D2H at all;
    #   (b) flush the prepare-layout frontier with one mark_step BEFORE the
    #       schedule is built — the bridged build_row_timesteps' index pulls
    #       then run as small standalone D2H copy graphs and the plan's
    #       `.to(device)` outbound copies are plain unfused H2D (verified
    #       compile-clean on this stack).
    # If upstream changes, raise loudly (same policy as patch_decode_steps).
    upstream_call = cls.__call__
    src_call = inspect.getsource(upstream_call)
    if (
        "components.scheduler.set_timesteps(block_state.num_inference_steps, device=device)"
        not in src_call
        or "float(timestep)" not in src_call
    ):
        raise RuntimeError(
            "upstream MiniMaxH3SetTimestepsStep.__call__ changed; "
            "re-audit patch_set_timesteps_step()"
        )

    @torch.no_grad()
    @functools.wraps(upstream_call)
    def cpu_schedule_call(self, components, state):
        block_state = self.get_block_state(state)
        device = components._execution_device

        # (b) close the prepare-layout frontier BEFORE anything scalarizes.
        _flush_lazy_frontier()

        # (a) schedule on CPU — all consumers convert back explicitly.
        components.scheduler.set_timesteps(
            block_state.num_inference_steps, device="cpu"
        )
        components.audio_scheduler.set_timesteps(
            block_state.num_inference_steps, device="cpu"
        )
        block_state.timesteps = components.scheduler.timesteps
        block_state.audio_timesteps = components.audio_scheduler.timesteps

        block_state.row_timestep_plan = [
            tuple(
                tensor.to(device)
                for tensor in self.build_row_timesteps(
                    block_state.video_indices,
                    block_state.audio_indices,
                    block_state.num_condition_video_rows,
                    block_state.num_condition_audio_rows,
                    block_state.text_indices.numel(),
                    float(timestep),
                    float(audio_timestep),
                    max(float(timestep), components.keyframe_noise_aug),
                    1.0,
                )
            )
            for timestep, audio_timestep in zip(
                block_state.timesteps, block_state.audio_timesteps
            )
        ]

        self.set_block_state(state, block_state)
        return components, state

    cls.__call__ = cpu_schedule_call
    cls._h3_cpu_schedule = True


def _flush_lazy_frontier() -> None:
    """Close the pending lazy-mode frontier with one `htcore.mark_step()`.

    No-op in eager mode or without the Habana plugin (mark_step is a lazy-only
    op; calling it elsewhere just warns per call). Used before any op that
    scalarizes an HPU tensor (`float()`, `.item()`, `.cpu()`) so the sync pulls
    a small clean graph instead of fusing the whole pending frontier into one
    FusedOpLazy mega-graph (the HabanaFusedOpLazy_0_1 GC compile failure seen
    on the first card runs).
    """
    try:
        from habana_frameworks.torch.utils.internal import is_lazy as _is_lazy
        if _is_lazy():
            import habana_frameworks.torch.core as htcore
            htcore.mark_step()
            if _env_flag("H3_DEBUG_FLUSH", "0") == "1":
                print("[h3:flush] mark_step fired", flush=True)
        elif _env_flag("H3_DEBUG_FLUSH", "0") == "1":
            print("[h3:flush] skipped: is_lazy() False", flush=True)
    except ImportError as exc:
        if _env_flag("H3_DEBUG_FLUSH", "0") == "1":
            print(f"[h3:flush] skipped: import failed ({exc})", flush=True)


def patch_timestep_embedding_host_basis():
    _ensure_torch()
    """Hoist the sinusoidal basis build of `get_timestep_embedding` to host.

    The DiT forward is one lazy frontier until the post-forward mark_step; the
    `aten.arange.start` inside it (time_proj's fp32 basis) lowers to a range
    node whose creation fails when the fused graph is giant
    (synNodeCreateWithId range_f32, synStatus 26). Small-graph repros of the
    same arange pass, so the node is only poison inside the mega-graph.

    Fix: compute `exponent = -log(max_period) * arange(half) / (half - shift)`
    on the CPU host (pure constants — deterministic fp32, bit-exact vs the
    device-built vector) and move the finished vector to the timesteps'
    device. `exp`, the mul by timesteps and sin/cos stay device-side, so the
    device arithmetic is unchanged.

    Shows loudly (getsource drift check) if upstream changes the basis math.
    """
    import diffusers.models.embeddings as emb_mod
    if getattr(emb_mod, "_h3_host_basis", False):
        return
    orig = emb_mod.get_timestep_embedding
    src = inspect.getsource(orig)
    if (
        "torch.arange(" not in src
        or "start=0, end=half_dim" not in src
        or "torch.exp(exponent)" not in src
    ):
        raise RuntimeError(
            "upstream get_timestep_embedding changed; "
            "re-audit patch_timestep_embedding_host_basis()"
        )

    @functools.wraps(orig)
    def host_basis(timesteps, embedding_dim, flip_sin_to_cos=False,
                   downscale_freq_shift=1, scale=1, max_period=10000):
        half_dim = embedding_dim // 2
        exponent = -math.log(max_period) * torch.arange(
            half_dim, dtype=torch.float32
        )
        exponent = exponent / (half_dim - downscale_freq_shift)
        exponent = exponent.to(timesteps.device)
        emb = torch.exp(exponent)
        emb = timesteps[:, None].float() * emb[None, :]
        emb = scale * emb
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if flip_sin_to_cos:
            emb = torch.cat([emb[:, embedding_dim // 2 :], emb[:, : embedding_dim // 2]], dim=-1)
        if timesteps.dtype == torch.float64 and emb.dtype == torch.float32:
            emb = emb.to(torch.float64)
        return emb

    emb_mod.get_timestep_embedding = host_basis
    emb_mod._h3_host_basis = True


def patch_rope_forward_per_axis():
    _ensure_torch()
    """Rewrite MiniMaxH3RotaryPosEmbed.forward's freq build as per-axis 2-way
    broadcast outer products.

    Card repro (v42-v44, isolated with PT_HPU_GRAPH_DUMP=1): the stock
    3-way broadcast mul `position_ids.unsqueeze(-1) * inv_freq.view(1,1,-1)`
    ((seq,3,1) x (1,1,16)) fails IR conversion in the PT bridge —
    `attribute "attr::value" has unexpected kind: ival` — flakily (the same
    mul passed in one process and failed in the next, and pre-expanded
    operands fail too). The generic GC failure (synStatus 26) logged on the
    card runs was this same lowering bug with its message swallowed by the
    launch thread.

    The per-axis rewrite computes the SAME elementwise fp32 products
    (p[:, i] * inv_freq[j], no reduction anywhere), concatenated in the stock
    t,h,w order — bit-identical output, only the broadcast shape changes.
    Drift-checked against the upstream source like the other patches.
    """
    mod = importlib.import_module(
        "diffusers.models.transformers.transformer_minimax_h3"
    )
    cls = mod.MiniMaxH3RotaryPosEmbed
    if getattr(cls, "_h3_per_axis", False):
        return
    orig = cls.forward
    src = inspect.getsource(orig)
    if (
        "freqs = position_ids.unsqueeze(-1) * self.inv_freq.view(1, 1, -1)"
        not in src
        or "freqs_t, freqs_h, freqs_w = freqs.unbind(dim=1)" not in src
    ):
        raise RuntimeError(
            "upstream MiniMaxH3RotaryPosEmbed.forward changed; "
            "re-audit patch_rope_forward_per_axis()"
        )

    @functools.wraps(orig)
    def per_axis_forward(self, position_ids):
        position_ids = position_ids.to(torch.float32)
        # Same elementwise products as the stock 3-way broadcast mul, built as
        # three 2-way broadcasts (one per rotary axis) — avoids the bridge's
        # flaky 3-way broadcast lowering (attr::value ival IR failure).
        parts = [
            position_ids[:, axis : axis + 1] * self.inv_freq.view(1, -1)
            for axis in range(3)
        ]
        freqs = torch.cat(parts, dim=-1)
        freqs = torch.cat((freqs, freqs), dim=-1)
        return freqs.cos(), freqs.sin()

    cls.forward = per_axis_forward
    cls._h3_per_axis = True


def _late_decode_enabled() -> bool:
    """True when an in-process late decode mode is active (fused/queue/inproc)."""
    return (
        _env_bool("H3_FUSED_DECODE", False)
        or _env_bool("H3_QUEUE_DECODE", False)
        or _env_bool("H3_INPROC_DECODE", False)
    )


def install_block_mark_steps(transformer, extra_targets=()) -> int:
    """Close the lazy frontier after every transformer/refiner block.

    The DiT forward enqueues ~5.4k aten ops into ONE lazy graph (the
    post-forward mark_step is the first frontier break). Giant fused-op graph
    builds fail on this stack in several ways (memcpy node engine replace,
    range node creation, cross-compound input validation), while every
    small-graph repro passes. Per-block mark_step — the standard vLLM-HPU
    pattern — keeps each graph to one block's ops. No-op on CPU and skipped
    under HPU graphs (the graph wrap owns the dispatch boundary).

    v97: the video VAE decoder is the SAME graph class (36-layer ViT with
    atDecoder3d blocks) and never got these breaks — its in-process decode
    compiled one unbroken giant frontier per clip and never returned
    (v54/v58/v72/v73c/v86-v96 hangs; the hang looks like idle engines + free
    memory because a compile that never returns looks exactly like that).
    install_block_mark_steps(pipe.vae.decoder) decomposes it identically.
    """
    if transformer is None or not _hpu_ok():
        return 0
    try:
        import habana_frameworks.torch.core as htcore
    except Exception:
        return 0

    def hook(module, args, output):
        htcore.mark_step()
        if _env_flag("H3_SYNC_IN_HOOK", "0") == "1":
            # Sync inside the hook so an async compile failure surfaces HERE
            # with the module name, not at the next container boundary.
            try:
                torch.hpu.synchronize()
            except Exception as exc:
                raise RuntimeError(
                    f"HPU graph compile failed in module "
                    f"{module.__class__.__name__} (index unknown): "
                    f"{str(exc).splitlines()[0][:300]}"
                ) from exc
        return output

    count = 0
    targets = ["transformer_blocks", "token_refiner", *extra_targets]
    if _env_flag("H3_MARK_PER_MODULE", "0") == "1":
        # Finer bisection: also give each entry/exit submodule its own graph.
        targets += ["rope", "proj_in", "audio_proj_in", "context_embedder",
                    "time_proj", "time_embedder", "norm_out", "proj_out",
                    "audio_proj_out"]
    for group_name in targets:
        group = getattr(transformer, group_name, None)
        if group is None:
            continue
        children = list(group.children()) if hasattr(group, "children") else []
        if isinstance(group, (list, tuple)):
            blocks = list(group)
        elif children:
            # A container module (e.g. MiniMaxH3TokenRefiner): hook its direct
            # children (the refiner blocks).
            blocks = children
        else:
            # Scalar submodule (e.g. rope): hook it directly.
            key = f"_h3_block_mark_step_{group_name}"
            if getattr(group, key, False):
                continue
            group.register_forward_hook(hook)
            setattr(group, key, True)
            count += 1
            continue
        for index, block in enumerate(blocks):
            key = f"_h3_block_mark_step_{group_name}_{index}"
            if getattr(block, key, False):
                continue
            block.register_forward_hook(hook)
            setattr(block, key, True)
            count += 1
    if count:
        print(f"[h3] per-block mark_step installed on {count} blocks", flush=True)
    return count


def install_dit_trace(transformer) -> None:
    """Diagnostic traces around the DiT forward, env-gated:

    * H3_TRACE_DIT=1  — pre-forward trace of the DiT's direct children so a
      mid-forward failure names the submodule being entered last.
    * H3_TRACE_OPS=1  — a TorchDispatchMode around the forward logging every
      aten op with a sequence number ([h3:op] N func); the tail right before a
      failure names the lowering that poisoned the graph.
    Zero cost when disabled.
    """
    if not transformer or (
        _env_flag("H3_TRACE_DIT", "0") != "1"
        and _env_flag("H3_TRACE_OPS", "0") != "1"
        and not _env_flag("H3_STOP_AFTER", "")
    ):
        return
    if _env_flag("H3_TRACE_DIT", "0") == "1":
        for name, child in transformer.named_children():
            def make_hook(n):
                def hook(module, args):
                    print(f"[h3:dit] > {n}", flush=True)
                return hook
            child.register_forward_pre_hook(make_hook(name), with_kwargs=False)
        print("[h3] DiT child trace installed", flush=True)
    if _env_flag("H3_TRACE_OPS", "0") == "1":
        from torch.utils._python_dispatch import TorchDispatchMode

        class _OpTracer(TorchDispatchMode):
            seq = 0

            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                _OpTracer.seq += 1
                print(f"[h3:op] {_OpTracer.seq} {func}", flush=True)
                return func(*args, **(kwargs or {}))

        orig_fwd = transformer.forward

        @functools.wraps(orig_fwd)
        def traced_forward(*args, **kwargs):
            with _OpTracer():
                return orig_fwd(*args, **kwargs)

        transformer.forward = traced_forward
        print("[h3] DiT op trace installed", flush=True)

    stop_after = _env_flag("H3_STOP_AFTER", "")
    if stop_after:
        # Frontier-truncation bisect on the REAL pipeline: run the forward
        # prologue submodule by submodule in upstream order, mark_step + sync
        # after each, then stop (raise _BisectStop, caught by the runner shim
        # below) once the named module ran. The last PASSING module is the
        # context in which the failure lives; the first FAILING sync names it.
        orig_fwd = transformer.forward

        @functools.wraps(orig_fwd)
        def bisect_forward(self, *args, **kwargs):
            import habana_frameworks.torch.core as htcore
            position_ids = kwargs.get("position_ids")
            if position_ids is None and len(args) >= 6:
                position_ids = args[5]
            order = [
                ("rope", lambda: self.rope(position_ids)),
                ("proj_in", lambda: self.proj_in(
                    kwargs.get("hidden_states", args[0] if args else None).to(
                        next(self.proj_in.parameters()).dtype))),
                ("audio_proj_in", lambda: self.audio_proj_in(
                    kwargs.get("audio_hidden_states", args[1] if len(args) > 1 else None).to(
                        next(self.audio_proj_in.parameters()).dtype))),
                ("context_embedder", lambda: self.context_embedder(
                    kwargs.get("encoder_hidden_states", args[2] if len(args) > 2 else None).to(
                        next(self.context_embedder.parameters()).dtype))),
                ("time_proj", lambda: self.time_proj(
                    kwargs.get("timestep", args[3] if len(args) > 3 else None))),
            ]
            results = {}
            for name, fn in order:
                try:
                    results[name] = fn()
                    htcore.mark_step()
                    torch.hpu.synchronize()
                    print(f"[h3:bisect] {name}: PASS", flush=True)
                except Exception as exc:
                    print(
                        f"[h3:bisect] {name}: FAIL "
                        f"{str(exc).splitlines()[-1][:160]}",
                        flush=True,
                    )
                    raise
                if name == stop_after:
                    raise _BisectStop(name)
            # default: run the whole forward (stop point not reached)
            return orig_fwd(*args, **kwargs)

        transformer.forward = types.MethodType(bisect_forward, transformer)
        print(f"[h3] DiT bisect installed (stop_after={stop_after!r})", flush=True)


class _BisectStop(Exception):
    """Sentinel for H3_STOP_AFTER frontier bisection (not an error)."""


def install_block_logging(pipe, enabled: bool | None = None) -> None:
    """Hierarchical console logging of modular-pipeline execution.

    Wraps the `__call__` of every concrete block class reachable from the
    pipeline's block tree with an indent-aware enter/exit/failed envelope
    (`[h3:block]`), giving per-phase attribution on a card-window run: the
    denoise loop logs one line pair per iteration (step index + timestep via
    the loop's i/t kwargs) and the post-loop decode sub-blocks (video/audio
    VAE) show individually. Disable with H3_LOG_BLOCKS=0.
    """
    if enabled is None:
        enabled = _env_flag("H3_LOG_BLOCKS", "1").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
    sync_per_block = _env_bool("H3_SYNC_PER_BLOCK", False)
    flush_per_block = _env_bool("H3_FLUSH_PER_BLOCK", True)
    if not enabled:
        return

    depth = [0]
    seen = set()

    def wrap_class(cls) -> None:
        if cls in seen:
            return
        seen.add(cls)
        if getattr(cls, "_h3_logwrap", False) or not hasattr(cls, "__call__"):
            return
        orig = cls.__call__
        name = cls.__name__

        @functools.wraps(orig)
        def logged(self, *args, **kwargs):
            depth[0] += 1
            ind = "  " * (depth[0] - 1)
            extra = ""
            if "i" in kwargs and "t" in kwargs:
                extra = f" i={int(kwargs['i'])}"
                try:
                    extra += f" t={float(kwargs['t']):.4f}"
                except (TypeError, ValueError):
                    extra += f" t={kwargs['t']!r}"
            print(f"{ind}[h3:block] > {name}{extra}", flush=True)
            t0 = time.perf_counter()
            try:
                out = orig(self, *args, **kwargs)
                if flush_per_block and torch is not None:
                    # Standard Habana pattern (vLLM-HPU does the same): close
                    # the lazy frontier at block boundaries. Without this, the
                    # first H2D copy of a later block fuses the ENTIRE pending
                    # frontier into its graph and the GC's memcpy engine
                    # replacement fails on the fused blob
                    # (REPLACE_FAILED_INVALID_NEW_NODES).
                    _flush_lazy_frontier()
                if sync_per_block and torch is not None and getattr(torch, "hpu", None) is not None:
                    # Diagnostic mode (H3_SYNC_PER_BLOCK=1): lazy-mode graph
                    # compiles are async and their failures surface at the
                    # next host sync, misattributing the failing block. A
                    # sync here pins each failure to the block that owns it.
                    try:
                        torch.hpu.synchronize()
                    except Exception as exc:
                        print(
                            f"{ind}[h3:block] ! {name}{extra}: HPU sync FAILED after "
                            f"{time.perf_counter() - t0:.3f}s: {type(exc).__name__}: {str(exc).splitlines()[-1][:200]}",
                            flush=True,
                        )
                        raise
                dt = time.perf_counter() - t0
                print(f"{ind}[h3:block] < {name}{extra}: {dt:.3f}s", flush=True)
                if _BLOCK_TIME_RECORDS is not None:
                    # CP timing merge (PLAN_CP.md phase 1.4): workers record
                    # their per-block wall times for the generate run; the
                    # parent prints the combined table. Host-side times under
                    # lazy mode are enqueue costs, not device costs (v58
                    # lesson) -- balance information, not a benchmark.
                    _BLOCK_TIME_RECORDS.append((name, extra.strip(), round(dt, 3)))
                return out
            except Exception as exc:
                dt = time.perf_counter() - t0
                print(
                    f"{ind}[h3:block] ! {name}{extra}: FAILED after {dt:.3f}s: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
                raise
            finally:
                depth[0] -= 1

        cls.__call__ = logged
        cls._h3_logwrap = True

    def walk(block) -> None:
        if block is None:
            return
        wrap_class(block.__class__)
        sub_blocks = getattr(block, "sub_blocks", None)
        if sub_blocks:
            for sub in sub_blocks.values():
                walk(sub)

    walk(getattr(pipe, "_blocks", None))
    print(
        f"[h3] block logging installed ({len(seen)} block classes; "
        "H3_LOG_BLOCKS=0 to disable)",
        flush=True,
    )


def patch_text_encoder_device():
    """Bridge a CPU-resident (or offloaded) text encoder: encoders.py builds the
    presentation tensors (`input_ids`, `mm_token_type_ids`, vision kwargs) on
    `device=` given by the caller — the pipeline's `_execution_device` (HPU once
    the DiT is placed there) — but then calls `text_encoder.model(...)`, whose
    weights live on the encoder's OWN device. Mixed devices crash the embedding
    kernel ("Expected all tensors to be on the HPU device … input[idx=0] on cpu").

    Fix: run the encoder on its own device and lift the returned hidden states
    to the caller's requested device afterwards, so the packed-row scatter on
    the HPU side still sees exec-device embeddings.
    """
    encoders = _import_h3_module("encoders")
    if getattr(encoders, "_h3_te_device_bridged", False):
        return
    orig = encoders.get_qwen3vl_prompt_embeds

    @functools.wraps(orig)
    def bridged(*args, **kwargs):
        text_encoder = kwargs.get("text_encoder") or (args[0] if args else None)
        # te=stream: the streamed conditioner replaces the whole call.
        stream = getattr(text_encoder, "_h3_stream", None) if text_encoder is not None else None
        if stream is not None:
            # orig signature: (text_encoder, processor, token_ids,
            #  vision_inputs=None, text_encoder_layer=50, device=None, dtype=None)
            _pos = args[2:] if len(args) > 2 else []
            token_ids = kwargs.get("token_ids", _pos[0] if len(_pos) > 0 else None)
            if token_ids is None:
                raise ValueError("[te-stream] no token_ids in get_qwen3vl_prompt_embeds call")
            return stream.encode(
                token_ids=token_ids,
                vision_inputs=kwargs.get("vision_inputs", _pos[1] if len(_pos) > 1 else None),
                text_encoder_layer=kwargs.get("text_encoder_layer", _pos[2] if len(_pos) > 2 else None),
                device=kwargs.get("device"),
                dtype=kwargs.get("dtype"),
            )
        target_device = kwargs.get("device")
        if (
            text_encoder is not None
            and target_device is not None
            and any(True for _ in text_encoder.parameters())
        ):
            enc_device = next(text_encoder.parameters()).device
            if enc_device != target_device:
                run_kwargs = dict(kwargs)
                run_kwargs["device"] = enc_device
                out = orig(*args, **run_kwargs)
                return out.to(device=torch.device(target_device))
        return orig(*args, **kwargs)

    encoders.get_qwen3vl_prompt_embeds = bridged
    encoders._h3_te_device_bridged = True


def apply_text_budget(
    budget: int, cache_dir: Path | None = None, cache_key: str | None = None, cp: int = 1
):
    _ensure_torch()
    """Zero-embed text-run padding to `budget` rows (design 3.7 (i)).

    Pads both `prompt_embeds` and `text_token_tags` after the text-encoder
    step; `build_packed_sequence` derives the text run purely from
    `text_token_tags.shape[0]`, so padded rows flow through the layout, the
    rotary grid and the row scatter untouched. `budget <= 0` disables padding
    (exact mode: one graph per prompt length, lazy fallback).

    Also integrates the prompt-embed cache (design 4.1) keyed on the request
    fingerprint, and promotes the embeddings to the execution device so the
    packed-row scatter works when the text encoder stays on CPU.
    """
    encoders = _import_h3_module("encoders")
    for cls in (
        encoders.MiniMaxH3TextEncoderStep,
        encoders.MiniMaxH3FL2VATextEncoderStep,
    ):
        if getattr(cls, "_h3_budget_patched", False):
            cls._h3_budget = int(budget)
            cls._h3_cache_dir = cache_dir
            cls._h3_cache_key = cache_key
            cls._h3_cp = int(cp)
            continue
        orig_call = cls.__call__

        @functools.wraps(orig_call)
        def wrapped(self, components, state, _orig=orig_call):
            cache_dir_ = getattr(self.__class__, "_h3_cache_dir", None)
            cache_key_ = getattr(self.__class__, "_h3_cache_key", None)
            blob = (
                load_cached_embeds(cache_dir_, cache_key_)
                if (cache_dir_ and cache_key_)
                else None
            )
            if (
                blob is None
                and cache_dir_
                and cache_key_
                and int(getattr(self.__class__, "_h3_cp", 1)) > 1
                and (_RUNTIME.get("rank", 0) or 0) != 0
            ):
                # v99 encode-once-broadcast: rank0 encodes and stores; ranks
                # 1..N-1 wait for the cache file instead of re-encoding.
                # Fail-open: if rank0's store never lands (crash), the rank
                # computes locally — a slow run beats a hung one.
                _path = Path(cache_dir_) / f"{cache_key_}.pt"
                _t = time.perf_counter()
                while not _path.exists() and time.perf_counter() - _t < 660.0:
                    time.sleep(2.0)
                blob = load_cached_embeds(cache_dir_, cache_key_)
                if blob is None:
                    print(
                        f"[h3] WARNING: TE broadcast wait timed out ({_path}); "
                        "recomputing the text embedding locally",
                        flush=True,
                    )
                else:
                    print(
                        f"[h3] TE broadcast: received embeds from rank0 "
                        f"(waited {time.perf_counter() - _t:.1f}s)",
                        flush=True,
                    )
            if blob is None:
                components, state = _orig(self, components, state)
                embeds = state.get("prompt_embeds")
                tags = state.get("text_token_tags")
                if (
                    cache_dir_
                    and cache_key_
                    and embeds is not None
                    and tags is not None
                ):
                    store_cached_embeds(
                        cache_dir_,
                        cache_key_,
                        {
                            "prompt_embeds": embeds.detach().to("cpu"),
                            "text_token_tags": tags.detach().to("cpu"),
                        },
                    )
            else:
                import torch

                state.set("prompt_embeds", blob["prompt_embeds"])
                state.set("text_token_tags", blob["text_token_tags"])

            embeds = state.get("prompt_embeds")
            tags = state.get("text_token_tags")
            if (
                getattr(self.__class__, "_h3_budget", 0)
                and embeds is not None
                and tags is not None
            ):
                import torch

                rows = embeds.shape[1]
                pad = int(self.__class__._h3_budget) - rows
                if pad > 0:
                    zero_rows = torch.zeros(
                        (1, pad, embeds.shape[-1]),
                        dtype=embeds.dtype,
                        device=embeds.device,
                    )
                    last_tag = int(tags[-1].item()) if tags.numel() else 1
                    state.set("prompt_embeds", torch.cat([embeds, zero_rows], dim=1))
                    state.set(
                        "text_token_tags",
                        torch.cat(
                            [
                                tags,
                                torch.full(
                                    (pad,),
                                    last_tag,
                                    dtype=tags.dtype,
                                    device=tags.device,
                                ),
                            ]
                        ),
                    )
                elif pad < 0:
                    if int(getattr(self.__class__, "_h3_cp", 1)) > 1:
                        # Under CP the packed sequence must stay divisible by
                        # the mesh (EquipartitionSharder asserts it too); an
                        # over-budget prompt silently running exact-length
                        # would hang the all-to-all. Loud failure instead.
                        raise RuntimeError(
                            f"presentation has {rows} text rows > budget {self.__class__._h3_budget}; "
                            "under CP the padded seq must be divisible by the CP degree -- "
                            "raise --text-budget"
                        )
                    print(
                        f"[h3] WARNING: presentation has {rows} text rows > budget {self.__class__._h3_budget}; "
                        "running exact-length (no DiT graph shape reuse)."
                    )

            embeds = state.get("prompt_embeds")
            if embeds is not None and embeds.device != components._execution_device:
                state.set("prompt_embeds", embeds.to(components._execution_device))
            return components, state

        cls.__call__ = wrapped
        cls._h3_budget = int(budget)
        cls._h3_cache_dir = cache_dir
        cls._h3_cache_key = cache_key
        cls._h3_cp = int(cp)
        cls._h3_budget_patched = True


# ---------------------------------------------------------------------------
# Graphs + mark_step (design 3.4, 3.5 #4)
# ---------------------------------------------------------------------------


import torch.nn as _nn


def wrap_transformer_graphs(pipe, transformer):
    """`wrap_in_hpu_graph(transformer)` per bucket (Wan precedent
    `pipeline_wan_i2v.py:154-160`). Static per-bucket shapes (B=1, seq) make
    the DiT a per-bucket graph asset.

    v131: H3_PER_BLOCK_GRAPHS=1 wraps EACH transformer block instead of the
    whole transformer — the GC compiles one recipe per block, so cold-capture
    host RSS is bounded by the LARGEST single block instead of the 5.4k-node
    whole-DiT graph (v121 kernel-OOM at 92 GB). Combined with
    PT_HPUGRAPH_DISABLE_TENSOR_CACHE=1 (intermediates freed after each replay
    instead of retained per graph) this attacks both the host-IR and device
    ceilings h3expert identified. Replay boundary overhead ~0.5 ms/block
    (vllm-gaudi measurement) — negligible vs the device-bound step.
    """
    per_block = _env_bool("H3_PER_BLOCK_GRAPHS", False)
    shim = _try_module("graphs")
    apply = getattr(shim, "wrap_in_hpu_graph", None) if shim is not None else None
    if apply is None:
        try:
            from habana_frameworks.torch.hpu import wrap_in_hpu_graph
        except Exception as exc:
            print(
                f"[h3] WARNING: wrap_in_hpu_graph unavailable ({exc}); staying in eager lazy mode"
            )
            return transformer
        apply = wrap_in_hpu_graph
    if per_block:
        import habana_frameworks.torch.core as _htcore

        _htcore.mark_step()
        blocks = list(transformer.transformer_blocks)
        for _bi, _blk in enumerate(blocks):
            blocks[_bi] = apply(_blk, disable_tensor_cache=True)
        transformer.transformer_blocks = _nn.ModuleList(blocks)
        wrapped = transformer
        print(
            f"[h3] DiT wrapped per-block: {len(blocks)} block graphs "
            "(disable_tensor_cache=True)"
        )
    else:
        wrapped = apply(transformer)
    if wrapped is not transformer:
        try:
            pipe.update_components(transformer=wrapped)
        except Exception as exc:
            print(
                f"[h3] WARNING: could not re-register the graph-wrapped transformer ({exc})"
            )
    print("[h3] DiT wrapped into HPU graphs (per-bucket static shapes)")
    return wrapped


def apply_mark_step_pipe(pipe, transformer, text_encoder, include_transformer: bool):
    """Eager lazy-mode mark_step wrappers (design 3.5 #4): post-call
    `htcore.mark_step()` on the model entry points. mark_step is a no-op on
    CPU (single harmless warning), so this is safe to install unconditionally
    in the no-graph path."""
    if not _hpu_ok():
        return
    try:
        import habana_frameworks.torch.core as htcore
    except Exception:
        return

    def wrap_method(obj, name):
        orig = getattr(obj, name, None)
        if orig is None:
            return

        @functools.wraps(orig)
        def wrapped(*args, **kwargs):
            out = orig(*args, **kwargs)
            htcore.mark_step()
            return out

        with suppress(Exception):
            setattr(obj, name, wrapped)

    if (
        transformer is not None
        and include_transformer
        and not getattr(transformer, "_h3_mark_step_wrapped", False)
    ):
        wrap_method(transformer, "forward")
        transformer._h3_mark_step_wrapped = True
    if (
        text_encoder is not None
        and getattr(text_encoder, "model", None) is not None
        and not getattr(text_encoder.model, "_h3_mark_step_wrapped", False)
    ):
        wrap_method(text_encoder.model, "forward")
        text_encoder.model._h3_mark_step_wrapped = True
    for name in ("vae", "audio_vae"):
        component = getattr(pipe, name, None)
        if component is not None and not getattr(
            component, "_h3_mark_step_wrapped", False
        ):
            wrap_method(component, "decode")
            component._h3_mark_step_wrapped = True


# ---------------------------------------------------------------------------
# Embed cache (design 4.1)
# ---------------------------------------------------------------------------


def _fit_anchor_image(img, width: int, height: int, mode: str):
    """v117: fl2va anchor conditioning geometry.

    Stock fl2va stretches the geometry anchor onto the canvas (diffusers
    before_encoder.py MiniMaxH3FL2VASetupStep, resize_mode="default" = PIL
    LANCZOS resize). For an anchor whose aspect differs from the canvas that
    distorts subjects (4:3 photo on the 1.75 r768 canvas -> 31% horizontal
    stretch, measured frame0-vs-stretched-init cos 0.995). Server-side fit
    modes replace the stretch:
      stretch  stock behavior: plain resize to (width, height)
      crop     cover-crop with the released model's own follower-anchor
               rounding (scale=max, round, (src-w)//2 centering) — full-bleed,
               top/bottom content lost
      pad      ImageOps.pad: undistorted fit + letterbox bars. The bars are
               part of the conditioning (frame0 matches the padded anchor).
    Called per request in the server claim block; a same-size anchor passes
    through untouched for every mode.
    """
    from PIL import Image, ImageOps

    w, h = img.size
    if (w, h) == (width, height):
        return img
    if mode == "stretch":
        return img.resize((width, height), Image.LANCZOS)
    if mode == "crop":
        scale = max(width / w, height / h)
        rw = max(width, round(w * scale))
        rh = max(height, round(h * scale))
        left = max(0, (rw - width) // 2)
        top = max(0, (rh - height) // 2)
        return img.resize((rw, rh), Image.LANCZOS).crop(
            (left, top, left + width, top + height)
        )
    if mode == "pad":
        return ImageOps.pad(img, (width, height), method=Image.LANCZOS, color=(0, 0, 0))
    raise ValueError(f"bad anchor fit mode {mode!r} (stretch|crop|pad)")


def embed_cache_key(
    workflow: str, prompt: str, keyframes: list[str], fingerprint: str
) -> str:
    digest = hashlib.sha256()
    digest.update(b"h3-embed-cache-v1|")
    digest.update(workflow.encode())
    digest.update(b"|")
    digest.update(prompt.encode())
    digest.update(b"|")
    for kf in keyframes or []:
        path = Path(kf)
        digest.update(path.name.encode())
        digest.update(b"|")
        digest.update(hashlib.sha256(path.read_bytes()).hexdigest().encode())
        digest.update(b"|")
    digest.update(fingerprint.encode())
    return digest.hexdigest()


def load_cached_embeds(cache_dir: Path | None, key: str | None):
    if not cache_dir or not key:
        return None
    path = Path(cache_dir) / f"{key}.pt"
    if not path.exists():
        return None
    try:
        blob = torch.load(path, map_location="cpu", weights_only=True)
        print(f"[h3] embed cache hit: {path}")
        return blob
    except Exception as exc:
        print(
            f"[h3] WARNING: embed cache read failed ({exc}); recomputing the text embedding"
        )
        return None


def store_cached_embeds(cache_dir: Path | None, key: str | None, blob: dict) -> None:
    if not cache_dir or not key:
        return
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{key}.pt"
    # PID-unique tmp name: under CP every rank stores the same key and a shared
    # tmp file would race (torch.save interleave -> corrupt blob). The final
    # .replace is atomic; distinct tmp names make the whole write race-free.
    tmp = path.with_suffix(f".tmp.rank{os.getpid()}")
    torch.save(blob, tmp)
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Context parallelism (CP) + decode-subprocess hatch (PLAN_CP.md rev 2)
# ---------------------------------------------------------------------------

# Runner context shared with the patched decode blocks: the blocks are class-
# level methods and cannot see main()'s locals, so the export paths and the
# workdir (which the decode subprocess re-loads the VAEs from) live here.
_RUNTIME: dict = {}
# Per-block wall-time records for the CP timing merge; None = not recording.
_BLOCK_TIME_RECORDS: list | None = None
_BLOCK_TIMES_LAST: list | None = None
_CP_TEST_ENV = "H3_CP_TEST"


def _cp_skip_decode() -> bool:
    """True when a CP rank other than rank0 must skip the decode phase."""
    return _RUNTIME.get("world_size", 1) > 1 and _RUNTIME.get("rank", 0) not in (None, 0)


def _pick_free_port() -> int:
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class _RankTaggedStdout:
    """Line-buffered stdout wrapper: any line containing `[h3` gets the rank
    tag (`[h3:rN]`) spliced in, so N workers interleaving on one console stay
    attributable (PLAN_CP.md phase 1.4). Everything else passes through
    untouched; non-log lines are printed verbatim."""

    def __init__(self, stream, rank: int):
        self._stream = stream
        self._rank = rank
        self._buf = ""

    def _tag(self, line: str) -> str:
        if "[h3" in line:
            return line.replace("[h3", f"[h3:r{self._rank}]", 1)
        return line

    def write(self, s: str) -> int:
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._stream.write(self._tag(line) + "\n")
        return len(s)

    def flush(self) -> None:
        if self._buf:
            self._stream.write(self._tag(self._buf))
            self._buf = ""
        self._stream.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


def _cp_init_process_group(hpu_ok: bool, world_size: int) -> str:
    """Initialize the CP process group; returns the backend in use.

    hccl under an approved HPU run (1 proc : 1 HPU, same as vLLM-Gaudi);
    gloo when HPU is unavailable AND the run is explicitly a CPU harness
    (H3_CP_TEST=1) or an approved CPU fallback -- that gloo path is what keeps
    the WHOLE CP code path exercisable on a CPU-only box (PLAN_CP.md 1.5).
    RANK/WORLD_SIZE/MASTER_ADDR/MASTER_PORT come from the worker env set by
    _cp_worker_entry()."""
    import torch.distributed as dist

    if dist.is_initialized():
        return str(dist.get_backend())
    if hpu_ok:
        try:
            import habana_frameworks.torch.distributed.hccl  # noqa: F401  (registers hccl)
        except Exception as exc:
            raise RuntimeError(
                f"CP requires the hccl backend and the Habana plugin failed to provide it: {exc}"
            ) from exc
        backend = "hccl"
    elif _env_bool(_CP_TEST_ENV, False) or _env_bool("H3_ALLOW_CPU_FALLBACK", False):
        backend = "gloo"
    else:
        raise RuntimeError(
            "CP>1 requires approved HPU (H3_ALLOW_HPU=1) or the CPU harness "
            "(H3_CP_TEST=1 / H3_ALLOW_CPU_FALLBACK=1); refusing an ambiguous run."
        )
    dist.init_process_group(backend=backend)
    if dist.get_world_size() != world_size:
        raise RuntimeError(
            f"CP world mismatch: process group world {dist.get_world_size()} != requested {world_size}"
        )
    return backend


def _cp_enable_parallelism(transformer, world_size: int, backend: str):
    """Build the device mesh and switch the DiT to diffusers context parallel.

    Actual enable_parallelism API (diffusers 0.40.0, recorded for PLAN_CP.md):
    keyword-only `config: ParallelConfig | ContextParallelConfig |
    TensorParallelConfig` plus optional `cp_plan` (the model's class-level
    `_cp_plan` applies when omitted). It derives device_type from
    torch._C._get_accelerator() internally, so we ALWAYS hand it our own mesh
    (built for the backend actually in use) and never let it create a second
    one. Under CP the device map puts dit/vae/audio on the SAME device set;
    the text encoder stays on the host."""
    from torch.distributed.device_mesh import init_device_mesh

    from diffusers.models._modeling_parallel import ContextParallelConfig

    device_type = "hpu" if backend == "hccl" else "cpu"
    mesh = init_device_mesh(
        device_type, mesh_shape=(1, world_size), mesh_dim_names=("ring", "ulysses")
    )
    cp_cfg = ContextParallelConfig(ulysses_degree=world_size, mesh=mesh)
    if backend == "hccl":
        # v122: equal-split all-to-alls — the HCCL JobThread validates capture
        # split lists against the tensor BASE (view dim0 mismatch abort). See
        # _install_cp_static_allsplit in hpu_patches.py.
        import hpu_patches as _hp

        _hp._install_cp_static_allsplit()
    transformer.enable_parallelism(config=cp_cfg)
    # enable_parallelism stamps `_parallel_config` on EVERY attention processor
    # in the model -- including the token refiner's. The refiner runs on the
    # unsharded text run BEFORE the packed buffer is built (the _cp_plan splits
    # only at transformer_blocks.0), so its attention must stay replicated:
    # CP-wrapping a full-seq input through the Ulysses all-to-all shrinks the
    # head dim (ws*S_full seq with repeated blocks) and breaks shapes/math the
    # same way the stock native backend would. Un-stamp exactly the refiner's
    # processors; the packed-sequence attention keeps the config.
    cleared = 0
    refiner = getattr(transformer, "token_refiner", None)
    if refiner is not None:
        for module in refiner.modules():
            processor = getattr(module, "processor", None)
            if processor is not None and getattr(processor, "_parallel_config", None) is not None:
                processor._parallel_config = None
                cleared += 1
    if cleared:
        print(
            f"[h3] CP: token refiner kept replicated ({cleared} processors un-stamped; "
            "refiner runs pre-packing on the full text run)",
            flush=True,
        )
    print(
        f"[h3] context parallelism ON: ulysses={world_size} backend={backend} "
        f"mesh={mesh} (seq sharded inside diffusers via the model's _cp_plan)",
        flush=True,
    )
    return cp_cfg


def _cp_verify_identical_cpu(tensor, tag: str) -> None:
    """CP noise-determinism check (PLAN_CP.md phase 1 item 8): every rank must
    hold the SAME full latent tensor before the model's _cp_plan sharding kicks
    in (the plan slices dim-1 inside diffusers; nothing here hand-slices).
    Cheap sha256 of the CPU bytes, one all_gather of digests, loud failure."""
    import hashlib

    import torch.distributed as dist

    if tensor is None or not dist.is_initialized() or dist.get_world_size() == 1:
        return
    t = tensor.detach()
    if t.device.type != "cpu":
        t = t.to("cpu")
    digest = hashlib.sha256(t.contiguous().to(torch.float32).numpy().tobytes()).digest()[:8]
    # v117: fold the server job id into the gathered payload. In the v115
    # desync the ranks held IDENTICAL noise (same seed) while serving
    # DIFFERENT jobs, so the tensor digest alone passed spuriously. Gathering
    # 8 B tensor digest + 4 B job-id digest makes a cross-job desync fail
    # loudly instead of silently cross-contaminating outputs.
    _srv_job = str(_RUNTIME.get("_server_job") or "")
    job_digest = hashlib.sha256(_srv_job.encode()).digest()[:4]
    local = torch.frombuffer(bytearray(digest + job_digest), dtype=torch.uint8).clone()
    world = dist.get_world_size()
    # hccl collectives only accept HPU-resident tensors ("No backend type
    # associated with device type cpu"); gloo (the CPU harness) only cpu.
    # Place the 12-byte digest on the backend's device for the gather.
    if dist.get_backend() == "hccl":
        gather_dev = torch.device("hpu")
    else:
        gather_dev = torch.device("cpu")
    local_dev = local.to(gather_dev)
    gathered = [torch.empty_like(local_dev) for _ in range(world)]
    dist.all_gather(gathered, local_dev)
    tensor_dg = {bytes(g.cpu().tolist()[:8]) for g in gathered}
    job_dg = {bytes(g.cpu().tolist()[8:]) for g in gathered}
    if len(tensor_dg) != 1:
        raise RuntimeError(
            f"CP noise determinism FAILED for {tag}: {len(tensor_dg)} distinct draws across "
            f"{world} ranks (seed/generator diverged)"
        )
    if len(job_dg) != 1:
        raise RuntimeError(
            f"CP noise determinism FAILED for {tag}: RANK DESYNC — ranks are serving "
            f"{len(job_dg)} different job contexts across {world} ranks "
            f"(stale serve_sync state or missed claim broadcast)"
        )
    if dist.get_rank() == 0:
        print(f"[h3] CP noise determinism: {tag} identical on all {world} ranks", flush=True)


def _pick_free_hpu_card(preferred: int, skip: int = -1, timeout_s: float = 300.0) -> int:
    """Pick an HPU card with only the idle ~768 MiB driver footprint.

    CP=8 uses every card for workers; ranks 1..7 exit after denoise, and a
    dead process releases ALL its device memory (unlike the parent's stale
    caching-allocator pool). So scan all cards, wait (up to timeout) for one
    to drain, prefer `preferred` (= world_size+rank when it exists).
    """
    import subprocess as sp
    import time as _t

    deadline = _t.monotonic() + timeout_s
    order = [preferred] + [c for c in range(8) if c not in (preferred, skip)]
    # Only in-range candidates are legal; `preferred` may point past the last
    # card when every card is in the CP set.
    order = [c for c in order if 0 <= c < 8 and c != skip]
    if not order:
        return 0
    while True:
        used = {}
        try:
            out = sp.run(
                ["hl-smi", "-Q", "index,memory.used", "-f", "csv"],
                capture_output=True,
                text=True,
                timeout=30,
            ).stdout
            for ln in out.splitlines()[1:]:
                parts = [p.strip() for p in ln.split(",")]
                if len(parts) == 2 and parts[1].endswith("MiB"):
                    used[int(parts[0])] = int(parts[1].split(" ")[0])
        except Exception as e:  # hl-smi unavailable -> trust the preferred card
            print(f"[h3] hl-smi probe failed ({e}); using card {preferred}", flush=True)
            return preferred
        for c in order:
            if used.get(c, 1 << 30) <= 2048:
                return c
        if _t.monotonic() > deadline:
            # Never return an invalid card: pick the least-used in-range one.
            return min(order, key=lambda c: used.get(c, 1 << 30))
        _t.sleep(5.0)


def _shard_heartbeat(device) -> None:
    """Tiny completed device op so a polling rank's card keeps showing engine
    progress (the Synapse no-progress watchdog kills cards with enqueued-but
    -stalled work; a completed 64-element add every poll resets it)."""
    t = torch.ones(64, device=device)
    float((t + 1).sum().item())


def _shard_wait_file(path: Path, device, timeout_s: float = 600.0) -> Path:
    """Poll for a handoff file (host-side wait; heartbeat keeps the card
    ticking). Returns the path or raises TimeoutError."""
    t0 = time.perf_counter()
    while not path.exists():
        if time.perf_counter() - t0 > timeout_s:
            raise TimeoutError(f"shard handoff file never appeared: {path}")
        try:
            _shard_heartbeat(device)
        except Exception:
            pass
        time.sleep(1.0)
    return path


def _wait_card_free(card: int, timeout_s: float = 180.0) -> None:
    """Wait until the card shows ~idle usage. At spawn time the CP parent
    ranks still hold ~72 GiB (freed-DiT pages stay in each parent's pool
    while the process lives; v74's workers died on 394 MB allocs). Ranks
    1..7 hard-exit ~1 s after the farm spawns, so this wait is short."""
    t0 = time.perf_counter()
    while True:
        try:
            out = sp.run(
                ["hl-smi", "-Q", "index,memory.used", "-f", "csv"],
                capture_output=True,
                text=True,
                timeout=30,
            ).stdout
            used = {}
            for ln in out.splitlines()[1:]:
                parts = [p.strip() for p in ln.split(",")]
                if len(parts) == 2 and parts[1].endswith("MiB"):
                    used[int(parts[0])] = int(parts[1].split(" ")[0])
            # "Enough free for a worker" — workers are no_grad (~10 GiB:
            # VAE 9.7 + decode workspace), so they coexist with a resident
            # 62 GiB DiT + 15 GiB VAEs on a CP serve card (87 of 94.6).
            # The gate exists to avoid the v74 autograd-OOM class of
            # failures and grossly oversubscribed cards.
            if used.get(card, 1 << 30) <= 88000:
                return
        except Exception:
            return  # hl-smi unavailable: spawn anyway (old behavior)
        if time.perf_counter() - t0 > timeout_s:
            print(
                f"[h3] card {card} still busy after {timeout_s}s; spawning anyway",
                flush=True,
            )
            return
        time.sleep(1.0)


def _respawn_farm_worker(argv, card, device_tip, log_path=None):
    """Respawn one farm worker (self-healing: a worker that died during the
    marker wait is respawned once). Worker output goes to a log FILE under
    frames_dir/.shard_tmp so the death reason is always on disk (v82 lesson:
    PIPE output is lost when the parent never reads it)."""
    import subprocess as sp

    env = dict(os.environ)
    env["PT_HPU_LAZY_MODE"] = "0"
    env.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "1" if device_tip == "hpu" else "0")
    if device_tip == "hpu":
        # v82: HABANA_VISIBLE_DEVICES is IGNORED by Synapse device acquire — a
        # process with no pin first-free-scans ALL cards and FAILS (synStatus=8,
        # "already acquired by PID") when every card is held, e.g. by resident
        # CP parents in serve mode. HABANA_VISIBLE_MODULES=<physical card> is
        # the Synapse-level selector that actually works, and explicitly
        # acquiring a pinned card PERMITS same-card coexistence with the
        # resident parent (probe-proven, two eager processes on one card).
        env["HABANA_VISIBLE_DEVICES"] = str(card)
        env["HABANA_VISIBLE_MODULES"] = str(card)
    if log_path is None:
        # Derive from argv (--frames-dir + .shard_tmp) when caller didn't say.
        fi = argv.index("--frames-dir")
        shard_i = int(argv[argv.index("--shard") + 1])
        log_path = Path(argv[fi + 1]) / ".shard_tmp" / f"worker_{shard_i}_re.log"
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    fh = open(log_path, "w")
    _RUNTIME.setdefault("_farm_log_fhs", []).append(fh)
    return sp.Popen(argv, env=env, stdout=fh, stderr=sp.STDOUT, text=True)


def _decode_sharded_core(vae, z_raw, pixel_mean, pixel_std, frames_dir, rank, world, device):
    """Temporal-sharded MiniMax-H3 video VAE decode across CP ranks.

    AutoencoderKLMiniMaxH3._decode walks the latents in `tokens_chunk_size`
    latent-frame chunks; the ONLY cross-chunk state is a `frame_overlap`-frame
    pixel tensor (the un-blended tail of the previous clip). So the internal
    loop distributes exactly: chunk i -> rank i % world, every clip decode
    runs concurrently on the VAE already resident on that rank's card. Each
    rank denormalizes + clamps its own frames (identical elementwise math to
    the stock path) and writes them straight to the shared frames_dir with
    global numbering -- no gather.

    Cross-rank handoff is FILE-BASED, deliberately: v72 showed that after the
    giant denoise graph drains, the first big post-drain enqueue (the cast
    feeding an HCCL broadcast) stalls -- enqueued-but-stalled work is also
    what trips the Synapse no-progress watchdog (idle cards are safe, v70).
    So the decode phase uses NO collectives at all: each owner drops its
    clip's 5-frame tail as a bf16 .pt under frames_dir/.shard_tmp/, the next
    owner polls for it (host wait + heartbeat), and completion is signaled
    with marker files. Latents are NOT broadcast either: CP replication plus
    the verified per-step determinism (v61) makes them identical by
    construction.

    Geometry notes: decoder unpatchify expands patch_size pixels per latent
    row (16) and patch_size_t per latent frame (= temporal ratio, 4); clip i
    covers latents [i*ts, i*ts+ts+tok_ov) and contributes cnf-pre blended
    frames at pixel offset i*(cnf-pre), with the last clip's un-blended tail
    appended at the end. Returns (frames_saved, clip_decode_seconds,
    phase_seconds)."""
    vae.eval()
    cfg = vae.config
    temporal_ratio = int(vae.temporal_compression_ratio)
    ts = int(vae.tokens_chunk_size)
    tok_ov = int(vae.token_overlap)
    pre = int(vae.frame_pre_padding)
    fov = int(vae.frame_overlap)
    cnf = ts * temporal_ratio
    token_drop = int(getattr(cfg, "token_drop", 0) or 0)

    # Replicated + deterministic under CP (verified at the noise level, v61);
    # identical inputs + identical deterministic steps => identical latents.
    latents = z_raw.to(device).contiguous()

    lm_cfg = getattr(cfg, "latents_mean", None)
    if lm_cfg is not None:
        latents_mean = torch.tensor(cfg.latents_mean, device=device).view(1, -1, 1, 1, 1)
        latents_std = torch.tensor(cfg.latents_std, device=device).view(1, -1, 1, 1, 1)
        z = (latents * latents_std + latents_mean).to(
            next(vae.decoder.parameters()).dtype
        )
    else:
        # Synthetic configs (CPU tests) may omit the normalization constants;
        # the real checkpoint always carries them.
        z = latents.to(next(vae.decoder.parameters()).dtype)

    num_tokens = z.shape[2] + token_drop
    pad = (-num_tokens) % ts
    n_chunks = (num_tokens + pad) // ts - int(token_drop > 0)
    if pad > 0:
        z = torch.cat([z, z[:, :, -1:].repeat(1, 1, pad, 1, 1)], dim=2)

    pm = torch.tensor(pixel_mean, device=device).view(1, -1, 1, 1, 1)
    ps = torch.tensor(pixel_std, device=device).view(1, -1, 1, 1, 1)
    frames_dir = Path(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    tmp = frames_dir / ".shard_tmp"
    tmp.mkdir(parents=True, exist_ok=True)

    import numpy as np
    from PIL import Image

    t0 = time.perf_counter()
    clip_s = 0.0
    frames_saved = 0
    stride = cnf - pre
    # Inference only: without no_grad the 36-layer decoder saves activations
    # for backward and fills the card (~94 GiB at 480p clip shapes -> the
    # 394 MB PT_DEVMEM failures in v74/v75 farm workers; the in-process path
    # was shielded by the block's @torch.no_grad decorator, the raw core was
    # not).
    _grad_prev = torch.is_grad_enabled()
    torch.set_grad_enabled(False)

    def _decode_sharded_body():
        nonlocal frames_saved, clip_s

        def _write_frames(pix_uint8, base):
            nonlocal frames_saved
            for k in range(pix_uint8.shape[2]):
                arr = pix_uint8[0, :, k].contiguous().cpu().numpy().transpose(1, 2, 0)
                Image.fromarray(arr).save(frames_dir / f"frame_{base + k:06d}.png")
                frames_saved += 1

        def _write_tail(ov_out, i):
            # Save the UN-blended tail in the clip's own dtype (stock blends in
            # the clip dtype before denormalization). Per-frame D2H (~2.5 MB) --
            # the small-copy pattern that reliably completes.
            frames = [ov_out[0, :, k].contiguous().cpu() for k in range(ov_out.shape[2])]
            tail = torch.stack(frames, dim=1).unsqueeze(0)
            torch.save(tail, tmp / f"tail_{i:03d}.pt")

        # ---- Phase A: pre-decode OWNED chunks (no collectives inside) ----
        # The clip decode never needs the previous clip's overlap -- only the
        # blend does -- so every owned clip is enqueued FIRST and all cards
        # work concurrently. (v71's decode-after-broadcast order serialized
        # the clips into a world-long chain; v72's broadcast handoff then
        # stalled on the post-drain enqueue.) Chunk refs stay alive until
        # written.
        owns = [i for i in range(n_chunks) if i % world == rank]
        decoded = {}
        for i in owns:
            t_clip = time.perf_counter()
            start = i * ts
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16 if device.type == "hpu" else torch.float16,
                enabled=device.type in ("cuda", "hpu"),
            ):
                clip = vae._decode_clip(z[:, :, start : start + ts + tok_ov])
            clip_s += time.perf_counter() - t_clip
            # v97: per-clip frontier break (see _decode_fused_core).
            _flush_lazy_frontier()
            part = clip[:, :, :cnf][:, :, pre:]
            ov_out = clip[:, :, cnf:][:, :, pre:]
            decoded[i] = (part, ov_out)

        # ---- Phase B: file handoff + blend + frame writes (ascending) ----
        for i in owns:
            part, ov_out = decoded.pop(i)
            if i > 0:
                tail_path = _shard_wait_file(tmp / f"tail_{i - 1:03d}.pt", device)
                overlap_in = torch.load(tail_path, map_location=device, weights_only=True)
                if overlap_in.dtype != part.dtype:
                    # Stock blends in the clip's own dtype; the transport
                    # cast is logged so parity deltas stay attributable.
                    print(
                        f"[h3:r{rank}]] sharded overlap cast "
                        f"{overlap_in.dtype} -> {part.dtype}",
                        flush=True,
                    )
                    overlap_in = overlap_in.to(part.dtype)
                part = vae._blend(overlap_in, part, fov, dim=-3)
            last = i == n_chunks - 1
            if not last:
                _write_tail(ov_out, i)
            pix = torch.cat([part, ov_out], dim=2) if last else part
            # Denormalize + clamp on device, quantize to uint8 ON DEVICE so
            # each per-frame D2H stays ~1.5 MB (the small-copy pattern that
            # reliably completes; the monolithic 616 MB pull is the v50
            # wedge).
            pix = ((pix.float() * ps + pm).clamp(0, 1) * 255.0).to(torch.uint8)
            _write_frames(pix, i * stride)
        # Done marker for ALL ranks (chunkless ranks included) -- the export
        # phase polls these instead of a dist barrier (post-drain collectives
        # stall).
        (tmp / f"done_r{rank}.json").write_text(
            json.dumps({"rank": rank, "chunks": owns, "frames": frames_saved})
        )
        return frames_saved, clip_s, time.perf_counter() - t0

    try:
        return _decode_sharded_body()
    finally:
        torch.set_grad_enabled(_grad_prev)


def _decode_fused_core(vae, z_raw, pixel_mean, pixel_std, frames_dir, rank, world, device, write_marker=True):
    """v93: IN-PROCESS FUSED decode for CP serve with resident DiT weights.

    One lazy frontier per rank, enqueued while the denoise queue is still
    draining (no boundary sync, no DiT free -- the decode fits beside the
    resident DiT at ~+0.1 GiB activations, v74):

        [normalize z] [decode OWNED clips (cached recipes)] [stack own tails]
        [zeros-padded tail buffer] [all_gather tails across ranks]
        [blend owned chunks with gathered neighbor tails] [denorm + quantize
        to uint8] [permute frames-first + contiguous]  -> mark_step

    Rationale, from the evidence:
    - collectives inside a lazy frontier are the PROVEN denoise pattern
      (every denoise-step all_reduce runs in one); the all_gather is the
      same class of op, tiny (world x n_chunks x 5 frames bf16).
    - the post-drain wedge (v72/v73c/v86/v92) hits device-work enqueues or
      D2Hs AFTER the queue drains at a variable point; by fusing the blend +
      quantize into the frontier and emitting ONE contiguous frames tensor,
      the post-frontier work is PURE contiguous D2H pulls (~0.8 MB/frame) --
      the same class as the 8/8-reliable blob save.
    - the tail file-handoff (and its H2D torch.load) disappears entirely:
      the neighbor tail arrives via the in-frontier all_gather.

    Determinism: same clip math + exact-copy gather + same blend/denorm math
    as _decode_sharded_core, so output is bit-identical modulo the gather.
    Chunkless ranks contribute zero tails and write no frames; every rank
    writes the same done_r{rank}.json marker the export phase polls.
    """
    import torch.distributed as dist

    vae.eval()
    cfg = vae.config
    temporal_ratio = int(vae.temporal_compression_ratio)
    ts = int(vae.tokens_chunk_size)
    tok_ov = int(vae.token_overlap)
    pre = int(vae.frame_pre_padding)
    fov = int(vae.frame_overlap)
    cnf = ts * temporal_ratio
    token_drop = int(getattr(cfg, "token_drop", 0) or 0)

    latents = z_raw.to(device).contiguous()
    lm_cfg = getattr(cfg, "latents_mean", None)
    if lm_cfg is not None:
        latents_mean = torch.tensor(cfg.latents_mean, device=device).view(1, -1, 1, 1, 1)
        latents_std = torch.tensor(cfg.latents_std, device=device).view(1, -1, 1, 1, 1)
        z = (latents * latents_std + latents_mean).to(next(vae.decoder.parameters()).dtype)
    else:
        z = latents.to(next(vae.decoder.parameters()).dtype)

    num_tokens = z.shape[2] + token_drop
    pad = (-num_tokens) % ts
    n_chunks = (num_tokens + pad) // ts - int(token_drop > 0)
    if pad > 0:
        z = torch.cat([z, z[:, :, -1:].repeat(1, 1, pad, 1, 1)], dim=2)

    pm = torch.tensor(pixel_mean, device=device).view(1, -1, 1, 1, 1)
    ps = torch.tensor(pixel_std, device=device).view(1, -1, 1, 1, 1)
    frames_dir = Path(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    tmp = frames_dir / ".shard_tmp"
    tmp.mkdir(parents=True, exist_ok=True)

    import numpy as np
    from PIL import Image

    t0 = time.perf_counter()
    clip_s = 0.0
    frames_saved = 0
    stride = cnf - pre
    owns = [i for i in range(n_chunks) if i % world == rank]

    _grad_prev = torch.is_grad_enabled()
    torch.set_grad_enabled(False)
    try:
        decoded = {}
        for i in owns:
            t_clip = time.perf_counter()
            start = i * ts
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type in ("cuda", "hpu")):
                clip = vae._decode_clip(z[:, :, start : start + ts + tok_ov])
            clip_s += time.perf_counter() - t_clip
            # v97: per-clip frontier break — without it the stack/blend/gather
            # ops fuse INTO the clip's frontier, and even with per-block
            # mark_steps the tail of the frontier re-grows a giant graph.
            _flush_lazy_frontier()
            decoded[i] = (
                clip[:, :, :cnf][:, :, pre:],
                clip[:, :, cnf:][:, :, pre:],
            )
        # FLUSH 1 (v95 lesson): close the clip frontier BEFORE the collective.
        # A lazy collective only executes once a mark_step closes its frontier;
        # all_gather into an OPEN frontier deadlocks the host in work.wait()
        # (the mark_step can never be issued from inside the wait). Denoise
        # collectives never hit this: every denoise step flushes before the
        # next all_reduce is enqueued -- collectives always arrive in an
        # empty queue. The flush also keeps the device draining (denoise +
        # clips execute while the host continues), so no drain-to-idle wedge
        # window ever opens.
        _flush_lazy_frontier()
        # Tail exchange: rank r's slot holds tail(i) for every owned chunk i
        # (zeros elsewhere); all_gather lands the whole table on every rank.
        # Tails are PIXEL frames: slot shape (n_chunks, 3, fov, Hp, Wp).
        # Chunkless ranks derive pixel dims from the spatial unpatchify
        # factor (16) applied to the latent grid.
        _, _, _, Hl, Wl = z.shape
        patch = int(getattr(cfg, "patch_size", 16) or 16)
        Hp, Wp = Hl * patch, Wl * patch
        tail_dtype = torch.bfloat16 if device.type == "hpu" else torch.float32
        my_tails = torch.zeros(n_chunks, 3, fov, Hp, Wp, device=device, dtype=tail_dtype)
        for i in owns:
            my_tails[i] = decoded[i][1][0].to(my_tails.dtype)
        # FLUSH 2: close the tiny tail-buffer frontier BEFORE the collective --
        # with lazy collectives disabled (default) all_gather executes EAGERLY
        # at enqueue and blocks the host until the device completes it, so it
        # must enter a clean queue that is already stream-ordered after the
        # tail stacks (same discipline as a denoise step's all_reduce after
        # its mark_step).
        _flush_lazy_frontier()
        gathered = [torch.empty_like(my_tails) for _ in range(world)]
        dist.all_gather(gathered, my_tails)
        # Blend + quantize IN-FRONTIER using gathered neighbor tails.
        pix_out = {}
        for i in owns:
            part, ov_out = decoded.pop(i)
            if i > 0:
                src_rank = (i - 1) % world
                overlap_in = gathered[src_rank][i - 1].unsqueeze(0)
                if overlap_in.dtype != part.dtype:
                    overlap_in = overlap_in.to(part.dtype)
                part = vae._blend(overlap_in, part, fov, dim=-3)
            last = i == n_chunks - 1
            pix = torch.cat([part, ov_out], dim=2) if last else part
            pix = ((pix.float() * ps + pm).clamp(0, 1) * 255.0).to(torch.uint8)
            pix_out[i] = pix
        # One contiguous frames-first tensor per chunk IN-FRONTIER, so every
        # post-frontier D2H is a pure contiguous DMA pull (no device kernels).
        frames_c = {
            i: p.permute(2, 0, 1, 3, 4).contiguous() for i, p in pix_out.items()
        }
        del decoded, pix_out, gathered, my_tails
        # FLUSH 3: close the blend/quantize frontier, then sync. After this,
        # every frame tensor is materialized in device memory and the ONLY
        # remaining work is pure contiguous D2H pulls -- the small-copy
        # post-sync pattern that has been 8/8 reliable (blob saves).
        _flush_lazy_frontier()
        torch.hpu.synchronize()
        for i in owns:
            p = frames_c.pop(i)
            base = i * stride
            for k in range(p.shape[0]):
                arr = p[k, 0].cpu().numpy().transpose(1, 2, 0)  # (3,H,W) -> HWC
                Image.fromarray(arr).save(frames_dir / f"frame_{base + k:06d}.png")
                frames_saved += 1
        if write_marker:
            (tmp / f"done_r{rank}.json").write_text(
                json.dumps({"rank": rank, "chunks": owns, "frames": frames_saved})
            )
        return frames_saved, clip_s, time.perf_counter() - t0
    finally:
        torch.set_grad_enabled(_grad_prev)


def _decode_sharded_audio(components, state, block_state, rank) -> None:
    """Audio decode for the sharded path (strict fp32, wav written directly).

    Mirrors _H3AudioDecodeStep's math: denormalize with the audio VAE's
    latents_mean/std, decode, permute to (1, 2, S) stereo, save wav.
    """
    device = components.audio_vae.device
    audio_latents = getattr(block_state, "audio_latents", None)
    if audio_latents is None:
        audio_latents = state.get("audio_latents") if hasattr(state, "get") else None
    if audio_latents is None:
        print(f"[h3:r{rank}]] sharded audio: no audio latents; skipping wav", flush=True)
        return
    audio_latents = audio_latents.to(device)
    a_mean = torch.tensor(components.audio_vae.config.latents_mean, device=device).view(1, -1, 1)
    a_std = torch.tensor(components.audio_vae.config.latents_std, device=device).view(1, -1, 1)
    audio = components.audio_vae.decode(audio_latents * a_std + a_mean, return_dict=False)[0]
    audio = audio.float().permute(1, 0, 2)  # (1, 2, S) stereo
    sampling_rate = int(components.audio_sampling_rate)
    save_audio_wav(audio, sampling_rate, Path(_RUNTIME["wav_path"]))
    # v97b: every caller (audio step stub / fused / queue branches) previously
    # had to set sharded_audio_done itself; the fused and queue branches forgot,
    # so rank0 re-decoded the audio post-drain (a second post-drain enqueue
    # waiting behind the first). The function owns the flag now.
    _RUNTIME["sharded_audio_done"] = True
    print(f"[h3:r{rank}]] sharded audio decode: wav -> {_RUNTIME['wav_path']}", flush=True)


def _spawn_decode_farm(self, components, state, block_state):
    """H3_DECODE_SHARDED=1, parent side: temporal-sharded decode via a FARM of
    fresh eager subprocesses.

    In-process sharded decode is not viable on this stack: after the giant
    denoise graph drains, post-drain device work in the SAME process wedges
    at a variable point (v72: clip + 17 frame D2Hs then the cast feeding an
    HCCL broadcast; v73c: the first tail D2H). Fresh processes are immune
    (the single-process decode subprocess has been 8/8 reliable), and the
    clip recipes are cached, so a farm of one eager worker per chunk decodes
    the whole video in ~load+one-clip wall (~35 s vs 112 s single-process).

    Protocol (all file-based, NO post-drain collectives):
    - rank0 saves the latents blob (the same small post-drain D2H that has
      been 8/8 reliable) and Popens nshards video workers (chunk i -> worker
      i) + one audio worker; workers run decode_subproc.py --shard/--audio-shard.
    - Workers share frames_dir; tail handoff + done markers live under
      frames_dir/.shard_tmp (identical to the in-process protocol).
    - The parent's export phase polls the markers (host-side, safe) and muxes.
    """
    import subprocess as sp

    rank = _RUNTIME.get("rank", 0) or 0
    world = _RUNTIME.get("world_size", 1) or 1
    if rank not in (None, 0):
        # Non-zero ranks skip: their idle cards host farm workers instead.
        block_state.videos = None
        self.set_block_state(state, block_state)
        return components, state
    workdir = _RUNTIME.get("workdir")
    frames_dir = _RUNTIME.get("frames_dir")
    wav_path = _RUNTIME.get("wav_path")
    if not (workdir and frames_dir and wav_path):
        raise RuntimeError("sharded decode farm requires the runner context")
    frames_dir = Path(frames_dir)
    wav_path = Path(wav_path)
    scratch = frames_dir / ".shard_tmp"
    scratch.mkdir(parents=True, exist_ok=True)
    # Wipe stale markers/tails from previous runs sharing this artifact dir.
    for stale in scratch.glob("*"):
        stale.unlink()

    vae_cfg = components.vae.config
    ts = int(components.vae.tokens_chunk_size)
    token_drop = int(getattr(vae_cfg, "token_drop", 0) or 0)
    num_tokens = block_state.latents.shape[2] + token_drop
    n_chunks = (num_tokens + (-num_tokens) % ts) // ts - int(token_drop > 0)
    nshards = min(n_chunks, max(world, 1))
    audio_marker = n_chunks % world

    # Blob save: the ONE post-drain D2H pattern proven reliable (small,
    # immediately after the boundary sync + DiT free).
    audio_latents = getattr(block_state, "audio_latents", None)
    if audio_latents is None:
        audio_latents = state.get("audio_latents") if hasattr(state, "get") else None
    blob_path = wav_path.parent / (wav_path.stem + "_decode_blob.pt")
    blob = {
        "latents": block_state.latents.detach().to("cpu", torch.float32),
        "audio_latents": (
            audio_latents.detach().to("cpu", torch.float32)
            if audio_latents is not None
            else None
        ),
        "pixel_mean": list(components.pixel_mean),
        "pixel_std": list(components.pixel_std),
    }
    torch.save(blob, blob_path)
    meta_path = wav_path.parent / (wav_path.stem + "_decode_meta.json")
    subproc_py = Path(__file__).resolve().parent / "decode_subproc.py"
    device_tip = "hpu" if block_state.latents.device.type == "hpu" else "cpu"

    env = dict(os.environ)
    env["PT_HPU_LAZY_MODE"] = "0"  # eager plugin; decode_subproc.py double-checks pre-import
    env.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "1" if device_tip == "hpu" else "0")

    def _farm_env(card: int) -> dict:
        e = dict(env)
        if device_tip == "hpu":
            # v82: pin the physical card at the Synapse level (see the
            # HABANA_VISIBLE_MODULES note in _respawn_farm_worker). Without the
            # pin the worker first-free-scans and dies when all 8 cards are
            # held by resident CP parents (v80/v81 worker massacre).
            e["HABANA_VISIBLE_DEVICES"] = str(card)
            e["HABANA_VISIBLE_MODULES"] = str(card)
        return e

    procs = []
    _RUNTIME["_farm_log_fhs"] = []
    # v85 settle delay: the denoise boundary sync (0.8 s) is NOT enough for the
    # driver to settle every card (v84-req1: the whole first batch died at
    # synDeviceAcquire inside the drain window; respawns seconds later
    # succeeded). Give the cards a beat before the first spawn.
    time.sleep(8.0)
    if audio_latents is not None:
        # Audio worker FIRST on card 0 (rank0's own card has ~22 GiB free
        # after the DiT release; the audio VAE + activations need ~5 GiB).
        procs.append(
            sp.Popen(
                [
                    sys.executable,
                    str(subproc_py),
                    "--blob",
                    str(blob_path),
                    "--workdir",
                    str(workdir),
                    "--frames-dir",
                    str(frames_dir),
                    "--wav",
                    str(wav_path),
                    "--meta",
                    str(meta_path),
                    "--device",
                    device_tip,
                    "--audio-shard",
                    "--shard",
                    str(audio_marker),
                ],
                env=_farm_env(0),
                stdout=open(scratch / "worker_audio.log", "w"),
                stderr=sp.STDOUT,
                text=True,
            )
        )
        _RUNTIME["_farm_log_fhs"].append(None)  # audio log fh owned by the file object
    for i in range(nshards):
        # Video worker i -> cards 1..7 (the CP parents there hard-exit moments
        # after the spawn; wait for each card to drain before handing it
        # over). Overflow workers (longer videos, nshards > 7) share card 0
        # with rank0 + the audio worker (~15 GiB more; tight but fits).
        card = (i + 1) if i < 7 else 0
        if device_tip == "hpu":
            _wait_card_free(card)
        procs.append(
            sp.Popen(
                [
                    sys.executable,
                    str(subproc_py),
                    "--blob",
                    str(blob_path),
                    "--workdir",
                    str(workdir),
                    "--frames-dir",
                    str(frames_dir),
                    "--wav",
                    str(wav_path),
                    "--meta",
                    str(meta_path),
                    "--device",
                    device_tip,
                    "--shard",
                    str(i),
                    "--nshards",
                    str(nshards),
                ],
                env=_farm_env(card),
                stdout=open(scratch / f"worker_{i}.log", "w"),
                stderr=sp.STDOUT,
                text=True,
            )
        )
    print(
        f"[h3] decode farm: {nshards} video workers + "
        f"{1 if audio_latents is not None else 0} audio worker spawned "
        f"(audio card 0; video cards 1..{min(nshards, 7)})",
        flush=True,
    )
    # Fire-and-forget: the export phase polls done markers (with timeout) and
    # reports failures; the parent process itself must NOT touch the device
    # again before exiting (post-drain wedge).
    _RUNTIME["_farm_procs"] = procs
    _RUNTIME["decode_sharded_done"] = True
    _RUNTIME["sharded_audio_rank"] = audio_marker
    block_state.videos = None
    self.set_block_state(state, block_state)
    return components, state


def _decode_sharded_h3(self, components, state, block_state):
    """H3_DECODE_SHARDED=1 block wrapper around _decode_sharded_core.

    Runs on EVERY CP rank inside _H3VideoDecodeStep (before the rank0-only
    skip): every rank decodes its chunks and writes its own frame range;
    handoff is file-based (no collectives -- post-drain HCCL broadcasts
    stall, v72). Frames land on disk from all ranks; videos/audio are left
    unset and the audio step + export phase handle the rest via the
    decode_sharded_done flag.
    """
    rank = _RUNTIME.get("rank", 0) or 0
    world = _RUNTIME.get("world_size", 1) or 1
    device = components.vae.device
    latents = block_state.latents
    if latents is None:
        raise RuntimeError("sharded decode: no latents in block state")
    ts = int(components.vae.tokens_chunk_size)
    token_drop = int(getattr(components.vae.config, "token_drop", 0) or 0)
    num_tokens = latents.shape[2] + token_drop
    n_chunks = (num_tokens + (-num_tokens) % ts) // ts - int(token_drop > 0)
    audio_rank = n_chunks % world
    owns = [i for i in range(n_chunks) if i % world == rank]
    if not owns and rank == audio_rank:
        # Chunkless rank (world > n_chunks): it IS the audio rank. Decode the
        # audio NOW -- concurrently with the clip decodes -- so the audio is
        # ready when the export phase looks for it (and the card shows
        # progress instead of idling). The audio step then stubs via
        # sharded_audio_done.
        _decode_sharded_audio(components, state, block_state, rank)
        _RUNTIME["sharded_audio_done"] = True
    frames_saved, clip_s, total_s = _decode_sharded_core(
        components.vae,
        latents,
        components.pixel_mean,
        components.pixel_std,
        _RUNTIME["frames_dir"],
        rank,
        world,
        device,
    )
    _RUNTIME["decode_sharded_done"] = True
    _RUNTIME["sharded_audio_rank"] = audio_rank
    block_state.videos = None
    self.set_block_state(state, block_state)
    print(
        f"[h3:r{rank}]] sharded decode: {frames_saved} frames written, "
        f"clip decode {clip_s:.1f}s, phase {total_s:.1f}s",
        flush=True,
    )
    return components, state


def _decode_via_subprocess(self, components, state, block_state):
    """H3_DECODE_SUBPROCESS=1 escape hatch (PLAN_CP.md): hand the VAE decode to
    a FRESH eager-mode process.

    The lazy-mode launch thread can stop accepting enqueues after the giant
    denoise graph drains (JoinPendingLaunchThread wedge, v50/51/52/58) -- the
    big post-decode D2H never completes, silently, with plenty of free memory.
    Small D2Hs and fresh processes are reliable, so this writes the final
    latents to disk (small pull, post-sync, post-DiT-free) and runs
    decode_subproc.py, which loads the VAEs itself, decodes in eager mode,
    writes frames + wav + a meta json, and exits. The parent keeps muxing."""
    import json as _json

    from PIL import Image

    workdir = _RUNTIME.get("workdir")
    frames_dir = _RUNTIME.get("frames_dir")
    wav_path = _RUNTIME.get("wav_path")
    if not (workdir and frames_dir and wav_path):
        raise RuntimeError(
            "H3_DECODE_SUBPROCESS=1 requires the runner context (workdir/out paths); "
            "use run_t2va.py, not a bare pipeline"
        )
    frames_dir = Path(frames_dir)
    wav_path = Path(wav_path)
    device_tip = "hpu" if block_state.latents.device.type == "hpu" else "cpu"
    blob_path = wav_path.parent / (wav_path.stem + "_decode_blob.pt")
    meta_path = wav_path.parent / (wav_path.stem + "_decode_meta.json")
    # The video step's BlockState carries latents only; audio latents live in
    # the pipeline state under the audio branch's name.
    audio_latents = getattr(block_state, "audio_latents", None)
    if audio_latents is None:
        audio_latents = state.get("audio_latents") if hasattr(state, "get") else None
    blob = {
        "latents": block_state.latents.detach().to("cpu", torch.float32),
        "audio_latents": (
            audio_latents.detach().to("cpu", torch.float32)
            if audio_latents is not None
            else None
        ),
        "pixel_mean": list(components.pixel_mean),
        "pixel_std": list(components.pixel_std),
    }
    torch.save(blob, blob_path)
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / "decode_subproc.py"),
        "--blob",
        str(blob_path),
        "--workdir",
        str(workdir),
        "--frames-dir",
        str(frames_dir),
        "--wav",
        str(wav_path),
        "--meta",
        str(meta_path),
        "--device",
        device_tip,
    ]
    env = dict(os.environ)
    env["PT_HPU_LAZY_MODE"] = "0"  # eager plugin; decode_subproc.py double-checks pre-import
    env.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "1" if device_tip == "hpu" else "0")
    if device_tip == "hpu":
        # The parent process's freed DiT pages return to ITS OWN caching
        # allocator pool, not to the device — a second process on the same
        # card sees it as nearly full (PT_DEVMEM 394 MB alloc failure in
        # v64's subprocess). Give the subprocess a card of its own: with CP
        # workers on cards 0..N-1, rank r's decoder takes card N+r (plenty
        # idle on an 8-card box); CP=1 takes card 1.
        _rank = _RUNTIME.get("rank", 0) or 0
        _ws = _RUNTIME.get("world_size", 1) or 0
        card = _pick_free_hpu_card(_ws + _rank, skip=_rank)
        env["HABANA_VISIBLE_DEVICES"] = str(card)
        env["HABANA_VISIBLE_MODULES"] = str(card)
        print(f"[h3] decode subprocess card: HABANA_VISIBLE_MODULES={card}", flush=True)
    print(f"[h3] decode subprocess: {device_tip} eager decode via {cmd[2]}", flush=True)
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"decode subprocess failed (rc={proc.returncode}):\n{proc.stdout[-1500:]}\n{proc.stderr[-1500:]}"
        )
    meta = _json.loads(meta_path.read_text())
    _RUNTIME["decode_meta"] = meta
    _RUNTIME["decode_subproc_done"] = True
    frames = [Image.open(p) for p in sorted(frames_dir.glob("frame_*.png"))]
    block_state.videos = [frames]  # postprocess_video shape: list per batch item
    block_state.sampling_rate = int(meta["sampling_rate"])
    state.set("sampling_rate", int(meta["sampling_rate"]))
    with suppress(OSError):
        blob_path.unlink()
    return block_state


def _write_rank_timings(out_path: Path, rank: int, timings: dict) -> Path:
    path = Path(out_path).parent / f"{Path(out_path).stem}_cp_timings_rank{rank}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"rank": rank, "phases": timings, "blocks": list(_BLOCK_TIMES_LAST or [])})
    )
    return path


def _cp_test_worker_steps(rank: int, world_size: int, info: dict, payload: dict) -> None:
    """H3_CP_TEST=1 worker body: no model, no HPU. Exercises the plumbing the
    real workers run through -- process-group init, barrier, plan accounting,
    rank-prefixed logging and the per-rank timings file."""
    import torch.distributed as dist

    seq = int(info["expected_packed_seq"])
    base = int(info.get("cp_seq_base", seq))
    pad = int(info.get("cp_seq_pad", 0))
    assert seq == base + pad, (seq, base, pad)
    assert seq % world_size == 0, (seq, world_size)
    assert pad == (-base) % world_size, (pad, base, world_size)
    dist.barrier()
    print(
        f"[h3] CP rank {rank}/{world_size}: process group ok "
        f"(backend={dist.get_backend()}); padded seq {seq} = {base} + {pad} pad",
        flush=True,
    )
    global _BLOCK_TIME_RECORDS, _BLOCK_TIMES_LAST
    _BLOCK_TIME_RECORDS = [("FakeStep", " i=0", round(0.01 * (rank + 1), 3))]
    time.sleep(0.02)
    _BLOCK_TIMES_LAST = list(_BLOCK_TIME_RECORDS)
    _BLOCK_TIME_RECORDS = None
    dist.barrier()
    _write_rank_timings(Path(payload["out_path"]), rank, {"smoke": 0.02})


def _cp_worker_entry(rank: int, payload: dict) -> None:
    """Spawn target. Runs in a FRESH interpreter: the per-rank env MUST be set
    before torch is imported anywhere in this process (Habana plugin pairing
    rules are read at import/device-init time, see launch_h3.sh)."""
    os.environ["HABANA_VISIBLE_DEVICES"] = str(rank)
    # v82: pin the physical card at the Synapse level too. HABANA_VISIBLE_DEVICES
    # alone is ignored by device acquire (first-free scan); the pin makes
    # rank->card deterministic instead of spawn-race dependent.
    os.environ["HABANA_VISIBLE_MODULES"] = str(rank)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(payload["cp"])
    os.environ["MASTER_ADDR"] = payload["master_addr"]
    os.environ["MASTER_PORT"] = str(payload["master_port"])
    # Per-rank recipe cache (PLAN_CP.md phase 2 item): the recipe cache config
    # is per-process, and N workers writing one directory race on the same
    # recipes. Under ulysses the per-rank shapes are identical, but one dir per
    # rank is cheap insurance either way.
    recipes = os.environ.get("H3_RECIPES_DIR")
    if recipes:
        base = Path(recipes)
        rank_dir = base.with_name(base.name + f"_rank{rank}")
        try:
            rank_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        os.environ["H3_RECIPES_DIR"] = str(rank_dir)
        if "PT_HPU_RECIPE_CACHE_CONFIG" in os.environ:
            parts = os.environ["PT_HPU_RECIPE_CACHE_CONFIG"].split(",")
            parts[0] = str(rank_dir)
            os.environ["PT_HPU_RECIPE_CACHE_CONFIG"] = ",".join(parts)
    sys.stdout = _RankTaggedStdout(sys.stdout, rank)
    try:
        _cp_worker_main(rank, payload)
    except SystemExit:
        raise
    except BaseException:
        import traceback

        traceback.print_exc()
        sys.exit(1)


def _cp_worker_main(rank: int, payload: dict) -> None:
    args = types.SimpleNamespace(**payload["args"])
    info = payload["info"]
    world = int(payload["cp"])
    _RUNTIME.update(rank=rank, world_size=world)
    _ensure_torch()
    hpu_ok = _hpu_ok() and (args.allow_hpu or _env_bool("H3_ALLOW_HPU", False))
    if _env_bool(_CP_TEST_ENV, False):
        backend = _cp_init_process_group(hpu_ok, world)
        assert backend == "gloo", f"test harness must stay on gloo, got {backend}"
        _cp_test_worker_steps(rank, world, info, payload)
        import torch.distributed as dist

        dist.barrier()
        dist.destroy_process_group()
        return
    # Full CP run: process group first (hccl under an approved HPU run, gloo on
    # the CPU harness path), then the ordinary pipeline with rank/world set.
    _RUNTIME["backend"] = _cp_init_process_group(hpu_ok, world)
    device_map = parse_device_map(
        args.device_map or _env_flag("H3_DEVICE_MAP", DEFAULT_DEVICE_MAP)
    )
    timings = run_pipeline(
        args,
        info,
        rank=rank,
        world_size=world,
        device_map=device_map,
        out_path=Path(payload["out_path"]),
        frames_dir=Path(payload["frames_dir"]),
        wav_path=Path(payload["wav_path"]),
    )
    _write_rank_timings(Path(payload["out_path"]), rank, timings)
    if _RUNTIME.get("backend") == "hccl":
        # Hard exit WITHOUT HCCL teardown and WITHOUT interpreter shutdown:
        # destroy_process_group blocks until every rank joins, so ranks
        # 1..N-1 hang holding ~70 GiB while rank0 decodes (v69: card picker
        # then found nothing free and timed out). Rank0 has its own wedge:
        # the interpreter that ran the giant lazy graph hangs in atexit
        # teardown (v70: 237 idle threads, empty py-spy stack, artifact
        # already written). By this point every artifact + timing file is on
        # disk, so os._exit(0) is safe for all ranks.
        print(f"[h3:r{rank}]] hard exit (skip HCCL/interpreter teardown)", flush=True)
        sys.stdout.flush()
        os._exit(0)


def _server_claim_request(rank, world_size, server_dir, last_seq, apply_params=None):
    """Server-mode request claim (v100). Blocks until the next request or
    shutdown.

    Protocol (files only, parent's HTTP front-end writes the spool):
    - rank0: polls spool/req_*.json in seq order; claims by atomic rename to
      .claimed; atomically overwrites serve_sync/current.json with the request
      dict (the broadcast to ranks 1..N-1) AFTER the previous request's
      req.done marker exists (rank0 exports before claiming next, so
      current.json appearing implies prev done) — ranks pace off this, same
      discipline as the --serve serve-sync (v84).
    - ranks 1..N-1: poll serve_sync/current.json until its seq differs from
      the last seen one.
    - SHUTDOWN: server_dir/SHUTDOWN file → rank0 writes serve_sync/SHUTDOWN
      and every rank returns None (clean loop exit).
    Returns the request dict (with seq/id/prompt/steps/seed) or None.
    """
    root = Path(server_dir)
    spool = root / "spool"
    sync = root / "serve_sync"
    jobs = root / "jobs"
    sync.mkdir(parents=True, exist_ok=True)

    def _hb(device_wait=False):
        # Idle-card heartbeat during host-side polling (v70: idle cards are
        # safe; this keeps the watchdog fed on ranks that sit in long waits).
        if device_wait and _RUNTIME.get("backend") == "hccl":
            try:
                _shard_heartbeat(torch.device("hpu"))
            except Exception:
                pass

    if rank in (None, 0):
        if (root / "SHUTDOWN").exists():
            (sync / "SHUTDOWN").write_text("1")
            return None
        n_poll = 0
        while True:
            for cand in sorted(spool.glob("req_*.json")):
                claimed = cand.with_suffix(".claimed")
                try:
                    cand.rename(claimed)
                except FileNotFoundError:
                    continue  # another claimer got it (never happens: single claimer)
                try:
                    req = json.loads(claimed.read_text())
                except Exception as e:
                    _bad_id = claimed.stem.replace("req_", "").split("_", 1)[-1]
                    (jobs / f"{_bad_id}.json").write_text(
                        json.dumps({"id": _bad_id, "status": "failed", "error": f"bad request json: {e}"})
                    )
                    continue
                if (spool / f"{req['id']}.cancel").exists():
                    (jobs / f"{req['id']}.json").write_text(
                        json.dumps({"id": req["id"], "status": "cancelled"})
                    )
                    print(f"[h3] server: {req['id']} cancelled before run", flush=True)
                    continue
                # v103b: validate/apply per-request params BEFORE publishing —
                # a gated/unsupported shape fails THIS job only; nothing is
                # published, so ranks 1..N-1 never see the seq (no desync).
                if apply_params is not None:
                    try:
                        apply_params(req)
                    except Exception as _perr:
                        (jobs / f"{req['id']}.json").write_text(
                            json.dumps({"id": req["id"], "status": "failed", "error": str(_perr)})
                        )
                        print(f"[h3] server: {req['id']} rejected: {_perr}", flush=True)
                        continue
                # prev req.done must exist (rank0 finished export+done) — true
                # by construction; assert defensively.
                tmp = sync / ".current.tmp"
                tmp.write_text(json.dumps(req))
                tmp.replace(sync / "current.json")
                (jobs / f"{req['id']}.json").write_text(
                    json.dumps({**req, "status": "running"})
                )
                print(f"[h3] server: claimed {req['id']} (seq {req['seq']})", flush=True)
                return "run", req
            if (root / "SHUTDOWN").exists():
                (sync / "SHUTDOWN").write_text("1")
                return "shutdown", None
            n_poll += 1
            _hb(device_wait=True)
            time.sleep(1.0)
    else:
        sync_shutdown = sync / "SHUTDOWN"
        n_poll = 0
        while True:
            if sync_shutdown.exists():
                return "shutdown", None
            cur = sync / "current.json"
            if cur.exists():
                try:
                    req = json.loads(cur.read_text())
                    if int(req.get("seq", -1)) != last_seq:
                        print(f"[h3:r{rank}]] server: received claim {req['id']} (seq {req['seq']})", flush=True)
                        try:
                            if apply_params is not None:
                                apply_params(req)
                            return "run", req
                        except Exception as _perr:
                            # rank0 validated before publish, so this should
                            # never fire; guard anyway and wait for the next
                            # seq (do NOT reprocess this one).
                            print(f"[h3:r{rank}]] server: param apply failed: {_perr}", flush=True)
                            return "skip", req
                except Exception:
                    pass
            n_poll += 1
            _hb(device_wait=True)
            time.sleep(1.0)


def _server_finish_job(server_dir, job_id, out_path, timings, status="completed", error=None):
    """Rank0-side job status update for the HTTP front-end."""
    jobs = Path(server_dir) / "jobs"
    jobs.mkdir(parents=True, exist_ok=True)
    frame_count = 0
    meta_path = out_path.parent / (out_path.stem + "_decode_meta.json")
    try:
        if meta_path.exists():
            frame_count = int(json.loads(meta_path.read_text()).get("num_frames") or 0)
    except Exception:
        frame_count = 0
    if not frame_count:
        # Fused/in-proc decode writes pngs directly (no meta file); count them.
        frames_dir = out_path.parent / (out_path.stem + "_frames")
        try:
            frame_count = len(list(frames_dir.glob("frame_*.png")))
        except Exception:
            frame_count = 0
    blob = {
        "id": job_id,
        "status": status,
        "artifact": str(out_path) if out_path else None,
        "frame_count": frame_count,
        "timings": timings or {},
        "error": error,
    }
    tmp = jobs / f".{job_id}.tmp"
    tmp.write_text(json.dumps(blob))
    tmp.replace(jobs / f"{job_id}.json")


def _cp_parent_main(args, info, cp: int, out_path: Path, frames_dir: Path, wav_path: Path) -> int:
    """CP parent: spawn N workers (multiprocessing spawn -- fork is unsafe with
    the Habana plugin threads), wait, merge the per-rank timing files, and
    report. Artifacts (frames/wav/mp4) land via rank0's shared out paths; the
    parent itself never imports torch."""
    import multiprocessing as mp

    payload = {
        "args": dict(vars(args)),
        "info": info,
        "cp": cp,
        "master_addr": "127.0.0.1",
        "master_port": _pick_free_port(),
        "out_path": str(out_path),
        "frames_dir": str(frames_dir),
        "wav_path": str(wav_path),
    }
    ctx = mp.get_context("spawn")
    procs = []
    for rank in range(cp):
        proc = ctx.Process(
            target=_cp_worker_entry, args=(rank, payload), name=f"h3-cp-rank{rank}"
        )
        proc.start()
        procs.append(proc)
    for proc in procs:
        proc.join()
    codes = {proc.name: proc.exitcode for proc in procs}
    _cp_merge_timings(out_path, cp)
    failed = [name for name, code in codes.items() if code not in (0, None)]
    if failed:
        print(f"[h3] CP FAILED: {failed} (exit codes {codes})", flush=True)
        # v100: mark any in-flight server jobs failed so HTTP clients unblock.
        if _env_bool("H3_SERVER_MODE", False):
            _sdir = os.environ.get("H3_SERVER_DIR", "")
            if _sdir:
                for jf in (Path(_sdir) / "jobs").glob("*.json"):
                    try:
                        _job = json.loads(jf.read_text())
                    except Exception:
                        continue
                    if _job.get("status") in ("queued", "running"):
                        _job["status"] = "failed"
                        _job["error"] = f"worker pool died: {failed}"
                        jf.write_text(json.dumps(_job))
        return 1
    print(f"[h3] CP run complete: {cp} ranks; artifacts via rank0 -> {out_path}", flush=True)
    return 0


def _cp_merge_timings(out_path: Path, cp: int) -> None:
    """Merge the per-rank timing JSONs into one console table."""
    blobs = []
    for rank in range(cp):
        path = Path(out_path).parent / f"{Path(out_path).stem}_cp_timings_rank{rank}.json"
        try:
            blobs.append(json.loads(path.read_text()))
        except (OSError, json.JSONDecodeError):
            pass
    if not blobs:
        print("[h3] CP timing merge: no per-rank timing files found", flush=True)
        return
    phase_names = []
    for blob in blobs:
        for name in blob.get("phases", {}):
            if name not in phase_names:
                phase_names.append(name)
    print(f"[h3] CP timing merge (world={len(blobs)}):", flush=True)
    print("  phase".ljust(16) + "".join(f"r{blob['rank']}".rjust(10) for blob in blobs) + "   (max)")
    for name in phase_names:
        vals = [blob["phases"].get(name) for blob in blobs]
        row = ("  " + name).ljust(16) + "".join(
            ("-" if v is None else str(v)).rjust(10) for v in vals
        )
        nums = [v for v in vals if isinstance(v, (int, float))]
        if nums:
            row += f"   {max(nums):.2f}"
        print(row)
    agg = {}
    for blob in blobs:
        per = {}
        for name, _extra, dt in blob.get("blocks", []):
            per[name] = per.get(name, 0.0) + float(dt)
        agg[blob["rank"]] = per
    names = sorted({n for per in agg.values() for n in per})
    if names:
        print("[h3] CP block totals (summed per class, generate run; host-side enqueue cost):", flush=True)
        for name in names:
            row = ("  " + name).ljust(44) + "".join(
                f"{agg[blob['rank']].get(name, 0.0):9.2f}" for blob in blobs
            )
            print(row)


# ---------------------------------------------------------------------------
# Output export
# ---------------------------------------------------------------------------


def save_frames(videos, out_dir: Path) -> list[Path]:
    """`videos` is the post-`postprocess_video` output. For `output_type="pil"`
    that is a flat list of PIL frames (list per batch item)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = videos[0] if isinstance(videos, (list, tuple)) else videos
    paths = []
    if isinstance(frames, (list, tuple)):
        for i, frame in enumerate(frames):
            path = out_dir / f"frame_{i:06d}.png"
            frame.save(path)
            paths.append(path)
        return paths

    import numpy as np
    from PIL import Image

    array = (
        frames
        if isinstance(frames, np.ndarray)
        else frames.detach().float().cpu().numpy()
    )
    array = np.clip(array * 255.0, 0, 255).astype("uint8")
    if array.ndim != 4:
        raise ValueError(f"Unsupported video tensor shape {array.shape}")
    if (
        array.shape[0] in (1, 3) and array.shape[-1] != 3
    ):  # (C, T, H, W) -> (T, H, W, C)
        array = array.transpose(1, 2, 3, 0)
    for i, frame in enumerate(array):
        path = out_dir / f"frame_{i:06d}.png"
        Image.fromarray(
            frame if frame.shape[-1] == 3 else frame.transpose(2, 0, 1)
        ).save(path)
        paths.append(path)
    return paths


def save_audio_wav(audio, sampling_rate: int, out_path: Path) -> Path:
    """`audio`: `(1, 2, num_samples)` float -> 16-bit PCM stereo WAV."""
    import numpy as np

    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = (
        audio if isinstance(audio, np.ndarray) else audio.detach().float().cpu().numpy()
    )
    data = np.clip(data.reshape(-1, data.shape[-1]), -1.0, 1.0)  # (2, S)
    pcm = (data.T.reshape(-1) * 32767.0).astype("<i2").tobytes()
    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(data.shape[0])
        wf.setsampwidth(2)
        wf.setframerate(int(sampling_rate))
        wf.writeframes(pcm)
    return out_path


def mux_mp4(frames_dir: Path, wav_path: Path, out_path: Path, fps: int = 24) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        print("[h3] WARNING: ffmpeg not found; leaving raw frames + wav unmuxed")
        return False
    cmd = [
        ffmpeg,
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frames_dir / "frame_%06d.png"),
        "-i",
        str(wav_path),
        "-filter_complex",
        "[0:v]format=yuv420p[v];[1:a]aresample=48000[a]",
        "-map",
        "[v]",
        "-map",
        "[a]",
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        "17",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-movflags",
        "+faststart",
        "-shortest",
        str(out_path),
    ]
    print("[h3] mux:", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"[h3] WARNING: ffmpeg failed:\n{proc.stderr[-2000:]}")
        return False
    return True


# ---------------------------------------------------------------------------
# CLI + main
# ---------------------------------------------------------------------------


def parse_device_map(raw: str) -> dict:
    device_map = {}
    for part in raw.split(","):
        if not part.strip():
            continue
        key, _, value = part.partition("=")
        key, value = key.strip(), value.strip().lower()
        if key not in ("te", "dit", "vae", "audio") or value not in ("cpu", "hpu", "stream"):
            raise ValueError(
                f"Bad H3_DEVICE_MAP entry {part!r} (expected te/dit/vae/audio = cpu|hpu)"
            )
        device_map[key] = value
    return device_map or {"te": "cpu", "dit": "cpu", "vae": "cpu", "audio": "cpu"}


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Run MiniMax-H3 t2va/fl2va (HPU port runner)."
    )
    p.add_argument("--prompt", required="--server" not in (argv or sys.argv[1:]))  # not required in --server mode (prompt is per-request)
    p.add_argument("--repo", default=REPO_DEFAULT)
    p.add_argument("--workflow", choices=["t2va", "fl2va"], default="t2va")
    p.add_argument(
        "--keyframe",
        action="append",
        default=[],
        metavar="IMG",
        help="fl2va keyframe: first is the video start anchor, the second the end anchor",
    )
    p.add_argument(
        "--duration",
        type=int,
        default=None,
        help="seconds; snapped to the nearest durations table row",
    )
    p.add_argument(
        "--bucket",
        default=None,
        metavar="NAME",
        help="named bring-up bucket (buckets.json `buckets` list); overrides --duration/--resolution",
    )
    p.add_argument(
        "--resolution",
        default=None,
        help="WxH or bucket id (r480b, r544, ...); default 864x480",
    )
    p.add_argument(
        "--steps",
        type=int,
        default=None,
        help="denoise steps; default = buckets.canonical_steps",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--out", default=None, help="target mp4; default /root/src/h3/out/<slug>.mp4"
    )
    p.add_argument(
        "--text-budget",
        type=int,
        default=2048, #aaron changed from None
        help="pad the text run to N rows (0 = exact)",
    )
    p.add_argument(
        "--no-graphs",
        action="store_true",
        help="eager lazy mode with post-call mark_step",
    )
    p.add_argument(
        "--audio-vae-device",
        choices=["hpu", "cpu"],
        default=None,
        help="kill-switch: 'cpu' pins the audio VAE off the HPU (fp32 strictly)",
    )
    p.add_argument(
        "--te-threads",
        type=int,
        default=None,
        help="torch.set_num_threads for the text encoder",
    )
    p.add_argument(
        "--embed-cache-dir",
        default=DEFAULT_EMBED_CACHE_DIR,
        help="prompt-embed cache dir; pass '' to disable",
    )
    p.add_argument(
        "--device-map",
        default=None,
        help="H3_DEVICE_MAP override, e.g. te=stream,dit=hpu,vae=hpu,audio=hpu (te: cpu|hpu|stream)",
    )
    p.add_argument(
        "--allow-hpu",
        action="store_true",
        help="permit HPU placement (launch_h3.sh sets H3_ALLOW_HPU=1 after the user approves a card)",
    )
    p.add_argument(
        "--cp",
        type=int,
        choices=[1, 2, 4, 8],
        default=1,
        help="context parallelism degree (PLAN_CP.md); 1 keeps the single-process "
        "behavior bit-identical. N>1 spawns N workers (spawn context, one card each) "
        "running Ulysses sequence parallelism via diffusers' _cp_plan",
    )
    p.add_argument(
        "--warmup",
        type=int,
        default=0,
        metavar="N",
        help="run an N-step warm pass first (recipe compile), then the real run",
    )
    p.add_argument("--serve", type=int, default=1, help="weight-resident serving probe: run N back-to-back requests in ONE process (DiT stays resident; decode via fresh subprocess each request). CP=1 only.")
    p.add_argument("--server", action="store_true", help="stable server mode (v100): sdcpp-compatible HTTP API in the parent process; requests flow to the resident-weight worker pool via a file spool. Duration/resolution fixed server-wide; prompt/steps/seed per request.")
    p.add_argument("--port", type=int, default=8021, help="HTTP port for --server (default 8021; 8000 is llama.cpp on this box)")
    p.add_argument(
        "--report-only",
        action="store_true",
        help="resolve the plan and exit without loading torch",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    buckets = load_buckets()

    if args.workflow == "fl2va" and not args.keyframe:
        raise SystemExit("--workflow fl2va requires at least one --keyframe")
    if len(args.keyframe) > 2:
        raise SystemExit("fl2va supports at most 2 keyframes (first + last)")

    bucket_steps = None
    if args.bucket:
        named = next(
            (b for b in buckets.get("buckets", []) if b.get("name") == args.bucket),
            None,
        )
        if named is None:
            names = [b.get("name") for b in buckets.get("buckets", [])]
            raise SystemExit(f"Unknown --bucket {args.bucket!r}; available: {names}")
        args.resolution = f"{named['width']}x{named['height']}"
        args.duration = int(named["seconds"])
        bucket_steps = int(named.get("num_inference_steps", 0)) or None
    args.resolution = args.resolution or "864x480"
    args.duration = args.duration or 5

    steps = int(args.steps) if args.steps is not None else bucket_steps
    if steps is None:
        steps = int(
            buckets.get("defaults", {}).get(
                "num_inference_steps", buckets.get("canonical_steps", 25)
            )
        )
    budget = (
        int(args.text_budget)
        if args.text_budget is not None
        else int(buckets.get("text_budget", 512))
    )

    res_name, width, height = resolve_resolution(buckets, args.resolution)
    frames, t_lat = resolve_duration(buckets, args.duration)
    rows_per_frame = int(buckets["resolutions"][res_name]["rows_per_frame"])
    dur_spec = next(v for v in buckets["durations"].values() if v["frames"] == frames)
    n_audio = int(dur_spec["A"])
    fl2va_extra = (2 * rows_per_frame + 16) if args.workflow == "fl2va" else 0
    expected_seq = budget + t_lat * rows_per_frame + 2 * n_audio + fl2va_extra

    device_map = parse_device_map(
        args.device_map or _env_flag("H3_DEVICE_MAP", DEFAULT_DEVICE_MAP)
    )

    # Context parallelism (PLAN_CP.md phase 1 item 2): the packed sequence is
    # equipartitioned across the CP mesh, so pad the TEXT budget upward to the
    # smallest value making seq % N == 0. apply_text_budget already pads
    # zero-embed rows up to the budget; the divisibility pad rides that exact
    # mechanism (zero-embed rows flow through layout/rope/scatter untouched).
    cp = int(args.cp)
    cp_pad = 0
    cp_seq_base = expected_seq
    if cp > 1:
        cp_pad = (-expected_seq) % cp
        budget += cp_pad
        expected_seq += cp_pad

    info = {
        "workflow": args.workflow,
        "bucket_res": res_name,
        "canvas": [width, height],
        "duration_frames": frames,
        "T_lat": t_lat,
        "audio_latents": n_audio,
        "rows_per_frame": rows_per_frame,
        "fl2va_extra_rows": fl2va_extra,
        "text_budget": budget,
        "expected_packed_seq": expected_seq,
        "steps": steps,
        "seed": args.seed,
        "device_map": device_map,
    }
    # CP keys appear ONLY when CP is active: the CP=1 plan print (and the slug
    # it feeds) stays byte-identical to the pre-CP runner.
    if cp > 1:
        info["cp"] = cp
        info["cp_seq_base"] = cp_seq_base
        info["cp_seq_pad"] = cp_pad
        print(f"[h3] CP: seq {cp_seq_base} padded by {cp_pad} text rows to {expected_seq} "
              f"(divisible by {cp})")
    print("[h3] plan:", json.dumps(info, indent=2))
    if args.report_only:
        return 0

    # Artifact paths are computed in the parent so the CP payload, the per-rank
    # timing files and rank0's export targets all agree.
    slug = "h3_" + hashlib.sha256(json.dumps(info, sort_keys=True).encode()).hexdigest()[:10]
    out_path = Path(args.out or f"/root/src/h3/out/{args.workflow}_{slug}.mp4")
    frames_dir = out_path.parent / (out_path.stem + "_frames")
    wav_path = out_path.parent / (out_path.stem + ".wav")

    if getattr(args, "server", False):
        # v100: server mode — the parent's HTTP front-end and the workers
        # communicate through a file spool under <out dir>/.server.
        _sdir = out_path.parent / ".server"
        (_sdir / "spool").mkdir(parents=True, exist_ok=True)
        (_sdir / "jobs").mkdir(parents=True, exist_ok=True)
        # v117 boot hygiene: spool/jobs/serve_sync state PERSISTS across boots
        # (the out dir is never wiped). Stale state caused the v115 desync:
        # ranks 1..N-1 initialized last_seq=0 and immediately adopted the
        # PREVIOUS boot's current.json (seq 6) while rank0's claim loop
        # scanned the spool and claimed seq 9 — a one-job offset. Identical
        # plans + seed made every collective shape-match and the noise digest
        # pass spuriously; both outputs were cross-job chimeras.
        _syncdir = _sdir / "serve_sync"
        _syncdir.mkdir(parents=True, exist_ok=True)
        # (a) a leftover SHUTDOWN marker would make the new pool exit instantly
        (_sdir / "SHUTDOWN").unlink(missing_ok=True)
        (_syncdir / "SHUTDOWN").unlink(missing_ok=True)
        # (b) the broadcast file must never survive a boot: ranks pace off it,
        # so it must not exist until rank0 publishes THIS boot's first claim
        (_syncdir / "current.json").unlink(missing_ok=True)
        for _stale in _syncdir.glob("req*.done"):
            _stale.unlink(missing_ok=True)
        # (b2) the serve-pacing markers live in <out>/.serve_sync (out_path
        # parent, NOT the .server dir) and ALSO persist across boots — a stale
        # req0.done lets ranks 1..N-1 skip the export-pacing wait on this
        # boot's request #2 and race rank0's export collectives.
        _servesync = out_path.parent / ".serve_sync"
        if _servesync.is_dir():
            for _stale in _servesync.glob("req*.done"):
                _stale.unlink(missing_ok=True)
        # (c) jobs still marked "running" belong to a dead pool
        for _jr in sorted((_sdir / "jobs").glob("job_*.json")):
            try:
                _jd = json.loads(_jr.read_text())
            except Exception:
                continue
            if _jd.get("status") == "running":
                _jd["status"] = "failed"
                _jd["error"] = "stale: worker pool exited before completion (previous boot)"
                _jr.write_text(json.dumps(_jd))
                print(f"[h3] server: stale running job {_jd.get('id')} marked failed at boot", flush=True)
        os.environ["H3_SERVER_MODE"] = "1"
        os.environ["H3_SERVER_DIR"] = str(_sdir)
        import h3_server as _srv
        import threading as _threading
        _t = _threading.Thread(
            target=_srv.serve, args=(str(_sdir), int(args.port)), daemon=True
        )
        _t.start()
        (_sdir / "server.json").write_text(
            json.dumps(
                {
                    "video_frames": int(info.get("duration_frames") or 0),
                    "fps": 24,
                    "width": int((info.get("canvas") or [0, 0])[0]),
                    "height": int((info.get("canvas") or [0, 0])[1]),
                    "steps": int(info.get("steps") or 4),
                }
            )
        )
        print(f"[h3] server: HTTP API on :{args.port} (spool {_sdir})", flush=True)

    if cp > 1:
        return _cp_parent_main(args, info, cp, out_path, frames_dir, wav_path)

    _ensure_torch()  # the engine import; launcher guarantees LD_LIBRARY_PATH has libpython3.12
    run_pipeline(
        args, info, out_path=out_path, frames_dir=frames_dir, wav_path=wav_path
    )
    return 0


def run_pipeline(
    args,
    info,
    rank=None,
    world_size=1,
    out_path=None,
    frames_dir=None,
    wav_path=None,
    device_map: dict | None = None,
) -> dict:
    """One full t2va/fl2va run; returns the phase timing dict.

    rank=None / world_size=1 is the legacy single-process path (CP=1) with
    unchanged behavior. rank >= 0 is a CP worker of world_size N: the process
    group is initialized by the caller (_cp_worker_main), the device map pins
    dit/vae/audio onto one device set, enable_parallelism() activates the
    model's _cp_plan, and only rank0 exports artifacts.
    """
    _ensure_torch()
    import torch.distributed as dist

    if world_size > 1:
        if not dist.is_initialized():
            raise RuntimeError(
                "CP workers must initialize the process group first "
                "(_cp_init_process_group in _cp_worker_main)"
            )
        if dist.get_world_size() != world_size:
            raise RuntimeError(
                f"CP world mismatch: process group world {dist.get_world_size()} "
                f"!= runner world {world_size}"
            )
        if rank is None:
            rank = dist.get_rank()
        _RUNTIME.update(rank=rank, world_size=world_size)

    # LAZY MODE POLICY (see launch_h3.sh comment block): PT_HPU_LAZY_MODE and
    # PT_HPU_AUTOLOAD must be set at process start (before the torch import) or
    # the plugin/lib pair mismatches at device init. The launcher HPU branch
    # exports the pair; DO NOT set them here post-import.

    hpu_ok = _hpu_ok() and (args.allow_hpu or _env_bool("H3_ALLOW_HPU", False))
    cp_gloo_test = world_size > 1 and _env_bool(_CP_TEST_ENV, False)
    if device_map is None:
        device_map = parse_device_map(
            args.device_map or _env_flag("H3_DEVICE_MAP", DEFAULT_DEVICE_MAP)
        )
    if not hpu_ok and any(v == "hpu" for v in device_map.values()):
        if _env_bool("H3_ALLOW_CPU_FALLBACK", False) or cp_gloo_test:
            print(
                "[h3] WARNING: HPU unavailable or not approved; collapsing to CPU "
                "(H3_ALLOW_CPU_FALLBACK=1 -- this will run the full 62 GiB DiT on "
                "host CPU for hours)."
                if not cp_gloo_test
                else "[h3] CP test harness: HPU unavailable; collapsing to CPU on the gloo path."
            )
            device_map = dict.fromkeys(device_map, "cpu")
        else:
            raise RuntimeError(
                "HPU requested but unavailable/not approved (H3_ALLOW_HPU=1, Habana "
                "plugin autoload); refusing the silent multi-hour CPU fallback. Set "
                "H3_ALLOW_CPU_FALLBACK=1 to allow it anyway."
            )

    if args.te_threads:
        torch.set_num_threads(int(args.te_threads))

    exec_device = torch.device(
        "hpu" if hpu_ok and any(v == "hpu" for v in device_map.values()) else "cpu"
    )
    audio_tip = (
        args.audio_vae_device or _env_flag("H3_AUDIO_VAE_DEVICE", "hpu")
    ).lower()
    if device_map.get("audio") == "hpu" or args.audio_vae_device is not None:
        # Kill-switch semantics: an explicit cpu request always wins; an hpu tip
        # collapses to cpu when HPU is unavailable.
        device_map["audio"] = audio_tip if (audio_tip == "cpu" or hpu_ok) else "cpu"

    repo = Path(args.repo).resolve()
    workdir = build_workdir(repo=repo, workflow=args.workflow)
    print(f"[h3] workdir: {workdir}")

    # --- patches BEFORE from_pretrained (design 3.5) ----------------------
    from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3ModularPipeline

    patch_execution_device(MiniMaxH3ModularPipeline)
    patch_decode_steps()
    patch_inproc_decode_prewarm()
    patch_text_encoder_device()
    patch_set_timesteps_step()
    patch_prepare_latents_cpu(cp_world_size=world_size)
    patch_timestep_embedding_host_basis()
    patch_rope_forward_per_axis()
    backend_name, _backend_fn = _register_attention_backend()

    # --- components -------------------------------------------------------
    # NOTE: build_workdir() writes `_blocks_class_name: MiniMaxH3Blocks` into the
    # remapped model_index.json; ModularPipeline.from_pretrained resolves the
    # blocks class from it. Do NOT pass `blocks=` here: from_pretrained forwards
    # unknown kwargs into the pipeline constructor alongside its own keyword,
    # raising TypeError: got multiple values for 'blocks'.
    pipe = MiniMaxH3ModularPipeline.from_pretrained(
        workdir, workflow=args.workflow, local_files_only=True
    )
    pipe.load_components(
        ["tokenizer", "processor", "scheduler", "audio_scheduler"],
        local_files_only=True,
    )
    pipe.load_components(
        ["text_encoder"],
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        local_files_only=True,
    )
    pipe.load_components(
        ["transformer"],
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        local_files_only=True,
    )
    # Video + audio VAE stay fp32 (`AutoencoderKLMiniMaxH3*`); no dtype kwarg.
    pipe.load_components(
        ["vae"], dtype=torch.float32, low_cpu_mem_usage=True, local_files_only=True
    )
    pipe.load_components(
        ["audio_vae"],
        dtype=torch.float32,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )

    # Video VAE standalone-repo fallback (design 3.1 + pre-GPU checklist #4):
    # if the load failed because the checkpoint is `minimax_h3_video_vae`
    # trust-remote-code format, load the repo-local class and register it as
    # `components.vae`; it must expose `.config.latents_mean/std` + `.decode()`.
    if getattr(pipe, "vae", None) is None:
        video_vae_dir = repo / "FL2VA" / "video_vae"
        if not (video_vae_dir / "model.safetensors").exists() and not (
            (repo / "vae").is_dir() and any((repo / "vae").glob("*safetensors"))
        ):
            raise RuntimeError(
                f"Video VAE weights are missing at {video_vae_dir}/model.safetensors and {repo}/vae; "
                "the run cannot proceed. Retry once the download completes."
            )
        from diffusers import AutoencoderKLMiniMaxH3

        try:
            vae = AutoencoderKLMiniMaxH3.from_pretrained(
                video_vae_dir,
                dtype=torch.float32,
                low_cpu_mem_usage=True,
                local_files_only=True,
            )
        except Exception:
            vae = AutoencoderKLMiniMaxH3.from_pretrained(
                video_vae_dir,
                dtype=torch.float32,
                low_cpu_mem_usage=True,
                local_files_only=True,
                trust_remote_code=True,
            )
        pipe.update_components(vae=vae)
        print("[h3] video VAE loaded via standalone-repo fallback into components.vae")
    if getattr(pipe, "audio_vae", None) is None:
        raise RuntimeError(
            f"audio_vae failed to load from {repo / 'FL2VA' / 'audio_vae'}"
        )

    # Full-frame video-VAE decode (speed option): stock decoding tiles the
    # spatial grid into ceil-dim 256x256 px tiles with blended overlaps -- for
    # the 864x480 bucket that is 15 tiles x 2 latent-clip shapes plus blend
    # passes, tripling the host-side graph-compilation work of the 36-layer ViT
    # decoder for no visual benefit at 480p. With H3_VAE_FULL_FRAME=1 the two
    # full-frame clip shapes (16x16 latents x {5,2} frames) compile once each
    # and run without stitching overhead. 96 GiB comfortably holds the
    # activations of one 864x480 chunk now that the DiT workspace is free.
    if _env_flag("H3_VAE_FULL_FRAME", "0") == "1":
        if getattr(pipe.vae, "use_tiling", True):
            pipe.vae.use_tiling = False
            print("[h3] video VAE full-frame decode enabled (tiling disabled)")

    # --- device map -------------------------------------------------------
    # Components are loaded on CPU by default; only hpu tips trigger a device
    # move (`.to(device)` changes no dtypes, so `_keep_in_fp32_modules` holds).
    tip_to_component = {
        "te": "text_encoder",
        "dit": "transformer",
        "vae": "vae",
        "audio": "audio_vae",
    }
    # v109: boot-chain serialization. Device acquire first-free-scans ALL
    # cards; when every CP rank scans in the same instant, two ranks can
    # target the same "first free" card and the loser dies with synStatus=8
    # at placement (v109: 3+ ranks died at boot). So: rank r waits for rank
    # r-1's acquired marker before its FIRST placement, then publishes its
    # own — boot becomes sequential, exactly one rank scanning at a time.
    # Weight deserialization (the slow part) is NOT serialized: it happened
    # during from_pretrained; only the acquire is chained.
    if world_size > 1 and exec_device.type == "hpu":
        _boot_dir = Path(_RUNTIME.get("out_path", "/tmp")).parent / ".bootchain"
        _boot_dir.mkdir(parents=True, exist_ok=True)
        _my_prev = _boot_dir / f"r{rank - 1}.acquired" if rank and rank > 0 else None
        _t_boot = time.perf_counter()
        while _my_prev is not None and not _my_prev.exists():
            if time.perf_counter() - _t_boot > 900.0:
                raise RuntimeError(f"boot chain stalled: {_my_prev} never appeared")
            time.sleep(1.0)

    def _place_hpu(component_name, component):
        # v113: max 2 attempts. Each synStatus=8 failure appears to LEAK a
        # partial pool (v113: 3 ranks x 6 retries -> five 74 GiB ghost pools
        # hl-smi showed with no owning process, poisoning every later scan).
        # Fail fast + loud beats retry-and-leak; the driver cycle is the
        # remedy for residue, not retries.
        for _attempt in range(2):
            try:
                component.to(exec_device)
                print(f"[h3] placed {component_name} on {exec_device}", flush=True)
                return
            except RuntimeError as _e:
                if "synStatus=8" not in str(_e) or _attempt == 1:
                    raise
                _wait = 10.0 * (_attempt + 1)
                print(
                    f"[h3] {component_name} placement hit synStatus=8 (attempt "
                    f"{_attempt + 1}/2); retrying in {_wait:.0f}s",
                    flush=True,
                )
                time.sleep(_wait)

    _boot_placed_any = False
    for tip, component_name in tip_to_component.items():
        component = getattr(pipe, component_name, None)
        if component is None:
            continue
        if device_map.get(tip) == "hpu" and exec_device.type == "hpu":
            _place_hpu(component_name, component)
            _boot_placed_any = True
        if (
            world_size > 1
            and exec_device.type == "hpu"
            and _boot_placed_any
            and not (_boot_dir / f"r{rank}.acquired").exists()
            and tip == "audio"  # v113: publish AFTER the LAST component, so
            # the next rank's scan never overlaps an in-progress placement
            # sequence (the v109 first-component publish let rank1 scan while
            # rank0 was mid-sequence).
        ):
            (_boot_dir / f"r{rank}.acquired").write_text(str(os.getpid()))
    # Audio VAE kill-switch (--audio-vae-device / H3_AUDIO_VAE_DEVICE):
    # pin it off the HPU, fp32 strictly — never bf16 (documented -20 dB hazard).
    audio_device = getattr(pipe.audio_vae, "device", None)
    if (
        device_map.get("audio") == "cpu"
        and audio_device is not None
        and audio_device != torch.device("cpu")
    ):
        pipe.audio_vae.to(torch.device("cpu"))
        print("[h3] audio_vae pinned to cpu (fp32 strictly, never bf16)")

    if world_size > 1:
        # CP device contract (PLAN_CP.md): replicated weights -- dit/vae/audio
        # all sit on the SAME per-rank device set; the text encoder stays on
        # the host. An explicit --audio-vae-device cpu above still wins.
        cp_tip = "hpu" if exec_device.type == "hpu" else "cpu"
        for tip_name in ("dit", "vae", "audio"):
            device_map[tip_name] = cp_tip
        print(f"[h3] CP device map: dit/vae/audio on {exec_device} (replicated); te on cpu")

    pipe.__dict__["_h3_exec_device"] = str(exec_device)
    _RUNTIME["workdir"] = str(workdir)
    transformer = getattr(pipe, "transformer", None)
    text_encoder = getattr(pipe, "text_encoder", None)
    # from_pretrained(dtype=bf16, low_cpu_mem_usage=True) already honors
    # `_keep_in_fp32_modules` (proj_out/audio_proj_out/time_embedder/rope/proj_in),
    # and `.to(hpu)` above moves devices without touching dtypes.

    # attention backend (DiT + video VAE ViT decoder; softmax mode resolved
    # inside hpu_patches.habana_fused_sdpa: bf16 inputs may use 'fast' when
    # H3_FAST_SOFTMAX=1; fp32 inputs (VAE ViT decode) must use mode None —
    # Habana asserts softmax_mode='fp32' is BF16-input-only)
    if (
        transformer is not None
        and (hpu_ok or _env_bool("H3_FORCE_ATTN_BACKEND", False))
        and _force_backend(transformer, backend_name)
    ):
        _force_backend(pipe.vae, backend_name)
        print(f"[h3] attention backend: {backend_name} (DiT + video VAE)")

    if install_rope_cache(transformer):
        print("[h3] rope cache installed on transformer.rope")
    install_dit_trace(transformer)

    if world_size > 1:
        # PLAN_CP.md phase 1: switch the DiT to diffusers context parallelism.
        # Runs AFTER the attention-backend force (enable_parallelism validates
        # the processor backends against _supports_context_parallel) and after
        # the rope cache install (its hooks wrap the patched forward).
        backend = str(dist.get_backend())
        _cp_enable_parallelism(transformer, world_size, backend)

    graphs_on = not args.no_graphs and hpu_ok
    if world_size > 1 and graphs_on and not _env_bool("H3_CP_GRAPHS", False):
        # First CP bring-up stays eager: graph capture with hccl collectives
        # inside is untested on this stack. H3_CP_GRAPHS=1 opts back in.
        graphs_on = False
        print(
            "[h3] CP: HPU graphs default OFF under CP "
            "(collectives inside graph capture are untested; H3_CP_GRAPHS=1 to force)"
        )
    apply_mark_step_pipe(
        pipe, transformer, text_encoder, include_transformer=not graphs_on
    )
    if graphs_on:
        wrap_transformer_graphs(pipe, transformer)
        print("[h3] graphs ON")
    else:
        print("[h3] graphs OFF: eager lazy mode with post-call mark_step")
        install_block_mark_steps(transformer)
    if _late_decode_enabled() and getattr(pipe, "vae", None) is not None:
        # v97: the video VAE decoder never got per-block frontier breaks — its
        # in-process decode compiled one unbroken giant graph per clip and
        # never returned (the v54/v58/v72/v73c/v86-v96 hang class). Same fix
        # as the DiT: one graph per decoder block.
        _vae_dec = getattr(pipe.vae, "decoder", None)
        if _vae_dec is not None:
            _n_vae = install_block_mark_steps(_vae_dec)
            print(f"[h3] VAE decoder per-block mark_step installed on {_n_vae} blocks", flush=True)
    if _env_bool("H3_SERVER_MODE", False) or args.workflow == "fl2va":
        # v105: the keyframe VAE ENCODER (fl2va) is a conv stack; give its
        # down_blocks frontier breaks too (cheap insurance under graphs-OFF).
        _vae_enc = getattr(pipe.vae, "encoder", None)
        if _vae_enc is not None:
            _n_enc = install_block_mark_steps(_vae_enc, extra_targets=("down_blocks",))
            print(f"[h3] VAE encoder per-block mark_step installed on {_n_enc} blocks", flush=True)
    if _env_bool("H3_SERVER_MODE", False):
        # v106: dual-pipe server — the second workflow's block graph shares
        # EVERY component instance (same DiT/VAE/audio/TE objects; identical
        # weights verified by shard md5), so holding both pipelines costs
        # only Python block objects. Per-request workflow: image present ->
        # fl2va (i2va), else t2va.
        _other = "fl2va" if args.workflow == "t2va" else "t2va"
        if _other == "fl2va":
            _owd = build_workdir(repo=repo, workflow="fl2va")
            _pipe2 = MiniMaxH3ModularPipeline.from_pretrained(
                _owd, workflow="fl2va", local_files_only=True
            )
            _pipe2.update_components(
                tokenizer=pipe.tokenizer,
                processor=pipe.processor,
                scheduler=pipe.scheduler,
                audio_scheduler=pipe.audio_scheduler,
                text_encoder=pipe.text_encoder,
                transformer=pipe.transformer,
                vae=pipe.vae,
                audio_vae=pipe.audio_vae,
            )
            apply_mark_step_pipe(_pipe2, transformer, text_encoder, include_transformer=not graphs_on)
            _RUNTIME["_server_pipes"] = {args.workflow: pipe, _other: _pipe2}
            print(f"[h3] server: dual-pipe ready ({args.workflow} + {_other}; shared weights)", flush=True)

    # Close the placement/setup frontier before the pipeline runs: the first
    # H2D copy inside the pipeline (e.g. randn_tensor's latents.to(device))
    # would otherwise fuse the placement ops plus every pending op into one
    # graph whose memcpy lowering the GC rejects.
    _flush_lazy_frontier()

    # --- run ---------------------------------------------------------------
    cache_dir = Path(args.embed_cache_dir) if args.embed_cache_dir else None
    if cache_dir is None and world_size > 1:
        # v99: encode-once-broadcast under CP. Without a shared cache dir every
        # rank redundantly runs the ~452 s CPU text encode (v69). The store is
        # already atomic (PID-unique tmp + .replace), so a shared dir plus a
        # wait loop on ranks 1..N-1 turns 8x452 s of host work into 1x452 s —
        # and frees 7 ranks' cores entirely.
        cache_dir = workdir / ".ecache"
        print(f"[h3] CP embed cache (encode-once): {cache_dir}", flush=True)
    te_layer = getattr(pipe, "text_encoder_layer", None)
    if device_map.get("te") == "stream":
        # te=stream: streamed Qwen3-VL conditioner — decoder layers pinned in
        # host RAM, ONE scratch layer on-card, H2D copy + eager forward per
        # layer, NO graphs (te_stream_probe5: 47 ms/layer steady => ~3 s for
        # the 51-layer prefill vs ~89 s CPU). RANK 0 ONLY: pinning makes the
        # weights private (the mmap page sharing across ranks is lost), so
        # activating on every rank would need 8x53 GB pinned.
        if exec_device.type != "hpu":
            print("[h3] te=stream requested without an HPU exec device -> te=cpu fallback")
            device_map["te"] = "cpu"
        elif world_size > 1 and (_RUNTIME.get("rank", 0) or 0) != 0:
            print(
                "[h3] te=stream on rank>0: encode-once-broadcast covers this rank "
                "(mmap CPU TE kept as fail-open fallback)"
            )
        else:
            from te_stream import StreamedQwen3VL

            _stream = StreamedQwen3VL(
                text_encoder,
                getattr(pipe, "processor", None),
                exec_device=exec_device,
                text_encoder_layer=int(te_layer) if te_layer is not None else 50,
            )
            _stream.activate()
            text_encoder._h3_stream = _stream
            print(
                f"[h3] TE streaming conditioner active (rank0; condition layer "
                f"{_stream.text_encoder_layer}; graphs OFF)",
                flush=True,
            )
    # Under CP the plan (resolution/duration/budget resolution) lives in the
    # parent's `info` dict; derive the fingerprint inputs from it.
    _canvas = info.get("canvas") or [args.width, args.height]
    width, height = int(_canvas[0]), int(_canvas[1])
    frames = int(info.get("duration_frames") or args.duration)
    budget = int(info.get("text_budget") or args.text_budget or 0)
    steps = int(info.get("steps") or args.steps or 0)
    # te device in the fingerprint: streamed-TE embeds are a DIFFERENT bf16
    # branch than CPU-TE embeds (sink-channel knife-edge layers flip on 1-ulp
    # accumulation noise — img_bisect7) and must not cross-contaminate caches.
    fingerprint = (
        f"te{te_layer}/{width}x{height}/{frames}/{args.workflow}/{device_map.get('te', 'cpu')}"
    )
    if args.prompt is None:
        # --server mode: no boot prompt (requests carry their own). The
        # boot-time embed-cache key is a placeholder; _server_apply_params
        # rebuilds it per request. Outside --server this is unreachable
        # (argparse requires --prompt).
        assert _env_bool("H3_SERVER_MODE", False), "--prompt is required outside --server mode"
        args.prompt = ""
    budget_cache_key = embed_cache_key(
        args.workflow, args.prompt, args.keyframe, fingerprint
    )
    apply_text_budget(
        budget, cache_dir=cache_dir, cache_key=budget_cache_key, cp=world_size
    )
    install_block_logging(pipe)

    kwargs = {
        "prompt": args.prompt,
        "height": height,
        "width": width,
        "num_frames": frames,
        "num_inference_steps": steps,
        "generator": torch.Generator("cpu").manual_seed(args.seed),
        "output_type": "pil",
    }
    if args.workflow == "fl2va":
        from PIL import Image

        images = [Image.open(kf).convert("RGB") for kf in args.keyframe]
        kwargs["image"] = images[0]
        if len(images) > 1:
            kwargs["last_image"] = images[1]

    out_path = (
        Path(out_path)
        if out_path is not None
        else Path(args.out or f"/root/src/h3/out/{args.workflow}_h3.mp4")
    )
    frames_dir = (
        Path(frames_dir) if frames_dir is not None else out_path.parent / (out_path.stem + "_frames")
    )
    wav_path = Path(wav_path) if wav_path is not None else out_path.parent / (out_path.stem + ".wav")
    _RUNTIME.update(
        out_path=str(out_path), frames_dir=str(frames_dir), wav_path=str(wav_path)
    )

    # --- weight-resident serving (--serve N) -------------------------------
    n_serve = max(1, int(getattr(args, "serve", 1) or 1))
    _server_mode = _env_bool("H3_SERVER_MODE", False)
    _server_dir = os.environ.get("H3_SERVER_DIR", "")
    if _server_mode:
        # v100: server loop runs until the SHUTDOWN file appears; requests come
        # from the spool (see _server_claim_request). Duration/resolution stay
        # fixed server-wide (per-request duration would change the DiT graph
        # shape and recompile the giant graph per request); prompt/steps/seed
        # are per-request.
        n_serve = 1_000_000
        os.environ["H3_SERVER_MODE"] = "1"
        if _server_dir:
            (Path(_server_dir) / "jobs").mkdir(parents=True, exist_ok=True)
    _base_out, _base_frames, _base_wav = out_path, frames_dir, wav_path
    _server_last_seq = 0

    def _server_apply_params(req):
        """v103b/v108: per-request frames/resolution/CP-budget/embed-key.

        Called on rank0 BEFORE current.json is published (a rejection fails
        the job and nothing is published, so ranks never see the seq — no
        desync) and on ranks 1..N-1 after receiving the claim. Deterministic
        from req + plan. Mutates kwargs and the encoder class attributes.

        v108: ALWAYS recomputes the CP pad — an fl2va (i2va) request prepends
        anchors*rows_per_frame conditioning rows (1 anchor = 405 rows at
        r480, ODD), which flips the packed sequence's parity. The v107 crash:
        the t2va boot pad left the 1-anchor fl2va seq odd (16317) and
        EquipartitionSharder asserted on hidden_states dim 1. seq = budget +
        anchors*rows + 2*A + T_lat*rows must be % world == 0. (Vision-block
        tokens ride inside the text run, which the budget pad normalizes, so
        they don't enter the parity math.)
        """
        _wf = "fl2va" if req.get("init_image") else "t2va"
        _anchors = (1 + (1 if req.get("end_image") else 0)) if _wf == "fl2va" else 0
        _buckets = load_buckets()
        _fps_b = int(_buckets.get("fps", 24))
        _base_budget = int(info["text_budget"]) - int(info.get("cp_seq_pad") or 0)
        _frames = int(info["duration_frames"])
        _rows = int(info["rows_per_frame"])
        _tlat, _A = int(info["T_lat"]), int(info["audio_latents"])
        if req.get("video_frames"):
            _sec = max(1, round(int(req["video_frames"]) / _fps_b))
            _frames, _tlat = resolve_duration(_buckets, _sec)
            _dur_spec = next(
                v for v in _buckets["durations"].values() if v["frames"] == _frames
            )
            _A = int(_dur_spec["A"])
        if req.get("width") and req.get("height"):
            _res_name, _w, _h = resolve_resolution(
                _buckets, f"{int(req['width'])}x{int(req['height'])}"
            )
            _rows = int(_buckets["resolutions"][_res_name]["rows_per_frame"])
            kwargs["width"], kwargs["height"] = _w, _h
            print(f"[h3] server: resolution -> {_res_name} ({_w}x{_h})", flush=True)
        _extra = _anchors * _rows
        _expected = _base_budget + _tlat * _rows + 2 * _A + _extra
        _pad = (-_expected) % world_size
        _encoders = _import_h3_module("encoders")
        for _cls in (
            _encoders.MiniMaxH3TextEncoderStep,
            _encoders.MiniMaxH3FL2VATextEncoderStep,
        ):
            _cls._h3_budget = _base_budget + _pad
        kwargs["num_frames"] = _frames
        print(
            f"[h3] server: wf {_wf} anchors {_anchors} frames {_frames} "
            f"(T_lat {_tlat}, A {_A}, budget {_base_budget}+{_pad}, "
            f"seq {_expected + _pad})",
            flush=True,
        )
        # v121: guard the graph-capture host-IR budget. Under CP=1 with HPU
        # graphs, a bucket NOT already in the recipe cache triggers a fresh
        # FULL graph capture whose host-side IR is roughly linear in seq
        # (r480/124f ~ seq 17.4k fits; r768/192f ~ seq 60k ballooned process
        # RSS to 86.6 GB and the kernel OOM-killed the whole server
        # mid-capture, leaving a 96 GiB ghost pool on the card). Reject HERE
        # (pre-publish, v103b discipline) so only this job fails. Remedies:
        # --cp 4/8 (per-rank graphs shrink by N), restart with --no-graphs
        # (per-block mark_steps bound the IR), or raise H3_MAX_GRAPH_SEQ.
        _gcap = int(os.environ.get("H3_MAX_GRAPH_SEQ") or 20000)
        if world_size == 1 and graphs_on and (_expected + _pad) > _gcap:
            raise ValueError(
                f"packed seq {_expected + _pad} exceeds H3_MAX_GRAPH_SEQ={_gcap} "
                f"under CP=1 with HPU graphs (graph-capture host IR OOM, v121): "
                f"use --cp 4/8, restart the server with --no-graphs, or set "
                f"H3_MAX_GRAPH_SEQ higher"
            )
        # Per-request embed cache key (encode-once-broadcast keyed on prompt
        # AND shape AND workflow AND keyframe bytes AND anchor fit mode).
        _kfs = [k for k in (req.get("init_image"), req.get("end_image")) if k]
        _fit = str(req.get("fit") or "stretch").lower()
        _te_layer = getattr(pipe, "text_encoder_layer", None)
        _fp = (
            f"te{_te_layer}/{kwargs.get('width') or info['canvas'][0]}x"
            f"{kwargs.get('height') or info['canvas'][1]}/"
            f"{kwargs.get('num_frames') or info['duration_frames']}/{_wf}/fit{_fit}"
        )
        _ck = embed_cache_key(_wf, req["prompt"], _kfs, _fp)
        _encoders = _import_h3_module("encoders")
        for _cls in (
            _encoders.MiniMaxH3TextEncoderStep,
            _encoders.MiniMaxH3FL2VATextEncoderStep,
        ):
            _cls._h3_cache_key = _ck

    for _serve_i in range(n_serve):
        if _server_mode and _server_dir:
            _action, _req = _server_claim_request(
                rank, world_size, _server_dir, _server_last_seq, _server_apply_params
            )
            if _action == "shutdown":
                print(f"[h3] server loop: shutdown (rank {rank})", flush=True)
                break
            if _action == "skip":
                # Ranks only: rank0 rejected this seq pre-publish; mark it seen
                # and wait for the next one. Nobody touches the device.
                _server_last_seq = int(_req.get("seq", _server_last_seq))
                continue
            _server_last_seq = int(_req.get("seq", _server_last_seq))
            # Per-request overrides: prompt/steps/seed; artifact named by job id.
            args = argparse.Namespace(**{**vars(args), "prompt": _req["prompt"]})
            steps = int(_req.get("steps") or info.get("steps") or args.steps or 4)
            _seed = int(_req.get("seed", 42))
            if _seed < 0:
                import secrets as _secrets
                _seed = _secrets.randbits(31)
            out_path = Path(_req["out_dir"]) / f"{_req['id']}.mp4"
            frames_dir = out_path.parent / f"{out_path.stem}_frames"
            wav_path = out_path.parent / f"{out_path.stem}.wav"
            _RUNTIME.update(out_path=str(out_path), frames_dir=str(frames_dir), wav_path=str(wav_path))
            _RUNTIME["_server_job"] = _req["id"]  # v101: finish_job keys on this
            _RUNTIME["serve_keep_dit"] = True  # server always keeps weights resident
            kwargs["prompt"] = _req["prompt"]
            kwargs["num_inference_steps"] = steps
            kwargs["generator"] = torch.Generator("cpu").manual_seed(_seed)
            # v106: per-request workflow — image present -> fl2va (i2va).
            kwargs.pop("image", None)
            kwargs.pop("last_image", None)
            _wf = args.workflow
            _pipe_i = pipe
            if _req.get("init_image"):
                _wf = "fl2va"
                _pipes = _RUNTIME.get("_server_pipes") or {}
                _pipe_i = _pipes.get("fl2va", pipe)
                from PIL import Image as _PILImage
                _fit = str(_req.get("fit") or "stretch").lower()
                kwargs["image"] = _fit_anchor_image(
                    _PILImage.open(_req["init_image"]).convert("RGB"),
                    kwargs["width"], kwargs["height"], _fit,
                )
                if _req.get("end_image"):
                    kwargs["last_image"] = _fit_anchor_image(
                        _PILImage.open(_req["end_image"]).convert("RGB"),
                        kwargs["width"], kwargs["height"], _fit,
                    )
                if _fit != "stretch":
                    print(f"[h3] server: anchor fit {_fit}", flush=True)
            if _req.get("negative_prompt"):
                print(f"[h3] server: negative_prompt ignored in v1", flush=True)
            print(f"[h3] === server request {out_path.stem} ({_wf}) ===", flush=True)
        elif n_serve > 1:
            _RUNTIME["serve_keep_dit"] = _serve_i < n_serve - 1
            out_path = _base_out.with_name(f"{_base_out.stem}_req{_serve_i}.mp4")
            frames_dir = _base_out.parent / f"{_base_out.stem}_req{_serve_i}_frames"
            wav_path = _base_out.parent / f"{_base_out.stem}_req{_serve_i}.wav"
            _RUNTIME.update(out_path=str(out_path), frames_dir=str(frames_dir), wav_path=str(wav_path))
            print(f"[h3] === serve request {_serve_i + 1}/{n_serve} ===", flush=True)
        if (
            n_serve > 1
            and _serve_i > 0
            and _RUNTIME.get("backend") == "hccl"
            and (_RUNTIME.get("rank", 0) or 0) != 0
        ):
            # CP serve pacing: ranks 1..N-1 wait (HOST-side poll + tiny device
            # heartbeat) for rank0's export-done marker before starting this
            # request. Racing ahead would enqueue denoise collectives that
            # stall while rank0 exports (~55 s) -- enqueued-but-stalled work
            # trips the Synapse no-progress watchdog (v72/v73b); an idle card
            # with a heartbeat does not (v70/v75).
            _sync = Path(_RUNTIME["out_path"]).parent / ".serve_sync"
            _mark = _sync / f"req{_serve_i - 1}.done"
            _t = time.perf_counter()
            _dev = torch.device("hpu")
            while not _mark.exists() and time.perf_counter() - _t < 600.0:
                try:
                    _shard_heartbeat(_dev)
                except Exception:
                    pass
                time.sleep(1.0)
            if not _mark.exists():
                # v97b: fail LOUD — a stuck export that silently fail-opens
                # sends ranks 1..N-1 into the next request's denoise
                # collectives without rank0, which wedges the whole run and
                # masks the original stall (v85's massacre).
                raise RuntimeError(
                    f"serve sync marker {_mark} never appeared (600 s); "
                    "rank0's export is stuck"
                )
            else:
                print(f"[h3] serve sync: rank export done ({time.perf_counter() - _t:.1f}s wait)", flush=True)
        if _serve_i > 0:
            # reset per-request bookkeeping from the previous request (both
            # --serve N and server mode need this; only the RESEED is
            # --serve-N-only — server mode sets a per-request generator in the
            # claim block above, and this block would clobber it)
            _RUNTIME.pop("decode_subproc_done", None)
            _RUNTIME.pop("decode_meta", None)
            _RUNTIME.pop("decode_sharded_done", None)
            _RUNTIME.pop("sharded_audio_done", None)
            if not _server_mode:
                # Re-seed per request: the shared kwargs generator advanced
                # during request 1, which changes the noise (fine for variety,
                # but it would mask a graph-replay correctness bug). A fresh
                # generator makes request N's noise identical to request 1's,
                # so artifacts must match bit-for-bit if replay is exact.
                kwargs["generator"] = torch.Generator("cpu").manual_seed(args.seed)
        timings = {}
        state = None
        global _BLOCK_TIME_RECORDS, _BLOCK_TIMES_LAST
        runs = ([("warmup", max(1, int(args.warmup)))] if args.warmup else []) + [
            ("generate", steps)
        ]
        for label, run_steps in runs:
            t0 = time.perf_counter()
            # Per-block wall-time recording (CP timing merge): generate run only.
            _BLOCK_TIME_RECORDS = [] if label == "generate" else None
            try:
                # v106: server mode may have switched the active pipeline
                # (fl2va for image requests); --serve mode always uses the
                # boot pipe.
                _active = (
                    _pipe_i if (_server_mode and _server_dir) else pipe
                )
                state = _active(**{**kwargs, "num_inference_steps": run_steps})
            except _BisectStop as stop:
                _BLOCK_TIME_RECORDS = None
                print(f"[h3] bisect stop after {stop}; diagnostics complete")
                return {}
            if label == "generate":
                _BLOCK_TIMES_LAST = list(_BLOCK_TIME_RECORDS or [])
            _BLOCK_TIME_RECORDS = None
            timings[label] = round(time.perf_counter() - t0, 2)
        # Phase-boundary sync (decode -> export): drain the device and surface
        # any swallowed error before the host-side export reads the outputs.
        if hpu_ok:
            t0 = time.perf_counter()
            torch.hpu.synchronize()
            print(f"[h3] pipeline complete: device synced before export "
                  f"({time.perf_counter() - t0:.2f}s)", flush=True)
        if state is None:
            raise RuntimeError("the pipeline returned no state")

        videos = state.get("videos")
        audio = state.get("audio")
        sampling_rate = int(state.get("sampling_rate") or 32000)

        if rank not in (None, 0):
            # CP with replicated weights: rank0 renders the artifact (frames +
            # wav + mux); the other ranks skip export. In serve mode they must
            # KEEP LOOPING (request 2's collectives need them), so continue
            # instead of returning; the serve-sync wait at the loop head paces
            # them against rank0's export.
            print(f"[h3] rank {rank}: rank0 owns export; nothing to write", flush=True)
            timings["export"] = 0.0
            timings["mux"] = 0.0
            print(f"[h3] timings: {json.dumps(timings)}")
            continue

        t0 = time.perf_counter()
        if _RUNTIME.get("decode_sharded_done"):
            # Sharded decode: every rank wrote its own frames + a done marker.
            # Rank0 waits for ALL markers before exporting (host-side poll, no
            # device work -- an idle drained card does not trip the watchdog,
            # v70; a stalled ENQUEUE does, which is why the decode phase has no
            # collectives, v72).
            ws = _RUNTIME.get("world_size", 1) or 1
            shardtmp = frames_dir / ".shard_tmp"
            subproc_py = Path(__file__).resolve().parent / "decode_subproc.py"
            blob_path = wav_path.parent / (wav_path.stem + "_decode_blob.pt")
            meta_path = wav_path.parent / (wav_path.stem + "_decode_meta.json")
            retried = set()
            # v85: per-shard attempt budget (max 8) instead of one-shot retry.
            # v84 showed the FIRST farm batch can die wholesale at device
            # acquire: the parents' giant denoise graph drains at decode entry
            # and the driver keeps the cards unsettled for a few seconds; a
            # worker spawned inside that window gets synStatus=8 even on a pin
            # whose card "looks" free. A respawn seconds later succeeds. The
            # old one-shot retry plus the per-proc `retried` set either
            # exhausted immediately or (via respawned procs re-entering the
            # pool) thrashed unboundedly, v81.
            _respawn_attempts = {}
            _RESPAWN_MAX = 8
            # Workers target HPU cards whenever this is an HPU run (hccl
            # backend implies HPU); CPU harness respawns stay on CPU.
            device_tip = "hpu" if _RUNTIME.get("backend") == "hccl" else "cpu"
            t_wait = time.perf_counter()
            missing = list(range(ws))
            while missing and time.perf_counter() - t_wait < 900.0:
                missing = [
                    r for r in range(ws) if not (shardtmp / f"done_r{r}.json").exists()
                ]
                if missing:
                    time.sleep(2.0)
                # Self-healing farm: a worker that died (e.g. synStatus=8 host
                # hugepage exhaustion when 8 workers spawn beside 8 resident
                # CP parents, v80) is respawned ONCE — by then the dead
                # workers' HPs are back in the pool.
                if missing:
                    for p in list(_RUNTIME.get("_farm_procs") or []):
                        if p.poll() is None or p in retried:
                            continue
                        retried.add(p)
                        argv = p.args
                        shard_i = int(argv[argv.index("--shard") + 1])
                        is_audio = "--audio-shard" in argv
                        if shard_i not in missing:
                            continue
                        key = (shard_i, is_audio)
                        n_att = _respawn_attempts.get(key, 0)
                        if n_att >= _RESPAWN_MAX:
                            continue
                        _respawn_attempts[key] = n_att + 1
                        card = 0 if is_audio else (shard_i + 1 if shard_i < 7 else 0)
                        if device_tip == "hpu":
                            _wait_card_free(card)
                        # Settle delay: let the parents' drain window close.
                        time.sleep(3.0 * (n_att + 1))
                        print(
                            f"[h3] respawn farm worker shard {shard_i}"
                            f"{' (audio)' if is_audio else ''} on card {card}"
                            f" (attempt {n_att + 1}/{_RESPAWN_MAX})",
                            flush=True,
                        )
                        new_p = _respawn_farm_worker(
                            argv, card, device_tip,
                            log_path=Path(argv[argv.index("--frames-dir") + 1])
                            / ".shard_tmp"
                            / (f"worker_{shard_i}_re{n_att + 1}.log"),
                        )
                        _RUNTIME.setdefault("_farm_procs", []).append(new_p)
                        # Autopsy: surface WHY the worker died, from its log file.
                        try:
                            fi = argv.index("--frames-dir")
                            wlog = Path(argv[fi + 1]) / ".shard_tmp" / (
                                "worker_audio.log" if is_audio else f"worker_{shard_i}.log"
                            )
                            tail = wlog.read_text(errors="replace")[-400:] if wlog.exists() else "(no log)"
                            print(f"[h3] dead worker shard {shard_i} tail: {tail}", flush=True)
                        except Exception as _e:
                            print(f"[h3] worker autopsy failed: {_e}", flush=True)
            # Reap the farm workers and surface any failures.
            for p in _RUNTIME.get("_farm_procs") or []:
                try:
                    out = p.communicate(timeout=30)[0] if p.stdout is not None else ""
                    if p.returncode != 0:
                        print(f"[h3] farm worker rc={p.returncode}: {out[-500:]}", flush=True)
                except Exception as e:
                    print(f"[h3] farm worker reap failed: {e}", flush=True)
            if missing:
                print(f"[h3] WARNING: shard markers missing for ranks {missing}; "
                      f"exporting with whatever frames exist", flush=True)
            else:
                print(f"[h3] all {ws} shard markers present ({time.perf_counter() - t_wait:.1f}s wait)", flush=True)
            # Frames are already on disk (written by every rank with global
            # numbering); nothing to save here.
            frame_paths = []
            print(f"[h3] frames already on disk from sharded decode: {frames_dir}")
        else:
            frame_paths = save_frames(videos, frames_dir)
        if frame_paths:
            print(f"[h3] wrote {len(frame_paths)} frames to {frames_dir}")
        if audio is None:
            # The decode subprocess hatch (and the sharded decode's audio rank)
            # write the wav themselves (state carries no audio tensors across);
            # adopt their artifact.
            if (
                _RUNTIME.get("decode_subproc_done") or _RUNTIME.get("decode_sharded_done")
            ) and wav_path.exists():
                audio_written = True
                meta = _RUNTIME.get("decode_meta") or {}
                sampling_rate = int(meta.get("sampling_rate") or sampling_rate)
                print(f"[h3] audio adopted from decode subprocess: {wav_path}", flush=True)
            else:
                print("[h3] WARNING: no audio in pipeline state; skipping wav export")
                audio_written = False
        else:
            save_audio_wav(audio, sampling_rate, wav_path)
            audio_written = True
        timings["export"] = round(time.perf_counter() - t0, 2)

        timings["mux"] = 0.0
        if audio_written:
            t0 = time.perf_counter()
            # `buckets` is parent-side plan state (not passed to CP workers); the
            # fps lives in buckets.json regardless, so re-read it here.
            fps = int(load_buckets().get("fps", 24))
            if mux_mp4(frames_dir, wav_path, out_path, fps=fps):
                timings["mux"] = round(time.perf_counter() - t0, 2)
                print(f"[h3] wrote {out_path}")

        print(f"[h3] timings: {json.dumps(timings)}")
        if _server_mode and _server_dir and rank in (None, 0):
            # v100: job completed — the HTTP front-end serves b64 mp4 from this.
            _server_finish_job(
                _server_dir, _RUNTIME.get("_server_job", "unknown"), out_path, timings
            )
        if n_serve > 1 and (_RUNTIME.get("backend") == "hccl") and (rank in (None, 0)):
            # Serve-sync publish: tell ranks 1..N-1 this request's export is
            # done (host-side marker; they heartbeat-poll it, v80 protocol).
            _sync = out_path.parent / ".serve_sync"
            _sync.mkdir(parents=True, exist_ok=True)
            (_sync / f"req{_serve_i}.done").write_text("ok")
        if _env_flag("LOG_LEVEL_PT_FALLBACK", "0") not in ("0", "", None):
            print(
                "[h3] fallback-op audit enabled (LOG_LEVEL_PT_FALLBACK=1): scan the process log for FALLBACK op "
                "reports (nn.RMSNorm sites, fp64 rope grid, F.pad, index_select/index_copy, FusedSDPA reshapes)."
            )
    return timings


# Ensure the CLI-facing parent dir import works for `hpu_port.*` sibling lookups.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception:
        import traceback

        traceback.print_exc()
        sys.exit(1)

# re-dispatch: pyright env fix verification (whitespace-only)

# pyright re-dispatch: pythonPath config verify (whitespace only)
