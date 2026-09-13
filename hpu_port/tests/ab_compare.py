#!/usr/bin/env python
"""ab_compare.py — pixel-level A/B of te=cpu vs te=stream artifacts."""

import sys
from pathlib import Path

import numpy as np
from PIL import Image


def load_dir(d: Path) -> np.ndarray:
    files = sorted(d.glob("frame_*.png"))
    return np.stack([np.asarray(Image.open(f).convert("RGB")) for f in files])


def main() -> None:
    a_dir, b_dir = Path(sys.argv[1]), Path(sys.argv[2])
    a, b = load_dir(a_dir), load_dir(b_dir)
    assert a.shape == b.shape, f"shape mismatch {a.shape} vs {b.shape}"
    a_f, b_f = a.astype(np.float32), b.astype(np.float32)
    mse = ((a_f - b_f) ** 2).mean()
    psnr = 10 * np.log10(255.0**2 / mse) if mse > 0 else float("inf")
    # per-frame cosine of flattened RGB
    cos = []
    for i in range(a.shape[0]):
        x, y = a_f[i].ravel(), b_f[i].ravel()
        cos.append(float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-9)))
    ident = int(np.array_equal(a, b))
    print(f"frames: {a.shape[0]} | identical bytes: {ident}")
    print(f"global PSNR: {psnr:.2f} dB | MSE: {mse:.2f}")
    print(f"per-frame RGB cosine: min={min(cos):.6f} mean={np.mean(cos):.6f}")
    worst = int(np.argmin(cos))
    print(f"worst frame: {worst} (cos {min(cos):.6f})")
    # audio
    return


if __name__ == "__main__":
    main()
