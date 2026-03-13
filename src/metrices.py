#!/usr/bin/env python3
"""
metrics.py  —  PSNR + SSIM evaluator for DMFFT

Reads comparison images from dmfft.py results/ folder automatically.
Splits each side-by-side image into baseline (left) and DMFFT (right).

Usage:
    python metrics.py --results ./results

Requirements:
    pip install numpy pillow scikit-image
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from skimage.metrics import structural_similarity as ssim_fn
    from skimage.metrics import peak_signal_noise_ratio as psnr_fn
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    print("WARNING: scikit-image not found — using numpy fallback.")
    print("         pip install scikit-image\n")


# ─────────────────────────────────────────────────────────────────────────────
# Split comparison images
# Must match dmfft.py _save_comparison() constants exactly
# ─────────────────────────────────────────────────────────────────────────────

LABEL_H  = 40
FOOTER_H = 60
GAP      = 6

def split_comparison(img_path):
    """Split one comparison PNG into (baseline, dmfft) PIL Images."""
    comp = Image.open(img_path).convert('RGB')
    cw, ch = comp.size
    img_w = (cw - GAP) // 2
    img_h = ch - LABEL_H - FOOTER_H
    baseline = comp.crop((0,           LABEL_H, img_w,            LABEL_H + img_h))
    dmfft    = comp.crop((img_w + GAP, LABEL_H, img_w * 2 + GAP,  LABEL_H + img_h))
    return baseline, dmfft


def load_pairs(results_dir):
    """Load all dmfft comparison images from results/ folder."""
    p = Path(results_dir)
    files = sorted(
        f for f in p.iterdir()
        if f.suffix.lower() in {'.png', '.jpg', '.jpeg'}
        and 'grid' not in f.name
        and f.name.startswith('dmfft_')
    )
    if not files:
        raise ValueError(
            f"No comparison images found in '{results_dir}'.\n"
            "Run dmfft.py first to generate results.")

    print(f"  Found {len(files)} comparison image(s) in {results_dir}\n")
    return [(split_comparison(f)[0], split_comparison(f)[1], f.name) for f in files]


# ─────────────────────────────────────────────────────────────────────────────
# PSNR & SSIM
# ─────────────────────────────────────────────────────────────────────────────

def to_numpy(img):
    return np.array(img, dtype=np.float32) / 255.0

def compute_psnr(a, b):
    a, b = to_numpy(a), to_numpy(b)
    if HAS_SKIMAGE:
        return psnr_fn(a, b, data_range=1.0)
    mse = np.mean((a - b) ** 2)
    return float('inf') if mse == 0 else 20 * np.log10(1.0 / np.sqrt(mse))

def compute_ssim(a, b):
    a, b = to_numpy(a), to_numpy(b)
    if HAS_SKIMAGE:
        return ssim_fn(a, b, data_range=1.0, channel_axis=-1)
    # Numpy grayscale fallback
    ag, bg = a.mean(axis=-1), b.mean(axis=-1)
    C1, C2 = 0.01**2, 0.03**2
    mu1, mu2 = ag.mean(), bg.mean()
    s1, s2   = np.var(ag), np.var(bg)
    s12      = np.cov(ag.flat, bg.flat)[0, 1]
    return ((2*mu1*mu2 + C1) * (2*s12 + C2)) / \
           ((mu1**2 + mu2**2 + C1) * (s1 + s2 + C2))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PSNR + SSIM evaluator for DMFFT",
        epilog="Example: python metrics.py --results ./results"
    )
    parser.add_argument('--results', required=True,
                        help='results/ folder produced by dmfft.py')
    args = parser.parse_args()

    if not os.path.exists(args.results):
        print(f"Error: path not found: {args.results}")
        sys.exit(1)

    print("=" * 65)
    print("DMFFT  Metrics  —  PSNR + SSIM  (Baseline vs DMFFT)")
    print("=" * 65 + "\n")

    pairs = load_pairs(args.results)

    rows = []
    for base, dmfft, name in pairs:
        psnr = compute_psnr(base, dmfft)
        ssim = compute_ssim(base, dmfft)
        rows.append((name[:45], psnr, ssim))

    # ── Print table ──────────────────────────────────────────────────────────
    print(f"  {'File':<45} {'PSNR (dB)':>10} {'SSIM':>8}")
    print("  " + "-" * 65)
    for name, psnr, ssim in rows:
        print(f"  {name:<45} {psnr:>10.2f} {ssim:>8.4f}")

    avg_psnr = np.mean([r[1] for r in rows])
    avg_ssim = np.mean([r[2] for r in rows])
    print("  " + "-" * 65)
    print(f"  {'AVERAGE':<45} {avg_psnr:>10.2f} {avg_ssim:>8.4f}")

    print("\n  PSNR : higher is better  (dB)")
    print("  SSIM : higher is better  (max 1.0)")
    print("=" * 65)


if __name__ == '__main__':
    main()
