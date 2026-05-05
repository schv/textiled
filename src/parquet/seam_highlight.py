#!/usr/bin/env python3
"""
Normalize uneven lighting on parquet-floor photos and overlay detected plank seams in red.

Uses OpenCV for I/O, color space, morphology, filtering, and gradients; NumPy only for
broadcasting blends. All per-pixel work is vectorized (no Python loops over pixels).

Requirements: numpy, opencv-python
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def geometry_luma_u8(bgr_u8: np.ndarray) -> np.ndarray:
    """Single-channel luma for geometry / border cues (BGR uint8 → gray uint8)."""
    return cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2GRAY)


def normalize_lighting_bgr(
    image_bgr: np.ndarray,
    flat_field_sigma: float = 35.0,
    clahe_clip: float = 2.5,
    clahe_grid: int = 8,
) -> np.ndarray:
    """
    Reduce shading / uneven illumination while preserving plank texture.

    1) Flat-field: divide each BGR channel by a large-Gaussian blurred luminance estimate.
    2) LAB: apply CLAHE on L only, then merge (perceptual lightness normalization).
    """
    bgr = image_bgr.astype(np.float32)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    ksize = 0  # derive from sigma
    blur_luma = cv2.GaussianBlur(gray, (ksize, ksize), flat_field_sigma)
    eps = 1e-3
    scale = np.mean(blur_luma) / (blur_luma + eps)
    flat = np.clip(bgr * scale[..., None], 0.0, 255.0).astype(np.uint8)

    lab = cv2.cvtColor(flat, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_grid, clahe_grid))
    l2 = clahe.apply(l)
    merged = cv2.merge((l2, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def seam_probability_map(gray_u8: np.ndarray) -> np.ndarray:
    """
    Float map in [0, 1]: high where plank seams / borders are likely. Fully OpenCV-based.
    """
    g = gray_u8

    # Local mean via box filter (fast, separable); seams often darker than neighborhood.
    k = 15
    local_mean = cv2.boxFilter(g, ddepth=-1, ksize=(k, k), normalize=True)
    dark_line = cv2.normalize(
        (local_mean.astype(np.float32) - g.astype(np.float32)),
        None,
        0.0,
        1.0,
        cv2.NORM_MINMAX,
        dtype=cv2.CV_32F,
    )

    # Morphological gradient = edge strength (vectorized).
    rect3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph_grad = cv2.morphologyEx(g, cv2.MORPH_GRADIENT, rect3)
    edge_strength = cv2.normalize(morph_grad, None, 0.0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    sob = cv2.magnitude(gx, gy)
    sob_n = cv2.normalize(sob, None, 0.0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # Combine cues with channel-wise max then slight blur to bridge small gaps.
    stacked = np.stack((dark_line, edge_strength, sob_n), axis=-1)
    combined = np.max(stacked, axis=-1).astype(np.float32)
    combined = cv2.GaussianBlur(combined, (0, 0), sigmaX=1.0)
    return np.clip(combined, 0.0, 1.0)


def seam_mask_from_map(
    prob: np.ndarray,
    quantile: float = 0.92,
    dilate_ksize: int = 3,
) -> np.ndarray:
    """Binary uint8 mask 0/255 from probability map; threshold adapts to image contrast."""
    flat = prob.reshape(-1)
    thresh = float(np.quantile(flat, quantile))
    thresh = max(thresh, 0.05)
    mask = (prob >= thresh).astype(np.uint8) * 255
    if dilate_ksize > 1:
        ker = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (dilate_ksize, dilate_ksize),
        )
        mask = cv2.dilate(mask, ker, iterations=1)
    return mask


def overlay_red_seams(
    image_bgr: np.ndarray,
    seam_mask_u8: np.ndarray,
    alpha: float = 0.65,
) -> np.ndarray:
    """Alpha-blend red (BGR) over seam pixels; vectorized with broadcasting."""
    m = (seam_mask_u8 > 0).astype(np.float32)[..., None]
    red = np.array([0.0, 0.0, 255.0], dtype=np.float32)
    base = image_bgr.astype(np.float32)
    tint = base * (1.0 - alpha * m) + red * (alpha * m)
    return np.clip(tint, 0.0, 255.0).astype(np.uint8)


def process_image(
    image_bgr: np.ndarray,
    flat_field_sigma: float,
    clahe_clip: float,
    seam_quantile: float,
    seam_dilate: int,
    overlay_alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (normalized_bgr, seam_mask_u8, overlay_bgr, seam_probability_f32).
    """
    norm = normalize_lighting_bgr(
        image_bgr,
        flat_field_sigma=flat_field_sigma,
        clahe_clip=clahe_clip,
    )
    gray = cv2.cvtColor(norm, cv2.COLOR_BGR2GRAY)
    prob = seam_probability_map(gray)
    mask = seam_mask_from_map(prob, quantile=seam_quantile, dilate_ksize=seam_dilate)
    overlay = overlay_red_seams(norm, mask, alpha=overlay_alpha)
    return norm, mask, overlay, prob


def main() -> None:
    p = argparse.ArgumentParser(description="Parquet lighting normalize + red seam overlay")
    p.add_argument("input", type=Path, help="Input image path (parquet photo)")
    p.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory for output images",
    )
    p.add_argument("--flat-sigma", type=float, default=35.0, help="Gaussian sigma for flat-field")
    p.add_argument("--clahe-clip", type=float, default=2.5, help="CLAHE clip limit")
    p.add_argument(
        "--seam-q",
        type=float,
        default=0.92,
        help="Quantile for adaptive seam threshold (0–1, higher = fewer seams)",
    )
    p.add_argument("--seam-dilate", type=int, default=3, help="Odd-ish kernel size to thicken lines")
    p.add_argument("--overlay-alpha", type=float, default=0.65, help="Red blend strength on seams")
    args = p.parse_args()

    img = cv2.imread(str(args.input), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"Could not read image: {args.input}")

    norm, mask, overlay, _prob = process_image(
        img,
        flat_field_sigma=args.flat_sigma,
        clahe_clip=args.clahe_clip,
        seam_quantile=args.seam_q,
        seam_dilate=args.seam_dilate,
        overlay_alpha=args.overlay_alpha,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.input.stem
    cv2.imwrite(str(args.output_dir / f"{stem}_lighting_norm.png"), norm)
    cv2.imwrite(str(args.output_dir / f"{stem}_seam_mask.png"), mask)
    cv2.imwrite(str(args.output_dir / f"{stem}_seams_red.png"), overlay)


if __name__ == "__main__":
    main()
