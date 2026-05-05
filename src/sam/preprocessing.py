"""Pre-SAM image preprocessing (swappable later without touching SAM code)."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import cv2
import numpy as np

from parquet.seam_highlight import normalize_lighting_bgr


def normalize_for_sam(
    image_bgr: np.ndarray,
    *,
    flat_field_sigma: float = 35.0,
    clahe_clip: float = 2.5,
    clahe_grid: int = 8,
) -> np.ndarray:
    """Flat-field + LAB CLAHE on L, same as the parquet seam helper."""
    return normalize_lighting_bgr(
        image_bgr,
        flat_field_sigma=flat_field_sigma,
        clahe_clip=clahe_clip,
        clahe_grid=clahe_grid,
    )


def normalize_for_sam_from_args(image_bgr: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    return normalize_for_sam(
        image_bgr,
        flat_field_sigma=args.flat_sigma,
        clahe_clip=args.clahe_clip,
        clahe_grid=args.clahe_grid,
    )


def bilateral_denoise_bgr(
    image_bgr: np.ndarray,
    d: int,
    sigma_color: float,
    sigma_space: float,
) -> np.ndarray:
    """Mild edge-preserving denoise; ``d <= 0`` skips."""
    if d <= 0:
        return image_bgr
    d2 = d if d % 2 == 1 else d + 1
    return cv2.bilateralFilter(
        image_bgr,
        d2,
        float(sigma_color),
        float(sigma_space),
    )


@dataclass(frozen=True)
class LetterboxMeta:
    """Map letterboxed (S×S) masks back to full-res H×W."""

    size: int
    orig_h: int
    orig_w: int
    pad_top: int
    pad_left: int
    nh: int
    nw: int


def align_imgsz_stride14(n: int) -> int:
    """Multiple of max stride 14 (Ultralytics-style 640 → 644)."""
    if n <= 0:
        return n
    return max(14, ((n + 13) // 14) * 14)


def letterbox_square_bgr(
    image_bgr: np.ndarray,
    size: int,
    pad_bgr: tuple[int, int, int],
) -> tuple[np.ndarray, LetterboxMeta]:
    """Resize to fit inside ``size``×``size``, pad with ``pad_bgr`` (BGR)."""
    h, w = image_bgr.shape[:2]
    scale = min(size / float(h), size / float(w))
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image_bgr, (nw, nh), interpolation=interp)
    pad_h = size - nh
    pad_w = size - nw
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    b, g, r = pad_bgr
    out = cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=(b, g, r),
    )
    meta = LetterboxMeta(
        size=size,
        orig_h=h,
        orig_w=w,
        pad_top=pad_top,
        pad_left=pad_left,
        nh=nh,
        nw=nw,
    )
    return out, meta


def unletterbox_masks(masks: np.ndarray, meta: LetterboxMeta | None) -> np.ndarray:
    """(N,S,S) on letterboxed canvas → (N, orig_h, orig_w)."""
    if meta is None:
        return masks
    n, mh, mw = int(masks.shape[0]), int(masks.shape[1]), int(masks.shape[2])
    if mh == meta.orig_h and mw == meta.orig_w:
        return masks
    y0, y1 = meta.pad_top, meta.pad_top + meta.nh
    x0, x1 = meta.pad_left, meta.pad_left + meta.nw
    cropped = masks[:, y0:y1, x0:x1]
    out = np.empty((n, meta.orig_h, meta.orig_w), dtype=np.float32)
    for i in range(n):
        out[i] = cv2.resize(
            cropped[i],
            (meta.orig_w, meta.orig_h),
            interpolation=cv2.INTER_LINEAR,
        )
    return out
