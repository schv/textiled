#!/usr/bin/env python3
"""
Equal-rectangle plank grid fitting from SAM mask boundaries + OpenCV border evidence.

Produces a compact lattice model (two dominant directions, spacings, phases) and
scored plank parallelograms for downstream extraction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from parquet.seam_highlight import geometry_luma_u8, seam_probability_map

JsonDict = dict[str, Any]
LineSegment = tuple[float, float, float, float]


@dataclass
class PlankGridFitParams:
    min_mask_area_frac: float = 0.001
    max_mask_area_frac: float = 0.85
    boundary_dilate: int = 3
    evidence_sam_weight: float = 0.45
    evidence_border_weight: float = 0.55
    evidence_quantile: float = 0.82
    ridge_dist_max: int = 5
    hough_min_len_frac: float = 28.0  # divisor of min(h,w)
    angle_hist_bins: int = 180
    orthogonality_tol_deg: float = 18.0
    lattice_spacing_search_steps: int = 48
    lattice_phase_steps: int = 36
    max_plank_cells: int = 8
    max_hough_segments: int = 600
    perimeter_sample_step: int = 2
    nms_iou: float = 0.55
    mask_interior_support_quantile: float = 0.35


def _segment_angle_deg(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180.0


def _circular_angle_distance_deg(a: float, b: float) -> float:
    d = abs((a - b) % 180.0)
    return min(d, 180.0 - d)


def mask_boundary_map(
    masks: np.ndarray,
    min_area_frac: float,
    max_area_frac: float,
    boundary_dilate: int,
) -> tuple[np.ndarray, list[dict[str, float | int]], list[dict[str, float | int | str]]]:
    """SAM instance mask boundaries as uint8 0/255 (same logic as sam3_segment_sample)."""
    h, w = masks.shape[1:]
    image_area = float(h * w)
    min_area = max(16, int(round(image_area * min_area_frac)))
    max_area = max(min_area + 1, int(round(image_area * max_area_frac)))
    edge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilate_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (max(1, boundary_dilate), max(1, boundary_dilate)),
    )

    boundary = np.zeros((h, w), dtype=np.uint8)
    kept: list[dict[str, float | int]] = []
    rejected: list[dict[str, float | int | str]] = []
    for idx, raw in enumerate(masks):
        mask = (raw >= 0.5).astype(np.uint8) * 255
        area = int(cv2.countNonZero(mask))
        area_frac = float(area / image_area)
        if area < min_area:
            rejected.append({"index": idx, "area_px": area, "area_frac": area_frac, "reason": "too_small"})
            continue
        if area > max_area:
            rejected.append({"index": idx, "area_px": area, "area_frac": area_frac, "reason": "too_large"})
            continue

        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, edge_kernel)
        edge = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, edge_kernel)
        if boundary_dilate > 1:
            edge = cv2.dilate(edge, dilate_kernel, iterations=1)
        boundary = np.maximum(boundary, edge)
        kept.append({"index": idx, "area_px": area, "area_frac": area_frac})

    if int(np.max(boundary)) > 0:
        boundary = cv2.morphologyEx(boundary, cv2.MORPH_CLOSE, edge_kernel)
    return boundary, kept, rejected


def build_boundary_evidence(
    image_bgr: np.ndarray,
    sam_boundary_u8: np.ndarray,
    params: PlankGridFitParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Weighted float evidence [0,1], thinned ridge uint8 for line detection, and SAM-only norm.
    """
    gray = geometry_luma_u8(image_bgr)
    border_prob = seam_probability_map(gray)

    sam_f = sam_boundary_u8.astype(np.float32)
    sm = float(np.max(sam_f))
    if sm > 0:
        sam_f = sam_f / sm

    evidence = np.clip(
        params.evidence_border_weight * border_prob + params.evidence_sam_weight * sam_f,
        0.0,
        1.0,
    ).astype(np.float32)
    evidence = cv2.GaussianBlur(evidence, (0, 0), sigmaX=0.6)

    flat = evidence.reshape(-1)
    t = float(np.quantile(flat, params.evidence_quantile))
    t = max(t, 0.05)
    binary = (evidence >= t).astype(np.uint8) * 255
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    if int(np.max(binary)) == 0:
        thinned = binary
        return evidence, thinned, sam_f

    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    k = 2 * params.ridge_dist_max + 1
    local_max = cv2.dilate(dist, np.ones((k, k), np.uint8))
    ridge = ((dist >= local_max - 1e-3) & (dist > 0.5)).astype(np.uint8) * 255
    ridge = cv2.morphologyEx(ridge, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return evidence, ridge, sam_f


def detect_weighted_segments(
    ridge_u8: np.ndarray,
    evidence_f32: np.ndarray,
    min_line_length: int,
    max_line_gap: int,
    max_segments: int,
) -> tuple[list[LineSegment], list[float]]:
    if int(np.max(ridge_u8)) == 0:
        return [], []
    edges = cv2.Canny(ridge_u8, 30, 90)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=max(15, min_line_length // 3),
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )
    if lines is None:
        return [], []
    segments: list[LineSegment] = []
    weights: list[float] = []
    h, w = evidence_f32.shape[:2]
    for ln in lines.reshape(-1, 4):
        x1, y1, x2, y2 = map(float, ln)
        mx, my = int((x1 + x2) * 0.5), int((y1 + y2) * 0.5)
        mx = int(np.clip(mx, 0, w - 1))
        my = int(np.clip(my, 0, h - 1))
        ew = float(evidence_f32[my, mx]) + 0.15
        L = math.hypot(x2 - x1, y2 - y1)
        segments.append((x1, y1, x2, y2))
        weights.append(L * ew)
    if len(segments) > max_segments:
        order = np.argsort(-np.array(weights, dtype=np.float64))[:max_segments]
        segments = [segments[int(k)] for k in order]
        weights = [weights[int(k)] for k in order]
    return segments, weights


def angle_histogram_weighted(
    segments: list[LineSegment],
    weights: list[float],
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    edges = np.linspace(0.0, 180.0, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    hist = np.zeros(n_bins, dtype=np.float64)
    for (x1, y1, x2, y2), w in zip(segments, weights, strict=False):
        a = _segment_angle_deg(x1, y1, x2, y2)
        idx = int(np.clip(a / 180.0 * n_bins, 0, n_bins - 1))
        hist[idx] += w
    smooth = hist + 0.5 * np.roll(hist, 1) + 0.5 * np.roll(hist, -1)
    return smooth, centers


def pick_two_orthogonal_peaks(
    hist: np.ndarray,
    centers: np.ndarray,
    ortho_tol_deg: float,
) -> tuple[float, float, float]:
    """Returns (angle_u_deg, angle_v_deg, combined_support)."""
    n = len(hist)
    if n == 0:
        return 0.0, 90.0, 0.0
    order = np.argsort(-hist)
    best_score = -1.0
    best_pair = (float(centers[0]), float(centers[min(1, n - 1)]))
    for _i, iu in enumerate(order[:12]):
        iu = int(iu)
        if hist[iu] <= 0:
            break
        ang_u = float(centers[iu])
        for _j, iv in enumerate(order[:12]):
            iv = int(iv)
            if iu == iv or hist[iv] <= 0:
                continue
            ang_v = float(centers[iv])
            d = _circular_angle_distance_deg(ang_u, ang_v)
            dev = abs(d - 90.0)
            if dev > ortho_tol_deg:
                continue
            score = float(hist[iu] * hist[iv])
            if score > best_score:
                best_score = score
                best_pair = (ang_u, ang_v)
    if best_score < 0:
        i0 = int(order[0])
        ang_u = float(centers[i0])
        target = (ang_u + 90.0) % 180.0
        j = int(np.argmin([_circular_angle_distance_deg(float(c), target) for c in centers]))
        best_pair = (ang_u, float(centers[j]))
        best_score = float(hist[i0] * hist[j])
    return best_pair[0], best_pair[1], best_score


def refine_angle_from_segments(
    segments: list[LineSegment],
    weights: list[float],
    angle_target_deg: float,
    tol_deg: float,
) -> float:
    sx = sy = sw = 0.0
    for (x1, y1, x2, y2), w in zip(segments, weights, strict=False):
        ang = _segment_angle_deg(x1, y1, x2, y2)
        if _circular_angle_distance_deg(ang, angle_target_deg) > tol_deg:
            continue
        dx, dy = x2 - x1, y2 - y1
        L = math.hypot(dx, dy)
        if L < 1e-6:
            continue
        sx += (dx / L) * w
        sy += (dy / L) * w
        sw += w
    if sw < 1e-6:
        return angle_target_deg
    return math.degrees(math.atan2(sy, sx)) % 180.0


def unit_normal_from_angle_deg(angle_deg: float) -> tuple[float, float]:
    phi = math.radians(angle_deg)
    return (-math.sin(phi), math.cos(phi))


def sample_offsets_for_angle(
    segments: list[LineSegment],
    weights: list[float],
    angle_deg: float,
    tol_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    nx, ny = unit_normal_from_angle_deg(angle_deg)
    offs: list[float] = []
    ws: list[float] = []
    for (x1, y1, x2, y2), w in zip(segments, weights, strict=False):
        ang = _segment_angle_deg(x1, y1, x2, y2)
        if _circular_angle_distance_deg(ang, angle_deg) > tol_deg:
            continue
        mx, my = (x1 + x2) * 0.5, (y1 + y2) * 0.5
        offs.append(mx * nx + my * ny)
        ws.append(w)
    if not offs:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    return np.array(offs, dtype=np.float64), np.array(ws, dtype=np.float64)


def _lattice_residuals(offsets: np.ndarray, weights: np.ndarray, spacing: float, phase: float) -> float:
    if spacing < 4.0 or len(offsets) == 0:
        return 1e18
    t = (offsets - phase) / spacing
    err = np.abs(t - np.round(t)) * spacing
    return float(np.sum(weights * err))


def fit_lattice_1d(
    offsets: np.ndarray,
    weights: np.ndarray,
    spacing_min: float,
    spacing_max: float,
    n_spacing: int,
    n_phase: int,
) -> tuple[float, float, float]:
    """Returns (spacing, phase, score) where score is negative mean weighted residual."""
    if len(offsets) < 4:
        return max(spacing_min, 8.0), 0.0, -1e18
    spacings = np.linspace(spacing_min, spacing_max, max(3, n_spacing))
    best = (spacings[len(spacings) // 2], 0.0, -1e18)
    for d in spacings:
        if d < 4.0:
            continue
        for ph in np.linspace(0.0, float(d), max(2, n_phase), endpoint=False):
            res = _lattice_residuals(offsets, weights, float(d), float(ph))
            score = -res / (np.sum(weights) + 1e-6)
            if score > best[2]:
                best = (float(d), float(ph), float(score))
    d0, ph0, _ = best
    for _ in range(12):
        improved = False
        for scale in (0.25, 0.1, 0.03):
            for dd in (-scale * d0, 0.0, scale * d0):
                d1 = max(4.0, d0 + dd)
                for dp in (-scale * d0, 0.0, scale * d0):
                    ph1 = (ph0 + dp) % d1
                    res = _lattice_residuals(offsets, weights, d1, ph1)
                    score = -res / (np.sum(weights) + 1e-6)
                    if score > best[2]:
                        best = (d1, ph1, float(score))
                        d0, ph0 = d1, ph1
                        improved = True
        if not improved:
            break
    return best[0], best[1], best[2]


def line_intersection(
    n1: tuple[float, float],
    rho1: float,
    n2: tuple[float, float],
    rho2: float,
) -> tuple[float, float] | None:
    mat = np.array([[n1[0], n1[1]], [n2[0], n2[1]]], dtype=np.float64)
    det = float(np.linalg.det(mat))
    if abs(det) < 1e-9:
        return None
    rhs = np.array([rho1, rho2], dtype=np.float64)
    x, y = np.linalg.solve(mat, rhs)
    return float(x), float(y)


def plank_corners_from_grid(
    nu: tuple[float, float],
    nv: tuple[float, float],
    u_phase: float,
    v_phase: float,
    u_spacing: float,
    v_spacing: float,
    i: int,
    j: int,
    cells_u: int,
    cells_v: int,
) -> list[tuple[float, float]]:
    """Parallelogram between u-lines i and i+cells_u, v-lines j and j+cells_v."""
    ru = [u_phase + (i + k) * u_spacing for k in (0, cells_u)]
    rv = [v_phase + (j + k) * v_spacing for k in (0, cells_v)]
    p00 = line_intersection(nu, ru[0], nv, rv[0])
    p10 = line_intersection(nu, ru[1], nv, rv[0])
    p11 = line_intersection(nu, ru[1], nv, rv[1])
    p01 = line_intersection(nu, ru[0], nv, rv[1])
    pts = [p00, p10, p11, p01]
    if any(x is None for x in pts):
        return []
    return [(float(p[0]), float(p[1])) for p in pts if p is not None]


def polygon_perimeter_sample_mask(
    corners: list[tuple[float, float]],
    mask_u8: np.ndarray,
    step: int,
) -> tuple[float, int]:
    """Mean mask value along polygon edges (subsampled)."""
    if len(corners) < 3:
        return 0.0, 0
    pts = np.array(corners + [corners[0]], dtype=np.float32)
    h, w = mask_u8.shape[:2]
    vals: list[float] = []
    for k in range(len(pts) - 1):
        p0, p1 = pts[k], pts[k + 1]
        L = float(np.linalg.norm(p1 - p0))
        n = max(1, int(L / step))
        for t in range(n + 1):
            a = t / n
            x = int(np.clip(round(p0[0] * (1 - a) + p1[0] * a), 0, w - 1))
            y = int(np.clip(round(p0[1] * (1 - a) + p1[1] * a), 0, h - 1))
            vals.append(float(mask_u8[y, x]) / 255.0)
    return float(np.mean(vals)) if vals else 0.0, len(vals)


def polygon_centroid_mask_value(
    corners: list[tuple[float, float]],
    mask_f32: np.ndarray,
) -> float:
    cnt = np.array(
        [[int(round(c[0])), int(round(c[1]))] for c in corners],
        dtype=np.int32,
    )
    m = np.zeros(mask_f32.shape[:2], dtype=np.uint8)
    cv2.fillPoly(m, [cnt], 255)
    ys, xs = np.where(m > 0)
    if len(xs) == 0:
        return 0.0
    return float(np.mean(mask_f32[ys, xs]))


def bbox_iou_xyxy(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    bb = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = aa + bb - inter + 1e-9
    return inter / union


def corners_bbox(corners: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    return min(xs), min(ys), max(xs), max(ys)


def _offset_range_on_rect(n: tuple[float, float], width: int, height: int) -> tuple[float, float]:
    vals = []
    for x, y in ((0, 0), (width, 0), (width, height), (0, height)):
        vals.append(float(x) * n[0] + float(y) * n[1])
    return min(vals), max(vals)


def _lattice_index_range(
    n: tuple[float, float],
    phase: float,
    spacing: float,
    width: int,
    height: int,
    cells: int,
    pad: int = 3,
) -> tuple[int, int]:
    lo, hi = _offset_range_on_rect(n, width, height)
    if spacing < 1e-6:
        return -pad, pad
    k0 = int(np.floor((lo - phase) / spacing)) - cells - pad
    k1 = int(np.ceil((hi - phase) / spacing)) + pad
    return k0, k1


def draw_fitted_grid_lines(
    image_bgr: np.ndarray,
    nu: tuple[float, float],
    nv: tuple[float, float],
    u_phase: float,
    v_phase: float,
    u_spacing: float,
    v_spacing: float,
) -> np.ndarray:
    out = image_bgr.copy()
    h, w = out.shape[:2]
    diag = math.hypot(w, h) * 2.0

    def draw_family(n: tuple[float, float], phase: float, spacing: float, color: tuple[int, int, int]) -> None:
        nx, ny = n
        # tangent direction (along the line)
        tx, ty = ny, -nx
        k_min = int(math.floor((-diag - phase) / spacing)) - 2
        k_max = int(math.ceil((diag - phase) / spacing)) + 2
        for k in range(k_min, k_max + 1):
            rho = phase + k * spacing
            px, py = nx * rho, ny * rho
            p1 = (int(px - tx * diag), int(py - ty * diag))
            p2 = (int(px + tx * diag), int(py + ty * diag))
            ok, q1, q2 = cv2.clipLine((0, 0, w, h), p1, p2)
            if ok:
                cv2.line(out, (int(q1[0]), int(q1[1])), (int(q2[0]), int(q2[1])), color, 1, cv2.LINE_AA)

    draw_family(nu, u_phase, u_spacing, (0, 220, 255))
    draw_family(nv, v_phase, v_spacing, (255, 140, 0))
    return out


def draw_grid_line_offsets(
    image_bgr: np.ndarray,
    nu: tuple[float, float],
    nv: tuple[float, float],
    u_lines: list[float],
    v_lines: list[float],
) -> np.ndarray:
    """Draw finite explicit line families instead of expanding an infinite lattice."""
    out = image_bgr.copy()
    h, w = out.shape[:2]
    diag = math.hypot(w, h) * 2.0

    def draw_family(n: tuple[float, float], offsets: list[float], color: tuple[int, int, int]) -> None:
        nx, ny = n
        tx, ty = ny, -nx
        for rho in offsets:
            px, py = nx * rho, ny * rho
            p1 = (int(px - tx * diag), int(py - ty * diag))
            p2 = (int(px + tx * diag), int(py + ty * diag))
            ok, q1, q2 = cv2.clipLine((0, 0, w, h), p1, p2)
            if ok:
                cv2.line(out, (int(q1[0]), int(q1[1])), (int(q2[0]), int(q2[1])), color, 2, cv2.LINE_AA)

    draw_family(nu, u_lines, (0, 220, 255))
    draw_family(nv, v_lines, (255, 140, 0))
    return out


def draw_seam_segments(
    image_bgr: np.ndarray,
    seam_segments: list[JsonDict],
    color: tuple[int, int, int] = (220, 80, 220),
) -> np.ndarray:
    """Draw explicit seam spans, including partial/orphan seams not represented by planks."""
    out = image_bgr.copy()
    for seg in seam_segments:
        p0 = seg.get("p0")
        p1 = seg.get("p1")
        if not isinstance(p0, list) or not isinstance(p1, list) or len(p0) != 2 or len(p1) != 2:
            continue
        cv2.line(
            out,
            (int(round(float(p0[0]))), int(round(float(p0[1])))),
            (int(round(float(p1[0]))), int(round(float(p1[1])))),
            color,
            2,
            cv2.LINE_AA,
        )
    return out


def draw_planks(
    image_bgr: np.ndarray,
    planks: list[JsonDict],
) -> np.ndarray:
    out = image_bgr.copy()
    for p in planks:
        poly = np.array(
            [[[int(round(pt[0])), int(round(pt[1]))] for pt in p["corners"]]],
            dtype=np.int32,
        )
        col = (80, 220, 80) if p.get("orientation") == "A" else (200, 80, 220)
        cv2.polylines(out, [poly[0]], True, col, 2, cv2.LINE_AA)
    return out


def _plank_corners_float32(corners: list[Any]) -> np.ndarray | None:
    if len(corners) != 4:
        return None
    pts: list[list[float]] = []
    for c in corners:
        if not isinstance(c, (list, tuple)) or len(c) != 2:
            return None
        pts.append([float(c[0]), float(c[1])])
    return np.array(pts, dtype=np.float32)


def plank_rectified_size_px(corners: list[Any]) -> tuple[int, int] | None:
    """Width and height (pixels) used to rectify a plank quad, same convention as ``plank_image_crop_bgr``."""
    src = _plank_corners_float32(corners)
    if src is None:
        return None
    c0, c1, _, c3 = src[0], src[1], src[2], src[3]
    w = max(1, int(round(float(np.linalg.norm(c1 - c0)))))
    h = max(1, int(round(float(np.linalg.norm(c3 - c0)))))
    return w, h


def plank_image_crop_bgr(
    image_bgr: np.ndarray,
    corners: list[Any],
) -> np.ndarray | None:
    """Perspective-rectify one plank parallelogram (four corners, cyclic order) to a WxH BGR crop."""
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        return None
    src = _plank_corners_float32(corners)
    if src is None:
        return None
    c0, c1, _, c3 = src[0], src[1], src[2], src[3]
    w = max(1, int(round(float(np.linalg.norm(c1 - c0)))))
    h = max(1, int(round(float(np.linalg.norm(c3 - c0)))))
    dst = np.array(
        [[0.0, 0.0], [w - 1.0, 0.0], [w - 1.0, h - 1.0], [0.0, h - 1.0]],
        dtype=np.float32,
    )
    m = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(
        image_bgr,
        m,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def draw_residual_heatmap(
    image_bgr: np.ndarray,
    evidence_f32: np.ndarray,
    nu: tuple[float, float],
    nv: tuple[float, float],
    u_phase: float,
    v_phase: float,
    u_spacing: float,
    v_spacing: float,
) -> np.ndarray:
    """Per-pixel min distance to nearest fitted lattice line, weighted by evidence."""
    h, w = evidence_f32.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    nxu, nyu = nu
    nxv, nyv = nv
    pu = xx * nxu + yy * nyu
    pv = xx * nxv + yy * nyv
    su = max(float(u_spacing), 1e-3)
    sv = max(float(v_spacing), 1e-3)
    du = np.abs((pu - u_phase + su * 0.5) % su - su * 0.5)
    dv = np.abs((pv - v_phase + sv * 0.5) % sv - sv * 0.5)
    dist = np.minimum(du, dv)
    err = dist * (0.25 + evidence_f32)
    e = err.astype(np.float64)
    e = (e - np.min(e)) / (np.ptp(e) + 1e-9)
    jet = cv2.applyColorMap((e * 255.0).clip(0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    return cv2.addWeighted(image_bgr, 0.55, jet, 0.45, 0.0)


@dataclass
class PlankGridFitResult:
    grid_model: JsonDict
    sam_boundary_u8: np.ndarray
    evidence_f32: np.ndarray
    ridge_u8: np.ndarray
    segments: list[LineSegment] = field(default_factory=list)
    segment_weights: list[float] = field(default_factory=list)
    legacy_geometry: JsonDict = field(default_factory=dict)


def fit_plank_grid(
    image_bgr: np.ndarray,
    masks: np.ndarray,
    params: PlankGridFitParams,
    forced_angles_deg: tuple[float, float] | None = None,
) -> PlankGridFitResult:
    h, w = image_bgr.shape[:2]
    masks = np.stack(
        [cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR) for m in masks],
        axis=0,
    ) if masks.shape[1:] != (h, w) else masks.astype(np.float32)

    sam_boundary, kept_masks, rejected_masks = mask_boundary_map(
        masks,
        params.min_mask_area_frac,
        params.max_mask_area_frac,
        params.boundary_dilate,
    )

    evidence_f32, ridge_u8, _ = build_boundary_evidence(image_bgr, sam_boundary, params)

    min_len = max(18, int(min(h, w) / params.hough_min_len_frac))
    max_gap = max(6, min_len // 2)
    segments, seg_w = detect_weighted_segments(
        ridge_u8, evidence_f32, min_len, max_gap, params.max_hough_segments
    )
    if not segments:
        empty_model: JsonDict = {
            "version": 1,
            "image_size": {"width": w, "height": h},
            "error": "no_line_segments",
            "planks": [],
            "plank_count": 0,
        }
        return PlankGridFitResult(
            grid_model=empty_model,
            sam_boundary_u8=sam_boundary,
            evidence_f32=evidence_f32,
            ridge_u8=ridge_u8,
            segments=[],
            segment_weights=[],
            legacy_geometry={
                "mask_count": int(masks.shape[0]),
                "kept_mask_count": len(kept_masks),
                "rejected_mask_count": len(rejected_masks),
                "boundary_line_count": 0,
            },
        )

    tol = max(12.0, params.orthogonality_tol_deg)
    if forced_angles_deg is not None:
        ang_u = float(forced_angles_deg[0]) % 180.0
        ang_v = float(forced_angles_deg[1]) % 180.0
        if _circular_angle_distance_deg(ang_u, ang_v) < 25.0:
            ang_v = (ang_u + 90.0) % 180.0
    else:
        hist, centers = angle_histogram_weighted(segments, seg_w, params.angle_hist_bins)
        ang_u0, ang_v0, _ = pick_two_orthogonal_peaks(hist, centers, params.orthogonality_tol_deg)
        ang_u = refine_angle_from_segments(segments, seg_w, ang_u0, tol)
        ang_v = refine_angle_from_segments(segments, seg_w, ang_v0, tol)
        if _circular_angle_distance_deg(ang_u, ang_v) < 25.0:
            ang_v = (ang_u + 90.0) % 180.0

    nu = unit_normal_from_angle_deg(ang_u)
    nv = unit_normal_from_angle_deg(ang_v)

    off_u, w_u = sample_offsets_for_angle(segments, seg_w, ang_u, tol)
    off_v, w_v = sample_offsets_for_angle(segments, seg_w, ang_v, tol)

    # Spacing search bounds from offset spread
    def bounds(off: np.ndarray) -> tuple[float, float]:
        if len(off) < 3:
            return 8.0, max(32.0, min(h, w) / 8.0)
        off_sorted = np.sort(off)
        diffs = np.diff(off_sorted)
        diffs = diffs[diffs > 2.0]
        if len(diffs) == 0:
            return 8.0, max(32.0, min(h, w) / 8.0)
        med = float(np.median(diffs))
        return max(4.0, med * 0.35), max(med * 2.5, med + 1.0)

    u_min, u_max = bounds(off_u)
    v_min, v_max = bounds(off_v)
    u_spacing, u_phase, u_score = fit_lattice_1d(
        off_u, w_u, u_min, u_max, params.lattice_spacing_search_steps, params.lattice_phase_steps
    )
    v_spacing, v_phase, v_score = fit_lattice_1d(
        off_v, w_v, v_min, v_max, params.lattice_spacing_search_steps, params.lattice_phase_steps
    )

    # Integer plank dimensions (cells along u and v)
    mask_union = np.zeros((h, w), dtype=np.float32)
    for m in masks:
        mask_union = np.maximum(mask_union, (m >= 0.5).astype(np.float32))

    margin = 6.0
    best_Nu, best_Nv = 2, 2
    best_combo = -1e18
    max_n = params.max_plank_cells
    for Nu in range(1, max_n + 1):
        for Nv in range(1, max_n + 1):
            if Nu == 1 and Nv == 1:
                continue
            scores: list[float] = []
            for i0 in (-2, -1, 0):
                for j0 in (-2, -1, 0):
                    n_try = 0
                    s_acc = 0.0
                    for i in range(i0, i0 + 12):
                        for j in range(j0, j0 + 12):
                            corners = plank_corners_from_grid(
                                nu, nv, u_phase, v_phase, u_spacing, v_spacing, i, j, Nu, Nv
                            )
                            if len(corners) != 4:
                                continue
                            bx = corners_bbox(corners)
                            if (
                                bx[0] < -margin
                                or bx[1] < -margin
                                or bx[2] > w + margin
                                or bx[3] > h + margin
                            ):
                                continue
                            perim, _ = polygon_perimeter_sample_mask(
                                corners, ridge_u8, params.perimeter_sample_step
                            )
                            interior = polygon_centroid_mask_value(corners, mask_union)
                            s = perim - 0.35 * interior
                            s_acc += s
                            n_try += 1
                            if n_try >= 24:
                                break
                        if n_try >= 24:
                            break
                    if n_try > 0:
                        scores.append(s_acc / n_try)
            combo = float(np.median(scores)) if scores else -1e18
            if combo > best_combo:
                best_combo = combo
                best_Nu, best_Nv = Nu, Nv

    if best_combo <= -1e17:
        best_Nu, best_Nv = 2, 6

    i_lo, i_hi = _lattice_index_range(nu, u_phase, u_spacing, w, h, best_Nu)
    j_lo, j_hi = _lattice_index_range(nv, v_phase, v_spacing, w, h, best_Nv)

    planks_a: list[JsonDict] = []
    planks_b: list[JsonDict] = []
    for i in range(i_lo, i_hi + 1):
        for j in range(j_lo, j_hi + 1):
            corners = plank_corners_from_grid(
                nu, nv, u_phase, v_phase, u_spacing, v_spacing, i, j, best_Nu, best_Nv
            )
            if len(corners) != 4:
                continue
            bx = corners_bbox(corners)
            if bx[2] < margin or bx[3] < margin or bx[0] > w - margin or bx[1] > h - margin:
                continue
            perim, _ = polygon_perimeter_sample_mask(corners, ridge_u8, params.perimeter_sample_step)
            interior = polygon_centroid_mask_value(corners, mask_union)
            score = perim - 0.4 * interior - 0.02 * max(0.0, interior - params.mask_interior_support_quantile)
            planks_a.append(
                {
                    "i": i,
                    "j": j,
                    "cells_u": best_Nu,
                    "cells_v": best_Nv,
                    "corners": [[float(c[0]), float(c[1])] for c in corners],
                    "score": float(score),
                    "orientation": "A",
                }
            )

    swapped_Nu, swapped_Nv = best_Nv, best_Nu
    i_lo2, i_hi2 = _lattice_index_range(nu, u_phase, u_spacing, w, h, swapped_Nu)
    j_lo2, j_hi2 = _lattice_index_range(nv, v_phase, v_spacing, w, h, swapped_Nv)
    for i in range(i_lo2, i_hi2 + 1):
        for j in range(j_lo2, j_hi2 + 1):
            corners = plank_corners_from_grid(
                nu, nv, u_phase, v_phase, u_spacing, v_spacing, i, j, swapped_Nu, swapped_Nv
            )
            if len(corners) != 4:
                continue
            bx = corners_bbox(corners)
            if bx[2] < margin or bx[3] < margin or bx[0] > w - margin or bx[1] > h - margin:
                continue
            perim, _ = polygon_perimeter_sample_mask(corners, ridge_u8, params.perimeter_sample_step)
            interior = polygon_centroid_mask_value(corners, mask_union)
            score = perim - 0.4 * interior - 0.02 * max(0.0, interior - params.mask_interior_support_quantile)
            planks_b.append(
                {
                    "i": i,
                    "j": j,
                    "cells_u": swapped_Nu,
                    "cells_v": swapped_Nv,
                    "corners": [[float(c[0]), float(c[1])] for c in corners],
                    "score": float(score),
                    "orientation": "B",
                }
            )

    def nms(planks: list[JsonDict]) -> list[JsonDict]:
        planks = sorted(planks, key=lambda p: float(p["score"]), reverse=True)
        kept: list[JsonDict] = []
        bboxes: list[tuple[float, float, float, float]] = []
        for p in planks:
            bb = corners_bbox([(c[0], c[1]) for c in p["corners"]])
            ok = True
            for bb2 in bboxes:
                if bbox_iou_xyxy(bb, bb2) > params.nms_iou:
                    ok = False
                    break
            if not ok:
                continue
            kept.append(p)
            bboxes.append(bb)
            if len(kept) >= 220:
                break
        return kept

    planks_merged = nms(planks_a + planks_b)

    grid_model: JsonDict = {
        "version": 1,
        "image_size": {"width": w, "height": h},
        "normal_u": [float(nu[0]), float(nu[1])],
        "normal_v": [float(nv[0]), float(nv[1])],
        "angle_u_deg": float(ang_u),
        "angle_v_deg": float(ang_v),
        "u_spacing_px": float(u_spacing),
        "v_spacing_px": float(v_spacing),
        "u_phase_px": float(u_phase),
        "v_phase_px": float(v_phase),
        "u_lattice_score": float(u_score),
        "v_lattice_score": float(v_score),
        "plank_cells_u": int(best_Nu),
        "plank_cells_v": int(best_Nv),
        "plank_cells_alt_u": int(swapped_Nu),
        "plank_cells_alt_v": int(swapped_Nv),
        "planks": planks_merged[:220],
        "plank_count": len(planks_merged),
    }

    legacy_geometry: JsonDict = {
        "mask_count": int(masks.shape[0]),
        "kept_mask_count": len(kept_masks),
        "rejected_mask_count": len(rejected_masks),
        "kept_masks": kept_masks,
        "rejected_masks": rejected_masks,
        "boundary_line_count": len(segments),
        "angle_peaks_deg": [float(ang_u), float(ang_v)],
        "lattice": {
            "u_spacing_px": float(u_spacing),
            "v_spacing_px": float(v_spacing),
            "u_phase_px": float(u_phase),
            "v_phase_px": float(v_phase),
        },
    }

    return PlankGridFitResult(
        grid_model=grid_model,
        sam_boundary_u8=sam_boundary,
        evidence_f32=evidence_f32,
        ridge_u8=ridge_u8,
        segments=segments,
        segment_weights=seg_w,
        legacy_geometry=legacy_geometry,
    )


def _cluster_weighted_offsets(
    offsets: list[float],
    weights: list[float],
    tol_px: float,
    min_support: float,
) -> list[dict[str, float | int]]:
    if not offsets:
        return []
    order = np.argsort(np.array(offsets, dtype=np.float64))
    clusters: list[dict[str, float | int]] = []
    cur_offsets: list[float] = []
    cur_weights: list[float] = []

    def flush() -> None:
        if not cur_offsets:
            return
        sw = float(np.sum(cur_weights))
        if sw < min_support:
            return
        rho = float(np.average(np.array(cur_offsets, dtype=np.float64), weights=np.array(cur_weights, dtype=np.float64)))
        clusters.append({"rho": rho, "support": sw, "count": len(cur_offsets)})

    for idx in order:
        off = float(offsets[int(idx)])
        wt = float(weights[int(idx)])
        if cur_offsets:
            center = float(np.average(np.array(cur_offsets, dtype=np.float64), weights=np.array(cur_weights, dtype=np.float64)))
            if abs(off - center) > tol_px:
                flush()
                cur_offsets = []
                cur_weights = []
        cur_offsets.append(off)
        cur_weights.append(wt)
    flush()
    return sorted(clusters, key=lambda c: float(c["rho"]))


def _median_positive_gap(values: list[float], fallback: float) -> float:
    vals = np.sort(np.array(values, dtype=np.float64))
    if len(vals) < 2:
        return fallback
    gaps = np.diff(vals)
    gaps = gaps[gaps > 1e-6]
    if len(gaps) == 0:
        return fallback
    return float(np.median(gaps))


def _nearest_index(values: list[float], target: float) -> int:
    arr = np.array(values, dtype=np.float64)
    return int(np.argmin(np.abs(arr - target)))


def _extract_sam_contour_segments(
    masks: np.ndarray,
    kept_masks: list[dict[str, float | int]],
    min_segment_len: float,
) -> tuple[list[LineSegment], list[float]]:
    segments: list[LineSegment] = []
    weights: list[float] = []
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    for meta in kept_masks:
        idx = int(meta["index"])
        mask = (masks[idx] >= 0.5).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if len(cnt) < 4:
                continue
            perim = float(cv2.arcLength(cnt, True))
            if perim < min_segment_len * 2.0:
                continue
            epsilon = max(2.0, 0.006 * perim)
            approx = cv2.approxPolyDP(cnt, epsilon, True).reshape(-1, 2)
            if len(approx) < 2:
                continue
            for i, p0 in enumerate(approx):
                p1 = approx[(i + 1) % len(approx)]
                x1, y1 = float(p0[0]), float(p0[1])
                x2, y2 = float(p1[0]), float(p1[1])
                length = math.hypot(x2 - x1, y2 - y1)
                if length < min_segment_len:
                    continue
                segments.append((x1, y1, x2, y2))
                weights.append(length)
    return segments, weights


def _line_offsets_for_family(
    segments: list[LineSegment],
    weights: list[float],
    angle_deg: float,
    tol_deg: float,
    normal: tuple[float, float],
) -> tuple[list[float], list[float]]:
    nx, ny = normal
    offsets: list[float] = []
    support: list[float] = []
    for seg, wt in zip(segments, weights, strict=False):
        if _circular_angle_distance_deg(_segment_angle_deg(*seg), angle_deg) > tol_deg:
            continue
        x1, y1, x2, y2 = seg
        mx, my = (x1 + x2) * 0.5, (y1 + y2) * 0.5
        offsets.append(mx * nx + my * ny)
        support.append(wt)
    return offsets, support


def _with_image_border_rails(
    lines: list[float],
    normal: tuple[float, float],
    width: int,
    height: int,
    merge_tol_px: float,
) -> list[float]:
    lo, hi = _offset_range_on_rect(normal, width, height)
    out = list(lines)
    for rail in (lo, hi):
        if not any(abs(rail - rho) <= merge_tol_px for rho in out):
            out.append(float(rail))
    return sorted(out)


def _clip_line_for_offset(
    normal: tuple[float, float],
    rho: float,
    width: int,
    height: int,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    nx, ny = normal
    tx, ty = ny, -nx
    diag = math.hypot(width, height) * 2.0
    px, py = nx * rho, ny * rho
    p1 = (int(px - tx * diag), int(py - ty * diag))
    p2 = (int(px + tx * diag), int(py + ty * diag))
    ok, q1, q2 = cv2.clipLine((0, 0, width, height), p1, p2)
    if not ok:
        return None
    return (float(q1[0]), float(q1[1])), (float(q2[0]), float(q2[1]))


def _snap_segment_to_offset(seg: LineSegment, normal: tuple[float, float], rho: float) -> LineSegment:
    nx, ny = normal
    x1, y1, x2, y2 = seg
    d1 = rho - (x1 * nx + y1 * ny)
    d2 = rho - (x2 * nx + y2 * ny)
    return (x1 + nx * d1, y1 + ny * d1, x2 + nx * d2, y2 + ny * d2)


def _build_supported_seam_segments(
    segments: list[LineSegment],
    weights: list[float],
    angle_u: float,
    angle_v: float,
    nu: tuple[float, float],
    nv: tuple[float, float],
    u_lines: list[float],
    v_lines: list[float],
    angle_tol_deg: float,
    snap_tol_px: float,
    min_len_px: float,
    width: int,
    height: int,
) -> list[JsonDict]:
    seam_segments: list[JsonDict] = []

    def add_family(
        family: str,
        angle: float,
        normal: tuple[float, float],
        lines: list[float],
    ) -> None:
        if not lines:
            return
        line_arr = np.array(lines, dtype=np.float64)
        for seg, wt in zip(segments, weights, strict=False):
            if _circular_angle_distance_deg(_segment_angle_deg(*seg), angle) > angle_tol_deg:
                continue
            x1, y1, x2, y2 = seg
            length = math.hypot(x2 - x1, y2 - y1)
            if length < min_len_px:
                continue
            mx, my = (x1 + x2) * 0.5, (y1 + y2) * 0.5
            off = mx * normal[0] + my * normal[1]
            j = int(np.argmin(np.abs(line_arr - off)))
            rho = float(line_arr[j])
            if abs(rho - off) > snap_tol_px:
                continue
            sx1, sy1, sx2, sy2 = _snap_segment_to_offset(seg, normal, rho)
            if max(sx1, sx2) < -2 or min(sx1, sx2) > width + 2 or max(sy1, sy2) < -2 or min(sy1, sy2) > height + 2:
                continue
            seam_segments.append(
                {
                    "family": family,
                    "rho": rho,
                    "p0": [float(np.clip(sx1, 0, width - 1)), float(np.clip(sy1, 0, height - 1))],
                    "p1": [float(np.clip(sx2, 0, width - 1)), float(np.clip(sy2, 0, height - 1))],
                    "support": float(wt),
                    "source": "sam_boundary",
                }
            )

    add_family("u", angle_u, nu, u_lines)
    add_family("v", angle_v, nv, v_lines)

    # Add crop rails as full spans; they help close planks at image edges without inventing interior seams.
    for family, normal, lines in (("u", nu, u_lines), ("v", nv, v_lines)):
        lo, hi = _offset_range_on_rect(normal, width, height)
        for rho in (lo, hi):
            if not any(abs(float(rho) - float(line_rho)) <= snap_tol_px for line_rho in lines):
                continue
            clipped = _clip_line_for_offset(normal, float(rho), width, height)
            if clipped is None:
                continue
            p0, p1 = clipped
            seam_segments.append(
                {
                    "family": family,
                    "rho": float(rho),
                    "p0": [p0[0], p0[1]],
                    "p1": [p1[0], p1[1]],
                    "support": float(max(width, height)),
                    "source": "image_border",
                }
            )

    return seam_segments


def _polygon_mask_overlap(corners: list[tuple[float, float]], mask_f32: np.ndarray) -> float:
    if len(corners) != 4:
        return 0.0
    h, w = mask_f32.shape[:2]
    pts = np.array(
        [[[int(np.clip(round(x), 0, w - 1)), int(np.clip(round(y), 0, h - 1))] for x, y in corners]],
        dtype=np.int32,
    )
    poly = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(poly, [pts[0]], 255)
    area = int(cv2.countNonZero(poly))
    if area == 0:
        return 0.0
    ys, xs = np.where(poly > 0)
    return float(np.mean(mask_f32[ys, xs]))


def fit_sam_border_grid(
    image_bgr: np.ndarray,
    masks: np.ndarray,
    params: PlankGridFitParams,
) -> PlankGridFitResult:
    """Fit a sparse plank grid from SAM mask borders, without image-texture Hough lines."""
    h, w = image_bgr.shape[:2]
    masks = np.stack(
        [cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR) for m in masks],
        axis=0,
    ) if masks.shape[1:] != (h, w) else masks.astype(np.float32)

    sam_boundary, kept_masks, rejected_masks = mask_boundary_map(
        masks,
        params.min_mask_area_frac,
        params.max_mask_area_frac,
        params.boundary_dilate,
    )
    evidence_f32 = (sam_boundary.astype(np.float32) / 255.0).clip(0.0, 1.0)
    ridge_u8 = sam_boundary.copy()

    min_dim = min(h, w)
    min_segment_len = max(64.0, min_dim * 0.09)
    max_segment_gap = max(12, int(round(min_dim * 0.025)))
    segments, seg_w = detect_weighted_segments(
        ridge_u8,
        evidence_f32,
        int(round(min_segment_len)),
        max_segment_gap,
        params.max_hough_segments,
    )
    if len(segments) < 4:
        contour_segments, contour_w = _extract_sam_contour_segments(masks, kept_masks, max(32.0, min_dim * 0.045))
        segments.extend(contour_segments)
        seg_w.extend(contour_w)
    if len(segments) < 4:
        empty_model: JsonDict = {
            "version": 1,
            "image_size": {"width": w, "height": h},
            "error": "not_enough_sam_border_segments",
            "planks": [],
            "plank_count": 0,
        }
        return PlankGridFitResult(
            grid_model=empty_model,
            sam_boundary_u8=sam_boundary,
            evidence_f32=evidence_f32,
            ridge_u8=ridge_u8,
            segments=segments,
            segment_weights=seg_w,
            legacy_geometry={
                "mask_count": int(masks.shape[0]),
                "kept_mask_count": len(kept_masks),
                "rejected_mask_count": len(rejected_masks),
                "boundary_line_count": len(segments),
                "sam_border_grid_debug": {"reason": "not_enough_sam_border_segments"},
            },
        )

    hist, centers = angle_histogram_weighted(segments, seg_w, params.angle_hist_bins)
    ang_u0, ang_v0, angle_support = pick_two_orthogonal_peaks(hist, centers, params.orthogonality_tol_deg)
    tol = max(10.0, params.orthogonality_tol_deg)
    ang_u = refine_angle_from_segments(segments, seg_w, ang_u0, tol)
    ang_v = refine_angle_from_segments(segments, seg_w, ang_v0, tol)
    if abs(_circular_angle_distance_deg(ang_u, ang_v) - 90.0) > params.orthogonality_tol_deg:
        ang_v = (ang_u + 90.0) % 180.0

    nu = unit_normal_from_angle_deg(ang_u)
    nv = unit_normal_from_angle_deg(ang_v)
    off_u, w_u = _line_offsets_for_family(segments, seg_w, ang_u, tol, nu)
    off_v, w_v = _line_offsets_for_family(segments, seg_w, ang_v, tol, nv)

    cluster_tol = max(8.0, min_dim * 0.014)
    min_cluster_support = max(min_segment_len * 0.75, min_dim * 0.04)
    clusters_u = _cluster_weighted_offsets(off_u, w_u, cluster_tol, min_cluster_support)
    clusters_v = _cluster_weighted_offsets(off_v, w_v, cluster_tol, min_cluster_support)
    u_lines = [float(c["rho"]) for c in clusters_u]
    v_lines = [float(c["rho"]) for c in clusters_v]

    min_spacing = max(32.0, min_dim * 0.045)
    u_lines = [rho for i, rho in enumerate(u_lines) if i == 0 or abs(rho - u_lines[i - 1]) >= min_spacing * 0.5]
    v_lines = [rho for i, rho in enumerate(v_lines) if i == 0 or abs(rho - v_lines[i - 1]) >= min_spacing * 0.5]
    u_lines = _with_image_border_rails(u_lines, nu, w, h, cluster_tol)
    v_lines = _with_image_border_rails(v_lines, nv, w, h, cluster_tol)

    if len(u_lines) < 2 or len(v_lines) < 2:
        empty_model = {
            "version": 1,
            "image_size": {"width": w, "height": h},
            "error": "not_enough_snapped_sam_grid_lines",
            "planks": [],
            "plank_count": 0,
        }
        return PlankGridFitResult(
            grid_model=empty_model,
            sam_boundary_u8=sam_boundary,
            evidence_f32=evidence_f32,
            ridge_u8=ridge_u8,
            segments=segments,
            segment_weights=seg_w,
            legacy_geometry={
                "mask_count": int(masks.shape[0]),
                "kept_mask_count": len(kept_masks),
                "rejected_mask_count": len(rejected_masks),
                "boundary_line_count": len(segments),
                "sam_border_grid_debug": {
                    "reason": "not_enough_snapped_sam_grid_lines",
                    "segment_count": len(segments),
                    "u_line_count": len(u_lines),
                    "v_line_count": len(v_lines),
                },
            },
        )

    seam_segments = _build_supported_seam_segments(
        segments,
        seg_w,
        ang_u,
        ang_v,
        nu,
        nv,
        u_lines,
        v_lines,
        tol,
        cluster_tol * 1.75,
        max(24.0, min_dim * 0.035),
        w,
        h,
    )

    mask_union = np.zeros((h, w), dtype=np.float32)
    for m in masks:
        mask_union = np.maximum(mask_union, (m >= 0.5).astype(np.float32))

    planks: list[JsonDict] = []
    for meta in kept_masks:
        idx = int(meta["index"])
        ys, xs = np.where(masks[idx] >= 0.5)
        if len(xs) < 16:
            continue
        pu = xs.astype(np.float64) * nu[0] + ys.astype(np.float64) * nu[1]
        pv = xs.astype(np.float64) * nv[0] + ys.astype(np.float64) * nv[1]
        u0 = _nearest_index(u_lines, float(np.quantile(pu, 0.02)))
        u1 = _nearest_index(u_lines, float(np.quantile(pu, 0.98)))
        v0 = _nearest_index(v_lines, float(np.quantile(pv, 0.02)))
        v1 = _nearest_index(v_lines, float(np.quantile(pv, 0.98)))
        if u0 == u1 or v0 == v1:
            continue
        if u0 > u1:
            u0, u1 = u1, u0
        if v0 > v1:
            v0, v1 = v1, v0
        ru0, ru1 = u_lines[u0], u_lines[u1]
        rv0, rv1 = v_lines[v0], v_lines[v1]
        pts = [
            line_intersection(nu, ru0, nv, rv0),
            line_intersection(nu, ru1, nv, rv0),
            line_intersection(nu, ru1, nv, rv1),
            line_intersection(nu, ru0, nv, rv1),
        ]
        if any(p is None for p in pts):
            continue
        corners = [p for p in pts if p is not None]
        bx = corners_bbox(corners)
        if bx[2] < -8 or bx[3] < -8 or bx[0] > w + 8 or bx[1] > h + 8:
            continue
        perim, _ = polygon_perimeter_sample_mask(corners, sam_boundary, params.perimeter_sample_step)
        own_overlap = _polygon_mask_overlap(corners, (masks[idx] >= 0.5).astype(np.float32))
        union_overlap = _polygon_mask_overlap(corners, mask_union)
        score = perim + 0.6 * own_overlap - 0.15 * max(0.0, union_overlap - own_overlap)
        if own_overlap < 0.25 and perim < 0.12:
            continue
        planks.append(
            {
                "mask_index": idx,
                "i": u0,
                "j": v0,
                "cells_u": u1 - u0,
                "cells_v": v1 - v0,
                "corners": [[float(c[0]), float(c[1])] for c in corners],
                "score": float(score),
                "orientation": "SAM",
            }
        )

    def nms(planks_in: list[JsonDict]) -> list[JsonDict]:
        ordered = sorted(planks_in, key=lambda p: float(p["score"]), reverse=True)
        kept: list[JsonDict] = []
        bboxes: list[tuple[float, float, float, float]] = []
        for p in ordered:
            bb = corners_bbox([(c[0], c[1]) for c in p["corners"]])
            if any(bbox_iou_xyxy(bb, existing) > params.nms_iou for existing in bboxes):
                continue
            kept.append(p)
            bboxes.append(bb)
            if len(kept) >= 250:
                break
        return kept

    planks = nms(planks)
    fallback_spacing = max(min_dim / 6.0, min_spacing)
    u_spacing = max(min_spacing, _median_positive_gap(u_lines, fallback_spacing))
    v_spacing = max(min_spacing, _median_positive_gap(v_lines, fallback_spacing))
    u_phase = float(u_lines[0] % u_spacing)
    v_phase = float(v_lines[0] % v_spacing)
    grid_model: JsonDict = {
        "version": 1,
        "image_size": {"width": w, "height": h},
        "normal_u": [float(nu[0]), float(nu[1])],
        "normal_v": [float(nv[0]), float(nv[1])],
        "angle_u_deg": float(ang_u),
        "angle_v_deg": float(ang_v),
        "u_spacing_px": float(u_spacing),
        "v_spacing_px": float(v_spacing),
        "u_phase_px": u_phase,
        "v_phase_px": v_phase,
        "u_lines_px": u_lines,
        "v_lines_px": v_lines,
        "min_spacing_px": float(min_spacing),
        "plank_cells_u": 1,
        "plank_cells_v": 1,
        "plank_cells_alt_u": 1,
        "plank_cells_alt_v": 1,
        "seam_segments": seam_segments,
        "seam_segment_count": len(seam_segments),
        "planks": planks,
        "plank_count": len(planks),
    }
    debug: JsonDict = {
        "segment_count": len(segments),
        "angle_support": float(angle_support),
        "angle_clusters_deg": [float(ang_u), float(ang_v)],
        "offset_cluster_tolerance_px": float(cluster_tol),
        "min_cluster_support": float(min_cluster_support),
        "min_spacing_px": float(min_spacing),
        "u_line_count": len(u_lines),
        "v_line_count": len(v_lines),
        "seam_segment_count": len(seam_segments),
        "u_lines_px": u_lines,
        "v_lines_px": v_lines,
    }
    legacy_geometry: JsonDict = {
        "mask_count": int(masks.shape[0]),
        "kept_mask_count": len(kept_masks),
        "rejected_mask_count": len(rejected_masks),
        "kept_masks": kept_masks,
        "rejected_masks": rejected_masks,
        "boundary_line_count": len(segments),
        "angle_peaks_deg": [float(ang_u), float(ang_v)],
        "lattice": {
            "u_spacing_px": float(u_spacing),
            "v_spacing_px": float(v_spacing),
            "u_phase_px": u_phase,
            "v_phase_px": v_phase,
        },
        "sam_border_grid_debug": debug,
    }
    return PlankGridFitResult(
        grid_model=grid_model,
        sam_boundary_u8=sam_boundary,
        evidence_f32=evidence_f32,
        ridge_u8=ridge_u8,
        segments=segments,
        segment_weights=seg_w,
        legacy_geometry=legacy_geometry,
    )
