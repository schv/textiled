"""Write SAM + grid debug images and JSON summaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import cv2
import numpy as np

from fitting.plank_grid_fit import (
    draw_fitted_grid_lines,
    draw_grid_line_offsets,
    draw_planks,
    draw_residual_heatmap,
    draw_seam_segments,
    plank_image_crop_bgr,
    plank_rectified_size_px,
)
from workflow_types import GridFitOutput, JsonDict, SamSegmentationResult


def mask_composite(masks: np.ndarray) -> np.ndarray:
    m = masks.astype(np.float32)
    acc = m.sum(axis=0)
    mx = float(np.max(acc))
    if mx > 0.0:
        acc = acc / mx
    return (acc * 255.0).astype(np.uint8)


def write_lighting_normalized_preview(out_dir: Path, stem: str, sam_input_bgr: np.ndarray) -> Path:
    path = out_dir / f"{stem}_lighting_normalized_for_sam.png"
    cv2.imwrite(str(path), sam_input_bgr)
    return path


def write_sam_plot(out_dir: Path, stem: str, plot_bgr: np.ndarray) -> Path:
    path = out_dir / f"{stem}_sam3_plot.png"
    cv2.imwrite(str(path), plot_bgr)
    return path


def write_mask_sum(out_dir: Path, stem: str, masks: np.ndarray) -> Path:
    path = out_dir / f"{stem}_sam3_mask_sum.png"
    cv2.imwrite(str(path), mask_composite(masks))
    return path


def write_grid_fit_artifacts(
    out_dir: Path,
    stem: str,
    image_bgr: np.ndarray,
    grid: GridFitOutput,
) -> dict[str, Path]:
    """Writes boundary, evidence, ridge, grid lines, planks, overlay, residuals, grid_model.json."""
    gm = grid.grid_model
    paths: dict[str, Path] = {}

    paths["boundary"] = out_dir / f"{stem}_sam3_boundary.png"
    cv2.imwrite(str(paths["boundary"]), grid.sam_boundary_u8)

    paths["evidence"] = out_dir / f"{stem}_boundary_evidence.png"
    cv2.imwrite(
        str(paths["evidence"]),
        (np.clip(grid.evidence_f32, 0.0, 1.0) * 255.0).astype(np.uint8),
    )

    paths["ridge"] = out_dir / f"{stem}_boundary_ridge.png"
    cv2.imwrite(str(paths["ridge"]), grid.ridge_u8)

    if grid.status == "failed" or "normal_u" not in gm:
        paths["grid_lines"] = out_dir / f"{stem}_grid_lines.png"
        cv2.imwrite(str(paths["grid_lines"]), image_bgr.copy())
        paths["planks"] = out_dir / f"{stem}_plank_rectangles.png"
        cv2.imwrite(str(paths["planks"]), image_bgr.copy())
        paths["overlay"] = out_dir / f"{stem}_sam3_grid_geometry.png"
        cv2.imwrite(str(paths["overlay"]), image_bgr.copy())
        paths["residuals"] = out_dir / f"{stem}_grid_residuals.png"
        cv2.imwrite(str(paths["residuals"]), image_bgr.copy())
        paths["grid_model"] = out_dir / f"{stem}_grid_model.json"
        paths["grid_model"].write_text(json.dumps(gm, indent=2), encoding="utf-8")
        summary: JsonDict = {
            **grid.legacy_geometry,
            "grid_model_path": paths["grid_model"].name,
            "grid_fit_strategy": grid.strategy,
            "grid_fit_status": grid.status,
        }
        if grid.debug_json:
            summary["grid_fit_debug"] = grid.debug_json
        paths["summary"] = out_dir / f"{stem}_sam3_grid_geometry.json"
        paths["summary"].write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return paths

    nu = (float(gm["normal_u"][0]), float(gm["normal_u"][1]))
    nv = (float(gm["normal_v"][0]), float(gm["normal_v"][1]))
    u_lines = gm.get("u_lines_px")
    v_lines = gm.get("v_lines_px")
    if isinstance(u_lines, list) and isinstance(v_lines, list):
        grid_lines = draw_grid_line_offsets(
            image_bgr,
            nu,
            nv,
            [float(v) for v in u_lines],
            [float(v) for v in v_lines],
        )
    else:
        grid_lines = draw_fitted_grid_lines(
            image_bgr,
            nu,
            nv,
            float(gm["u_phase_px"]),
            float(gm["v_phase_px"]),
            float(gm["u_spacing_px"]),
            float(gm["v_spacing_px"]),
        )
    paths["grid_lines"] = out_dir / f"{stem}_grid_lines.png"
    cv2.imwrite(str(paths["grid_lines"]), grid_lines)

    seam_segments = gm.get("seam_segments") or []
    seam_base = (
        draw_seam_segments(image_bgr, seam_segments)
        if isinstance(seam_segments, list)
        else image_bgr.copy()
    )
    seam_grid = (
        draw_seam_segments(grid_lines, seam_segments)
        if isinstance(seam_segments, list)
        else grid_lines
    )

    planks = gm.get("planks") or []
    if isinstance(planks, list):
        paths["planks"] = out_dir / f"{stem}_plank_rectangles.png"
        cv2.imwrite(str(paths["planks"]), draw_planks(seam_base, planks[:250]))
        paths["overlay"] = out_dir / f"{stem}_sam3_grid_geometry.png"
        cv2.imwrite(str(paths["overlay"]), draw_planks(seam_grid, planks[:150]))
    else:
        paths["planks"] = out_dir / f"{stem}_plank_rectangles.png"
        cv2.imwrite(str(paths["planks"]), seam_base)
        paths["overlay"] = out_dir / f"{stem}_sam3_grid_geometry.png"
        cv2.imwrite(str(paths["overlay"]), seam_grid)

    paths["residuals"] = out_dir / f"{stem}_grid_residuals.png"
    cv2.imwrite(
        str(paths["residuals"]),
        draw_residual_heatmap(
            image_bgr,
            grid.evidence_f32,
            nu,
            nv,
            float(gm["u_phase_px"]),
            float(gm["v_phase_px"]),
            float(gm["u_spacing_px"]),
            float(gm["v_spacing_px"]),
        ),
    )

    paths["grid_model"] = out_dir / f"{stem}_grid_model.json"
    paths["grid_model"].write_text(json.dumps(gm, indent=2), encoding="utf-8")

    summary: JsonDict = {
        **grid.legacy_geometry,
        "image_size": gm.get("image_size"),
        "grid_model_path": paths["grid_model"].name,
        "plank_count": gm.get("plank_count", 0),
        "grid_fit_strategy": grid.strategy,
        "grid_fit_status": grid.status,
    }
    if grid.debug_json:
        summary["grid_fit_debug"] = grid.debug_json

    paths["summary"] = out_dir / f"{stem}_sam3_grid_geometry.json"
    paths["summary"].write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return paths


# Max spread across a kept dimension (width or height): max/min <= this factor ≈ ±1%.
_PLANK_DIM_RATIO_MAX = 1.03


def _largest_ratio_cluster_indices(dim_value_and_idx: list[tuple[float, int]]) -> set[int]:
    """Largest subset whose ratio max/min is at most ``_PLANK_DIM_RATIO_MAX`` (sliding window on sorted values)."""
    if not dim_value_and_idx:
        return set()
    sorted_vals = sorted(dim_value_and_idx, key=lambda x: x[0])
    tol = _PLANK_DIM_RATIO_MAX
    best: set[int] = set()
    best_n = 0
    left = 0
    n = len(sorted_vals)
    for r in range(n):
        wr = sorted_vals[r][0]
        while wr > sorted_vals[left][0] * tol + 1e-12:
            left += 1
        cand = {sorted_vals[i][1] for i in range(left, r + 1)}
        cn = len(cand)
        if cn > best_n:
            best_n = cn
            best = cand
    return best


def write_plank_extractions(
    out_dir: Path,
    stem: str,
    image_bgr: np.ndarray,
    grid: GridFitOutput,
) -> Path | None:
    """Write PNG crops under ``{out_dir}/{stem}_planks/``; outliers go to ``discarded/``.

    Keeps the largest subset whose rectified widths **or** heights are mutually within ±1%
    (``max/min <= _PLANK_DIM_RATIO_MAX``); tie-break favors the width-consistent subset.
    """
    if grid.status == "failed":
        return None
    gm = grid.grid_model
    planks = gm.get("planks")
    if not isinstance(planks, list) or not planks:
        return None

    sized: list[tuple[int, JsonDict, int, int]] = []
    for idx, p in enumerate(planks):
        if not isinstance(p, dict):
            continue
        p_dict = cast(JsonDict, p)
        corners = p_dict.get("corners")
        if not isinstance(corners, list):
            continue
        wh = plank_rectified_size_px(corners)
        if wh is None:
            continue
        w, h = wh
        sized.append((idx, p_dict, w, h))

    if not sized:
        return None

    by_w = [(float(w), i) for i, _, w, _ in sized]
    by_h = [(float(h), i) for i, _, _, h in sized]
    w_keep = _largest_ratio_cluster_indices(by_w)
    h_keep = _largest_ratio_cluster_indices(by_h)
    if len(w_keep) > len(h_keep) or (len(w_keep) == len(h_keep)):
        keep_idx = w_keep
    else:
        keep_idx = h_keep

    plank_dir = out_dir / f"{stem}_planks"
    plank_dir.mkdir(parents=True, exist_ok=True)
    disc_dir = plank_dir / "discarded"
    n_written = 0

    for idx, p_dict, _w, _h in sized:
        corners = p_dict.get("corners")
        if not isinstance(corners, list):
            continue
        crop = plank_image_crop_bgr(image_bgr, corners)
        if crop is None:
            continue
        i_raw, j_raw = p_dict.get("i"), p_dict.get("j")
        if isinstance(i_raw, (int, float)) and isinstance(j_raw, (int, float)):
            name = f"{stem}_plank_{int(i_raw):04d}_{int(j_raw):04d}_{idx:04d}.png"
        else:
            name = f"{stem}_plank_{idx:04d}.png"
        if idx in keep_idx:
            path = plank_dir / name
        else:
            disc_dir.mkdir(parents=True, exist_ok=True)
            path = disc_dir / name
        cv2.imwrite(str(path), crop)
        n_written += 1

    if n_written == 0:
        try:
            plank_dir.rmdir()
        except OSError:
            pass
        return None
    return plank_dir


def write_sam_segmentation_artifacts(
    out_dir: Path,
    stem: str,
    plot_bgr: np.ndarray,
    sam: SamSegmentationResult | None,
) -> None:
    write_sam_plot(out_dir, stem, plot_bgr)
    if sam is not None:
        write_mask_sum(out_dir, stem, sam.masks)
