"""Pluggable grid fitting strategies (function-level, no scenario framework)."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

import fitting.plank_grid_fit as pgf
from fitting.plank_grid_fit import PlankGridFitParams, PlankGridFitResult, fit_plank_grid, fit_sam_border_grid
from workflow_types import GridFitOutput


def plank_grid_fit_params(
    *,
    min_mask_area_frac: float,
    max_mask_area_frac: float,
    boundary_dilate: int,
    angle_tol_deg: float,
) -> PlankGridFitParams:
    return PlankGridFitParams(
        min_mask_area_frac=min_mask_area_frac,
        max_mask_area_frac=max_mask_area_frac,
        boundary_dilate=boundary_dilate,
        orthogonality_tol_deg=max(12.0, angle_tol_deg),
    )


def plank_grid_fit_params_from_args(args: argparse.Namespace) -> PlankGridFitParams:
    return plank_grid_fit_params(
        min_mask_area_frac=args.min_mask_area_frac,
        max_mask_area_frac=args.max_mask_area_frac,
        boundary_dilate=args.boundary_dilate,
        angle_tol_deg=args.angle_tol_deg,
    )


def _plank_result_to_grid_output(strategy: str, result: PlankGridFitResult, debug_json: dict | None = None) -> GridFitOutput:
    gm = dict(result.grid_model)
    gm["strategy"] = strategy
    err = gm.get("error")
    status = "failed" if err else "ok"
    return GridFitOutput(
        strategy=strategy,
        status=status,
        grid_model=gm,
        sam_boundary_u8=result.sam_boundary_u8,
        evidence_f32=result.evidence_f32,
        ridge_u8=result.ridge_u8,
        legacy_geometry=dict(result.legacy_geometry),
        debug_json=debug_json,
    )


class GridFitter(Protocol):
    name: str

    def fit(self, image_bgr: np.ndarray, masks: np.ndarray, params: PlankGridFitParams) -> GridFitOutput: ...


@dataclass
class LatticeGridFitter:
    """Histogram peaks + lattice search (existing `fit_plank_grid`)."""

    name: str = "current_lattice"

    def fit(self, image_bgr: np.ndarray, masks: np.ndarray, params: PlankGridFitParams) -> GridFitOutput:
        result = fit_plank_grid(image_bgr, masks, params)
        return _plank_result_to_grid_output(self.name, result)


@dataclass
class SamBorderGridFitter:
    """SAM contour segments + snapped sparse grid lines."""

    name: str = "sam_border_grid"

    def fit(self, image_bgr: np.ndarray, masks: np.ndarray, params: PlankGridFitParams) -> GridFitOutput:
        result = fit_sam_border_grid(image_bgr, masks, params)
        debug = result.legacy_geometry.get("sam_border_grid_debug")
        return _plank_result_to_grid_output(self.name, result, debug_json=debug if isinstance(debug, dict) else None)


@dataclass
class RansacLineGridFitter:
    """RANSAC-style robust angle hypotheses, then reuse lattice fit from those angles."""

    name: str = "ransac_lines"
    n_angle_iterations: int = 450
    angle_inlier_tol_deg: float = 12.0
    u_band_exclude_deg: float = 14.0
    seed: int = 42

    def fit(self, image_bgr: np.ndarray, masks: np.ndarray, params: PlankGridFitParams) -> GridFitOutput:
        h, w = image_bgr.shape[:2]
        masks_f = (
            np.stack(
                [cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR) for m in masks],
                axis=0,
            )
            if masks.shape[1:] != (h, w)
            else masks.astype(np.float32)
        )
        sam_boundary, _, _ = pgf.mask_boundary_map(
            masks_f,
            params.min_mask_area_frac,
            params.max_mask_area_frac,
            params.boundary_dilate,
        )
        evidence_f32, ridge_u8, _ = pgf.build_boundary_evidence(image_bgr, sam_boundary, params)
        min_len = max(18, int(min(h, w) / params.hough_min_len_frac))
        max_gap = max(6, min_len // 2)
        segments, seg_w = pgf.detect_weighted_segments(
            ridge_u8, evidence_f32, min_len, max_gap, params.max_hough_segments
        )
        if not segments:
            r = fit_plank_grid(image_bgr, masks, params)
            return _plank_result_to_grid_output(
                self.name,
                r,
                debug_json={"angles_deg": None, "reason": "no_segments_before_ransac"},
            )

        rng = np.random.default_rng(self.seed)
        ang_u = self._ransac_best_angle(segments, seg_w, rng, self.n_angle_iterations, self.angle_inlier_tol_deg)

        segs_v: list = []
        w_v: list = []
        for seg, wt in zip(segments, seg_w, strict=False):
            if pgf._circular_angle_distance_deg(pgf._segment_angle_deg(*seg), ang_u) > self.u_band_exclude_deg:
                segs_v.append(seg)
                w_v.append(wt)
        if len(segs_v) < 4:
            ang_v = (ang_u + 90.0) % 180.0
            debug = {"angles_deg": [ang_u, ang_v], "second_family": "orthogonal_fallback"}
        else:
            ang_v = self._ransac_best_angle(
                segs_v,
                w_v,
                rng,
                min(self.n_angle_iterations, 300),
                self.angle_inlier_tol_deg,
            )
            ang_v = float(ang_v) % 180.0
            debug = {"angles_deg": [ang_u, ang_v], "second_family": "ransac"}

        if pgf._circular_angle_distance_deg(ang_u, ang_v) < 25.0:
            ang_v = (ang_u + 90.0) % 180.0

        result = fit_plank_grid(image_bgr, masks, params, forced_angles_deg=(ang_u, ang_v))
        return _plank_result_to_grid_output(self.name, result, debug_json=debug)

    def _ransac_best_angle(
        self,
        segments: list,
        weights: list[float],
        rng: np.random.Generator,
        n_iter: int,
        inlier_tol_deg: float,
    ) -> float:
        w_arr = np.array(weights, dtype=np.float64)
        p = w_arr / (np.sum(w_arr) + 1e-9)
        idx = np.arange(len(segments))
        best_ang = 0.0
        best_score = -1.0
        for _ in range(n_iter):
            j = int(rng.choice(idx, p=p))
            hyp = pgf._segment_angle_deg(*segments[j])
            score = 0.0
            for seg, wt in zip(segments, weights, strict=False):
                if pgf._circular_angle_distance_deg(pgf._segment_angle_deg(*seg), hyp) <= inlier_tol_deg:
                    score += wt
            if score > best_score:
                best_score = score
                best_ang = hyp
        return float(pgf.refine_angle_from_segments(segments, weights, best_ang, inlier_tol_deg))


_REGISTRY: dict[str, GridFitter] = {
    "current_lattice": LatticeGridFitter(),
    "ransac_lines": RansacLineGridFitter(),
    "sam_border_grid": SamBorderGridFitter(),
}


def get_grid_fitter(name: str) -> GridFitter:
    key = (name or "current_lattice").strip().lower()
    if key not in _REGISTRY:
        raise ValueError(
            f"unknown grid fitter {name!r}; choose one of: {', '.join(sorted(_REGISTRY))}"
        )
    return _REGISTRY[key]


def list_grid_fitters() -> list[str]:
    return sorted(_REGISTRY)
