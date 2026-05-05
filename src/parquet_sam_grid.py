"""
Parquet: preprocess (denoise + lighting + SAM letterbox) → SAM masks → plank grid.

CLI (automated): **`--image`**, **`--text`**, optional **`--device`**. Everything else is fixed below.
See `NOTES.md`.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np

from sam.artifacts import (
    write_grid_fit_artifacts,
    write_lighting_normalized_preview,
    write_plank_extractions,
    write_sam_segmentation_artifacts,
)
from sam.backend import load_image_bgr, pick_device, resolve_model_path, run_sam3_segmentation
from sam.grid_fitters import get_grid_fitter, plank_grid_fit_params
from sam.logging_setup import configure_logging
from sam.preprocessing import (
    LetterboxMeta,
    align_imgsz_stride14,
    bilateral_denoise_bgr,
    letterbox_square_bgr,
    normalize_for_sam,
    unletterbox_masks,
)
from workflow_types import SamInferenceConfig, SamSegmentationResult

# --- fixed automation knobs (not CLI) ------------------------------------------
SAM_WEIGHTS = Path("sam3.pt")
OUT_PARENT = Path("outputs")
FLAT_SIGMA = 35.0
CLAHE_CLIP = 2.5
CLAHE_GRID = 8
CONF = 0.25
HALF = False
VERBOSE = True
GRID_FIT = "sam_border_grid"
MIN_MASK_AREA_FRAC = 0.001
MAX_MASK_AREA_FRAC = 0.85
BOUNDARY_DILATE = 3
ANGLE_TOL_DEG = 10.0
# --- preprocessing (not CLI): substeps run in order in preprocessing() ------------
BILATERAL_D = 5
BILATERAL_SIGMA_COLOR = 50.0
BILATERAL_SIGMA_SPACE = 50.0
SAM_IMGSZ = 644
LETTERBOX_PAD_BGR = (114, 114, 114)

log = logging.getLogger("parquet_sam_grid")


def preprocessing(input_bgr: np.ndarray, out_dir: Path, stem: str) -> tuple[np.ndarray, Path, LetterboxMeta | None]:
    """Procedural pre-SAM pipeline; full-res image for grid, letterboxed file for SAM + mask unwarp."""
    log.info("preprocess 1/3: bilateral denoise (d=%s)", BILATERAL_D)
    after_denoise = bilateral_denoise_bgr(
        input_bgr,
        BILATERAL_D,
        BILATERAL_SIGMA_COLOR,
        BILATERAL_SIGMA_SPACE,
    )
    log.info("preprocess 2/3: lighting normalize (flat-field + CLAHE)")
    after_lighting = normalize_for_sam(
        after_denoise,
        flat_field_sigma=FLAT_SIGMA,
        clahe_clip=CLAHE_CLIP,
        clahe_grid=CLAHE_GRID,
    )
    log.info("preprocess 3/3: letterbox to %s (stride-aligned)", SAM_IMGSZ)
    letterbox_size = align_imgsz_stride14(SAM_IMGSZ)
    predict_bgr, letterbox_meta = letterbox_square_bgr(
        after_lighting,
        letterbox_size,
        LETTERBOX_PAD_BGR,
    )
    predict_path = write_lighting_normalized_preview(out_dir, stem, predict_bgr)
    return after_lighting, predict_path, letterbox_meta


def segmentation(
    lighting_path: Path,
    prompts: list[str],
    model_path: Path,
    device: str,
    out_dir: Path,
    stem: str,
) -> tuple[Any, SamSegmentationResult]:
    inference = SamInferenceConfig(
        model_path=model_path,
        device=device,
        half=HALF,
        conf=CONF,
        verbose=VERBOSE,
    )
    log.info("SAM inference (Ultralytics may print its own lines here) …")
    r0, sam = run_sam3_segmentation(
        image_path_for_predictor=str(lighting_path),
        bbox=None,
        text_prompts=prompts,
        inference=inference,
    )
    if sam is None or sam.mask_count == 0:
        log.info("write SAM plot (no masks)")
        write_sam_segmentation_artifacts(out_dir, stem, r0.plot(), sam)
        raise RuntimeError("SAM found no masks")
    log.info("SAM done; masks=%d", sam.mask_count)
    plot_bgr = sam.plot_bgr if sam.plot_bgr is not None else r0.plot()
    log.info("write SAM plot / mask_sum")
    write_sam_segmentation_artifacts(out_dir, stem, plot_bgr, sam)
    return r0, sam


def grid_construction(
    sam_input_bgr: np.ndarray,
    sam: SamSegmentationResult,
    out_dir: Path,
    stem: str,
) -> None:
    grid_params = plank_grid_fit_params(
        min_mask_area_frac=MIN_MASK_AREA_FRAC,
        max_mask_area_frac=MAX_MASK_AREA_FRAC,
        boundary_dilate=BOUNDARY_DILATE,
        angle_tol_deg=ANGLE_TOL_DEG,
    )
    fitter = get_grid_fitter(GRID_FIT)
    log.info("grid fit (%s) … can take a while on large images", GRID_FIT)
    grid_out = fitter.fit(sam_input_bgr, sam.masks.astype(np.float32), grid_params)
    log.info("grid fit done status=%s", grid_out.status)
    log.info("write grid artifacts")
    write_grid_fit_artifacts(out_dir, stem, sam_input_bgr, grid_out)
    crops_dir = write_plank_extractions(out_dir, stem, sam_input_bgr, grid_out)
    if crops_dir is not None:
        log.info("plank crops -> %s", crops_dir)


def run(image: Path, text: str, device: str | None = None) -> None:
    if not logging.getLogger().handlers:
        configure_logging()
    image_path = image.expanduser().resolve()
    out_dir = OUT_PARENT / image_path.stem
    log.info("out_dir=%s", out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem

    prompts = [s.strip() for s in text.split(",") if s.strip()]
    if not prompts:
        raise ValueError("text produced no prompts (need at least one non-empty comma-separated phrase)")
    log.info("load image %s", image_path)
    input_bgr = load_image_bgr(image_path)
    model_path = resolve_model_path(SAM_WEIGHTS)
    dev = pick_device(device)
    log.info("model=%s device=%s prompts=%s", model_path, dev, prompts)

    sam_input_bgr, predict_path, letterbox_meta = preprocessing(input_bgr, out_dir, stem)
    log.info("SAM predict image -> %s", predict_path)

    _r0, sam = segmentation(predict_path, prompts, model_path, dev, out_dir, stem)
    if letterbox_meta is not None:
        sam = SamSegmentationResult(
            masks=unletterbox_masks(sam.masks, letterbox_meta),
            plot_bgr=sam.plot_bgr,
            mask_count=sam.mask_count,
        )

    grid_construction(sam_input_bgr, sam, out_dir, stem)
    log.info("finished")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--text", type=str, required=True, help="comma-separated SAM prompts")
    p.add_argument("--device", default=None, help="omit for auto (CUDA→MPS→CPU); else must be available or RuntimeError")
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="stderr progress (default INFO)",
    )
    a = p.parse_args()
    configure_logging(getattr(logging, a.log_level))
    run(a.image, a.text, a.device)


if __name__ == "__main__":
    main()
