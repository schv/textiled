"""Shared datatypes for SAM parquet procedural workflow (no scenario framework)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# Re-export for convenience
__all__ = [
    "JsonDict",
    "SamInferenceConfig",
    "SamSegmentationResult",
    "GridFitOutput",
]

JsonDict = dict[str, Any]


@dataclass
class SamInferenceConfig:
    """Arguments needed to run SAM3 once (excluding prompts, which are passed separately)."""

    model_path: Path
    device: str
    half: bool
    conf: float
    verbose: bool


@dataclass
class SamSegmentationResult:
    """Normalized output from SAM3 semantic segmentation."""

    masks: np.ndarray  # (N, H, W) float32 in [0, 1] or logits-like
    plot_bgr: np.ndarray | None
    mask_count: int


@dataclass
class GridFitOutput:
    """Unified output from any grid fitting strategy (lattice, RANSAC, ...)."""

    strategy: str
    status: str  # "ok" | "partial" | "failed"
    grid_model: JsonDict
    sam_boundary_u8: np.ndarray
    evidence_f32: np.ndarray
    ridge_u8: np.ndarray
    legacy_geometry: JsonDict = field(default_factory=dict)
    debug_json: JsonDict | None = None
