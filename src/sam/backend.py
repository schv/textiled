"""SAM3 weights resolution, device selection, and Ultralytics inference."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from workflow_types import SamInferenceConfig, SamSegmentationResult


def resolve_device(user: str) -> str:
    u = (user or "auto").strip().lower()
    if u == "auto":
        if torch.cuda.is_available():
            return "0"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return user


def pick_device(explicit: str | None) -> str:
    """If `explicit` is None/empty/`auto`, use `resolve_device(\"auto\")` (CUDA → MPS → CPU). Else require that device."""
    if explicit is None or not str(explicit).strip() or str(explicit).strip().lower() == "auto":
        return resolve_device("auto")
    d = str(explicit).strip()
    low = d.lower()
    if low == "cpu":
        return "cpu"
    if low == "mps":
        if not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
            raise RuntimeError("device mps is not available")
        return "mps"
    if d.isdigit() or low.startswith("cuda") or low.startswith("gpu"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"device {d!r} requires CUDA, which is not available")
        return d
    return d


def parse_bbox(value: str | list[float] | tuple[float, ...] | None) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        raw = " ".join(str(x) for x in value)
    else:
        raw = value
    parts = [float(x) for x in raw.replace(",", " ").split()]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("bbox must be four numbers: x1 y1 x2 y2")
    x1, y1, x2, y2 = parts
    if x2 <= x1 or y2 <= y1:
        raise argparse.ArgumentTypeError("bbox must satisfy x2 > x1 and y2 > y1")
    return parts


def hf_hub_cache_dirs() -> list[Path]:
    roots: list[Path] = []
    if v := os.environ.get("HF_HUB_CACHE"):
        roots.append(Path(v).expanduser())
    if v := os.environ.get("HF_HOME"):
        roots.append(Path(v).expanduser() / "hub")
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    seen: set[Path] = set()
    out: list[Path] = []
    for r in roots:
        try:
            r = r.resolve()
        except OSError:
            continue
        if r in seen or not r.is_dir():
            continue
        seen.add(r)
        out.append(r)
    return out


def find_sam3_pt_in_hf_cache() -> Path | None:
    best: Path | None = None
    best_mtime = -1.0
    for hub in hf_hub_cache_dirs():
        repo_dir = hub / "models--facebook--sam3" / "snapshots"
        if not repo_dir.is_dir():
            continue
        for rev_dir in repo_dir.iterdir():
            if not rev_dir.is_dir():
                continue
            for name in ("sam3.pt", "SAM3.pt"):
                cand = rev_dir / name
                if cand.is_file():
                    m = cand.stat().st_mtime
                    if m > best_mtime:
                        best_mtime = m
                        best = cand.resolve()
    return best


def resolve_model_path(model_arg: Path) -> Path:
    p = model_arg.expanduser().resolve()
    if p.is_file():
        return p
    try:
        from huggingface_hub import hf_hub_download

        cached = Path(
            hf_hub_download(
                repo_id="facebook/sam3",
                filename="sam3.pt",
                local_files_only=True,
            )
        )
        if cached.is_file():
            return cached.resolve()
    except ImportError:
        pass
    except Exception:
        pass
    found = find_sam3_pt_in_hf_cache()
    if found is not None:
        return found
    raise FileNotFoundError(
        f"sam3 weights not found (tried {p!s}, Hugging Face cache, ~/.cache/huggingface/hub). "
        "Download sam3.pt or place it next to the project."
    )


def load_image_bgr(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"cannot read image (missing or unsupported format): {path}")
    return bgr


def run_sam3_segmentation(
    *,
    image_path_for_predictor: str,
    bbox: list[float] | None,
    text_prompts: list[str],
    inference: SamInferenceConfig,
) -> tuple[Any, SamSegmentationResult | None]:
    """
    Run Ultralytics SAM3SemanticPredictor.

    Returns (first_result_object, SamSegmentationResult | None if no masks).
    Caller may use result.plot() for visualization before masks are consumed.
    """
    from ultralytics.models.sam import SAM3SemanticPredictor

    overrides = dict(
        conf=inference.conf,
        task="segment",
        mode="predict",
        model=str(inference.model_path),
        device=inference.device,
        half=inference.half,
        save=False,
        verbose=inference.verbose,
        source=image_path_for_predictor,
    )
    predictor = SAM3SemanticPredictor(overrides=overrides)
    predictor.set_image(image_path_for_predictor)
    if bbox is not None:
        results = predictor(bboxes=[bbox])
    else:
        results = predictor(text=text_prompts)
    if not results:
        return None, None
    r0 = results[0]
    if r0.masks is None or r0.masks.data is None:
        return r0, None
    md = r0.masks.data.cpu().numpy()
    if md.ndim != 3 or md.shape[0] == 0:
        return r0, None
    plot_bgr = r0.plot()
    return r0, SamSegmentationResult(masks=md.astype(np.float32), plot_bgr=plot_bgr, mask_count=int(md.shape[0]))
