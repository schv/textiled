# Notes (non-code)

## SAM / weights

- **Ultralytics SAM3** imports **`clip`** (Ultralytics CLIP git) and **`timm`** for the ViT backbone — both are listed in `pyproject.toml`. If you see `ModuleNotFoundError: clip` / `timm`, run `uv sync`.
- Request HF access to `facebook/sam3`, then e.g. `HF_TOKEN=… hf download facebook/sam3`, or place `sam3.pt` / rely on hub cache. `SAM_WEIGHTS` in `parquet_sam_grid.py` points at `sam3.pt` by default.

## Device

- **Omit `--device`:** same order as `resolve_device("auto")` — CUDA if available, else MPS, else CPU.
- **`--device auto`:** same as omitting.
- **Explicit `--device`:** must be usable or **`RuntimeError`** (e.g. `mps` with no MPS, `0` / `cuda:0` with no CUDA).

## No masks

- If SAM returns no masks, the grid step dies on `sam.masks` — adjust weights / prompts / image (not a friendly message by design).

## Running the scenario file

- `uv run python src/parquet_sam_grid.py` is the same as `uv run parquet-sam-grid` (entry point is `main()` in that module).
- Progress on stderr: **`--log-level INFO`** (default). After Ultralytics lines, a long pause usually means **grid fitting**; use **`DEBUG`** if you add more logs later.
