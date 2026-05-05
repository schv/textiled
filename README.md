# textiled

SAM-based **parquet plank grid** pipeline: normalize lighting → Meta SAM 3 masks → fit a 2D lattice and write debug images / JSON.

## Setup

```bash
uv sync
```

Caveats: `**NOTES.md**`.

### SAM3 weights (`facebook/sam3`)

Ultralytics loads `**sam3.pt**` from Meta’s Hugging Face repo `**facebook/sam3**`. This pipeline resolves weights in order: a file named `**sam3.pt**` next to your working directory (see `SAM_WEIGHTS` in `parquet_sam_grid.py`), then the Hugging Face hub cache. It does **not** download from the Hub automatically—you must fetch the file once.

1. **Access** — Open [facebook/sam3](https://huggingface.co/facebook/sam3), sign in, and accept any license / request access if the card says the model is gated.
2. **Token** — Create a **read** token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
3. **Authenticate** (pick one):
  - `uv run hf auth login` and paste the token, or
  - `export HF_TOKEN=<your_token>` in the shell where you download and run.
4. **Download** (pick one; run from the repo root so `--local-dir .` lands beside `SAM_WEIGHTS`):
  - Into the project folder (matches the default `sam3.pt` filename):
  - Or only into the Hub cache (no `sam3.pt` in the repo; resolution still finds it under `~/.cache/huggingface/hub` unless you override cache dirs):
    ```bash
    hf download facebook/sam3 sam3.pt
    ```

Optional: `**HF_HOME**` / `**HF_HUB_CACHE**` change where cached snapshots live; see `hf_hub_cache_dirs()` in `sam/backend.py`.

## Run

```bash
python3 src/parquet-sam-grid.py --image samples/herringbone.png --text "wooden parquet rectangular planks"
```

Outputs go to `**outputs/<image_stem>/**`. Weights: `**SAM_WEIGHTS**` in `parquet_sam_grid.py` (default `sam3.pt` + hub resolution).
