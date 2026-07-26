# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MaterialsVision is a research codebase for automated quantitative analysis of
porous foam material microstructures from SEM images. It combines deep
learning pore segmentation (Cellpose-SAM, StarDist, PEFT-SAM/LoRA) with
morphological, topological, and spatial analysis implemented from scratch in
`materials_vision/quantitative_analysis/`. This is research code, not a
library with a stable public API — scripts are run directly against
locally-configured paths rather than through a packaged CLI.

## Environment Setup

```bash
# The project venv lives at .env/ (not the conventional .venv/)
source .env/bin/activate
pip install -r requirements.txt
```

Python 3.12+ is required. `mlflow`, `torch`/`torchvision`, `cellpose`, and
`stardist` are the heaviest dependencies.

## Common Commands

```bash
# Lint / format (flake8, isort, whitespace/EOF fixers)
pre-commit run --all-files

# Type check (config targets `MaterialsVision/`, but the actual package dir
# is lowercase `materials_vision/` — pass the real path explicitly)
mypy materials_vision/

# View MLflow experiment tracking UI
mlflow ui --backend-store-uri sqlite:///mlflow.db
# -> http://127.0.0.1:5000
```

**There is no automated test suite in this repository** (no `pytest`,
no `tests/` directory). Do not assume one exists or invent test commands;
verification for this codebase is manual (run the script/notebook against
sample data, inspect the plots/Excel report/MLflow run).

## Architecture

### Pipeline stages (data flows in this order)

1. **Data prep** (`side_scripts/`) — filter by magnification, group into
   per-material/magnification datasets, stratified train/test split.
2. **Segmentation** (`scripts/run_cellpose_inference.py`) — runs a trained
   Cellpose-SAM model over SEM images, produces `*_masks.tif`.
3. **Quantitative analysis** (`scripts/quantitative_analysis/`) — turns masks
   into per-pore metrics (Excel reports) and then aggregate macroscopic
   metrics across a whole material dataset.
4. **Model training/tuning** (`scripts/retrain_cellpose.py`,
   `scripts/grid_search_cellpose.py`, `scripts/finetune_peft_sam.py`) — all
   MLflow-tracked, configured via YAML rather than CLI flags.

### Package layout (`materials_vision/`)

- `quantitative_analysis/` — the analytical core.
  `quantitative_analysis.py` defines `PoreMorphologyMetrics` (per-pore shape
  descriptors: area, perimeter, circularity, solidity, Feret diameters via
  rotating calipers, ellipse fitting) and `PorousMaterialAnalyzer`
  (global descriptors: porosity, local porosity variance, anisotropy,
  nearest-neighbor spatial stats, fractal dimension via box-counting,
  Voronoi-based coordination number). `batch_analysis.py` wraps this into
  `BatchPorousMaterialAnalyzer` for whole-directory processing.
  `calculate_statistics.py` / `file_utils.py` support aggregation and I/O.
- `experiments/` — training/eval code per model family:
  `cellpose/retraining/` (YAML-configured fine-tuning), `cellpose/
  cellpose_grid_search/` (hyperparameter sweeps), `peft_sam/` (LoRA
  fine-tuning of SAM), `evaluate_morphological_operations/` (classic
  image-processing baseline vs. deep-learning segmentation comparison).
  **Note:** both `peft_sam/` and `peft-sam/` exist side by side — check
  which one is actually imported/current before editing; treat the other as
  possibly stale.
- `artificial_dataset/` — synthetic microstructure generation via Voronoi
  diagrams, used to create training data with ground-truth masks.
- `image_preprocessing/` — augmentation (rotation, flip, contrast, Poisson
  noise) for image/mask pairs.
- `config/sem_calibration.yaml` + `utils.load_pixel_sizes()` — the single
  source of truth for physical pixel size (µm/px) per SEM magnification.
  Analysis code resolves the right pixel size by parsing the magnification
  out of the input filename (e.g. `AS2_40_...` → 40x); a fallback
  `pixel_size` constant in the calling script is used if that parse fails.
- `logging_config.setup_logging()` — centralized console + rotating file
  logging (10MB/5 backups) writing to `logs/materials_vision.log` by default.

### Configuration philosophy

There is no single global config. Each concern has one dedicated home:

| What | Where |
|------|-------|
| SEM pixel calibration | `materials_vision/config/sem_calibration.yaml` |
| Cellpose fine-tuning hyperparameters/paths | `materials_vision/experiments/cellpose/retraining/cellpose_retraining_config.yaml` |
| Grid search parameter space | `materials_vision/experiments/cellpose/cellpose_grid_search/grid_search_config.yaml` |
| Inference/analysis I/O paths | CLI flags (`run_cellpose_inference.py`, `calculate_macroscopic_metrics.py`) or constants at the top of the script's `if __name__ == "__main__"` block (the two `quantitative_analysis/*.py` scripts) |

When editing a script that follows the "constants at the top" pattern, don't
add argparse plumbing unless asked — it's an intentional convention here, not
an oversight.

### Notebooks (`notebooks/`)

Exploratory/experimental work: Cellpose and StarDist fine-tuning + evaluation,
sensitivity analysis, SAM experiments, preprocessing tuning. These are
scratch/research notebooks, not part of the production pipeline — treat them
as reference material rather than code to keep in sync with the package.

## Code Quality Guidelines

- Write small, atomic functions that are easy to debug in isolation — one
  responsibility per function.
- Use NumPy-style docstrings (matching the existing convention throughout
  `materials_vision/`).
- Max line length: 79 characters (matches flake8 default already enforced by
  pre-commit).
- No emoticons/emoji in code, comments, or docstrings.
- Use logging, not `print`, for all runtime output.
- Reuse the repo-level logging setup (`materials_vision.logging_config.
  setup_logging()`) instead of creating new logging configuration.
- Avoid unnecessary comments — only comment to explain a non-obvious *why*
  (a subtle invariant, a workaround, a hard-to-guess reason), never to
  restate what the code does.
- Don't use multiple libraries for the same job (e.g. `cv2` and `tifffile`
  and `PIL` all for image I/O). Stick to whichever is already used for that
  purpose in this codebase; only introduce a new library when the existing
  one genuinely can't do the job.
- When writing new functionality, explicitly identify and handle edge cases
  rather than assuming happy-path input.
- Use explicit units in variable names and docstrings (`_px` vs `_um`) —
  this codebase constantly converts between pixel and µm values via
  `load_pixel_sizes()`, and unmarked units are a recurring source of silent
  bugs.
- Prefer vectorized numpy/skimage/scipy operations over manual per-pixel or
  per-pore Python loops.
- Don't mutate caller-owned numpy arrays or mask data in place — return new
  arrays instead, to keep state changes traceable across the pipeline.