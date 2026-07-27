# MaterialsVision

**Automated Microstructure Analysis Tool for Porous Foam Materials**

MaterialsVision is a comprehensive Python package for quantitative analysis of
foam material microstructures from scanning electron microscopy (SEM) images.
It combines state-of-the-art deep learning pore segmentation (Cellpose-SAM,
StarDist) with advanced morphological, topological, and spatial analysis
methods from materials science.

## Research Goals

This is a **research project** with a primary goal of developing an automated
porous material microstructure segmentation tool that can
**replace the manual expert annotation process**. Traditional microscopy
image analysis requires extensive manual labeling by domain experts, which is:

- **Time-consuming**: Manual annotation of pores in SEM images can take hours per sample
- **Costly**: Requires specialized expertise and significant labor investment
- **Prone to inconsistency**: Inter-annotator variability and fatigue affect results
- **Not scalable**: Large-scale materials characterization studies are impractical

By leveraging deep learning (Cellpose-SAM and StarDist) fine-tuned on foam
materials, MaterialsVision aims to **automate the segmentation pipeline**,
enabling researchers to:
- Analyze hundreds of samples in the time it previously took to process one
- Reduce costs associated with manual annotation
- Ensure consistent, reproducible segmentation across datasets
- Focus expert time on high-level analysis rather than tedious labeling

## Features

### AI-Powered Segmentation

**Cellpose-SAM**:
- **Cellpose-SAM Integration**: Instance segmentation for automated pore
  detection
- **Custom Model Training**: Fine-tune pretrained models on your foam datasets
  with MLflow experiment tracking
- **Hyperparameter Grid Search**: Systematic search over training and inference
  parameters with full MLflow logging

**StarDist**:
- **StarDist2D Training**: Star-convex polygon instance segmentation with
  U-Net backbone (32 rays, 400 epochs) fine-tuned on foam SEM images
- **Threshold Optimization**: Automatic optimization of probability and NMS
  thresholds on the validation set
- **Quantitative Evaluation**: IoU-based and boundary-score metrics across
  the full test set (128 images)

**General**:
- **Multi-Magnification Support**: Handles 40x, 50x SEM images, but others
are experimentally possible
- **GPU Acceleration**: CUDA support with automatic CPU fallback

### Comprehensive Quantitative Analysis

**Individual Pore Morphology**:
- Area and perimeter measurements
- Shape descriptors: circularity, solidity, roundness
- Feret diameters (maximum and minimum via rotating calipers)
- Ellipse fitting: major/minor axes, aspect ratio, orientation

**Global Microstructure Descriptors**:
- Porosity calculation
- Local porosity variance (homogeneity measure)
- Anisotropy (preferred orientation analysis)

**Spatial Relations**:
- Nearest neighbor distance distributions
- Centroid-based spatial analysis

**Topology & Connectivity**:
- Fractal dimension (box-counting method)
- Coordination number (Voronoi tessellation)

**Macroscopic Metrics** (aggregate across full material dataset):
- Size distribution: mean, median, D10, D90, span, skewness
- Elongation: mean aspect ratio, fraction of elongated pores (> 2.0)
- Regularity: mean circularity, D10 of most irregular pores
- Orientation entropy (0 = aligned, 1 = isotropic)
- Topology: coordination number mean and standard deviation

### Research Tools
- **Synthetic Data Generation**: Create realistic training data using Voronoi
  diagrams
- **Batch Processing**: Analyze multiple samples efficiently
- **Automated Reporting**: Excel reports with statistical summaries and
  visualizations
- **Experiment Tracking**: MLflow integration for reproducible research

## Installation

### Prerequisites
- Python 3.12+
- CUDA-compatible GPU (optional but recommended for segmentation)

### Setup

```bash
# Clone the repository
git clone https://github.com/DWalicki95/MaterialsVision.git
cd MaterialsVision

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Key Dependencies
- `cellpose`: Cellpose-SAM deep learning segmentation
- `stardist`: StarDist2D star-convex polygon segmentation
- `tensorflow` / `keras`: StarDist backend
- `mlflow`: Experiment tracking
- `GPUtil`: GPU monitoring
- `psutil`: System monitoring
- `openpyxl`: Excel report generation
- Scientific stack: `numpy`, `scipy`, `scikit-image`, `pandas`, `matplotlib`

## Configuration

There is no global config file. Each type of setting has a specific,
predictable home:

| What to configure | Where |
|-------------------|-------|
| SEM pixel calibration (µm/px) | `materials_vision/calibration/sem_calibration.yaml` |
| Training hyperparameters & dataset paths | `materials_vision/experiments/cellpose/retraining/cellpose_retraining_config.yaml` |
| Grid search parameter space | `materials_vision/experiments/cellpose/cellpose_grid_search/grid_search_config.yaml` |
| Segmentation runtime paths | CLI flags to `run_cellpose_inference.py` |
| Analysis runtime paths | Constants at the top of each analysis script |

### SEM Calibration (`sem_calibration.yaml`)

Physical pixel sizes for each microscope magnification, derived from SEM
metadata. **Edit only when switching to a different instrument.**

```yaml
# materials_vision/calibration/sem_calibration.yaml
pixel_sizes:
  40:   3.24023   # µm/px
  50:   2.59219
  100:  1.29609
  250:  0.51844
  500:  0.25922
  1000: 0.12961
```

Loaded automatically by all analysis code via `load_pixel_sizes()`. The right
pixel size is resolved from the magnification encoded in the filename
(e.g. `AS2_40_10_jpg.rf.abc_masks.tif` → 40x → 3.24023 µm/px). If
magnification cannot be extracted, the fallback `pixel_size` set in the
calling script is used.

### Retraining Config (`cellpose_retraining_config.yaml`)

Edit before running `retrain_cellpose.py`. No CLI arguments needed — the
script reads this file directly.

```yaml
dataset:
  train_dir: /path/to/train      # ← set your path
  test_dir: /path/to/test        # ← set your path
  name: AS                       # material series identifier
  version: 1

training:
  epochs: 200
  learning_rate: 0.001
  weight_decay: 0.0001
  model_name: cpsam              # Cellpose-SAM architecture

logging:
  experiment_name: "cellpose retraining AS - augmentation"

general:
  output_dir: outputs
  gpu: true
```

### Grid Search Config (`grid_search_config.yaml`)

Defines the parameter space for systematic hyperparameter search. Edit
before running `grid_search_cellpose.py`.

```yaml
mode: full  # training | evaluation | full

data:
  train_dir: /path/to/train      # ← set your path
  test_dir: /path/to/test        # ← set your path

training_grid:
  learning_rate: [0.01]
  weight_decay: [0.0001, 0.01]
  n_epochs: [160]
  batch_size: [1, 8]
  normalize: [true, {percentile: [3, 98]}, {tile_norm_blocksize: 128}]

eval_grid:
  flow_threshold: [0.2, 0.4, 0.6, 0.8]
  cellprob_threshold: [-2.0, -1.0, 0.0, 1.0, 2.0]
  min_size: [5, 15, 30]
```

### Analysis Script Paths

The quantitative analysis scripts are not pure CLI tools — they contain a
`if __name__ == "__main__"` block with paths set as Python constants. Before
running, open the script and set the variables at the top of the block:

```python
# scripts/quantitative_analysis/batch_quantitative_analysis.py
parent_directory = "/path/to/masks/organized"   # ← input dir
output_directory = Path("/path/to/results")      # ← output dir
magnification    = 40                            # ← fallback magnification
```

```python
# scripts/quantitative_analysis/single_image_quantitative_analysis.py
mask_path        = "/path/to/single_mask.tif"   # ← input mask
output_directory = Path("/path/to/results")      # ← output dir
magnification    = 40                            # ← SEM magnification
```

## Full Analysis Pipeline

End-to-end workflow from raw SEM images to macroscopic material metrics:

```
Step 1 — Data Preparation  (data_prep/)
   filter_magnifications.py         Filter images by microscope magnification
   group_material_into_datasets.py  Organize by material type and magnification
   split_dataset_into_subsets.py    Stratified train/test split

Step 2 — Segmentation  (scripts/)
   run_cellpose_inference.py        Generate pore masks from SEM images

Step 3 — Quantitative Analysis  (scripts/quantitative_analysis/)
   batch_quantitative_analysis.py       Per-material Excel reports (pore-level)
   calculate_macroscopic_metrics.py     17 aggregate metrics per material
                                         (core logic in materials_vision/
                                         quantitative_analysis/macroscopic_metrics.py)

Optional — Model Training  (scripts/)
   retrain_cellpose.py              Fine-tune Cellpose-SAM (YAML-configured)
   grid_search_cellpose.py          Systematic hyperparameter search
```

## Quick Start

### 1. Run Cellpose Segmentation

All three path arguments are required:

```bash
python scripts/run_cellpose_inference.py \
    --path-to-files /path/to/sem/images \
    --model-path /path/to/cellpose/model \
    --output-path /path/to/output/masks
```

Optional flags:
- `--no-flows` — skip saving optical flow fields
- `--no-styles` — skip saving style vectors
- `-v` / `--verbose` — enable DEBUG logging

### 2. Batch Quantitative Analysis

Set paths inside the script, then run:

```bash
# 1. Open the script and set parent_directory and output_directory
#    scripts/quantitative_analysis/batch_quantitative_analysis.py

# 2. Run
python scripts/quantitative_analysis/batch_quantitative_analysis.py
```

Generates per-material Excel reports with pore-level metrics.
The pixel size per image is resolved automatically from the magnification
in the filename using `materials_vision/calibration/sem_calibration.yaml`.

### 3. Calculate Macroscopic Metrics

```bash
python scripts/quantitative_analysis/calculate_macroscopic_metrics.py \
    --input-dir /path/to/analysis/results \
    --output-file macroscopic_metrics.xlsx \
    --generate-plots
```

Optional flags:
- `--merge-only` — combine raw pore datasets without calculating statistics
- `-v` / `--verbose` — enable DEBUG logging

### 4. Fine-tune Cellpose (YAML-configured)

```bash
# 1. Edit the config file:
#    materials_vision/experiments/cellpose/retraining/cellpose_retraining_config.yaml

# 2. Run — no CLI arguments needed
python scripts/retrain_cellpose.py
```

Training includes MLflow experiment tracking, real-time resource monitoring,
automatic model checkpointing, and loss visualization.

### 5. Hyperparameter Grid Search

```bash
# 1. Edit the config file:
#    materials_vision/experiments/cellpose/cellpose_grid_search/grid_search_config.yaml

# 2. Preview all combinations without running
python scripts/grid_search_cellpose.py \
    --config materials_vision/experiments/cellpose/cellpose_grid_search/grid_search_config.yaml \
    --mode full \
    --dry-run

# 3. Run the full grid search
python scripts/grid_search_cellpose.py \
    --config materials_vision/experiments/cellpose/cellpose_grid_search/grid_search_config.yaml \
    --mode full \
    --gpu-device 0
```

Modes:
- `training` — train models for all hyperparameter combinations
- `evaluation` — evaluate a single pre-trained model with different inference
  parameters
- `full` — training grid followed by evaluation grid on top-N models

### 6. View Experiment Results (MLflow)

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Open: http://127.0.0.1:5000
```

## Project Structure

```
MaterialsVision/
├── materials_vision/              # Main Python package
│   ├── calibration/
│   │   └── sem_calibration.yaml  # SEM pixel size calibration (µm/px)
│   ├── synthetic_dataset/        # Synthetic data generation
│   │   ├── create_voronoi_diagrams.py
│   │   ├── synthetic_microstructures.py
│   │   └── data_loader_synthetic_dataset.py
│   ├── experiments/               # Experiment code and configs
│   │   ├── cellpose/
│   │   │   ├── retraining/
│   │   │   │   ├── cellpose_retraining_config.yaml  # ← edit before training
│   │   │   │   └── training.py
│   │   │   ├── cellpose_grid_search/
│   │   │   │   ├── grid_search_config.yaml          # ← edit before search
│   │   │   │   └── grid_search.py
│   │   │   ├── inference.py
│   │   │   └── utils.py
│   │   ├── evaluate_morphological_operations/  # Watershed vs. DL segmentation comparison
│   │   │   ├── main.py
│   │   │   ├── alignment.py
│   │   │   ├── data_loading.py
│   │   │   ├── ground_truth.py
│   │   │   ├── reporting.py
│   │   │   └── visualization.py
│   │   └── peft_sam/               # LoRA fine-tuning of SAM
│   ├── quantitative_analysis/     # Core analysis modules
│   │   ├── quantitative_analysis.py   # PorousMaterialAnalyzer
│   │   ├── batch_analysis.py          # BatchPorousMaterialAnalyzer
│   │   ├── macroscopic_metrics.py     # Aggregate per-material metrics
│   │   ├── stats_utils.py
│   │   └── file_utils.py
│   ├── image_preprocessing/       # Data augmentation
│   │   ├── image_transformation.py
│   │   ├── transform.py
│   │   └── helpers.py
│   ├── logging_config.py          # Centralized logging setup
│   ├── metrics.py                 # Evaluation metrics (IoU, boundary scores)
│   └── utils.py                   # General utilities (load_pixel_sizes, ...)
├── scripts/                       # Entry point scripts
│   ├── run_cellpose_inference.py      # CLI: --path-to-files --model-path --output-path
│   ├── retrain_cellpose.py            # YAML-configured
│   ├── grid_search_cellpose.py        # CLI: --config --mode --gpu-device
│   ├── finetune_peft_sam.py           # LoRA fine-tuning of SAM
│   ├── augment_dataset.py             # Dataset augmentation
│   ├── generate_synthetic_microstructures.py  # Synthetic SEM image generation
│   ├── evaluate_morphological_operations.py   # Watershed vs. DL comparison
│   └── quantitative_analysis/
│       ├── single_image_quantitative_analysis.py  # set paths in script
│       ├── batch_quantitative_analysis.py         # set paths in script
│       ├── calculate_macroscopic_metrics.py       # CLI: --input-dir --output-file
│       └── batch_quantitative_analysis_random_pore_removal.py  # sensitivity analysis
├── data_prep/                     # Standalone data preparation scripts
│   ├── filter_magnifications.py
│   ├── group_material_into_datasets.py
│   └── split_dataset_into_subsets.py
├── notebooks/                     # Jupyter notebooks
│   ├── cellpose_finetuned_evaluation.ipynb
│   ├── cellpose_finetuned_inferece.ipynb
│   ├── cellpose_sensitivity_analysis.ipynb
│   ├── stardist_training.ipynb
│   ├── stardist_finetuned_evaulation_.ipynb
│   ├── SAM_tests.ipynb
│   └── preprocessing.ipynb
├── mlflow.db                      # MLflow experiment database
└── requirements.txt               # Python dependencies
```

## Detailed Usage

### Quantitative Analysis Metrics

#### Pore Morphology (Individual Pore Metrics)

| Metric | Formula | Description |
|--------|---------|-------------|
| **Area** | - | Pore area in μm² |
| **Perimeter** | - | Pore perimeter in μm |
| **Circularity** | 4π × Area / Perimeter² | Shape roundness (1.0 = perfect circle) |
| **Solidity** | Area / Convex Hull Area | Compactness measure |
| **Max Feret Diameter** | - | Longest distance between any two boundary points |
| **Min Feret Diameter** | - | Shortest distance (rotating calipers algorithm) |
| **Ellipse Major Axis** | - | Length of fitted ellipse major axis |
| **Ellipse Minor Axis** | - | Length of fitted ellipse minor axis |
| **Aspect Ratio** | Major Axis / Minor Axis | Elongation measure |
| **Orientation** | - | Ellipse orientation angle (degrees) |
| **Roundness** | 4 × Area / (π × Major Axis²) | Alternative shape descriptor |

#### Global Microstructure Descriptors

- **Porosity**: Total pore area / Total image area
- **Local Porosity Variance**: Standard deviation of porosity in sub-regions
  (measures homogeneity)
- **Anisotropy**: Preferred orientation strength (0 = isotropic,
  1 = highly directional)

#### Spatial Relations

- **Nearest Neighbor Distance**: Distance from each pore centroid to its
  closest neighbor
- **Distribution Plots**: Histograms and density plots of spatial metrics

#### Topology & Connectivity

- **Fractal Dimension**: Minkowski-Bouligand box-counting method (measures
  structural complexity)
- **Coordination Number**: Average number of neighboring pores via Voronoi
  tessellation (quantifies connectivity)

### Synthetic Data Generation

Generate training data with realistic microstructure features:

```bash
python scripts/generate_synthetic_microstructures.py \
    --n_samples 1000 \
    --dataset_name my_dataset \
    --save_path /path/to/synthetic_data
```

**Features**:
- Voronoi-based pore structure (40-70 pores per image)
- SEM-realistic texture and noise
- Boundary imperfections with jitter
- Pore perforations and contaminants
- Automatic ground truth mask generation
- Automatic train/test directory organization for Cellpose

### Data Preparation Utilities (`data_prep/`)

| Script | Purpose |
|--------|---------|
| `filter_magnifications.py` | Filter images by microscope magnification extracted from filenames |
| `group_material_into_datasets.py` | Organize images into subdirectories by material type and magnification |
| `split_dataset_into_subsets.py` | Stratified train/test split preserving image-mask pairs |

### Additional Scripts (`scripts/`)

| Script | Purpose |
|--------|---------|
| `generate_synthetic_microstructures.py` | Generate synthetic SEM images with ground truth masks |
| `augment_dataset.py` | Augment image/mask pairs (rotation, flip, contrast, Poisson noise) |
| `evaluate_morphological_operations.py` | Classic image-processing baseline (watershed) vs. deep-learning segmentation comparison |
| `quantitative_analysis/batch_quantitative_analysis_random_pore_removal.py` | Robustness analysis — measure sensitivity to missing pore detections |

### Output Structure

Analysis generates organized output directories:

```
output_dir/
├── plots/                    # Visualizations
│   ├── area_distribution.png
│   ├── perimeter_distribution.png
│   ├── circularity_distribution.png
│   ├── spatial_relations.png
│   ├── voronoi_diagram.png
│   └── fractal_dimension.png
├── data/                     # Raw numerical data (CSV)
│   ├── morphology_metrics.csv
│   ├── global_descriptors.csv
│   └── spatial_metrics.csv
└── reports/                  # Excel reports
    └── {filename}_analysis_report.xlsx
```

The Excel report includes:
- **Individual_Pores sheet**: Complete per-pore measurements
- **Summary sheet**: Statistical aggregates (mean, median, std, min, max)
- **Metadata**: Analysis parameters and settings

## Notebooks

Explore example workflows in [notebooks/](notebooks/):

| Notebook | Description |
|----------|-------------|
| [cellpose_finetuned_evaluation.ipynb](notebooks/cellpose_finetuned_evaluation.ipynb) | Evaluate fine-tuned Cellpose model (IoU, boundary scores) |
| [cellpose_finetuned_inferece.ipynb](notebooks/cellpose_finetuned_inferece.ipynb) | Run inference with a fine-tuned model |
| [cellpose_sensitivity_analysis.ipynb](notebooks/cellpose_sensitivity_analysis.ipynb) | Analyze sensitivity of metrics to inference parameters |
| [stardist_training.ipynb](notebooks/stardist_training.ipynb) | Train StarDist2D (U-Net backbone, 32 rays, 400 epochs) on foam SEM images with data augmentation and threshold optimization |
| [stardist_finetuned_evaulation_.ipynb](notebooks/stardist_finetuned_evaulation_.ipynb) | Evaluate fine-tuned StarDist model (IoU micro F1 ~0.794, boundary F1 ~0.824) |
| [SAM_tests.ipynb](notebooks/SAM_tests.ipynb) | Experiments with Segment Anything Model (SAM) |
| [preprocessing.ipynb](notebooks/preprocessing.ipynb) | Preprocessing experiments and parameter tuning |

## Technical Details

### Algorithms Implemented

- **Cellpose-SAM**: Generalist deep learning segmentation (cpsam model
  architecture)
- **StarDist2D**: Star-convex polygon segmentation via U-Net backbone;
  radial distance regression with 32 rays, trained with TensorFlow/Keras
- **Voronoi Tessellation**: Coordination number and spatial connectivity
- **Box-Counting Method**: Fractal dimension estimation with R² quality metric
- **Rotating Calipers**: Efficient O(n) minimum Feret diameter on convex hull
- **Ellipse Fitting**: Least-squares fitting to pore contours
- **Circular Statistics**: π-periodic statistics for anisotropy measurement

### GPU Acceleration

- Cellpose automatically leverages CUDA when available
- GPU memory monitoring via GPUtil
- Graceful CPU fallback for systems without GPU

### Experiment Tracking

MLflow integration tracks:
- Training and validation losses
- Hyperparameters
- System resource usage (CPU, RAM, GPU)
- Model artifacts and checkpoints
- Configuration files

Access the MLflow dashboard locally:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Open: http://127.0.0.1:5000
```

## Development

### Code Quality

This project uses:
- **Pre-commit hooks**: Configured in [.pre-commit-config.yaml](.pre-commit-config.yaml)
  - `flake8`: PEP 8 style checking
  - `isort`: Import sorting
  - Trailing whitespace removal
- **Type checking**: mypy configuration in [mypy.ini](mypy.ini)
- **Documentation**: NumPy-style docstrings throughout
- **Logging**: Comprehensive logging via `materials_vision.logging_config`

## Citation

If you use MaterialsVision in your research, please cite:

```
[Citation information to be added]
```

## Contact & Support

- **Issues**: Report bugs or request features via GitHub Issues
- **Author**: dwalicki.ai@gmail.com

---

**MaterialsVision** - Advancing materials science through automated microstructure analysis
