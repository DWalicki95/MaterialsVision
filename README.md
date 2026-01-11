# MaterialsVision

**Automated Microstructure Analysis Tool for Porous Foam Materials**

MaterialsVision is a comprehensive Python package for quantitative analysis of
foam material microstructures from scanning electron microscopy (SEM) images.
It combines state-of-the-art deep learning pores segmentation (Cellpose) with
advanced morphological, topological, and spatial analysis methods from
materials science.

## Research Goals

This is a **research project** with a primary goal of developing an automated
porous material microstructure segmentation tool that can 
**replace the manual expert annotation process**. Traditional microscopy 
image analysis requires extensive manual labeling by domain experts, which is:

- **Time-consuming**: Manual annotation of pores in SEM images can take hours per sample
- **Costly**: Requires specialized expertise and significant labor investment
- **Prone to inconsistency**: Inter-annotator variability and fatigue affect results
- **Not scalable**: Large-scale materials characterization studies are impractical

By leveraging deep learning (Cellpose) fine-tuned on foam materials,
 MaterialsVision aims to **automate the segmentation pipeline**, enabling researchers to:
- Analyze hundreds of samples in the time it previously took to process one
- Reduce costs associated with manual annotation
- Ensure consistent, reproducible segmentation across datasets
- Focus expert time on high-level analysis rather than tedious labeling

## Features

### AI-Powered Segmentation
- **Cellpose Integration**: Instance segmentation for automated pore detection
- **Custom Model Training**: Fine-tune pretrained models on your foam datasets with MLflow experiment tracking
- **Multi-Magnification Support**: Handles 40x, 50x, 100x, 250x, 500x, and 1000x SEM images
- **GPU Acceleration**: CUDA support with automatic CPU fallback

### Comprehensive Quantitative Analysis

**Individual Pore Morphology**:
- Area and perimeter measurements
- Shape descriptors: circularity, solidity, roundness
- Feret diameters (maximum and minimum)
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

### Research Tools
- **Synthetic Data Generation**: Create realistic training data using Voronoi diagrams
- **Batch Processing**: Analyze multiple samples efficiently
- **Automated Reporting**: Excel reports with statistical summaries and visualizations
- **Experiment Tracking**: MLflow integration for reproducible research

## Installation

### Prerequisites
- Python 3.12+
- CUDA-compatible GPU (optional but recommended for segmentation)

### Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd MaterialsVision

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Key Dependencies
- `cellpose`: Deep learning segmentation
- `mlflow`: Experiment tracking
- `GPUtil`: GPU monitoring
- `psutil`: System monitoring
- `openpyxl`: Excel report generation
- Scientific stack: `numpy`, `scipy`, `scikit-image`, `pandas`, `matplotlib`

## Quick Start

### 1. Run Cellpose Segmentation

```bash
python scripts/run_cellpose_inference.py \
    --model_path /path/to/cellpose/model \
    --input_dir /path/to/sem/images \
    --output_dir /path/to/output
```

### 2. Quantitative Analysis (Single Sample)

```python
from materials_vision.quantitative_analysis.quantitative_analysis import PorousMaterialAnalyzer

# Initialize analyzer
analyzer = PorousMaterialAnalyzer(
    mask_path='/path/to/segmentation_mask.tif',
    pixel_size=3.24023,  # μm/px for 40x magnification
    sample_name='AS1',
    output_dir='/path/to/output'
)

# Run comprehensive analysis
analyzer.run_analysis(reject_boundary_pores=True)
```

This generates:
- Excel report with all metrics
- Statistical summaries (mean, median, std, min, max)
- Visualization plots (distributions, Voronoi diagram, fractal analysis)

### 3. Batch Analysis

```bash
python scripts/quantitative_analysis/batch_quantitative_analysis.py \
    --material_series AS \
    --magnification 40
```

### 4. Fine-tune Cellpose on Custom Data

```bash
python scripts/retrain_cellpose.py \
    --train_dir /path/to/train \
    --test_dir /path/to/test \
    --epochs 500 \
    --learning_rate 0.1
```

Training includes:
- MLflow experiment tracking
- Real-time system resource monitoring
- Automatic model checkpointing
- Loss visualization

## Project Structure

```
MaterialsVision/
├── materials_vision/              # Main Python package
│   ├── artificial_dataset/        # Synthetic data generation
│   │   ├── create_voronoi_diagrams.py
│   │   └── synthetic_microstructures.py
│   ├── cellpose/                 # Cellpose integration & training
│   │   ├── training.py           # Model fine-tuning with MLflow
│   │   └── utils.py
│   ├── quantitative_analysis/    # Core analysis modules
│   │   ├── quantitative_analysis.py  # Main analyzer classes
│   │   ├── batch_analysis.py
│   │   └── calculate_statistics.py
│   ├── image_preprocessing/      # Data augmentation
│   │   └── image_transformation.py
│   ├── config.py                 # Data paths configuration
│   ├── data_loader.py            # Dataset loading utilities
│   ├── logging_config.py         # Logging setup
│   ├── metrics.py                # Evaluation metrics (IoU, boundary scores)
│   └── utils.py                  # General utilities
├── scripts/                       # Entry point scripts
│   ├── run_cellpose_inference.py
│   ├── retrain_cellpose.py
│   └── quantitative_analysis/
│       ├── single_image_quantitative_analysis.py
│       └── batch_quantitative_analysis.py
├── notebooks/                     # Jupyter notebooks for experiments
│   ├── cellpose_finetuned_evaluation.ipynb
│   ├── cellpose_out_of_the_box_evaluation.ipynb
│   ├── preprocessing.ipynb
│   └── manual_annotations_evaluation.ipynb
├── side_scripts/                 # Utility scripts
│   ├── generate_synthetic_microstructures.py
│   └── augment_dataset.py
├── config.py                     # Global configuration
└── requirements.txt              # Python dependencies
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
| **Max Feret Diameter** | - | Longest distance between any two points on boundary |
| **Min Feret Diameter** | - | Shortest distance (rotating calipers algorithm) |
| **Ellipse Major Axis** | - | Length of fitted ellipse major axis |
| **Ellipse Minor Axis** | - | Length of fitted ellipse minor axis |
| **Aspect Ratio** | Major Axis / Minor Axis | Elongation measure |
| **Orientation** | - | Ellipse orientation angle (degrees) |
| **Roundness** | 4 × Area / (π × Major Axis²) | Alternative shape descriptor |

#### Global Microstructure Descriptors

- **Porosity**: Total pore area / Total image area
- **Local Porosity Variance**: Standard deviation of porosity in sub-regions (measures homogeneity)
- **Anisotropy**: Preferred orientation strength (0 = isotropic, 1 = highly directional)

#### Spatial Relations

- **Nearest Neighbor Distance**: Distance from each pore centroid to its closest neighbor
- **Distribution Plots**: Histograms and density plots of spatial metrics

#### Topology & Connectivity

- **Fractal Dimension**: Minkowski-Bouligand box-counting method (measures structural complexity)
- **Coordination Number**: Average number of neighboring pores via Voronoi tessellation (quantifies connectivity)

### Configuration

Edit [config.py](config.py) to configure paths and pixel sizes:

```python
# Pixel sizes for different SEM magnifications (μm/px)
PIXEL_SIZES = {
    40: 3.24023,
    50: 2.59219,
    100: 1.29609,
    250: 0.51844,
    500: 0.25922,
    1000: 0.12961
}

# Segmentation inference settings
MODEL_PATH_INFERENCE = '/path/to/cellpose/model'
OUTPUT_PATH_INFERENCE = '/path/to/output/masks'
PATH_TO_FILES_INFERENCE = '/path/to/input/images'

# Quantitative analysis output
OUTPUT_PATH = '/path/to/analysis/results'
```

### Synthetic Data Generation

Generate training data with realistic microstructure features:

```bash
python side_scripts/generate_synthetic_microstructures.py \
    --n_samples 1000 \
    --output_dir /path/to/synthetic_data
```

**Features**:
- Voronoi-based pore structure (40-70 pores per image)
- SEM-realistic texture and noise
- Boundary imperfections with jitter
- Pore perforations and contaminants
- Automatic ground truth mask generation

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
    └── comprehensive_analysis.xlsx
```

The Excel report includes:
- **Summary sheet**: Statistical aggregates (mean, median, std, min, max)
- **Raw data sheets**: Complete per-pore measurements
- **Metadata**: Analysis parameters and settings

## Example Workflows

### Load and Filter Data

```python
from materials_vision.data_loader import DataLoader

# Load AS series foam materials
loader = DataLoader('AS')

# Filter by magnification
data_40x = loader.keep_magnification(40)

# Access image paths
for sample_name, image_path in data_40x.items():
    print(f"Sample: {sample_name}, Path: {image_path}")
```

### Evaluate Segmentation Model

See [notebooks/cellpose_finetuned_evaluation.ipynb](notebooks/cellpose_finetuned_evaluation.ipynb) for comprehensive model evaluation workflow:
- IoU-based metrics (precision, recall, F1)
- Boundary score evaluation
- Visualization of results

## Technical Details

### Algorithms Implemented

- **Cellpose**: Generalist deep learning segmentation (cyto3 model architecture)
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

## Notebooks

Explore example workflows in [notebooks/](notebooks/):

- [cellpose_finetuned_evaluation.ipynb](notebooks/cellpose_finetuned_evaluation.ipynb): Evaluate fine-tuned Cellpose model performance
- [cellpose_out_of_the_box_evaluation.ipynb](notebooks/cellpose_out_of_the_box_evaluation.ipynb): Test pretrained models on foam data
- [preprocessing.ipynb](notebooks/preprocessing.ipynb): Preprocessing experiments and parameter tuning
- [manual_annotations_evaluation.ipynb](notebooks/manual_annotations_evaluation.ipynb): Establish baseline performance from human annotations

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

### Contributing

Contributions welcome! The project follows standard Python best practices:
- Comprehensive docstrings and type hints
- Modular architecture with clear separation of concerns
- Extensive error handling and logging
- Scientific rigor in algorithm implementation

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
