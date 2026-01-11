from pathlib import Path

# QUANTITATIVE PORE ANALYSIS
PIXEL_SIZE = 3.24023  # from SEM metadata (um/px)
OUTPUT_PATH = Path(
    '/home/dwalicki/results/cellpose_finetuned'
)
PIXEL_SIZES = {
    40: 3.24023,
    50: 2.59219,
    100: 1.29609,
    250: 0.51844,
    500: 0.25922,
    1000: 0.12961
}

# SEGMENTATION
MODEL_PATH_INFERENCE = '/mnt/c/Projekty/cpsam-40-50_train augmented - only 40 & 50 magnifications_20251105_084436'
OUTPUT_PATH_INFERENCE = '/mnt/c/Projekty/results/maski_cellpose_inferencja'
PATH_TO_FILES_INFERENCE = '/mnt/c/Projekty/test_40_50'

