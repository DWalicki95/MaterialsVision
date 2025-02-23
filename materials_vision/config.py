import os
from pathlib import Path

if 'COLAB_GPU' in os.environ or 'COLAB_RUNTIME' in os.environ:
    MAIN_DIR = Path(
        r"/content/drive/MyDrive"
    )
    print('Wczytuje ściezki do danych w Google Drive')
else:
    MAIN_DIR = Path(
        r"/Users/dawidwalicki/Documents/OneDrive/",
        r"OneDrive - Politechnika Warszawska"
    )
    print('Wczytuje ściezki do danych w OneDrive')


# == DANE ==
DATA = MAIN_DIR / "Dane"
# PIANKI SERIA RL
DATA_DG = DATA / "DG"
DATA_DG_RL = DATA_DG / "Pianki RL"
DATA_DG_RL_IMAGES_DIR = DATA_DG_RL / "31_03"
DATA_DG_RL_RESULTS = DATA_DG_RL / "pianki_RL_wyniki.xlsx"
# PIANKI SERIA VAB
DATA_DG_VAB = DATA_DG / "Pianki VAB" / "Zdjęcia SEM i raporty"
DATA_DG_VAB_STEREO = DATA_DG_VAB / "Mikroskop stereoskopowy"
DATA_DG_VAB_SEM = DATA_DG_VAB / "SEM"
DATA_DG_VAB_SEM_PERPEND = DATA_DG_VAB_SEM / "11.10.2022"  # rownolegle do pow.
DATA_DG_VAB_SEM_PARALLEL = DATA_DG_VAB_SEM / "20.10.2022 GK"  # prostop. -//-
DATA_DG_VAB_RESULTS = DATA_DG_VAB_RESULTS = (
    DATA_DG_VAB_SEM_PARALLEL / "Zestawienie wielkości porów.xlsx"
)


DATA_MA = DATA / "MA"

# PIANKI SERIA AS
DATA_AS = DATA / "AS"
DATA_AS_RESULTS = DATA_AS / "Dataset.xlsx"

DATA_PATH_DICT = {"AS": DATA_AS}

# == MODELE ==
SAM_CHECKPOINTS = MAIN_DIR / "SAM_checkpoints"
SAM_VIT_B_WEIGHTS = SAM_CHECKPOINTS / "sam_vit_b_01ec64.pth"

# == OUTPUT_PATHS ==

# VORONOI DIAGRAMS
SYNTHETIC_DATASET_PATH = DATA / "synthetic_dataset"
