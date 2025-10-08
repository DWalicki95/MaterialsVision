from materials_vision.config import DATA_TRAIN_TEST
from datetime import datetime
from pathlib import Path


def get_train_and_test_dir(dataset_name: str):
    '''
    Returns train and test directories that are neccesary f.e. for cellpose
    model retraining.
    '''
    common_dir = DATA_TRAIN_TEST / dataset_name
    train_dir = common_dir / 'train'
    test_dir = common_dir / 'train'
    return str(train_dir), str(test_dir)


def create_current_time_output_directory(dir_base_path: Path):
    '''Creates output directory for f.e. for trained model'''
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(dir_base_path) / f"output_{now}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
