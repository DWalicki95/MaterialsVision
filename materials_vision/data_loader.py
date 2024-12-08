from pathlib import Path

from config import DATA_PATH_DICT


class DataLoader:
    def __init__(self):
        pass

    def load_data_dict(self, series: str):
        if series not in ["AS", "VAB", "RL"]:
            raise ValueError("Incorrect `series` parameter.")
        main_series_dir_path = DATA_PATH_DICT[series]
        data_dict = self._create_data_dict(main_series_dir_path)
        return data_dict

    def _create_data_dict(self, main_series_dir_path: Path):
        data_dict = {}
        subdirs = [
            sdir for sdir in main_series_dir_path.iterdir() if sdir.is_dir()]
        for subdir in subdirs:
            series_name = subdir.name
            img_paths = list(subdir.glob("*.jpg")) + list(subdir.glob("jpeg"))
            data_dict[series_name] = img_paths
        return data_dict
