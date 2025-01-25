from pathlib import Path
from typing import Dict, Optional

from materials_vision.config import DATA_PATH_DICT


class DataLoader:
    """Object handling microstructure images data"""
    def __init__(self, series):
        self.data_dict = self.load_data_dict(series)

    def load_data_dict(self, series: str) -> Dict[str, list]:
        """
        Loads dictionary where key is material series name and values are
        paths to images.


        Args:
            series (str): material series name

        Raises:
            ValueError: raises error when incorrect series name

        Returns:
            dict: dictionary where key is material series name and values are
                  paths to images.
        """
        if series not in ["AS", "VAB", "RL"]:
            raise ValueError("Incorrect `series` parameter.")
        main_series_dir_path = DATA_PATH_DICT[series]
        data_dict = self._create_data_dict(main_series_dir_path)
        return data_dict

    def _create_data_dict(self, main_series_dir_path: Path) -> Dict[str, list]:
        '''Creates dictionary to load. Details in load_data_dict method.'''
        data_dict = {}
        subdirs = [
            sdir for sdir in main_series_dir_path.iterdir() if sdir.is_dir()]
        for subdir in subdirs:
            series_name = subdir.name
            img_paths = list(subdir.glob("*.jpg")) + list(subdir.glob("jpeg"))
            data_dict[series_name] = img_paths
        return data_dict

    def keep_magnification(self, magnification: int,
                           series: Optional[str] = None) -> Dict[str, list]:
        """
        Method filters initial data dictionary.

        Args:
            magnification (int): maginification user would like to keep
            series (Optional[str], optional): user can also define exact
                                              material series. Defaults to None

        Returns:
            dict: filtered initial dictionary (to desired magnification and
                  series (if given) )
        """
        pattern = f"_{str(magnification)}_"
        data_dict = {}
        filtered_path_list = []
        filtered_data_dict = {}
        if series:
            data_dict[series] = self.data_dict[series]
        else:
            data_dict = self.data_dict
        for material_series, path_list in data_dict.items():
            for path in path_list:
                if pattern in path.name:
                    filtered_path_list.append(path)
            filtered_data_dict[material_series] = filtered_path_list
        return filtered_data_dict
