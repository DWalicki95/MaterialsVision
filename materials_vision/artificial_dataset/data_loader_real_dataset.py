from pathlib import Path
from typing import Dict, Optional


class DataLoader:
    """Object handling microstructure images data."""

    def __init__(self, series_path: Path):
        """
        Parameters
        ----------
        series_path : Path
            Path to directory containing subdirectories with images
            for a given material series.
        """
        self.series_path = Path(series_path)
        self.data_dict = self._create_data_dict(self.series_path)

    def _create_data_dict(
            self, main_series_dir_path: Path
    ) -> Dict[str, list]:
        """
        Build dictionary mapping series-subdirectory name to image paths.

        Parameters
        ----------
        main_series_dir_path : Path
            Root directory of a material series.

        Returns
        -------
        dict
            Keys are subdirectory names, values are lists of image paths.
        """
        data_dict = {}
        subdirs = [
            sdir for sdir in main_series_dir_path.iterdir()
            if sdir.is_dir()
        ]
        for subdir in subdirs:
            series_name = subdir.name
            img_paths = (
                list(subdir.glob("*.jpg"))
                + list(subdir.glob("jpeg"))
            )
            data_dict[series_name] = img_paths
        return data_dict

    def keep_magnification(
            self,
            magnification: int,
            series: Optional[str] = None
    ) -> Dict[str, list]:
        """
        Filter the data dictionary to a desired magnification.

        Parameters
        ----------
        magnification : int
            Magnification value to keep.
        series : str, optional
            Restrict filtering to a single series subdirectory.

        Returns
        -------
        dict
            Filtered initial dictionary (to desired magnification and
            series, if given).
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
