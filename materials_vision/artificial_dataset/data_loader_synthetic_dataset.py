from torch.utils.data import Dataset
from pathlib import Path
import torch
import numpy as np
from materials_vision.config import SYNTHETIC_DATASET_PATH_CLOUD
import logging


logger = logging.getLogger(__name__)


class SynthteticMicrostructuresDataset(Dataset):
    def __init__(
            self,
            root_dir: Path = (
                SYNTHETIC_DATASET_PATH_CLOUD / 'synthetic_dataset_'
            ),
            transform: bool = None
    ):
        self.root_dir = root_dir
        self.transform = transform
        # for evaluating dataset size
        self.mask_name_pattern = '*combined_mask.npy'
        self.num_of_masks_in_dataset = sum(
            1 for _ in self.root_dir.glob(self.mask_name_pattern)
        )

    def __len__(self):
        return self.num_of_masks_in_dataset

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name_pattern = f'sample_{idx+1}_image.npy'
        image = self._load_file_(img_name_pattern, idx+1)
        mask_name_pattern = f'sample_{idx+1}_combined_mask.npy'
        mask = self._load_file_(mask_name_pattern, idx)
        sample = {'image': image, 'mask': mask}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def _load_file_(self, file_name_pattern: str, idx: int):
        file_name = list(self.root_dir.glob(file_name_pattern))
        img_name = self._validate_file_presence_in_directory_(file_name, idx)
        return torch.from_numpy(np.load(img_name))

    def _validate_file_presence_in_directory_(self, file_name: str, idx: int):
        if len(file_name) == 1:
            file_name = file_name[0]
        elif len(file_name) > 1:
            logger.warning(
                f'File duplicat found: {file_name}. First of them loaded'
            )
            file_name = file_name[0]
        else:
            raise ValueError(f'Cannot find file of idx: {idx}')
        return file_name
