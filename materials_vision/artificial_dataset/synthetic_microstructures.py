import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from PIL import ImageDraw
from tqdm import tqdm

from materials_vision.artificial_dataset.create_voronoi_diagrams import \
    generate_artifical_images
from materials_vision.config import SYNTHETIC_DATASET_PATH_LOCAL_DRIVE

logger = logging.getLogger(__name__)


class SyntheticMicrostructuresGenerator():
    '''Synthetic microstructures creation and management object. '''
    def __init__(self, n_samples: int = 1, dataset_name: str = ''):
        """Initialize class.

        Parameters
        ----------
        n_samples : int, optional
            Number of samples to create, by default 1
        """
        self.n_samples = n_samples
        self.dataset_name = dataset_name

    def generate_artificial_microstructures(
            self,
            save: bool = False,
            save_path: Path = SYNTHETIC_DATASET_PATH_LOCAL_DRIVE
    ) -> dict:
        """
        Generates synthetic microstructures and store them in dictionary.

        Parameters
        ----------
        save : bool, optional
            Saves dataset if True else False, by default False

        Returns
        -------
        dict
            Dictionary which contains image index, pixel array, metadata,
            number of pores and params used for image creation
        """
        self.dataset_dict = {}
        for n in tqdm(range(self.n_samples), desc='Generowanie zbioru',
                      mininterval=10):
            img_dict = {}
            img, metadata, params, combined_mask, invalid_mask_pore_id = (
                generate_artifical_images(
                    plot_sample=False,
                    seed=False,
                    add_boundary_noise=False
                )
            )
            invalid_pores_in_mask = len(invalid_mask_pore_id)
            if invalid_pores_in_mask > 0:
                logger.warning(
                    f'Invalid pores in combined mask detected ! \n'
                    f'Number of invalid pores in mask: '
                    f'{invalid_pores_in_mask} \n'
                    f'Invalid pores ids: {invalid_mask_pore_id}.'
                )
            img_idx = n + 1
            img_dict['image'] = img
            img_dict['metadata'] = metadata
            img_dict['n_pores'] = len(metadata)
            img_dict['params'] = params
            img_dict['mask'] = combined_mask
            self.dataset_dict[img_idx] = img_dict
            if save:
                self._save_dataset_(
                    img=img,
                    img_idx=img_idx,
                    metadata=metadata,
                    combined_mask=combined_mask,
                    params=params,
                    dataset_name=self.dataset_name,
                    dataset_path=save_path
                )
        return self.dataset_dict

    def visualize_pores_mask(
        self,
        dataset_dict: Dict[int, Dict[str, Any]],
        img_idx: int = 1,
        visualization_type: str = 'all'
    ) -> None:
        """
        Visualize pore masks with multiple display options to verify boundary
        exclusion.

        Parameters
        ----------
        dataset_dict : Dict[int, Dict[str, Any]]
            Dictionary containing image and mask data
        img_idx : int
            Index of the image to visualize
        visualization_type : str
            Type of visualization to show:
            - 'all': Shows original, mask, and overlay in separate subplots
            - 'overlay': Shows mask overlaid on original image
            - 'boundaries': Shows mask edges to verify boundary exclusion
            - 'mask_only': Shows only the colored mask
        """
        img_dict = dataset_dict[img_idx]
        original_img = img_dict['image'].convert('RGB')
        mask = img_dict['mask']
        cmap = cm.rainbow

        if visualization_type == 'all':
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

            # original image
            ax1.imshow(original_img)
            ax1.set_title('Original Image')
            ax1.axis('off')

            # mask
            masked = ax2.imshow(mask, cmap=cmap)
            ax2.set_title('Pore Masks')
            ax2.axis('off')
            plt.colorbar(masked, ax=ax2, label='Pore ID')

            # overlay
            ax3.imshow(original_img)
            overlay = ax3.imshow(mask, alpha=0.5, cmap=cmap)
            ax3.set_title('Overlay')
            ax3.axis('off')
            plt.colorbar(overlay, ax=ax3, label='Pore ID')

        elif visualization_type == 'boundaries':
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            # create edge detection mask
            mask_edges = np.zeros_like(mask)
            mask_edges[:-1, :] |= (mask[1:, :] != mask[:-1, :])
            mask_edges[1:, :] |= (mask[:-1, :] != mask[1:, :])
            mask_edges[:, :-1] |= (mask[:, 1:] != mask[:, :-1])
            mask_edges[:, 1:] |= (mask[:, :-1] != mask[:, 1:])

            # original with detected boundaries
            ax1.imshow(original_img)
            ax1.imshow(mask_edges, alpha=0.5, cmap='Reds')
            ax1.set_title('Boundary Check')
            ax1.axis('off')

            # zoom into a region to check boundaries
            height, width = mask.shape
            zoom_size = min(width, height) // 4
            center_y, center_x = height // 2, width // 2

            zoomed_original = np.array(original_img)[
                center_y-zoom_size:center_y+zoom_size,
                center_x-zoom_size:center_x+zoom_size
            ]
            zoomed_edges = mask_edges[
                center_y-zoom_size:center_y+zoom_size,
                center_x-zoom_size:center_x+zoom_size
            ]

            ax2.imshow(zoomed_original)
            ax2.imshow(zoomed_edges, alpha=0.5, cmap='Reds')
            ax2.set_title('Zoomed Boundary Check')
            ax2.axis('off')

        elif visualization_type == 'mask_only':
            plt.figure(figsize=(8, 8))
            masked = plt.imshow(mask, cmap=cmap)
            plt.title('Pore Masks')
            plt.axis('off')
            plt.colorbar(masked, label='Pore ID')

        elif visualization_type == 'overlay':
            plt.figure(figsize=(8, 8))
            plt.imshow(original_img)
            overlay = plt.imshow(mask, alpha=0.5, cmap=cmap)
            plt.title('Mask Overlay')
            plt.axis('off')
            plt.colorbar(overlay, label='Pore ID')

        plt.tight_layout()
        plt.show()

    def _save_dataset_(
            self,
            img: ImageDraw.Draw,
            img_idx: int,
            metadata: List[dict],
            combined_mask: np.ndarray,
            params: dict,
            dataset_name: str = '',
            dataset_path: Path = SYNTHETIC_DATASET_PATH_LOCAL_DRIVE
    ) -> None:
        '''
        Saves synthetic image and pores mask as array (npy) and fially metadata
        of each sample as json file.
        '''
        save_path_suffix_main = dataset_path / (
            f'synthetic_dataset_{dataset_name}'
        )
        save_path_suffix_main.mkdir(parents=True, exist_ok=True)
        save_path_suffix = save_path_suffix_main / f'sample_{img_idx}'
        # save original image
        save_path_img = f'{save_path_suffix}_image.npy'
        np.save(save_path_img, np.array(img))
        # save combined mask
        save_path_combined_mask_npy = f'{save_path_suffix}_combined_mask.npy'
        np.save(save_path_combined_mask_npy, combined_mask)
        save_path_metadata = f'{save_path_suffix}_metadata.json'
        with open(save_path_metadata, 'w') as f:
            json.dump(
                {
                    'n_pores': len(metadata),
                    'pores_metadata': metadata,
                    'params': params
                }, f, indent=2
            )
