import json
import random
from typing import List

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from PIL import ImageDraw
from tqdm import tqdm

from materials_vision.artificial_dataset.create_voronoi_diagrams import \
    generate_artifical_images
from materials_vision.config import ARTIFICAL_DATASET_PATH


class SyntheticMicrostructuresGenerator():
    '''Synthetic microstructures creation and management object. '''
    def __init__(self, n_samples: int = 1):
        """Initialize class.

        Parameters
        ----------
        n_samples : int, optional
            Number of samples to create, by default 1
        """
        self.n_samples = n_samples

    def generate_artificial_microstructures(self, save: bool = False) -> dict:
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
        for n in tqdm(range(self.n_samples), desc='Generowanie zbioru'):
            img_dict = {}
            img, metadata, params = generate_artifical_images(
                plot_sample=False,
                seed=False,
                add_boundary_noise=False
            )
            img_idx = n + 1
            img_dict['image'] = img
            img_dict['metadata'] = metadata
            img_dict['n_pores'] = len(metadata)
            img_dict['params'] = params
            self.dataset_dict[img_idx] = img_dict
            if save:
                self._save_dataset_(
                    img=img,
                    img_idx=img_idx,
                    metadata=metadata,
                    polygon_=[x['polygon'] for x in metadata],
                    params=params
                )

        return self.dataset_dict

    def visualize_pores_mask(self, dataset_dict: dict, img_idx: int = 1):
        '''Plot image of synthetic image and precise pore segmentation mask'''
        img_dict = dataset_dict[img_idx]
        img = img_dict['image']
        n_pores = img_dict['n_pores']
        fig, ax = plt.subplots()
        cmap = cm.rainbow
        colors = [cmap(i/n_pores) for i in range(n_pores)]
        ax.imshow(img, cmap='gray', origin='upper', alpha=0.8)
        for j, pore in enumerate(img_dict['metadata']):
            polygon_ = [(x, y) for (x, y) in pore['polygon']]
            pore_color = colors[j % n_pores]
            ax.fill(
                *zip(*polygon_),
                alpha=0.5,
                edgecolor='black',
                facecolor=pore_color
            )
            centroid = pore['centroid']
            pore_id = pore['id']
            ax.text(
                centroid[0], centroid[1], str(pore_id),
                ha='center', va='center',
                fontsize=8, color='black'
            )
        plt.title(
            'Segmentacja porów sztucznej mikrostruktury. \n'
            f'Idx zdjęcia: {img_idx}. \n'
            f'Liczba porów: {n_pores}'
        )
        ax.axis('off')
        plt.show()

    def _save_dataset_(
            self,
            img: ImageDraw.Draw,
            img_idx: int,
            metadata: List[dict],
            polygon_: List[tuple],
            params: dict,
            dataset_name: str = ''
    ):
        '''
        Saves synthetic image as array, pores mask as array of arrays of
        vertices and metadata as json file.
        '''
        save_path_suffix = ARTIFICAL_DATASET_PATH / (
            f'synthetic_dataset_{dataset_name}'
        )
        save_path_suffix.mkdir(parents=True, exist_ok=True)
        save_path_suffix = save_path_suffix / f'sample_{img_idx}'
        save_path_img = f'{save_path_suffix}_image.npy'
        save_path_mask = f'{save_path_suffix}_mask.npy'
        save_path_metadata = f'{save_path_suffix}_metadata.json'
        np.save(save_path_img, np.array(img))
        polygon_np = np.array([np.array(p) for p in polygon_], dtype=object)
        np.save(save_path_mask, polygon_np)
        with open(save_path_metadata, 'w') as f:
            json.dump(
                {
                    'n_pores': len(metadata),
                    'pores_metadata': metadata,
                    'params': params
                }, f, indent=2
            )

    def visually_check_saved_sample(self, dataset_name: str = ''):
        """
        Choose, loads and plots randomly selected image stored in directory

        Parameters
        ----------
        dataset_name : str, optional
            In accordance to adopted convetion dataset name is a suffix in
            directory name of generated dataset, by default ''
        """
        dataset_path = ARTIFICAL_DATASET_PATH / (
            f'synthetic_dataset_{dataset_name}'
        )
        images = list(dataset_path.glob('*image*'))
        chosen_img = random.choice(images)
        chosen_img_num = chosen_img.name.split('_')[1]
        chosen_metadata = dataset_path / (
            f'sample_{chosen_img_num}_metadata.json'
        )
        loaded_img = np.load(chosen_img)
        with open(chosen_metadata, 'r') as f:
            loaded_metadata = json.load(f)
        sample_img_dict = {
            'image': loaded_img,
            'n_pores': loaded_metadata['n_pores'],
            'metadata': loaded_metadata['pores_metadata']

        }
        sample_dataset_dict = {
            1: sample_img_dict
        }
        self.visualize_pores_mask(sample_dataset_dict, 1)
