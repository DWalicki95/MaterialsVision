import numpy as np


def crop_image(img: np.ndarray, height_after_cropping: int = 960):
    """Crops image to remove the scale."""
    return img[0:height_after_cropping]
