import random
import warnings
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy.spatial import Voronoi
from shapely import affinity
from shapely.geometry import Point, Polygon

from materials_vision.data_loader import DataLoader
from materials_vision.preprocessing_utils import crop_image


def generate_artifical_images(
    plot_sample: bool = False,
    seed: bool = True,
    perforation_params: dict = {
        'min_holes_per_pore': 0,
        'max_holes_per_pore': 2,
        'min_major': 10,
        'max_major': 40,
        'min_aspect': 0.4,
        'max_deform': 0.5,
        'rotation': True,
        'max_attempts_per_hole': 50
    },
    contamination_params: dict = {
        'min_contaminants': 2,
        'max_contaminants': 8,
        'min_radius': 2,
        'max_radius': 8,
        'intensity_variation': 20  # max diff. of contaminant color to pore
    },
    add_boundary_noise: bool = True
) -> Tuple[
    ImageDraw.Draw, List[dict], Dict[str, dict], np.ndarray, np.ndarray
]:
    """
    Generate artificial images of porous material microstructure using
    Voronoi tesselation.


    Parameters
    ----------
    plot_sample : bool, optional
        Shows created image while True, by default False
    seed : bool, optional
        True ensures deterministic behavior, by default True
    perforation_params : dict, optional
        Dictionary of parameters used during adding perforations to the
        structure
    contamination_params : dict, optional
        Dictionary of parameters used during adding contaminations to the
        structure
    add_boundary_noise : bool, optional
        Flag parameter, if True then add some noise on boundary,
        by default True
    """
    metadata = []
    params = {
        'pertforation_params': perforation_params,
        'contamination_params': contamination_params
    }
    if seed:
        random.seed(42)
        np.random.seed(42)
    real_img = load_sample_real_img()
    real_img_cropped = crop_image(real_img)
    real_img_shape = real_img_cropped.shape
    # assumed other important parameters
    margin = 1
    boundary_width = 4
    gray_low = 100
    gray_high = 160
    img_width = real_img_shape[1]
    img_height = real_img_shape[0]
    pores_num = random.randint(40, 70)
    # handle black spaces in the corners of the image
    points = generate_points_with_mirroring(pores_num, margin)
    vor = Voronoi(points)
    img = Image.new('L', (img_width, img_height), color=0)
    draw = ImageDraw.Draw(img)
    # variables for pores counting
    pore_id = 1
    pore_registry = {}  # track mirrored duplicates
    for i, point in enumerate(vor.point_region):
        region = vor.regions[point]
        if -1 in region or len(region) == 0:
            continue
        polygon_ = [vor.vertices[v] for v in region]
        scaled_polygon = [
            (x * img_width, y * img_height) for (x, y) in polygon_
        ]
        # count pores
        if is_pore_valid(
            i, pores_num, scaled_polygon, img_width, img_height
        ):
            pore_registry[i] = True
            shapely_poly = Polygon(scaled_polygon)
            poly_centroid_x = shapely_poly.centroid.x
            poly_centroid_y = shapely_poly.centroid.y
            metadata.append(
                {
                    'id': pore_id,
                    'area': shapely_poly.area,
                    'centroid': (poly_centroid_x, poly_centroid_y),
                    'polygon': scaled_polygon

                }
            )
            pore_id += 1
        # fill with random grayscale
        gray = np.random.randint(gray_low, gray_high)
        draw.polygon(scaled_polygon, fill=gray)
        # operations to make similar to real image microstructure features
        add_contaminants(draw, scaled_polygon, gray, contamination_params)
        add_perforations(draw, scaled_polygon, perforation_params)

    masks = generate_pore_masks(
        metadata,
        img_width,
        img_height,
        (boundary_width + 1) // 2
    )
    combined_mask = create_combined_mask(masks)

    img = draw_boundaries(
        vor=vor,
        img=img,
        img_width=img_width,
        img_height=img_height,
        boundary_width=boundary_width,
        add_boundary_noise=add_boundary_noise
    )
    img = add_sem_effects(img)
    if plot_sample:
        img.show()
    return img, metadata, params, combined_mask, masks


def load_sample_real_img():
    '''Loads one of the collected raw real images.'''
    real_img = DataLoader('AS').keep_magnification(40, 'AS1')['AS1'][0]
    return cv2.imread(real_img)


def generate_points_with_mirroring(n: int, margin: int) -> np.ndarray:
    """
    Generates n points which form Voronoi cells. For each base point function
    creates 4 mirrored point in order to fully cover image (avoid black spaces
    on the edges).

    Parameters
    ----------
    n : int
        The base number of cells before adding mirrored points
    margin : int
        The number by which the base range (0, 1) of te Voronoi point
        coordinates can draw is reduced; how much from the edge of the image a
        point can appear


    Returns
    -------
    np.ndarray
        array containing base point coords and mirrored coords
    """
    points = np.random.uniform(low=margin, high=1-margin, size=(n, 2))
    mirrored = []
    for x, y in points:
        mirrored += [
            (x, 1 + (1 - y)),
            (x, -y),
            (1 + (1 - x), y),
            (-x, y)
        ]
    return np.vstack([points, np.array(mirrored)])


def is_pore_valid(
    i: int,
    pores_num: int,
    polygon_: List[tuple],
    img_width: int,
    img_height: int
) -> bool:
    """
    Checks if pore (Voronoi cell) is Valid (within original image pixel range).

    Parameters
    ----------
    i : int
       Number of Voronoi cell region
    pores_num : int
        Expected number of pores
    polygon_ : List[tuple]
        Coords of cell vertices
    img_width : int
        Width of original / real image
    img_height : int
        Height of original / real image

    Returns
    -------
    bool
        True if pore is valid else False
    """
    is_pore_valid = False
    if i < pores_num:
        min_x = min(p[0] for p in polygon_)
        max_x = max(p[0] for p in polygon_)
        min_y = min(p[1] for p in polygon_)
        max_y = max(p[1] for p in polygon_)
        if (
            max_x > 0 and min_x < img_width and
            max_y > 0 and min_y < img_height
        ):
            is_pore_valid = True
    return is_pore_valid


def draw_boundaries(
    vor: Voronoi,
    img: np.ndarray,
    img_width: int,
    img_height: int,
    boundary_width: float = 4.0,
    add_boundary_noise: bool = True,
    noise_intensity: float = 1.5,
    jitter_strength: float = 1.2
) -> np.ndarray:
    """
    Draw Voronoi cells boundaries.

    Parameters
    ----------
    vor : Voronoi
        Voronoi diagram object that contains all geometric information
        (vertices, ridges, regions)
    img : np.ndarray
        Empty image array of width and height like original microstructure
        image where boundaries will be drawn
    img_width : int
        Width of real microstructure image (used to scale Voronoi coords to
        pixel coords)
    img_height : int
        Height of real microstructure image (used to scale Voronoi coords to
        pixel coords)
    boundary_width : float, optional
        Width of cell ("pore") boundary in pixels, by default 3.0
    add_boundary_noise : bool, optional
        Flag parameter, if True then add some noise on boundaries to simulate
        imperfections seen in real images, by default True
    noise_intensity : float, optional
        Intensity of boundary noise (how much boundaries deviate from straight
        line), by default 1.5
    jitter_strength : float, optional
        Variable added to boundary width, by default 1.2

    Returns
    -------
    np.ndarray
        Image with pores' boundaries
    """
    draw = ImageDraw.Draw(img)
    max_deviation = boundary_width * 0.4
    for ridge in vor.ridge_vertices:
        # filter out "infinite" ridges at diagram edges
        if -1 in ridge:
            continue
        v0, v1 = vor.vertices[ridge[0]], vor.vertices[ridge[1]]
        start = (v0[0] * img_width, v0[1] * img_height)
        end = (v1[0] * img_width, v1[1] * img_height)

        if add_boundary_noise:
            start_np = np.array(start)
            end_np = np.array(end)
            vec = end_np - start_np
            length = np.linalg.norm(vec)
            direction = vec / length if length > 0 else vec
            perp = np.array([-direction[1], direction[0]])
            num_segments = max(3, int(length / 15))
            t = np.linspace(0, 1, num_segments)
            points = start_np + t[:, None] * vec.reshape(1, -1)
            noise = np.clip(
                noise_intensity * np.random.randn(num_segments),
                -max_deviation, max_deviation
            )
            points += noise[:, None] * perp.reshape(1, -1)
            # ensure points stay within Voronoi polygon bounds
            for i in range(len(points)):
                points[i] = np.clip(
                    points[i],
                    [boundary_width, boundary_width],
                    [img_width-boundary_width, img_height-boundary_width]
                )
            for i in range(len(points)-1):
                seg_start = tuple(points[i])
                seg_end = tuple(points[i+1])
                current_width = max(
                    1, int(
                        boundary_width + jitter_strength * np.random.randn()
                    )
                )
                draw.line([seg_start, seg_end], fill=180, width=current_width)
        else:
            draw.line([start, end], fill=180, width=boundary_width)
    return img


def add_sem_effects(img: np.ndarray) -> np.ndarray:
    """
    Add some operations to make artificial images more like real ones

    Parameters
    ----------
    img : np.ndarray
        Image with artificial pores before any global image effects

    Returns
    -------
    np.ndarray
        Transformed image
    """
    texture = Image.effect_noise(img.size, 32).convert('L')
    img = Image.blend(img, texture, alpha=0.1)  # 10% of texture and 90% of
    # original image
    img = img.filter(ImageFilter.SHARPEN)
    img = img.filter(ImageFilter.GaussianBlur(radius=0.8))
    img = img.point(lambda x: 0 if x < 50 else 255 if x > 200 else x)
    return img


def add_perforations(
    draw: ImageDraw.Draw,
    polygon_: List[tuple],
    params: dict
):
    """
    Adds perforations (small dark holes) inside Voronoi pores (cells).

    Parameters
    ----------
    draw : ImageDraw.Draw
        Drawing context for the PIL Image where perforations will be rendered.
    polygon_ : List[tuple]
        Pore boundary coordinates.
    params : dict
        Configuration dictionary with following keys: min and max number of
        perforations per pore, max attempts to render a perforations before
        giving up, min and max major axis lengths in pixels, min aspect ratio
        (min/ major) [0-1], max shape deformation (0 = perfect ellipse,
        0.2=+/-20% distortion, rotation if True then randlomly rotate ellipse)
    """
    # it's neccessary not to show warnings generated by shapely module
    # for completion task it's not a problem that perforations overlap or
    # doesn't stay within cell boundaries (it happens in real image)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            module="shapely"
        )
        spoly = Polygon(polygon_)
        if not spoly.is_valid:
            return
        min_x, min_y, max_x, max_y = spoly.bounds
        num_holes = np.random.randint(params['min_holes_per_pore'],
                                      params['max_holes_per_pore']+1)

        for _ in range(num_holes):
            added = False
            attempts = 0
            while not added and attempts < params['max_attempts_per_hole']:
                major = np.random.uniform(
                    params['min_major'], params['max_major']
                )
                aspect = np.random.uniform(params['min_aspect'], 1)
                minor = major * aspect

                angle = np.random.uniform(0, 360) if params['rotation'] else 0

                deform_x = 1 + np.random.uniform(
                    -params['max_deform'], params['max_deform'])
                deform_y = 1 + np.random.uniform(
                    -params['max_deform'], params['max_deform'])

                x = np.random.uniform(min_x + major, max_x - major)
                y = np.random.uniform(min_y + minor, max_y - minor)

                ellipse = Point(x, y).buffer(1)
                ellipse = affinity.scale(
                    ellipse, major*deform_x, minor*deform_y
                )
                ellipse = affinity.rotate(ellipse, angle)

                if spoly.contains(ellipse.buffer(-1)):
                    coords = np.array(ellipse.exterior.coords.xy).T
                    perf_color = np.random.randint(0, 30)
                    draw.polygon([tuple(p) for p in coords], fill=perf_color)
                    added = True
                attempts += 1


def add_contaminants(
    draw: ImageDraw.Draw,
    polygon_: List[tuple],
    base_gray: int,
    params: dict
):
    """
    Adds oval contaminants to te microstructure.

    Parameters
    ----------
    draw : ImageDraw.Draw
        Drawing context for the PIL images where contaminants will be rendered
    polygon_ : List[tuple]
        Pore boundary coords
    base_gray : int
        Pore gray color number
    params : dict
        Configuration dictionary contain: min and max numbers of contaminations
        per pore, min and max contaminations
    """
    # it's neccessary not to show warnings generated by shapely module
    # for completion task it's not a problem that contaminations overlap or
    # doesn't stay within cell boundaries (it happens in real image)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            module="shapely"
        )
        spoly = Polygon(polygon_)
        if not spoly.is_valid:
            return
        min_x, min_y, max_x, max_y = spoly.bounds
        num_contaminants = np.random.randint(params['min_contaminants'],
                                             params['max_contaminants']+1)
        for _ in range(num_contaminants):
            for attempt in range(50):
                radius = np.random.uniform(
                    params['min_radius'], params['max_radius'])
                x = np.random.uniform(min_x + radius, max_x - radius)
                y = np.random.uniform(min_y + radius, max_y - radius)
                contaminant = Point(x, y).buffer(radius)
                if spoly.contains(contaminant):
                    contaminant_gray = base_gray + np.random.randint(
                        -params['intensity_variation'],
                        params['intensity_variation']
                    )
                    contaminant_gray = np.clip(contaminant_gray, 0, 255)
                    # draw with structural noise
                    bbox = [x-radius, y-radius, x+radius, y+radius]
                    draw.ellipse(bbox, fill=int(contaminant_gray))
                    break


def generate_pore_masks(
    metadata: List[dict],
    img_width: int,
    img_height: int,
    boundary_margin: int = 2
) -> np.ndarray:
    """
    Generate binary masks for pores in the Voronoi diagram,
    excluding boundaries.

    Parameters
    ----------
    metadata : List[dict]
        List of dictionaries containing information about each valid pore
    img_width : int
        Width of the target image in pixels
    img_height : int
        Height of the target image in pixels
    boundary_margin : int, optional
        Number of pixels to exclude around cell boundaries to ensure
        clean separation between pores, by default 2

    Returns
    -------
    np.ndarray
        3D numpy array where each channel is a binary mask for a single pore
        Shape: (num_pores, height, width)
    """
    # create empty array to store all masks
    num_pores = len(metadata)
    masks = np.zeros((num_pores, img_height, img_width), dtype=np.uint8)

    # generate mask for each pore
    for idx, pore_data in enumerate(metadata):
        # create new image for this pore's mask
        pore_img = Image.new('L', (img_width, img_height), color=0)
        pore_draw = ImageDraw.Draw(pore_img)

        # draw the polygon for this pore
        polygon_ = pore_data['polygon']

        # Create slightly smaller polygon to avoid boundaries
        if boundary_margin > 0:
            # Convert to shapely polygon for inward buffer
            shapely_poly = Polygon(polygon_)
            smaller_poly = shapely_poly.buffer(-boundary_margin)
            # handle potential geometry splits from buffer
            if smaller_poly.geom_type == 'Polygon':
                # convert back to coordinate list
                smaller_poly = list(smaller_poly.exterior.coords)
                pore_draw.polygon(smaller_poly, fill=255)
            elif smaller_poly.geom_type == 'MultiPolygon':
                # handle each part separately
                for part in smaller_poly.geoms:
                    coords = list(part.exterior.coords)
                    pore_draw.polygon(coords, fill=255)
        else:
            # use original polygon if no margin needed
            pore_draw.polygon(polygon_, fill=255)

        masks[idx] = np.array(pore_img)
    return masks


def create_combined_mask(masks: np.ndarray) -> np.ndarray:
    """
    Create a single mask where each pore has a unique ID label.

    Parameters
    ----------
    masks : np.ndarray
        3D array of individual pore masks

    Returns
    -------
    np.ndarray
        2D array where each pore is labeled with a unique integer (
        1 to num_pores)
    """
    combined_mask = np.zeros((masks.shape[1], masks.shape[2]), dtype=np.uint16)
    for i in range(masks.shape[0]):
        combined_mask[masks[i] > 0] = i + 1
    return combined_mask


if __name__ == '__main__':
    generate_artifical_images(
        plot_sample=True,
        seed=False,
        add_boundary_noise=False
    )
