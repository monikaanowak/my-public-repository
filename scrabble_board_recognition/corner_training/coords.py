from typing import Tuple

from numpy import ndarray
from numpy.random import random_sample


############ Size operations ############

def ceil_size(size: Tuple[float, float], padding: int):
    import numpy as np
    return (
        int(np.ceil((size[0] + padding - .01) / padding)) * padding,
        int(np.ceil((size[1] + padding - .01) / padding)) * padding,
    )


def scale_size(size: Tuple[float, float], max_size: Tuple[float, float]):
    scale = min(max_size[0] / size[0], max_size[1] / size[1])
    return (
        int(size[0] * scale),
        int(size[1] * scale),
    )


def rand_size(size: Tuple[float, float], k: float):
    return (
        int(size[0] * (1 + k * (random_sample() - .5))),
        int(size[1] * (1 + k * (random_sample() - .5))),
    )


############ Coordinates operations ############

def coords_flip(label_coords: ndarray, shape, axis):
    label_coords = label_coords.copy()
    label_coords[:, axis] = shape[axis] - label_coords[:, axis]
    return label_coords


def coords_rot90(label_coords: ndarray, shape, k):
    import numpy as np

    for _ in range(k):
        label_coords = np.stack([
            shape[1] - label_coords[:, 1],
            label_coords[:, 0],
        ], axis=-1)
    return label_coords


def coords_resize(label_coords: ndarray, old_shape, new_shape):
    label_coords = label_coords.copy()
    label_coords[:, 0] *= new_shape[0] / old_shape[0]
    label_coords[:, 1] *= new_shape[1] / old_shape[1]
    return label_coords
