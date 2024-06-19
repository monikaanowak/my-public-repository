from typing import Tuple
import numpy as np


def make_box(minx, miny, maxx, maxy):
    return (minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)


def find_perspective_transform(source_points, im_shape: Tuple[float, float]):
    import cv2 as cv

    source_points = np.array(source_points[:4], dtype=np.float32)
    destination_points = np.array(make_box(0, 0, im_shape[0], im_shape[1]), dtype=np.float32)
    return cv.getPerspectiveTransform(source_points, destination_points)


def perspective_transform_points(transformation, points):
    import cv2 as cv

    if not len(points):
        return []
    points = np.array([points], dtype=np.float64)
    return cv.perspectiveTransform(points, transformation)[0]

def perspective_transform_points_swap(transformation, points):
    import cv2 as cv

    if not len(points):
        return []
    points = np.array([points], dtype=np.float64)[..., ::-1]
    return cv.perspectiveTransform(points, transformation)[0][..., ::-1]


# first image shape vertex must be in the top left corner (0,0)
def cut_image_box(
        im: np.ndarray, corners, out_size: Tuple[int, int]
):
    import cv2 as cv

    if len(corners) != 4:
        raise ValueError(f'Provided coords are not a rectangle: {corners}')

    to_box_pixels_transform = find_perspective_transform(
        corners, out_size
    )
    return cv.warpPerspective(
        im, to_box_pixels_transform, out_size, flags=cv.INTER_LANCZOS4
    )
