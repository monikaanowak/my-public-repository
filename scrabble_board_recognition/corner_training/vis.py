################ Display images ################
from contextlib import contextmanager
from typing import Dict, Optional, Tuple

from numpy import ndarray


@contextmanager
def opencv_display(wait_timeout=10., read_input=True, opencv_wait_ms=50):
    import cv2 as cv
    try:
        yield
        if cv.waitKey(0) == 27:  # ESC!
            raise KeyboardInterrupt('ESC clicked')
    finally:
        cv.destroyAllWindows()


def format_to_show(im: ndarray, max_size=(800, 600)) -> ndarray:
    import numpy as np
    import cv2 as cv

    im = im.copy()
    if im.dtype == np.float32:
        im = (im * 255).astype(np.uint8)
    else:
        im = np.clip(im, 0, 255)
        im = im.astype(np.uint8)
    w, h = im.shape[:2]
    w_max, h_max = max_size
    if w > w_max or h > h_max:
        scale = min(w_max/w, h_max/h)
        im = cv.resize(im, (int(h*scale), int(w*scale)), interpolation=cv.INTER_CUBIC)

    return im


def draw_corners(im, corners):
    import numpy as np
    import cv2 as cv

    corners = np.int0(corners)

    for i in corners:
        cv.circle(im, i[-2:][::-1], 10, (0, 0, 255), 3)


def plot_time_graphs(
    data: Dict[str, ndarray],
    draw_keys: Tuple[str, ...],
    time_key: Optional[str] = None
):
    import numpy as np
    from matplotlib import pyplot as plt

    if time_key is not None:
        time = data[time_key]
    else:
        first_data = data[draw_keys[0]]
        time = np.arange(first_data.shape[0])

    fig, axs = plt.subplots(len(draw_keys), sharex=True, figsize=(10, 6))

    for ax, key in zip(axs, draw_keys):
        ax.plot(time, data[key], label=key)
        ax.set_ylabel(key)
        ax.legend(loc='upper left')
        ax.grid(True)

    plt.tight_layout()

    # close graph on key press
    plt.waitforbuttonpress()
    plt.close(fig)


def draw_grid(im, corners, n):
    import cv2 as cv
    import numpy as np
    from .perspective import find_perspective_transform, perspective_transform_points

    to_box_transform = find_perspective_transform(
        corners, (1., 1.)
    )
    from_box_transform = np.linalg.inv(to_box_transform)
    grid_points = np.array([
        (i / n, j / n)
        for i in range(n + 1)
        for j in range(n + 1)
    ])

    for x, y in perspective_transform_points(from_box_transform, grid_points):
        cv.circle(im, (int(y), int(x)), 10, (0, 0, 255), 3)
