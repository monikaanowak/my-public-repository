from os import listdir


def find_points_by_color_range(im, col_range_l, col_range_h):
    """
    Find points in image by color range and return coordinates
    """
    import cv2 as cv
    import numpy as np

    mask = cv.inRange(im, col_range_l[::-1], col_range_h[::-1])
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    pts = np.stack(tuple(
        c.mean(axis=0).mean(axis=0)[::-1] for c in contours
    ), axis=0)
    return pts


def load_ims_with_labels(im_dir, label_dir, fnames, debug=False):
    import cv2 as cv

    for fname in fnames:
        if debug:
            print(f'Loading {fname}...')
        im = cv.imread(f'{im_dir}/{fname}')
        im_label = cv.imread(f'{label_dir}/{fname}')
        if im_label is None:
            raise ValueError(f"Could not find label file {fname}")
        if im.shape == im_label.shape:
            try:
                yield im, find_points_by_color_range(im_label, (220, 0, 0), (255, 30, 30))
            except Exception:
                print(f'Loading {fname} failed')


def find_dataset_ims(dataset_path):
    for fname in listdir(dataset_path):
        if fname.lower().endswith('.jpg') or fname.lower().endswith('.jpeg'):
            yield fname
