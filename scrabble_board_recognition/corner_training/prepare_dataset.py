import json
from os import path, replace
from sys import argv
from typing import Dict, Tuple

import numpy as np
from click import argument, command, option
from h5py import File, special_dtype
from numpy import ndarray
from numpy.random import random_sample
from tqdm import tqdm

from .boards import find_dataset_ims, load_ims_with_labels, find_points_by_color_range
from .letters import load_sheet_letters, find_dataset_board_excel, count_dataset_letters, compute_letter_repeat
from .coords import scale_size, rand_size, ceil_size
from .corners import get_perspective_corners


################ Dataset writing ################

def copy_shuffled(arrays_src, arrays_dst):
    """
    In-place shuffle many arrays in same order
    """
    n = arrays_src[0].shape[0]
    assert len(arrays_src) > 0
    assert all(n == a.shape[0] for a in arrays_src)

    indices = np.arange(n)
    np.random.shuffle(indices)
    for src, dst in zip(arrays_src, arrays_dst):
        dst[:, ...] = src[indices, ...]


# TODO: copying and shuffling saved dataset

def append_to_dataset(f: File, dataset_name: str, offset: int, data: ndarray, chunk_size):
    if dataset_name in f:
        dataset = f[dataset_name]
    else:
        if np.issubdtype(data.dtype, np.number):
            dtype = data.dtype
        elif np.issubdtype(data.dtype, np.string_):
            dtype = special_dtype(vlen=str)
        else:
            raise ValueError(f'Unsupported dtype: {data.dtype}')
        dataset = f.create_dataset(
            dataset_name, (chunk_size,) + data.shape[1:],
            maxshape=(None,) + data.shape[1:], dtype=dtype,
            chunks=(chunk_size,) + data.shape[1:]
        )
    while dataset.shape[0] < offset + data.shape[0]:
        dataset.resize(offset + chunk_size, axis=0)
    dataset[offset:offset + data.shape[0]] = data
    return offset + data.shape[0]


def append_data(
    f: File, offsets: Dict, shape_key, data_by_key: Dict[str, ndarray],
    chunk_size
):
    shape_key = '_'.join(map(str, shape_key))
    n = data_by_key[list(data_by_key.keys())[0]].shape[0]
    assert all(data.shape[0] == n for data in data_by_key.values())

    for key, data in data_by_key.items():
        dataset_name = f'{key}_{shape_key}'
        current_size = offsets.get(dataset_name, 0)
        offsets[dataset_name] = append_to_dataset(
            f, dataset_name, current_size, data, chunk_size
        )


def finalize_dataset(
    fname: str,
    epoch_offsets: Tuple[Dict, ...],
    data_keys: Tuple[str, ...],
    chunk_size=16, copy_chunk_size=None
):
    if copy_chunk_size is None:
        copy_chunk_size = chunk_size

    sizes = epoch_offsets[-1]
    fname_tmp = fname + '.tmp'
    with File(fname, 'r') as f, File(fname_tmp, 'w') as f_tmp:
        for dataset_name, size in sizes.items():
            if dataset_name.startswith(data_keys[0]):
                suf = dataset_name[len(data_keys[0]):]

                out_datasets = tuple(
                    f_tmp.create_dataset(
                        pref + suf, (size,) + f[pref + suf].shape[1:],
                        dtype=f[pref + suf].dtype,
                        chunks=(min(chunk_size, size),) + f[pref + suf].shape[1:]
                    ) for pref in data_keys
                )

                last_offset = 0
                for offsets in epoch_offsets:
                    offset = offsets.get(dataset_name, 0)
                    epoch_size = offset - last_offset
                    if not epoch_size:
                        continue

                    indexes = np.arange(epoch_size, dtype=np.int64) + last_offset
                    np.random.shuffle(indexes)

                    for src, dst in zip(data_keys, out_datasets):
                        for it in range(0, len(indexes), copy_chunk_size):
                            chunk_idx = indexes[it:it + copy_chunk_size]
                            chunk_idx_query = np.sort(chunk_idx)

                            chunk_values = f[src + suf][chunk_idx_query]
                            chunk_values_reordered = chunk_values[np.argsort(chunk_idx)]
                            dst[last_offset+it : last_offset+it+len(chunk_idx)] = chunk_values_reordered

                    last_offset = offset

        f_tmp.create_dataset('offsets', data=json.dumps(epoch_offsets))

    replace(fname_tmp, fname)


################ Augmentations ################

def random_decision(p) -> bool:
    from numpy.random import uniform
    return uniform() < p


def spot_mask(shape, spot_size=16.0):
    import cv2 as cv
    import numpy as np

    w, h = shape[:2]
    rand_mask = random_sample((
        int(np.ceil(w / spot_size)),
        int(np.ceil(h / spot_size)),
    ))
    return cv.resize(rand_mask, (h, w), interpolation=cv.INTER_CUBIC)


def sin_mask(shape, ww=16., wh=16., dtype=np.float32):
    import numpy as np

    w, h = shape[:2]
    dw = np.array(2 * np.pi / ww, dtype=dtype)
    dh = np.array(2 * np.pi / wh, dtype=dtype)
    grid = (
            np.arange(h, dtype=dtype)[np.newaxis, :] * dh + np.arange(w, dtype=dtype)[:, np.newaxis] * dw
    )
    return np.sin(grid) * .5 + .5


def apply_fog(im: ndarray, float_mask: ndarray, fog_col=(155, 155, 155)) -> ndarray:
    import numpy as np
    import cv2 as cv

    fog_col = np.array(fog_col, dtype=np.float32)
    rand_mask = random_sample(im.shape[:2])
    rand_mask *= float_mask

    rand_mask = rand_mask[:, :, np.newaxis]
    im_ret = (
            (1. - rand_mask) * im + rand_mask * fog_col
    ).astype(dtype=im.dtype)

    cv.GaussianBlur(im_ret, (5, 5), 0., dst=im_ret)
    return im_ret


def warp_colors(im: ndarray, float_mask) -> ndarray:
    import cv2 as cv
    import numpy as np

    im = cv.cvtColor(im, cv.COLOR_BGR2HSV)
    im = im.astype(np.float32)

    im[:, :, 1] = im[:, :, 1] * (float_mask * .5 + .5)
    im[:, :, 2] = im[:, :, 2] * (float_mask * .5 + .5)

    im = cv.cvtColor(im.astype(np.uint8), cv.COLOR_HSV2BGR)
    return im


def warp_perspective(
        im: ndarray, label_coords: ndarray,
        x_start=0.0, y_start=0.0, x_end=0.0, y_end=0.0,
        rotate=True, flip=True, outsize=None
):
    import cv2 as cv
    from .perspective import find_perspective_transform, perspective_transform_points_swap
    x, y = im.shape[:2]
    if outsize is None:
        outsize = x, y
    xmin = x * x_start
    xmax = x * x_end
    ymin = y * y_start
    ymax = y * y_end
    points = [
        [np.random.uniform(-xmax, -xmin), np.random.uniform(-ymax, -ymin)],
        [np.random.uniform(x+xmin, x+xmax), np.random.uniform(-ymax, -ymin)],
        [np.random.uniform(x+xmin, x+xmax), np.random.uniform(y+ymin, y+ymax)],
        [np.random.uniform(-xmax, -xmin), np.random.uniform(y+ymin, y+ymax)]
    ]
    if rotate:
        n = np.random.randint(4)
        points = points[n:] + points[:n]
    if flip and random_decision(.5):
        points = points[::-1]
    t_im = find_perspective_transform(
        [p[::-1] for p in points], outsize[::-1]
    )
    return cv.warpPerspective(
        im, t_im, outsize[::-1], flags=cv.INTER_LANCZOS4
    ), perspective_transform_points_swap(t_im, label_coords)


def augment_image_corners(im: ndarray, label_coords: ndarray, padding) -> Tuple[ndarray, ndarray]:
    import cv2 as cv
    import numpy as np

    shape = scale_size(im.shape, (
        np.random.uniform(800, 1400),
        np.random.uniform(800, 1400)
    ))
    if random_decision(.5):
        shape = rand_size(shape, .3)
    w, h = ceil_size(shape, padding)

    if random_decision(.5):
        im, label_coords = warp_perspective(im, label_coords, x_start=0.4, y_start=0.4, outsize=(w, h))
    else:
        im, label_coords = warp_perspective(im, label_coords, x_start=0.1, y_start=0.1, outsize=(w, h))

    if random_decision(.4):
        im = apply_fog(im, sin_mask(im.shape, np.random.uniform(64, 256), np.random.uniform(64, 256)))
    if random_decision(.4):
        im = apply_fog(im, spot_mask(im.shape, np.random.uniform(64, 256)))
    if random_decision(.4):
        im = warp_colors(im, sin_mask(im.shape, np.random.uniform(64, 256), np.random.uniform(64, 256)))
    if random_decision(.4):
        im = warp_colors(im, spot_mask(im.shape, np.random.uniform(64, 256)))
    if random_decision(.4):
        im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
        im = cv.cvtColor(im, cv.COLOR_GRAY2BGR)

    return im, label_coords


def augment_image_letter(im: ndarray) -> ndarray:
    import cv2 as cv
    import numpy as np

    w, h = im.shape[:2]

    im, _ = warp_perspective(
        im, (),
        x_start=-0.3, y_start=-0.3, x_end=0.3, y_end=0.3,
        outsize=(w, h)
    )

    p = .25
    if random_decision(p):
        im = apply_fog(im, sin_mask(im.shape, np.random.uniform(8, 32), np.random.uniform(8, 32)))
    if random_decision(p):
        im = apply_fog(im, spot_mask(im.shape, np.random.uniform(8, 32)))
    if random_decision(p):
        im = warp_colors(im, sin_mask(im.shape, np.random.uniform(8, 32), np.random.uniform(8, 32)))
    if random_decision(p):
        im = warp_colors(im, spot_mask(im.shape, np.random.uniform(8, 32)))
    if random_decision(p):
        im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
        im = cv.cvtColor(im, cv.COLOR_GRAY2BGR)

    return im


@command()
@option('--dataset-dir', type=str)
def vis_aug_corners(dataset_dir):
    import cv2 as cv
    from .vis import opencv_display, format_to_show, draw_grid

    im_dir = path.join(dataset_dir, 'photos')
    ims = list(find_dataset_ims(im_dir))
    np.random.shuffle(ims)
    for im, l in load_ims_with_labels(im_dir, path.join(dataset_dir, 'photos_labeled'), ims, debug=True):
        im_aug, l_aug = augment_image_corners(im, l, 108)
        corners = get_perspective_corners(l_aug)
        draw_grid(im_aug, corners, 15)

        print(f'Points count: {l_aug.shape[0]}')
        for pt in l_aug:
            cv.circle(im_aug, (int(pt[1]), int(pt[0])), 10, (0, 255, 0), 3)
        with opencv_display(wait_timeout=5.):
            cv.imshow('input', format_to_show(im_aug))


@command()
@option('--dataset-dir', type=str)
@option('--out-file', type=str)
@option('--num-epochs', default=1, type=int)
@option('--repeat-count', default=1, type=int)
def create_dataset_corners(dataset_dir, num_epochs, repeat_count, out_file):
    im_dir = path.join(dataset_dir, 'photos')
    ims = list(find_dataset_ims(im_dir))

    epoch_offsets = []
    offsets = {}
    with File(out_file, 'w') as f:
        for epoch in range(num_epochs):
            np.random.shuffle(ims)
            for im, l in tqdm(load_ims_with_labels(im_dir, path.join(dataset_dir, 'photos_labeled'), ims),
                              desc=f'Augmenting epoch {epoch + 1}/{num_epochs}', total=len(ims)):
                for _ in range(repeat_count):
                    im_aug, l_aug = augment_image_corners(im, l, 108)

                    append_data(f, offsets, im_aug.shape, {
                        'im': im_aug[np.newaxis, ...],
                        'pos': np.array([json.dumps(l_aug.tolist())], dtype='S'),
                    }, chunk_size=4)

            epoch_offsets.append(dict(offsets))

    print('Finalizing dataset...')
    finalize_dataset(out_file, epoch_offsets, ('im', 'pos'), chunk_size=4, copy_chunk_size=1024)


def load_board_letters(
        im_dir, label_dir, board_excel_dir, board, letter_shape
):
    import cv2 as cv
    from .corners import get_perspective_corners
    from .letters import cut_letters

    im = cv.imread(f'{im_dir}/IMG_{board}.JPEG')
    im_label = cv.imread(f'{label_dir}/IMG_{board}.JPEG')
    if im_label is None:
        raise ValueError(f"Could not find label file IMG_{board}.JPEG")

    sheet_letters = load_sheet_letters(f'{board_excel_dir}/{board}.xlsx')

    try:
        corners = find_points_by_color_range(im_label, (220, 0, 0), (255, 30, 30))
    except Exception:
        print(f'Loading IMG_{board}.JPEG failed')
        raise
    if len(corners) != 5:
        print(f'Board: {board}, corners: {len(corners)}')
    corners = get_perspective_corners(corners)
    corners = np.roll(corners, -1, axis=0)
    letters = cut_letters(im, corners, 15, letter_shape)

    for letter_im, cell_value in zip(letters, sheet_letters):
        yield letter_im, cell_value


@command()
@option('--dataset-dir', type=str)
@argument('boards', nargs=-1)
def vis_aug_letters(dataset_dir, boards):
    import cv2 as cv
    from .vis import opencv_display, format_to_show
    from .model_structure import LETTER_SHAPE

    board_exel_dir = path.join(dataset_dir, 'boards')
    if not boards:
        boards = list(find_dataset_board_excel(board_exel_dir))

    for board in boards:
        print(f'Board {board}')
        for im, letter in load_board_letters(
            path.join(dataset_dir, 'photos'), path.join(dataset_dir, 'photos_labeled'),
            board_exel_dir, board, LETTER_SHAPE
        ):
            im_aug = augment_image_letter(im)

            print(f'Letter: `{letter}`')
            with opencv_display(wait_timeout=5.):
                cv.imshow(f'Letter `{letter}`', format_to_show(im_aug))


@command()
@option('--dataset-dir', type=str)
@option('--out-file', type=str)
@option('--num-epochs', default=1, type=int)
@option('--min-repeat', default=1, type=int)
@option('--max-repeat', default=1, type=int)
@option('--repeat-scale', default=1, type=float)
def create_dataset_letters(
    dataset_dir, num_epochs, min_repeat, max_repeat, repeat_scale, out_file
):
    from .model_structure import LETTERS, LETTER_SHAPE
    board_exel_dir = path.join(dataset_dir, 'boards')
    boards = list(find_dataset_board_excel(board_exel_dir))

    letter_counts = count_dataset_letters(board_exel_dir, boards)
    letter_repeat = compute_letter_repeat(letter_counts, min_repeat, max_repeat, repeat_scale)
    for letter in LETTERS:
        print(f'Letter `{letter}` count: {letter_counts.get(letter, 0)}, repeat: {letter_repeat[letter]}')

    epoch_offsets = []
    offsets = {}
    with File(out_file, 'w') as f:
        for epoch in range(num_epochs):
            np.random.shuffle(boards)
            for board in tqdm(boards, desc=f'Augmenting epoch {epoch + 1}/{num_epochs}'):
                for im, letter in load_board_letters(
                    path.join(dataset_dir, 'photos'), path.join(dataset_dir, 'photos_labeled'),
                    board_exel_dir, board, LETTER_SHAPE
                ):
                    if letter not in LETTERS:
                        raise ValueError(f'Unknown letter: `{letter}` on board {board}')
                    letter_n = LETTERS.index(letter)
                    for _ in range(letter_repeat[letter]):
                        im_aug = augment_image_letter(im)

                        append_data(f, offsets, im_aug.shape, {
                            'im': im_aug[np.newaxis, ...],
                            'letter': np.array([letter_n], dtype=np.int32),
                        }, chunk_size=4)

            epoch_offsets.append(dict(offsets))

    print('Finalizing dataset...')
    finalize_dataset(
        out_file, epoch_offsets, ('im', 'letter'),
        chunk_size=256, copy_chunk_size=int(2 ** 8)
    )


if __name__ == '__main__':
    func_name = argv.pop(1)
    locals()[func_name]()
