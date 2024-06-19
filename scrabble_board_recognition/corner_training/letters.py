from collections import defaultdict
from math import floor
from os import listdir

import numpy as np
from .perspective import find_perspective_transform, perspective_transform_points, cut_image_box


def cut_letters(im, corners, n, out_size):
    to_box_transform = find_perspective_transform(
        np.array(corners)[:, ::-1], (1., 1.)
    )
    from_box_transform = np.linalg.inv(to_box_transform)
    for yy in range(n):
        for xx in range(n):
            letter_coords = np.array([
                (xx / n, yy / n),
                ((xx + 1) / n, yy / n),
                ((xx + 1) / n, (yy + 1) / n),
                (xx / n, (yy + 1) / n),
            ])
            yield cut_image_box(
                im, perspective_transform_points(from_box_transform, letter_coords),
                out_size
            )


def find_dataset_board_excel(dataset_path):
    for fname in listdir(dataset_path):
        if fname.lower().endswith('.xlsx'):
            yield fname.rsplit('.', 1)[0]


def load_sheet_letters(board_excel):
    import pandas as pd
    df = pd.read_excel(board_excel, header=None)
    for _, row in df.iterrows():
        for cell_value in row.values:
            letter = str(cell_value)
            if letter[0] == '"':
                letter = '~'
            yield letter


def count_dataset_letters(board_excel_dir, boards):
    counts = defaultdict(int)
    for board in boards:
        sheet_letters = load_sheet_letters(f'{board_excel_dir}/{board}.xlsx')
        for letter in sheet_letters:
            counts[letter] += 1
    return dict(counts)


def compute_letter_repeat(
    letter_counts, min_repeat, max_repeat, repeat_scale
):
    max_count = max(letter_counts.values()) * repeat_scale
    return {
        letter: int(max(min(floor(max_count / count), max_repeat), min_repeat))
        for letter, count in letter_counts.items()
    }

