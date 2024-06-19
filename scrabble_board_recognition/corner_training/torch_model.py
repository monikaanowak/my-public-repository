# python -m corner_training.torch_model train_corners --train-set-path=$DATASET_PATH/monia-mgr/corners_train.h5 --testset-dir=$DATASET_PATH/monia-mgr/test_data --num-epochs=256
# python -m corner_training.torch_model test_ims $DATASET_PATH/ploter/IMG_1_image.jpg
# python -m corner_training.torch_model train_letters --train-set-path=$DATASET_PATH/monia-mgr/letters_train.h5 --batch-size=128 --num-epochs=64
# python -m corner_training.torch_model score_checkpoint_letters --letters-checkpoint=letters_model.pt --im_dir=C:/Users/Monika/Desktop/PJATK/MGR/photos --im_labelled_dir=C:/Users/Monika/Desktop/PJATK/MGR/photos_labeled --board_excel_dir=C:/Users/Monika/Desktop/PJATK/MGR/boards --alert_loss=0.7 --device=cpu  6306 6307 6308 6309 6310
# python -m corner_training.torch_model score_checkpoint --checkpoint-path=corners_model.pt --im-dir=C:\Users\Monika\Desktop\PJATK\MGR\photos --im-labelled-dir=C:\Users\Monika\Desktop\PJATK\MGR\photos_labeled

import json
from collections import defaultdict
from os import path
from sys import argv
from typing import Tuple, Iterable

import numpy as np
from click import command, option, argument
from h5py import File
from sklearn.metrics import f1_score
from skimage.metrics import hausdorff_distance
from torch import Tensor, mean, from_numpy, float32, cat, meshgrid, arange, tensor, exp, stack, save, load, no_grad, \
    uint8, int64
from torch.linalg import norm
from torch.nn import Module
from torch.optim import Optimizer, Adam, ASGD
from torch.nn.functional import cross_entropy
from tqdm import tqdm

from .boards import find_dataset_ims, find_points_by_color_range, load_ims_with_labels
from .letters import cut_letters, find_dataset_board_excel, load_sheet_letters
from .corners import corners_distance_sum, filter_corners_by_weights, get_perspective_corners, mask_corners_nms
from .model_structure import LETTER_SHAPE, LETTERS, CornerModel, LetterClassifier
from .coords import scale_size, ceil_size, coords_resize
from .vis import draw_corners, draw_grid, format_to_show, opencv_display, plot_time_graphs
from .perspective import find_perspective_transform, perspective_transform_points


################ Feed data ################

# TODO: dynamic batch size based on image size
def get_indexes(old_offsets, new_offsets, batch_size, selected_keys):
    """
    Get indexes inside dataset to feed data to model
    """
    key_prefix = selected_keys[0] + '_'
    for key, new_offset in new_offsets.items():
        if key.startswith(key_prefix):
            old_offset = old_offsets.get(key, 0)
            size_key = key[len(key_prefix):]
            item_keys = tuple(f'{k}_{size_key}' for k in selected_keys)
            for r in range(old_offset, new_offset, batch_size):
                yield item_keys, r


def feed_epoch(f: File, prev_offsets, new_offsets, device, batch_size, selected_keys, desc):
    """
    Feed randomly ordered data to model for one epoch
    """
    indexes = list(get_indexes(
        prev_offsets, new_offsets,
        batch_size=batch_size, selected_keys=selected_keys
    ))

    np.random.shuffle(indexes)
    for keys, idx in tqdm(indexes, desc=desc):
        batch_sample = []
        for key in keys:
            x = f[key][idx:idx + batch_size]
            if np.issubdtype(x.dtype, np.number):
                x = from_numpy(x).to(device)
            elif np.issubdtype(x.dtype, np.string_):
                x = tuple(json.loads(str(s.astype(str))) for s in x)
            elif isinstance(x, (bytes, object)):
                x = tuple(json.loads(v.decode('utf-8')) for v in x)
            batch_sample.append(x)
        yield tuple(batch_sample)


def open_dataset(path: str):
    f = File(path, 'r')
    return f, json.loads(str(f['offsets'][...].astype(str)))


################ Model training ################

def set_optimizer_params(optimizer, new_params):
    """
    Set optimizer parameters for every parameter group
    """
    for param_group in optimizer.param_groups:
        param_group.update(new_params)


def compute_gt_output(
        shape: Tuple[int, ...],
        pts: Tuple[Tuple[Tuple[float, float], ...], ...],
        exp_factor: float, device
):
    """
    Compute ground truth mask for model from points coordinates
    """
    exp_mul = -1. / exp_factor

    xx = arange(shape[0], dtype=float32, device=device)
    yy = arange(shape[1], dtype=float32, device=device)
    positions = stack(meshgrid(xx, yy, indexing="ij"), dim=-1)

    sample_gts = []
    for sample in pts:
        datas = []
        for pt in sample:
            pos_diff = positions - tensor(pt, dtype=float32, device=device)
            dist = norm(pos_diff, axis=-1) + .1
            out_data = exp(dist * exp_mul)
            datas.append(out_data)
        sample_gts.append(
            stack(datas, dim=-1).amax(dim=-1)
        )

    return stack(sample_gts, dim=0)


def compute_loss_corners(y_out: Tensor, y_gt: Tensor):
    """
    Compute loss for model
    weight important points by number of points in image
    """
    y_out = y_out.squeeze(-3)

    high_mask = (y_gt > .07) | (y_out > .07)
    elem_count = y_gt.numel()

    high_count = high_mask.sum().to(dtype=float32)
    high_weight = elem_count / high_count

    loss_base = (y_out - y_gt).abs()

    loss_weighted = loss_base * (~high_mask) + loss_base * high_mask * high_weight
    return loss_weighted.view(loss_weighted.shape[0], -1).mean(dim=-1)


def training_loop_corners(
        model: Module, optimizer: Optimizer,
        data_feed: Iterable[Tuple]
):
    """
    Training loop for model with optimizer in one epoch
    """
    model.train()
    losses = []
    for input_im, label_coords in data_feed:
        label_gt = compute_gt_output(input_im.shape[1:], label_coords, 10., input_im.device)

        optimizer.zero_grad()
        y_out = model(input_im)
        batch_losses = compute_loss_corners(y_out, label_gt)
        batch_losses.mean().backward()
        optimizer.step()
        # gc.collect()
        losses.append(batch_losses.detach())
    return cat(losses, dim=-1)


def score_corners_model(
    model: CornerModel,
    im_dir: str, im_labelled_dir: str,
    boards: Tuple[str, ...], debug=False, device='cpu'
):
    import cv2 as cv

    loss_scores = []
    dist_scores = []
    hausdorff_scores = []

    with no_grad():
        # Initialize mode on gpu
        model.eval()
        model.to(device)

        for im, label_coords in tqdm(
            load_ims_with_labels(im_dir, im_labelled_dir, boards, debug=debug),
            desc='Scoring corners model', total=len(boards)
        ):
            new_size = scale_size(im.shape, (
                np.random.uniform(800, 1400),
                np.random.uniform(800, 1400)
            ))
            new_size = ceil_size(new_size, padding=108)
            label_coords = coords_resize(label_coords, im.shape[:2], new_size)
            im = cv.resize(im, new_size[::-1], interpolation=cv.INTER_CUBIC)

            x = from_numpy(im).to(device=device).unsqueeze(0)
            y = model(x)
            out_im = y[0].cpu().numpy()

            label_gt = compute_gt_output(out_im.shape[1:], [label_coords], 10., x.device)
            corners = mask_corners_nms(out_im[0], threshold=.5)
            corners = filter_corners_by_weights(corners, 100)[:5]

            loss = compute_loss_corners(y, label_gt).mean().item()
            hausdorff_score = hausdorff_distance(out_im[0] > .7, label_gt[0].cpu().numpy() > .7)
            if len(corners):
                corner_dist = corners_distance_sum(label_coords, corners)
            else:
                corner_dist = 1e3
            if debug:
                print(f'corner dist: {corner_dist}')
                print(f'loss: {loss}')

            loss_scores.append(loss)
            dist_scores.append(corner_dist)
            hausdorff_scores.append(hausdorff_score)

    return np.mean(loss_scores), np.mean(dist_scores), np.mean(hausdorff_score)


def training_loop_letters(
        model: Module, optimizer: Optimizer,
        data_feed: Iterable[Tuple]
):
    """
    Training loop for model with optimizer in one epoch
    """
    model.train()
    losses = []
    for input_im, label_gt in data_feed:
        if not len(input_im):
            continue

        optimizer.zero_grad()
        y_out = model(input_im)
        batch_losses = cross_entropy(
            y_out, label_gt.to(dtype=int64),
            reduction='none', label_smoothing=.02
        )
        batch_losses.mean().backward()
        optimizer.step()
        losses.append(batch_losses.detach())

    return cat(losses, dim=-1)


def score_letters_model(
    model: LetterClassifier,
    im_dir: str, im_labelled_dir: str, board_excel_dir: str,
    boards: Tuple[str, ...], debug=False, alert_loss=1e6, device='cpu'
):
    import cv2 as cv

    with no_grad():
        model.eval()
        model.to(device)
        all_losses = []
        all_ns = []
        all_ns_gt = []
        error_cnt = defaultdict(int)

        for board in tqdm(boards, desc='Evaluating model'):
            if debug:
                print(f'Processing {board}...')
            im = cv.imread(f'{im_dir}/IMG_{board}.JPEG')
            if im is None:
                print(f"Could not image file IMG_{board}.JPEG")
            im_label = cv.imread(f'{im_labelled_dir}/IMG_{board}.JPEG')
            if im_label is None:
                print(f"Could not find label file IMG_{board}.JPEG")

            try:
                corners = find_points_by_color_range(im_label, (220, 0, 0), (255, 30, 30))
            except Exception:
                print(f'Loading IMG_{board}.JPEG failed')
                raise
            if len(corners) != 5:
                print(f'Board: {board}, corners: {len(corners)}')
            corners = get_perspective_corners(corners)
            corners = np.roll(corners, -1, axis=0)
            letters_tensors = tuple(
                from_numpy(arr) for arr in cut_letters(im, corners, 15, LETTER_SHAPE)
            )
            letters_batch = stack(letters_tensors, dim=0).to(device)

            sheet_letters = tuple(load_sheet_letters(f'{board_excel_dir}/{board}.xlsx'))
            sheet_letters_gt = tensor(
                tuple(LETTERS.index(l) for l in sheet_letters),
                dtype=int64, device=device
            )
            all_ns_gt.append(sheet_letters_gt.cpu())

            letters_result = model(letters_batch)
            letters_y = letters_result.argmax(dim=-1).cpu()
            all_ns.append(letters_y)
            letters_losses = cross_entropy(
                letters_result, sheet_letters_gt,
                reduction='none', label_smoothing=.02
            )
            all_losses.append(letters_losses)
            if debug:
                for yy in range(15):
                    for xx in range(15):
                        letter = letters_y[yy * 15 + xx].item()
                        print(LETTERS[letter], end='')
                    print()
            alert_text = []
            for pos, (letter_gt, letter_y, letter_loss) in enumerate(zip(
                    sheet_letters, letters_y, letters_losses
            )):
                error_cnt[(LETTERS.index(letter_gt), int(letter_y))] += 1
                if letter_loss >= alert_loss:
                    alert_text.append(((pos % 15, pos // 15), letter_gt))
                    print(
                        f'{board}:{(pos % 15) + 1}{chr(ord("A") + pos // 15)}: loss: {letter_loss} '
                        f'gt:{letter_gt} ? model:{LETTERS[letter_y]}'
                    )
            if alert_text:
                put_text(im, corners, 15, alert_text)
                with opencv_display(wait_timeout=100.):
                    cv.imshow(f'mistakes? {board}', format_to_show(im))

        return (
            float(cat(all_losses, dim=0).mean(dim=0)), dict(error_cnt),
            f1_score(cat(all_ns_gt, dim=0), cat(all_ns, dim=0), average='micro'),
            f1_score(cat(all_ns_gt, dim=0), cat(all_ns, dim=0), average='weighted'),
        )


def compute_optim_params(epoch, end_epoch, scale=1.4e-3):
    """
    Compute optimizer parameters schedule for epoch
    """
    k = (1.01 - np.tanh((epoch / end_epoch) * 3.)) * (1.1 + np.sin(epoch / 8))
    return dict(
        lr=scale * k,
        weight_decay=scale * 1e-2 * k,
    )


@command()
@option('--train-set-path', default='train.h5', help='Path to train dataset .h5 file')
@option('--model-save-path', default='corners_model.pt', help='Path to save model')
@option('--stats-save-path', default='corner_model_losses.npz', help='Path to save stats as .npz file')
@option('--batch-size', default=4, help='Batch size')
@option('--num-epochs', default=64, help='Number of epochs')
@option('--testset-dir', default=None, help='Test set path')
@option('--device', default='cuda:0', help='Device to use')
def train_corners(
        train_set_path: str, model_save_path: str, stats_save_path: str, testset_dir: str,
        batch_size: int, num_epochs: int, device: str
):
    # Open dataset and load offsets
    train_set, train_offsets = open_dataset(train_set_path)

    # Create model and optimizer
    net = CornerModel()
    net.to(device)
    net.train()
    optimizer = Adam(net.parameters())

    best_loss = 1e10
    train_losses_all = []
    test_losses_all = []
    test_dist_all = []
    test_hausdorff_all = []
    lrs = []

    try:
        for e1 in range(0, num_epochs, len(train_offsets)):
            last_offsets = {}
            for e2, offsets in enumerate(train_offsets):
                epoch = e1 + e2 + 1

                # Set optimizer params for current epoch
                optim_params = compute_optim_params(epoch, num_epochs)
                set_optimizer_params(optimizer, optim_params)

                # Run training loop for this epoch
                train_losses = training_loop_corners(
                    net, optimizer, feed_epoch(
                        train_set, last_offsets, offsets, device,
                        batch_size=batch_size, selected_keys=('im', 'pos'), desc=f'Train epoch {epoch}'
                    )
                )
                train_loss_mean = float(mean(train_losses))
                last_offsets = offsets

                if testset_dir:
                    im_dir = path.join(testset_dir, 'photos')
                    ims = list(find_dataset_ims(im_dir))
                    test_loss, test_dist, test_hausdorff = score_corners_model(
                        net,
                        im_dir, path.join(testset_dir, 'photos_labeled'),
                        ims, device=device
                    )
                else:
                    test_loss = 0
                    test_dist = 0
                    test_hausdorff = 0
                print(
                    f'Epoch {epoch}, train loss: {train_loss_mean}, '
                    f'test loss: {test_loss}, test distance: {test_dist}, '
                    f'learning rate: {optim_params["lr"]}'
                )

                # Select model with best loss
                select_loss = test_loss or train_loss_mean
                if select_loss < best_loss:
                    best_loss = select_loss
                    print(f'Current best loss {best_loss} - saving model')
                    save(net.state_dict(), model_save_path)

                lrs.append(optim_params['lr'])
                train_losses_all.append(train_loss_mean)
                test_losses_all.append(test_loss)
                test_dist_all.append(test_dist)
                test_hausdorff_all.append(test_hausdorff)

    finally:
        np.savez(
            stats_save_path,
            train=np.array(train_losses_all),
            test=np.array(test_losses_all),
            dist=np.array(test_dist_all),
            hausdorff=np.array(test_hausdorff_all),
            lrs=np.array(lrs),
        )
        print(f'Best loss {best_loss}')


@command
@option('--stats-path', default='corner_model_losses.npz', help='Path of stats .npz file')
def show_corners_stats(stats_path: str):
    stats = dict(np.load(stats_path))
    nepochs = len(stats['lrs'])
    stats['epoch'] = np.arange(1, nepochs+1)
    plot_time_graphs(stats, ('lrs', 'train', 'test', 'dist', 'hausdorff'), time_key='epoch')


@command()
@option('--dataset-path', default='train.h5', help='Path to dataset .h5 file')
@option('--device', default='cpu', help='Device to use')
def show_dataset(dataset_path: str, device: str):
    """
    Visualize dataset to debug it
    """
    import cv2 as cv

    train_set, train_offsets = open_dataset(dataset_path)
    for im, pos in feed_epoch(
            train_set, {}, train_offsets[0], device,
            batch_size=1, selected_keys=('im', 'pos'), desc='Visualize'
    ):
        im = im[0].to(dtype=uint8).numpy().copy()
        pos = pos[0]
        for pt in pos:
            cv.circle(im, (int(pt[1]), int(pt[0])), 3, (0, 255, 0), -1)
        with opencv_display(wait_timeout=10.):
            cv.imshow('input', im)


@command()
@option('--checkpoint-path', default='corner_model.pt', help='Path to model checkpoint')
@option('--dataset-dir', type=str)
@option('--device', default='cpu', help='Device to use')
def score_checkpoint_corners(checkpoint_path: str, dataset_dir: str, device: str):
    im_dir = path.join(dataset_dir, 'photos')
    ims = list(find_dataset_ims(im_dir))

    with no_grad():
        # Initialize mode on gpu
        net = CornerModel()
        net.eval()
        net.load_state_dict(load(checkpoint_path, map_location='cpu'))
        net.to(device)

        avg_loss, avg_dist, avg_hausdorff = score_corners_model(
            net,
            im_dir, path.join(dataset_dir, 'photos_labeled'),
            ims, debug=True, device=device
        )

    print(f'Loss score: {avg_loss}')
    print(f'Corner distance score: {avg_dist}')
    print(f'Hausdorff score: {avg_hausdorff}')


@command()
@option('--train-set-path', default='train.h5', help='Path to train dataset .h5 file')
@option('--model-save-path', default='letters_model.pt', help='Path to save model')
@option('--stats-save-path', default='letters_model_losses.npz', help='Path to save stats as .npz file')
@option('--batch-size', default=64, help='Batch size')
@option('--num-epochs', default=64, help='Number of epochs')
@option('--testset-dir', default=None, help='Testset directory')
@option('--device', default='cuda:0', help='Device to use')
def train_letters(
        train_set_path: str, model_save_path: str, stats_save_path: str,
        batch_size: int, num_epochs: int, testset_dir: str, device: str
):
    # Open dataset and load offsets
    train_set, train_offsets = open_dataset(train_set_path)

    # Create model and optimizer
    net = LetterClassifier()
    net.to(device)
    net.train()
    optimizer = Adam(net.parameters())

    best_loss = 1e10
    train_losses_all = []
    test_losses_all = []
    test_f1_micro_all = []
    test_f1_weighted_all = []
    lrs = []

    try:
        for e1 in range(0, num_epochs, len(train_offsets)):
            last_offsets = {}
            for e2, offsets in enumerate(train_offsets):
                epoch = e1 + e2 + 1

                # Set optimizer params for current epoch
                optim_params = compute_optim_params(epoch, num_epochs, scale=.2 / batch_size)
                set_optimizer_params(optimizer, optim_params)

                # Run training loop for this epoch
                train_losses = training_loop_letters(
                    net, optimizer, feed_epoch(
                        train_set, last_offsets, offsets, device,
                        batch_size=batch_size, selected_keys=('im', 'letter'), desc=f'Train epoch {epoch}'
                    )
                )
                train_loss_mean = float(mean(train_losses))
                last_offsets = offsets

                # Run checkpoint test
                if testset_dir:
                    board_exel_dir = path.join(testset_dir, 'boards')
                    test_boards = list(find_dataset_board_excel(board_exel_dir))
                    test_loss_mean, _, f1_micro, f1_weighted = score_letters_model(
                        net,
                        path.join(testset_dir, 'photos'), path.join(testset_dir, 'photos_labeled'),
                        board_exel_dir, test_boards, device=device
                    )
                else:
                    test_loss_mean = 0
                    f1_micro = 0
                    f1_weighted = 0

                print(
                    f'Epoch {epoch}, train loss: {train_loss_mean}, '
                    f'test loss: {test_loss_mean}, learning rate: {optim_params["lr"]}'
                )

                # Select model with best loss
                select_loss = test_loss_mean or train_loss_mean
                if select_loss < best_loss:
                    best_loss = select_loss
                    print(f'Current loss {best_loss} - saving model')
                    save(net.state_dict(), model_save_path)

                lrs.append(optim_params['lr'])
                train_losses_all.append(train_loss_mean)
                test_losses_all.append(test_loss_mean)
                test_f1_micro_all.append(f1_micro)
                test_f1_weighted_all.append(f1_weighted)

    finally:
        np.savez(
            stats_save_path,
            train=np.array(train_losses_all),
            test=np.array(test_losses_all),
            test_f1_micro=np.array(test_f1_micro_all),
            test_f1_weighted=np.array(test_f1_weighted_all),
            lrs=np.array(lrs),
        )
        print(f'Best loss {best_loss}')


@command()
@option('--stats-path', default='letters_model_losses.npz', help='Path of stats .npz file')
def show_letters_stats(stats_path: str):
    stats = dict(np.load(stats_path))
    nepochs = len(stats['lrs'])
    stats['epoch'] = np.arange(1, nepochs+1)
    plot_time_graphs(stats, ('lrs', 'train', 'test', 'test_f1_micro', 'test_f1_weighted'), time_key='epoch')


@command()
@option('--corners-checkpoint', default='corners_model.pt', help='Path of the cornres checkpoint to test')
@option('--letters-checkpoint', default='letters_model.pt', help='Path of the letters checkpoint to test')
@option('--device', default='cpu', help='Device to use')
@argument('im_files', nargs=-1)
def test_ims(corners_checkpoint, letters_checkpoint, device, im_files):
    """
    Test model on images from files
    """
    import cv2 as cv

    with no_grad():
        # Initialize mode on gpu
        net = CornerModel()
        net.eval()
        net.load_state_dict(load(corners_checkpoint, map_location='cpu'))
        net.to(device)

        letters_classifier = LetterClassifier()
        letters_classifier.eval()
        letters_classifier.load_state_dict(load(letters_checkpoint, map_location='cpu'))
        letters_classifier.to(device)

        for fname in im_files:
            print(f'Processing {fname}...')
            im_orig = cv.imread(fname)
            if im_orig is None:
                print(f'Could not read image {fname}')
                continue

            # Scale image to lower random size
            shape = scale_size(im_orig.shape, (
                np.random.uniform(800, 1400),
                np.random.uniform(800, 1400)
            ))
            w, h = ceil_size(shape, padding=108)
            im = cv.resize(im_orig, (h, w), interpolation=cv.INTER_CUBIC)

            # Run model on image
            x = from_numpy(im).to(device=device).unsqueeze(0).to(dtype=float32)
            y = net(x)
            out_im = y[0, 0].cpu().numpy()

            corners = mask_corners_nms(out_im, threshold=.5)
            corners = filter_corners_by_weights(corners, 100)
            print(f'Found {len(corners)} corners')

            if len(corners) == 5:
                corners = get_perspective_corners(corners)
                corners = np.roll(corners, -1, axis=0)
                corners_orig = coords_resize(corners.astype(np.float64), im.shape, im_orig.shape)

                letters_tensors = tuple(
                    from_numpy(arr) for arr in cut_letters(im_orig, corners_orig, 15, LETTER_SHAPE)
                )
                letters_batch = stack(letters_tensors, dim=0).to(device).to(dtype=float32)
                draw_grid(im, corners, 15)

                letters_result = letters_classifier(letters_batch).softmax(dim=-1)
                letters_num = letters_result.argmax(dim=-1)
                for yy in range(15):
                    for xx in range(15):
                        letter_n = letters_num[yy * 15 + xx].item()
                        print(LETTERS[letter_n], end='')
                    print()
            else:
                draw_corners(im, corners)

            # Show image on screen
            with opencv_display(wait_timeout=100.):
                cv.imshow('output', format_to_show(out_im))
                cv.imshow('input', format_to_show(im))


def put_text(im, corners, n, field_text):
    import cv2 as cv

    to_box_transform = find_perspective_transform(
        corners, (1., 1.)
    )
    from_box_transform = np.linalg.inv(to_box_transform)
    coords, texts = zip(*field_text)
    coords = np.array([
        (x / n, (y + .9) / n)
        for x, y in coords
    ])

    for (x, y), text in zip(perspective_transform_points(from_box_transform, coords), texts):
        cv.putText(
            im, text, (int(y), int(x)), cv.FONT_HERSHEY_SIMPLEX,
            4, (0, 0, 255), 3
        )


@command()
@option('--letters-checkpoint', default='letters_model.pt', help='Path of the letters checkpoint to test')
@option('--dataset_dir', type=str)
@option('--alert_loss', type=float, default=1e9)
@option('--device', default='cpu', help='Device to use')
@argument('boards', nargs=-1)
def score_checkpoint_letters(
        letters_checkpoint, dataset_dir, alert_loss,
        device, boards
):
    board_exel_dir = path.join(dataset_dir, 'boards')
    if not boards:
        boards = list(find_dataset_board_excel(board_exel_dir))

    with no_grad():
        # Initialize mode on gpu
        letters_classifier = LetterClassifier()
        letters_classifier.eval()
        letters_classifier.load_state_dict(load(letters_checkpoint, map_location='cpu'))

        avg_loss, error_matrix, f1_score_micro, f1_score_weighted = score_letters_model(
            letters_classifier,
            path.join(dataset_dir, 'photos'), path.join(dataset_dir, 'photos_labeled'),
            board_exel_dir, boards,
            debug=True, alert_loss=alert_loss, device=device
        )

    error_matrix = defaultdict(int, error_matrix)
    with open('error-matrix.txt', 'w', encoding='utf-8') as f:
        print(f' ', end='', file=f)
        for letter in LETTERS:
            print(f'    {letter}', end='', file=f)
        print(file=f)
        for i in range(len(LETTERS)):
            print(f'{LETTERS[i]}', end='', file=f)
            for j in range(len(LETTERS)):
                # if error_matrix[(i, j)]:
                print(f' {error_matrix[(i, j)]: 4d}', end='', file=f)
                # else:
                #     print(' ' * 5, end='', file=f)
            print(file=f)

    print(f'Mean loss: {avg_loss}')
    print(f'Mean F1 score micro: {f1_score_micro}')
    print(f'Mean F1 score weighted: {f1_score_weighted}')


if __name__ == '__main__':
    func_name = argv.pop(1)
    locals()[func_name]()
