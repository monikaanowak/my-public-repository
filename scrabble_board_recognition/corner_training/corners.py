def mask_corners_nms(corners_mask, threshold):
    import numpy as np

    corners = []
    for i, j in zip(*np.where(corners_mask > threshold)):
        if i == 0 or j == 0 or i == corners_mask.shape[0] - 1 or j == corners_mask.shape[1] - 1:
            continue

        max_neighbour = max(
            corners_mask[i - 1, j - 1],
            corners_mask[i - 1, j],
            corners_mask[i - 1, j + 1],
            corners_mask[i, j - 1],
            corners_mask[i, j + 1],
            corners_mask[i + 1, j - 1],
            corners_mask[i + 1, j],
            corners_mask[i + 1, j + 1],
        )
        if max_neighbour > corners_mask[i, j]:
            continue
        corners.append((corners_mask[i, j], i, j))
    return corners


def filter_corners_by_weights(corners, max_dist):
    max_dist = max_dist ** 2

    found_corners = []
    for w, x, y in sorted(corners, reverse=True):
        found = False
        for xx, yy in found_corners:
            if (x - xx) ** 2 + (y - yy) ** 2 < max_dist:
                found = True
                break
        if not found:
            found_corners.append((x, y))

    return found_corners


def angle_sort_corners(corners):
    '''
    angle sort corners around their mass center
    '''
    import numpy as np
    corners = np.array(corners)
    center = corners.mean(axis=0)
    angles = np.arctan2(corners[:, 0] - center[0], corners[:, 1] - center[1])
    return corners[np.argsort(angles)]


def corners_distance_sum(c1, c2):
    import numpy as np
    c1 = np.array(c1)
    c2 = np.array(c2)

    dist = 0
    for p1 in c1:
        dist += np.linalg.norm(c2 - p1[None, ...], axis=1).min()
    return dist


def get_perspective_corners(corners):
    import numpy as np
    corners = tuple(angle_sort_corners(corners))
    eps = 1e10
    eps_pos = -1
    for it in range(len(corners)):
        ax, ay = corners[it-2]
        bx, by = corners[it-1]
        cx, cy = corners[it]
        ax -= bx
        ay -= by
        cx -= bx
        cy -= by
        cross = abs(ax*cy-cx*ay)
        if cross < eps:
            eps = cross
            eps_pos = it
    if eps_pos == 0:
        return np.stack(corners[:-1], axis=0)
    return np.stack(corners[eps_pos:] + corners[:eps_pos-1], axis=0)




