import cv2
import numpy as np
import torch
import collections
import matplotlib.cm as cm


def image2tensor(frame, device):
    return torch.from_numpy(frame / 255.).float()[None, None].to(device)


# --- VISUALIZATION ---
# based on: https://github.com/magicleap/SuperGluePretrainedNetwork/blob/master/models/utils.py
def plot_keypoints(image, kpts, scores=None):
    kpts = np.round(kpts).astype(int)

    if scores is not None:
        # get color
        smin, smax = scores.min(), scores.max()
        assert (0 <= smin <= 1 and 0 <= smax <= 1)

        color = cm.gist_rainbow(scores * 0.4)
        color = (np.array(color[:, :3]) * 255).astype(int)[:, ::-1]
        # text = f"min score: {smin}, max score: {smax}"

        for (x, y), c in zip(kpts, color):
            c = (int(c[0]), int(c[1]), int(c[2]))
            cv2.drawMarker(image, (x, y), tuple(c), cv2.MARKER_CROSS, 6)

    else:
        for x, y in kpts:
            cv2.drawMarker(image, (x, y), (0, 255, 0), cv2.MARKER_CROSS, 6)

    return image


# based on: https://github.com/magicleap/SuperGluePretrainedNetwork/blob/master/models/utils.py
def plot_matches(image0, image1, kpts0, kpts1, scores=None, layout="lr"):
    """
    plot matches between two images. If score is nor None, then red: bad match, green: good match
    :param image0: reference image
    :param image1: current image
    :param kpts0: keypoints in reference image
    :param kpts1: keypoints in current image
    :param scores: matching score for each keypoint pair, range [0~1], 0: worst match, 1: best match
    :param layout: 'lr': left right; 'ud': up down
    :return:
    """
    H0, W0 = image0.shape[0], image0.shape[1]
    H1, W1 = image1.shape[0], image1.shape[1]

    if layout == "lr":
        H, W = max(H0, H1), W0 + W1
        out = 255 * np.ones((H, W, 3), np.uint8)
        out[:H0, :W0, :] = image0
        out[:H1, W0:, :] = image1
    elif layout == "ud":
        H, W = H0 + H1, max(W0, W1)
        out = 255 * np.ones((H, W, 3), np.uint8)
        out[:H0, :W0, :] = image0
        out[H0:, :W1, :] = image1
    else:
        raise ValueError("The layout must be 'lr' or 'ud'!")

    kpts0, kpts1 = np.round(kpts0).astype(int), np.round(kpts1).astype(int)

    # get color
    if scores is not None:
        smin, smax = scores.min(), scores.max()
        assert (0 <= smin <= 1 and 0 <= smax <= 1)

        color = cm.gist_rainbow(scores * 0.4)
        color = (np.array(color[:, :3]) * 255).astype(int)[:, ::-1]
    else:
        color = np.zeros((kpts0.shape[0], 3), dtype=int)
        color[:, 1] = 255

    for (x0, y0), (x1, y1), c in zip(kpts0, kpts1, color):
        c = c.tolist()
        if layout == "lr":
            cv2.line(out, (x0, y0), (x1 + W0, y1), color=c, thickness=1, lineType=cv2.LINE_AA)
            # display line end-points as circles
            cv2.circle(out, (x0, y0), 2, c, -1, lineType=cv2.LINE_AA)
            cv2.circle(out, (x1 + W0, y1), 2, c, -1, lineType=cv2.LINE_AA)
        elif layout == "ud":
            cv2.line(out, (x0, y0), (x1, y1 + H0), color=c, thickness=1, lineType=cv2.LINE_AA)
            # display line end-points as circles
            cv2.circle(out, (x0, y0), 2, c, -1, lineType=cv2.LINE_AA)
            cv2.circle(out, (x1, y1 + H0), 2, c, -1, lineType=cv2.LINE_AA)

    return out


def Tcw2qvec(Tcw:np.ndarray):
    """ Convert 4x4 transformation matrix to quaternion

    Args:
        Tcw (np.ndarray): camera pose, 4x4 transformation matrix

    Returns:
        _type_: rotation quaternion
    """
    qvec = np.zeros(4)
    Rcw = Tcw[:3, :3]
    tr = np.trace(Rcw)
    if tr > 0:
        S = np.sqrt(tr+1.0) * 2 # S=4*qw
        qw = 0.25 * S
        qx = (Rcw[2,1] - Rcw[1,2]) / S
        qy = (Rcw[0,2] - Rcw[2,0]) / S
        qz = (Rcw[1,0] - Rcw[0,1]) / S
    elif (Rcw[0,0] > Rcw[1,1]) and (Rcw[0,0] > Rcw[2,2]):
        S = np.sqrt(1.0 + Rcw[0,0] - Rcw[1,1] - Rcw[2,2]) * 2
        qw = (Rcw[2,1] - Rcw[1,2]) / S
        qx = 0.25 * S
        qy = (Rcw[0,1] + Rcw[1,0]) / S
        qz = (Rcw[0,2] + Rcw[2,0]) / S
    elif Rcw[1,1] > Rcw[2,2]:
        S = np.sqrt(1.0 + Rcw[1,1] - Rcw[0,0] - Rcw[2,2]) * 2
        qw = (Rcw[0,2] - Rcw[2,0]) / S
        qx = (Rcw[0,1] + Rcw[1,0]) / S
        qy = 0.25 * S
        qz = (Rcw[1,2] + Rcw[2,1]) / S
    else:
        S = np.sqrt(1.0 + Rcw[2,2] - Rcw[0,0] - Rcw[1,1]) * 2
        qw = (Rcw[1,0] - Rcw[0,1]) / S
        qx = (Rcw[0,2] + Rcw[2,0]) / S
        qy = (Rcw[1,2] + Rcw[2,1]) / S
        qz = 0.25 * S
    qvec[0] = qx
    qvec[1] = qy
    qvec[2] = qz
    qvec[3] = qw
    return qvec