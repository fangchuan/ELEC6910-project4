import torch
import numpy as np
from typing import Union, Dict


def _mutual_nn_matcher(descriptors1, descriptors2):
    # Mutual nearest neighbors (NN) matcher for L2 normalized descriptors.
    device = descriptors1.device
    sim = torch.matmul(descriptors1, descriptors2.t())
    nn_sim, nn12 = torch.max(sim, dim=1)
    nn_dist = torch.sqrt(2 - 2 * nn_sim)
    nn21 = torch.max(sim, dim=0)[1]
    ids1 = torch.arange(0, sim.shape[0], device=device)
    mask = ids1 == nn21[nn12]
    matches = torch.stack([ids1[mask], nn12[mask]])
    return matches.t(), nn_dist[mask]


def _ratio_matcher(descriptors1, descriptors2, ratio=0.8):
    # Lowe's ratio matcher for L2 normalized descriptors.
    device = descriptors1.device
    sim = torch.matmul(descriptors1, descriptors2.t())
    nns_sim, nns = torch.topk(sim, 2, dim=1)
    nns_dist = torch.sqrt(2 - 2 * nns_sim)
    ids1 = torch.arange(0, sim.shape[0], device=device)
    matches = torch.stack([ids1, nns[:, 0]])
    ratios = nns_dist[:, 0] / (nns_dist[:, 1] + 1e-8)
    mask = ratios <= ratio
    matches = matches[:, mask]
    return matches.t(), nns_dist[mask, 0]


def _ratio_mutual_nn_matcher(descriptors1, descriptors2, ratio=0.8):
    # Lowe's ratio matcher + mutual NN for L2 normalized descriptors.
    device = descriptors1.device
    sim = torch.matmul(descriptors1, descriptors2.t())
    nns_sim, nns = torch.topk(sim, 2, dim=1)
    nn12 = nns[:, 0]
    nns_dist = torch.sqrt(2 - 2 * nns_sim)
    nn21 = torch.max(sim, dim=0)[1]
    ids1 = torch.arange(0, sim.shape[0], device=device)
    matches = torch.stack([ids1, nns[:, 0]])
    ratios = nns_dist[:, 0] / (nns_dist[:, 1] + 1e-8)
    mask = torch.min(ids1 == nn21[nn12], ratios <= ratio)
    matches = matches[:, mask]
    return matches.t(), nns_dist[mask, 0]


def _similarity_matcher(descriptors1, descriptors2, threshold=0.9):
    # Similarity threshold matcher for L2 normalized descriptors.
    device = descriptors1.device
    sim = descriptors1 @ descriptors2.t()
    nn_sim, nn12 = torch.max(sim, dim=1)
    nn_dist = torch.sqrt(2 - 2 * nn_sim)
    nn21 = torch.max(sim, dim=0)[1]
    ids1 = torch.arange(0, sim.shape[0], device=device)
    mask = nn_sim >= threshold
    matches = torch.stack([ids1[mask], nn12[mask]])
    return matches.t(), nn_dist[mask]


def desc_match(desc1, desc2, device=None, params={"method": "mutual"}):
    """

    Args:
        desc1: feature at frame 1, dim: (N, C)
        desc2: feature at frame 2, dim: (M, C)
        device: device to be used for computing feature matching
        params: the configuration of matching, can be:
                {'method': 'mutual'},
                {'method': 'ratio', 'ratio':0.8}
                {'method': 'mutual_nn_ratio', 'ratio':0.8}
                {'method': 'similarity', 'threshold':0.7}
    Returns:

    """
    return_ndarray = isinstance(desc1, np.ndarray)

    if isinstance(desc1, np.ndarray):
        desc1 = torch.from_numpy(desc1)
    if isinstance(desc2, np.ndarray):
        desc2 = torch.from_numpy(desc2)

    assert desc1.shape[1] == desc2.shape[1]
    if device is not None:
        desc1 = desc1.to(device)
        desc2 = desc2.to(device)

    if params["method"] == "similarity":
        thres = 0.8 if "threshold" not in params else params["threshold"]
        matches, scores = _similarity_matcher(desc1, desc2, threshold=thres)
    elif params["method"] == "mutual_nn_ratio":
        ratio = 0.8 if "ratio" not in params else params["ratio"]
        matches, scores = _ratio_mutual_nn_matcher(desc1, desc2, ratio=ratio)
    elif params["method"] == "ratio":
        ratio = 0.8 if "ratio" not in params else params["ratio"]
        matches, scores = _ratio_mutual_nn_matcher(desc1, desc2, ratio=ratio)
    elif params["method"] == "mutual":
        matches, scores = _mutual_nn_matcher(desc1, desc2)

    if return_ndarray:
        matches = matches.cpu().numpy()
        scores = scores.cpu().numpy()

    return matches, scores


# Define the Customized Correspondences Matcher
from skimage.measure import ransac as _ransac
from skimage.transform import FundamentalMatrixTransform 

# We need enough correspondences
_MIN_FUNDAMENTAL_MAT_SAMPLES = 50 

_FeatType = Union[np.ndarray, torch.Tensor] 

class BaseFeatureMatcher:
    """The base feature matcher."""

    def __init__(self) -> None:
        pass

    def match(self, feat_a: _FeatType, feat_b: _FeatType, **kwargs):
        pass

class KNNMatcherWithGeometricVerification(BaseFeatureMatcher):
    default_cfg = {'method': 'mutual_nn_ratio', 'ratio':0.8}
    
    def __init__(self, cfg: Dict):
        self._cfg = cfg
        
    @staticmethod
    def geometry_filtering(kpt_a, kpt_b, residual_threshold=5, max_trials=200):
        _, inliers = _ransac(
            (kpt_a, kpt_b),
            FundamentalMatrixTransform,
            min_samples=_MIN_FUNDAMENTAL_MAT_SAMPLES,
            residual_threshold=residual_threshold,
            max_trials=max_trials)
        return inliers        
        
    def match(self, feat_a, feat_b, **kwargs):
                
        # Matching using K nearest search         
        corres, _ = desc_match(feat_a, feat_b, params=self._cfg)
        
        # Run geometry verification (find a valid fundamental matrix)         
        # to remove correspondences outliers         
        if 'kpt_uv_a' in kwargs and 'kpt_uv_b' in kwargs and corres.shape[0] > _MIN_FUNDAMENTAL_MAT_SAMPLES:
            # print(kwargs['kpt_uv_a'][corres[:, 0]].shape[0])

            inlier_mask = KNNMatcherWithGeometricVerification.geometry_filtering(
                kwargs['kpt_uv_a'][corres[:, 0]],
                kwargs['kpt_uv_b'][corres[:, 1]])
            corres = corres[inlier_mask, :]
            
        return corres