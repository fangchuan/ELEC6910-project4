from typing import List
import numpy as np

class MapPoint:

    def __init__(self, position:np.ndarray, feature:np.ndarray, idx:int=0) -> None:
        assert position.shape == (1, 3)
        # assert feature.shape == (1, 256) if 

        # map point position
        self._map_pt_pos = position
        # map point feature
        self._map_pt_feat = feature
        # map point index
        self._map_pt_idx = idx
        # observed keyframe indices (frame index, keypoint index)
        self._map_pt_kf_obs = []
        
    @property
    def position(self) -> np.ndarray:
        return self._map_pt_pos
    @position.setter
    def position(self, new_pos:np.ndarray) -> None:
        if new_pos.shape == (3, 1):
            new_pos = new_pos.T
        self._map_pt_pos = new_pos

    @property
    def feature(self) -> np.ndarray:
        return self._map_pt_feat

    @property
    def observations(self) -> List:
        return self._map_pt_kf_obs
    
    @property
    def idx(self) -> int:
        return self._map_pt_idx
    
    def add_observation(self, frame_idx:int, keypt_idx:int) -> None:
        self._map_pt_kf_obs.append((frame_idx, keypt_idx))

    def observations(self) -> List:
        return self._map_pt_kf_obs