import os
import torch
import numpy as np
from joint_format import *

class convert2openpose():
    def __init__(self) -> None:
        self.smpl = None
        self.smplx = None
        self.openpose18 = None
        
        point_colors = []
        joint_pair_idxs = []
        for color_rgb in OPENPOSE_18JOINT_COLOR.values():
            point_colors.append(color_rgb)
        for joint_pair in OPENPOSE_18JOINT_PAIRS:
            joint_pair_idxs.append([OPENPOSE_18JOINT_MAP[joint_pair[0]],OPENPOSE_18JOINT_MAP[joint_pair[1]]])
            
        point_colors = np.asarray(point_colors)
        point_colors = np.divide(point_colors,255)
        joint_pair_idxs = np.asarray(joint_pair_idxs)
        joint_pair_colors = np.asarray(OPENPOSE_18JOINT_PAIRS_COLOR)
        joint_pair_colors = np.divide(joint_pair_colors,255)
        
    def convert(self):
        ...
    def norm_of_plane(self, v1, v2, v3, axis=1):
        from sklearn.preprocessing import normalize
        
        a = v1 - v2
        b = v2 - v3
        
        normal_vector = np.cross(a,b,axis=axis)
        
        normalize(normal_vector, axis=axis, copy=False)
            
        return normal_vector