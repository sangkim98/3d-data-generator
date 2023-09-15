import os
import numpy as np
import open3d as o3d
from joint_format import (
    MDM_JOINT_MAP, MDM_JOINT_PAIRS,
    OPENPOSE_JOINT_MAP, OPENPOSE_JOINT_PAIRS
)
from pathlib import Path
import colorutils


class mdmCustumVisualize():
    def __init__(self, npy_filepath: Path | str) -> None:
        if (type(npy_filepath) is str):
            npy_filepath = Path(npy_filepath)
        if not npy_filepath.exists():
            raise FileNotFoundError()

        mdm_data = np.load(npy_filepath, allow_pickle=True).item()

        # Initialize Data
        self.mdm_motion = mdm_data['motion']
        self.text = mdm_data['text']
        self.lengths = mdm_data['num_samples']
        self.num_samples = mdm_data['num_repetitions']
        self.num_repetitions = mdm_data['num_repetitions']

        # Convert Motion Diffusion Model (MDM) 22-keypoints to OpenPose 18-keypoints
        self.openpose_motion = self.convert_mdm2openpose()

    def visualize_all(self, keypoint_type: str):
        pass
        if keypoint_type == 'mdm':
            pass
        elif keypoint_type == 'coco-18':
            pass
    def convert_mdm2openpose(self):
        if self.mdm_motion.shape[3] == 22:
            def create_new_neck():
                """
                
                """
            
                left_shoulder = self.mdm_motion[:,MDM_JOINT_MAP['LeftShoulder'],:,:]
                right_shoulder = self.mdm_motion[:,MDM_JOINT_MAP['RightShoulder'],:,:]
                spine2 = self.mdm_motion[:,MDM_JOINT_MAP['Spine2'],:,:]
                
                self.mdm_motion[:,MDM_JOINT_MAP['Spine2'],:,:] = (left_shoulder+right_shoulder)*0.45 + spine2*0.1
                
            def create_eyes_ears():
                """
                
                """
                pass
            
                
            def adjust_uplegs2hips():
                """
                
                """
                left_upleg = self.mdm_motion[:,MDM_JOINT_MAP['LeftUpLeg'],:,:]
                right_upleg = self.mdm_motion[:,MDM_JOINT_MAP['RightUpLeg'],:,:]
                
                right_hip = (left_upleg*-0.1+right_upleg*1.1)
                left_hip = (left_upleg*1.1+right_upleg*-0.1)
                
                self.mdm_motion[:,MDM_JOINT_MAP['LeftUpLeg'],:,:] = left_hip
                self.mdm_motion[:,MDM_JOINT_MAP['RightUpLeg'],:,:] = right_hip
                
            create_new_neck()
            create_eyes_ears()
            adjust_uplegs2hips()
        elif self.mdm_motion.shape[3] == 18:
            print("Already in OpenPose format")
        else:
            print("Joint format not matching")
            
    def export_as_npy(self):
        pass