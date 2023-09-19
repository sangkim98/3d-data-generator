import os
import numpy as np
import open3d as o3d
from joint_format import *
from pathlib import Path
from sklearn.preprocessing import normalize


class mdmCustumVisualize():
    def __init__(self, npy_filepath: Path | str) -> None:
        if npy_filepath is not Path:
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
                
                new_neck = (left_shoulder+right_shoulder)*0.45 + spine2*0.1
                
                return new_neck
                
            def create_eyes_ears():
                """
                
                """
                
                def norm_of_3D_plane(v1, v2, v3):
                    """_summary_

                    Args:
                        v1 (_type_): _description_
                        v2 (_type_): _description_
                        v3 (_type_): _description_

                    Returns:
                        _type_: _description_
                    """
                    
                    a = v1 - v2
                    b = v2 - v3
                    
                    normal_vector = np.cross(a,b,axis=1)
                    
                    for frame in range(normal_vector.shape[-1]):
                        normal_vector[:,:,frame] = normalize(normal_vector[:,:,frame])
                        
                    return normal_vector
                
                head = self.mdm_motion[:,MDM_JOINT_MAP['Head'],:,:]
                neck = self.mdm_motion[:,MDM_JOINT_MAP['Neck'],:,:]
                spine2 = self.mdm_motion[:,MDM_JOINT_MAP['Spine2'],:,:]

                normal_vector = norm_of_3D_plane(head, neck, spine2)

                normal_vector_eyes = np.divide(normal_vector,20)
                normal_vector_ears = np.divide(normal_vector,13)

                left_eye = head - normal_vector_eyes
                right_eye = head + normal_vector_eyes

                left_ear = head - normal_vector_ears
                right_ear = head + normal_vector_ears
                
                norm_to_mv_eyes_ears = norm_of_3D_plane(neck, left_eye, right_eye)
                norm_to_mv_eyes = np.divide(norm_to_mv_eyes_ears, 22)
                norm_to_mv_ears = np.divide(norm_to_mv_eyes_ears, 20)
                
                left_eye = left_eye - norm_to_mv_eyes
                right_eye = right_eye - norm_to_mv_eyes
                
                left_ear = left_ear - norm_to_mv_ears
                right_ear = right_ear - norm_to_mv_ears
                
                norm_to_mv_eyes_ears_down = norm_of_3D_plane(head, left_eye, right_eye)
                # norm_to_mv_eyes_down = np.divide(norm_to_mv_eyes_ears_down, 30)
                norm_to_mv_ears_down = np.divide(norm_to_mv_eyes_ears_down, 13)
                
                # left_eye = left_eye - norm_to_mv_eyes_down
                # right_eye = right_eye - norm_to_mv_eyes_down
                
                left_ear = left_ear - norm_to_mv_ears_down
                right_ear = right_ear - norm_to_mv_ears_down
                
                return (right_eye, left_eye, right_ear, left_ear)            
                
            def create_rlHips():                
                """
                
                """
                
                left_upleg = self.mdm_motion[:,MDM_JOINT_MAP['LeftUpLeg'],:,:]
                right_upleg = self.mdm_motion[:,MDM_JOINT_MAP['RightUpLeg'],:,:]

                left_hip = (left_upleg*1.1+right_upleg*-0.1)                
                right_hip = (left_upleg*-0.1+right_upleg*1.1)
                
                return (right_hip, left_hip)
                
            new_neck = create_new_neck()
            (right_eye, left_eye, right_ear, left_ear) = create_eyes_ears()
            (right_hip, left_hip) = create_rlHips()
            
        elif self.mdm_motion.shape[3] == 18:
            print("Already in OpenPose format")
        else:
            print("Joint format not matching")
            
    def export_as_npy(self):
        pass