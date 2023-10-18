import os
import numpy as np
from convert2openpose import convert2openpose
from joint_format import *

class mdm2openpose(convert2openpose):
    def __init__(self, filepath: str) -> None:
        super().__init__()
        
        if not os.path.exists(filepath):
            raise FileNotFoundError()

        mdm_data = np.load(filepath, allow_pickle=True).item()

        # Initialize Data
        self.mdm_motion = mdm_data['motion']
        self.text = mdm_data['text']
        self.lengths = mdm_data['lengths']
        self.num_samples = mdm_data['num_repetitions']
        self.num_repetitions = mdm_data['num_repetitions']

        # Convert Motion Diffusion Model keypoints to OpenPose 18-keypoints
        self.openpose_motion = self.convert()
        
    def convert(self):
        if self.mdm_motion.shape[1] == 22:
            def norm_of_3D_plane(v1, v2, v3, num_frames: int):
                
                arrays_2d = []
                
                for frame in range(num_frames):
                    arrays_2d.append(self.norm_of_plane(v1[..., frame],
                                                        v2[..., frame],
                                                        v3[..., frame]
                                                        )
                                     )
                
                normal_vector = np.dstack(arrays_2d)
                    
                return normal_vector
            
            def create_new_neck():
                left_shoulder = self.mdm_motion[:,MDM_22JOINT_MAP['LeftShoulder'],:,:]
                right_shoulder = self.mdm_motion[:,MDM_22JOINT_MAP['RightShoulder'],:,:]
                spine2 = self.mdm_motion[:,MDM_22JOINT_MAP['Spine2'],:,:]
                
                new_neck = (left_shoulder+right_shoulder)*0.45 + spine2*0.1
                
                return new_neck
                
            def create_eyes_ears():   
                head = self.mdm_motion[:,MDM_22JOINT_MAP['Head'],:,:]
                neck = self.mdm_motion[:,MDM_22JOINT_MAP['Neck'],:,:]
                spine2 = self.mdm_motion[:,MDM_22JOINT_MAP['Spine2'],:,:]

                normal_vector = norm_of_3D_plane(head, neck, spine2, head.shape[-1])

                normal_vector_eyes = np.divide(normal_vector,20)
                normal_vector_ears = np.divide(normal_vector,13)

                left_eye = head - normal_vector_eyes
                right_eye = head + normal_vector_eyes

                left_ear = head - normal_vector_ears
                right_ear = head + normal_vector_ears
                
                norm_to_mv_eyes_ears = norm_of_3D_plane(neck, left_eye, right_eye, neck.shape[-1])
                norm_to_mv_eyes = np.divide(norm_to_mv_eyes_ears, 22)
                norm_to_mv_ears = np.divide(norm_to_mv_eyes_ears, 20)
                
                left_eye = left_eye - norm_to_mv_eyes
                right_eye = right_eye - norm_to_mv_eyes
                
                left_ear = left_ear - norm_to_mv_ears
                right_ear = right_ear - norm_to_mv_ears
                
                norm_to_mv_eyes_ears_down = norm_of_3D_plane(head, left_eye, right_eye, head.shape[-1])
                norm_to_mv_ears_down = np.divide(norm_to_mv_eyes_ears_down, 13)
                
                left_ear = left_ear - norm_to_mv_ears_down
                right_ear = right_ear - norm_to_mv_ears_down
                
                return (right_eye, left_eye, right_ear, left_ear)            
                
            def create_rlHips():
                left_upleg = self.mdm_motion[:,MDM_22JOINT_MAP['LeftUpLeg'],:,:]
                right_upleg = self.mdm_motion[:,MDM_22JOINT_MAP['RightUpLeg'],:,:]

                left_hip = (left_upleg*1.1+right_upleg*-0.1)                
                right_hip = (left_upleg*-0.1+right_upleg*1.1)
                
                return (right_hip, left_hip)
                
            new_neck = create_new_neck()
            (right_eye, left_eye, right_ear, left_ear) = create_eyes_ears()
            (right_hip, left_hip) = create_rlHips()
            
            data_shape = self.mdm_motion.shape
            
            openpose_data_shape = (data_shape[0], 18, data_shape[2], data_shape[3])
            
            openpose_data_edited = np.empty(openpose_data_shape, dtype=self.mdm_motion.dtype)
            
            new_joints = {
                'Neck': new_neck,
                'REye': right_eye, 'LEye': left_eye,
                'REar': right_ear, 'LEar': left_ear,
                'RHip': right_hip, 'LHip': left_hip
            }
            
            for mdm_joint, openpose_joint in MDM2OPENPOSE_KEYVAL.items():
                if openpose_joint != REMOVE:
                    openpose_data_edited[:,OPENPOSE_18JOINT_MAP[openpose_joint],:,:] = self.mdm_motion[:,MDM_22JOINT_MAP[mdm_joint],:,:]
                
            for new_openpose_joint, data in new_joints.items():
                openpose_data_edited[:,OPENPOSE_18JOINT_MAP[new_openpose_joint],:,:] = data
                
                    
            return openpose_data_edited
                    
        elif self.mdm_motion.shape[1] == 18:
            print("Already in OpenPose format")
        else:
            print("Joint format not matching")
            
    def export_as_npy(self, type: str):
        if type == 'mdm':
            pass
        elif type == 'openpose':
            pass