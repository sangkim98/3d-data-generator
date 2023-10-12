import os
import torch
import smplx
import numpy as np
from joint_format import SMPLX_JOINT_NAMES

JOINTS_IDX = [22,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54]

class smplx2openpose():
   def __init__(self, models_path: str, smplx_param_path: str, gender='neutral') -> None:
      def create_smplx_model(models_path: str, smplx_param_path: str, gender: str):
         model_name = f"SMPLX_{gender.upper()}.npz"
         smplx_model_template = smplx.SMPLX(os.path.join(models_path,model_name))
         
         thuman_joints = np.load(smplx_param_path, allow_pickle=True)
         
         betas = smplx.utils.Tensor(torch.as_tensor(thuman_joints['betas']))
         pose = smplx.utils.Tensor(torch.as_tensor(thuman_joints['body_pose']))
         global_orientation = smplx.utils.Tensor(torch.as_tensor(thuman_joints['global_orient']))
         expression = smplx.utils.Tensor(torch.as_tensor(thuman_joints['expression']))
         jaw_pose = smplx.utils.Tensor(torch.as_tensor(thuman_joints['jaw_pose']))
         leye_pose = smplx.utils.Tensor(torch.as_tensor(thuman_joints['leye_pose']))
         reye_pose = smplx.utils.Tensor(torch.as_tensor(thuman_joints['reye_pose']))

         smplx_model = smplx_model_template.forward(betas=betas,
                                                    global_orient=global_orientation,
                                                    body_pose=pose,
                                                    expression=expression,
                                                    jaw_pose=jaw_pose,
                                                    leye_pose=leye_pose,
                                                    reye_pose=reye_pose
                                                   )

         return smplx_model

      joints = create_smplx_model(models_path, smplx_param_path, gender).joints.detach().numpy()

      joints_reshaped = joints.reshape(127,3)
      
      self.joints = np.delete(joints_reshaped[:60,:], JOINTS_IDX, axis=0)
      
def main():
   test = smplx2openpose('/home/notingcode/Projects/3d_visualize/models/smplx', '/home/notingcode/Desktop/Thuman2_SMPLX/0000/smplx_param.pkl')
   print(test.joints.shape)
   
main()