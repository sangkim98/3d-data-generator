import os
import torch
import smplx
import numpy as np
from convert2openpose import convert2openpose
from joint_format import SMPLX2OPENPOSE_IDX
import open3d as o3d

JOINTS_IDX = [0,3,6,9,10,11,13,14,15,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54]

class smplx2openpose(convert2openpose):
   def __init__(self, models_path: str, smplx_param_path: str, gender='male') -> None:
      super().__init__()
      
      def create_smplx_model(models_path: str, smplx_param_path: str, gender: str):
         model_name = f"SMPLX_{gender.upper()}.npz"
         smplx_model_template = smplx.SMPLX(models_path, gender='male')
         
         thuman_joints = np.load(smplx_param_path, allow_pickle=True)
         
         betas = smplx.utils.Tensor(torch.as_tensor(thuman_joints['betas']))
         pose = smplx.utils.Tensor(torch.as_tensor(thuman_joints['body_pose']))
         global_orientation = smplx.utils.Tensor(torch.as_tensor(thuman_joints['global_orient']))
         expression = smplx.utils.Tensor(torch.as_tensor(thuman_joints['expression']))
         jaw_pose = smplx.utils.Tensor(torch.as_tensor(thuman_joints['jaw_pose']))
         # lh_pose = smplx.utils.Tensor(torch.as_tensor(thuman_joints['left_hand_pose']))
         # rh_pose = smplx.utils.Tensor(torch.as_tensor(thuman_joints['right_hand_pose']))
         leye_pose = smplx.utils.Tensor(torch.as_tensor(thuman_joints['leye_pose']))
         reye_pose = smplx.utils.Tensor(torch.as_tensor(thuman_joints['reye_pose']))
         transl = smplx.utils.Tensor(torch.as_tensor(np.asarray(thuman_joints['translation']).reshape(1,3)))
         scale = thuman_joints['scale'][0]

         smplx_model = smplx_model_template.forward(betas=betas,
                                                    global_orient=global_orientation,
                                                    body_pose=pose,
                                                    transl=transl,
                                                    expression=expression,
                                                    jaw_pose=jaw_pose,
                                                   #  left_hand_pose=lh_pose,
                                                   #  right_hand_pose=rh_pose,
                                                    leye_pose=leye_pose,
                                                    reye_pose=reye_pose,
                                                   )

         return smplx_model, scale

      model, scale = create_smplx_model(models_path, smplx_param_path, gender)

      print(model.vertices.shape)

      joints_reshaped = model.joints.detach().numpy().reshape(127,3)
      
      joints = np.delete(joints_reshaped[:60,:], JOINTS_IDX, axis=0)
      
      self.joints = joints[SMPLX2OPENPOSE_IDX]
      self.scale = scale